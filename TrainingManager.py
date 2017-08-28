# Class that manages the training loop. This class will also deal with logging
import NetworkManager
import time, copy, random # TODO change to numpy random
import BatchHandler
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import sys


class TrainingManager:
    def __init__(self, cf_pool, test_pool, parameter_dict):
        self.cf_pool = cf_pool
        self.test_pool = test_pool
        self.parameter_dict = parameter_dict
        self.hyper_results_logfile = "hyper.csv"

        return

    def train_network(self,netManager,training_batch_handler,validation_batch_handler,hyper_search=False):
        fold_time = time.time()
        current_step = 0
        previous_losses = []
        previous_val_losses = []
        step_time, loss = 0.0, 0.0
        steps_per_checkpoint = self.parameter_dict['steps_per_checkpoint']
        print "Starting Network training for:"
        print str(self.parameter_dict)
        overfitting_steps = 0
        final_run = False
        training_log_df = pd.DataFrame()

        while True:

            #### TRAINING
            if not final_run:
                step_start_time = time.time()
                batch_frame = training_batch_handler.get_minibatch()
                # print "Time to get batch: " + str(time.time()-step_start_time)
                train_x, _, weights, train_y = training_batch_handler.format_minibatch_data(batch_frame['encoder_sample'],
                                                                                            batch_frame['dest_1_hot'],
                                                                                            batch_frame['padding'])
                accuracy, step_loss, _ = netManager.run_training_step(train_x, train_y, weights, True)
                # print "Time to step: " + str(time.time() - step_start_time)

                # Periodically, run without training for the summary logs
                # This will always run in the same loop as the checkpoint fn below.
                # Explicit check in case of rounding errors
                step_time += (time.time() - step_start_time) / steps_per_checkpoint
                loss += step_loss / steps_per_checkpoint
                current_step += 1

            #### TENSORBOARD LOGGING
            if current_step % (steps_per_checkpoint/10) == 0 or \
                current_step % steps_per_checkpoint == 0 or \
                final_run:
                train_acc, train_step_loss, _ = netManager.run_training_step(train_x, train_y, weights, False,
                                                                                summary_writer=netManager.train_writer)
                #val_time = time.time()
                val_accuracy, val_step_loss, _ = netManager.run_validation(validation_batch_handler,
                                                                        summary_writer=netManager.val_writer,
                                                                           quick=(not final_run))
                #print "valbatch Time: " + str(time.time()-val_time)

            #### EVALUATION / CHECKPOINTING
            sys.stdout.write("\rg_step %06d " % (current_step))
            sys.stdout.flush()
            if (current_step % steps_per_checkpoint == 0) or final_run:
                sys.stdout.write("\rg_step %06d lr %.1e step %.4f avTL %.4f VL %.4f Acc %.3f v_acc %.3f "
                       % (netManager.get_global_step(),
                          netManager.get_learning_rate(),
                          step_time, loss, val_step_loss, accuracy, val_accuracy))
                sys.stdout.flush()

                # TODO make this run every n minutes, not a multiple of steps. Also add duration reporting to console
                if ((not self.parameter_dict['debug']) and current_step % (steps_per_checkpoint*10) == 0) or final_run:
                    # Compute Distance Metric
                    dist_results = netManager.compute_result_per_dis(validation_batch_handler, plot=False)
                    #f1_scores = netManager.compute_distance_f1_report(dist_results)
                    metric_results, metric_labels = netManager.evaluate_pdis_metric(dist_results)

                    metric_string = " "
                    for metric_idx in range(len(metric_results)):
                        metric_string += metric_labels[metric_idx][0]
                        metric_string += "%0.1f " % metric_results[metric_idx]
                    # DOn't log hyper search graphs, it explodes the log directory.
                    if not hyper_search:
                        graphs = netManager.draw_png_graphs_perf_dist(dist_results)
                        netManager.log_graphs_to_tensorboard(graphs)
                    netManager.log_metric_to_tensorboard(metric_results)
                    sys.stdout.write("p_dis" + metric_string)
                sys.stdout.write("\r\n")
                sys.stdout.flush()

                netManager.checkpoint_model()

                # Log all things
                results_dict = {'g_step':netManager.get_global_step(),
                                'training_loss':train_step_loss,
                                'training_acc':train_acc,
                                'validation_loss':val_step_loss,
                                'validation_acc':val_accuracy}
                training_log_df = training_log_df.append(results_dict,ignore_index=True)

                ### Decay learning rate checks
                if (len(previous_losses) > self.parameter_dict['decrement_steps']-1
                        and
                        loss > 0.99*(max(previous_losses))): #0.95 is float fudge factor
                    netManager.decay_learning_rate()
                    previous_losses = []
                previous_losses.append(loss)
                previous_losses = previous_losses[-self.parameter_dict['decrement_steps']:]
                previous_val_losses.append(val_step_loss)
                previous_val_losses = previous_val_losses[-self.parameter_dict['decrement_steps']:]

                ##### Training stop conditions:
                if final_run:
                    break
                # Check for significant divergence of val_loss and train_loss
                model_is_overfit = False
                if (loss < (val_step_loss)*0.9 and  # train / val have diverged
                    val_step_loss > 0.95*max(previous_val_losses)):  # val is ~increasing
                    overfitting_steps += 1
                    print "Warning, overfitting detected. Will stop training if it continues"
                    if overfitting_steps > 20:
                        model_is_overfit = True
                else:
                    overfitting_steps = 0

                learning_rate_too_low = (netManager.get_learning_rate() <
                                         self.parameter_dict['loss_decay_cutoff'] *
                                         self.parameter_dict['learning_rate'])
                out_of_time = time.time() - fold_time > 60 * self.parameter_dict['training_early_stop']

                if learning_rate_too_low or out_of_time or model_is_overfit:
                    # Lookup best model based on val_step_loss
                    # Load best model.
                    # Run one more loop for final network scores
                    best_g_step = training_log_df.sort_values('validation_loss',ascending=True).iloc[0].g_step
                    print "FINAL RUN, Best model was at step: " + str(best_g_step)
                    netManager.load_from_checkpoint(best_g_step)
                    netManager.clean_checkpoint_dir(best_g_step)
                    final_run = True

                step_time, loss = 0.0, 0.0

        fold_results = copy.copy(self.parameter_dict)
        fold_results['input_columns'] = ",".join(fold_results['input_columns'])
        fold_results['eval_accuracy'] = train_acc
        fold_results['final_learning_rate'] = netManager.get_learning_rate()
        fold_results['training_accuracy'] = accuracy
        fold_results['training_loss'] = train_step_loss
        fold_results['network_chkpt_dir'] = netManager.log_file_name
        fold_results['validation_accuracy'] = val_accuracy
        fold_results['validation_loss'] = val_step_loss

        for class_idx in range(len(metric_results)):
            key_str = 'perfect_distance_' + str(class_idx)
            fold_results[key_str] = metric_results[class_idx]

        fold_results['perfect_distance'] = np.max(metric_results) #worst distance

        return fold_results

    def test_network(self,netManager,test_batch_handler):
        # Function that takes the currently built network and runs the test data through it (each data point is run once
        #  and only once). Graphs are generated. Make it easy to generate many graphs as this will be helpful for the
        # sequence generation model

        test_accuracy, test_loss, _ = netManager.run_validation(test_batch_handler,quick=False)

        return test_accuracy, test_loss

    def run_hyperparameter_search(self):
        hyperparam_results_list = []
        hyper_time = time.time()
        self.parameter_dict['training_early_stop'] = self.parameter_dict['early_stop_cf']
        training_batch_handler_cache = {}
        validation_batch_handler_cache = {}
        while True:

            #Select new hyperparameters
            if self.parameter_dict['hyper_learning_rate_args'] is not None:
                self.parameter_dict['learning_rate'] = \
                    10 ** self.parameter_dict['hyper_learning_rate_fn'](
                        *self.parameter_dict['hyper_learning_rate_args'])
            if self.parameter_dict['hyper_rnn_size_args'] is not None:
                self.parameter_dict['rnn_size'] = \
                    int(self.parameter_dict['hyper_rnn_size_fn'](*self.parameter_dict['hyper_rnn_size_args']))
            if self.parameter_dict['hyper_reg_embedding_beta_args'] is not None:
                self.parameter_dict['reg_embedding_beta'] = \
                    10 ** self.parameter_dict['hyper_reg_embedding_beta_fn'](
                        *self.parameter_dict['hyper_reg_embedding_beta_args'])
            if self.parameter_dict['hyper_reg_l2_beta_args'] is not None:
                self.parameter_dict['l2_reg_beta'] = \
                    10 ** self.parameter_dict['hyper_reg_l2_beta_fn'](
                        *self.parameter_dict['hyper_reg_l2_beta_args'])

            # TODO obs steps needs to load a new dataset every time, as the dataset has a fixed step size
            # Actually it can just use the most recent t steps, but the dataset loaded needs to have the most encoder
            # steps
            #self.parameter_dict["observation_steps"] = random.choice(timestep_range)

            cf_fold = -1
            # I should call this outside the crossfold, so it occurs once
            # This way all the crossfolds for the same hyperparameters are adjacent in the checkpoint dirs
            log_file_time = str(time.time())
            cf_results_list = []

            for train_pool, val_pool in self.cf_pool:
                cf_fold += 1
                log_file_name = log_file_time + "-cf-" + str(cf_fold)

                print "Starting crossfold"
                try:
                    training_batch_handler = training_batch_handler_cache[hash(tuple(np.sort(train_pool.uniqueId.unique())))]
                except KeyError:
                    training_batch_handler = BatchHandler.BatchHandler(train_pool, self.parameter_dict, True)
                    training_batch_handler_cache[hash(tuple(np.sort(train_pool.uniqueId.unique())))] = training_batch_handler

                try:
                    validation_batch_handler = validation_batch_handler_cache[hash(tuple(np.sort(val_pool.uniqueId.unique())))]
                except KeyError:
                    validation_batch_handler = BatchHandler.BatchHandler(val_pool, self.parameter_dict, False)
                    validation_batch_handler_cache[hash(tuple(np.sort(val_pool.uniqueId.unique())))] = validation_batch_handler

                # Add input_size, num_classes
                self.parameter_dict['input_size'] = training_batch_handler.get_input_size()
                self.parameter_dict['num_classes'] = training_batch_handler.get_num_classes()

                netManager = NetworkManager.NetworkManager(self.parameter_dict, log_file_name)
                netManager.build_model()
                try:
                    cf_results = self.train_network(netManager,training_batch_handler,validation_batch_handler,hyper_search=True)
                except tf.errors.InvalidArgumentError:
                    print "**********************caught error, probably gradients have exploded"
                    continue

                cf_results['crossfold_number'] = cf_fold
                # As pandas does not like lists when adding a list to a row of a dataframe, set to None (the lists are
                # a large amount of redundant data). This is why I copy out parameters.py
                for key, value in cf_results.iteritems():
                    if (type(value) is list or
                         type(value) is np.ndarray or
                                type(value) is tuple):
                                  cf_results[key] = pd.Series([value],dtype=object)
                cf_results_list.append(pd.DataFrame(cf_results, index=[0]))

                # plot
                print "Drawing html graph"
                netManager.draw_html_graphs(validation_batch_handler)
                # netManager.draw_html_graphs(
                #     netManager.compute_distance_f1_report(
                #         netManager.compute_result_per_dis(
                #             validation_batch_handler)))

                #######
                # Here we have a fully trained model, but we are still in the cross fold.

                # FIXME Only do 1 fold per hyperparams. Its not neccessary to continue
                break

            cf_df = pd.concat(cf_results_list)
            # Condense results from cross fold (Average, best, worst, whatever selection method)
            hyperparam_results = copy.copy(self.parameter_dict)
            hyperparam_results['input_columns'] = ",".join(hyperparam_results['input_columns'])
            hyperparam_results['eval_accuracy'] = np.min(cf_df['eval_accuracy'])
            hyperparam_results['final_learning_rate'] = np.min(cf_df['final_learning_rate'])
            hyperparam_results['training_accuracy'] = np.min(cf_df['training_accuracy'])
            hyperparam_results['training_loss'] = np.average(cf_df['training_loss'])
            hyperparam_results['validation_accuracy'] = np.average(cf_df['validation_accuracy'])
            hyperparam_results['validation_loss'] =np.average(cf_df['validation_loss'])

            hyperparam_results['crossfold_number'] = -1
            #FIXME What is this line doing?
            hyperparam_results['network_chkpt_dir'] = (
                cf_df.sort_values('eval_accuracy',ascending=False).iloc[[0]]['network_chkpt_dir'])
            hyperparam_results['cf_summary'] = True
            for key, value in hyperparam_results.iteritems():
                if (type(value) is list or
                            type(value) is np.ndarray or
                            type(value) is tuple):
                    hyperparam_results[key] = pd.Series([value],dtype=object)  # str(cf_results[key])
            hyperparam_results_list.append(pd.DataFrame(hyperparam_results, index=[0]))
            hyperparam_results_list.append(cf_df)
            #Write results and hyperparams to hyperparameter_results_dataframe

            #Once cross folding has completed, select new hyperparameters, and re-run
            now = time.time()
            if now - hyper_time > 60 * 60 * self.parameter_dict['hyper_search_time']: # Stop the hyperparameter search after a set time
                break

        hyper_df = pd.concat(hyperparam_results_list,ignore_index=True)
        hyper_df.to_csv(os.path.join(self.parameter_dict['master_dir'],self.hyper_results_logfile))
        summary_df = hyper_df[hyper_df['cf_summary']==True]

        # Distance at which the classifier can make a sound judgement, lower is better
        if self.parameter_dict['evaluation_metric_type'] == 'perfect_distance':
            best_params = summary_df.sort_values('perfect_distance',ascending=True).iloc[0].to_dict()
        if self.parameter_dict['evaluation_metric_type'] == 'validation_accuracy': # Higher better
            best_params = summary_df.sort_values('validation_accuracy', ascending=False).iloc[0].to_dict()
        if self.parameter_dict['evaluation_metric_type'] == 'validation_loss': # Lower better
            best_params = summary_df.sort_values('validation_loss', ascending=True).iloc[0].to_dict()

        return best_params

    def long_train_network(self, params, train_pool, val_pool, test_pool, checkpoint=None, test_network_only=False):
        self.parameter_dict = params

        # Run for many minutes, or until loss decays significantly.
        self.parameter_dict['training_early_stop'] = self.parameter_dict['long_training_time']

        if checkpoint is not None:
            log_file_name = checkpoint
        else:
            log_file_name = "best-" + str(time.time())

        training_batch_handler = BatchHandler.BatchHandler(train_pool, self.parameter_dict, True)
        validation_batch_handler = BatchHandler.BatchHandler(val_pool, self.parameter_dict, False)
        test_batch_handler = BatchHandler.BatchHandler(test_pool, self.parameter_dict, False)

        # Add input_size, num_classes
        self.parameter_dict['input_size'] = training_batch_handler.get_input_size()
        self.parameter_dict['num_classes'] = training_batch_handler.get_num_classes()

        netManager = NetworkManager.NetworkManager(self.parameter_dict, log_file_name)
        netManager.build_model()

        if not test_network_only:
            best_results = self.train_network(netManager,training_batch_handler,validation_batch_handler)
        else:
            best_results = {}
        best_results['test_accuracy'], best_results['test_loss'] = self.test_network(netManager,test_batch_handler)

        print "Drawing html graph"
        #netManager.draw_html_graphs(netManager.compute_result_per_dis(test_batch_handler))

        netManager.draw_html_graphs(test_batch_handler)

        # FIXME maybe this needs its own function?
        for key, value in best_results.iteritems():
            if (type(value) is list or
                        type(value) is np.ndarray or
                        type(value) is tuple):
                best_results[key] = pd.Series([value], dtype=object)
        best_results = pd.DataFrame(best_results,index=[0])
        if not test_network_only:
            best_results.to_csv(os.path.join(self.parameter_dict['master_dir'],"best.csv"))

        return best_results

