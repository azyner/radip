# Class that manages the training loop. This class will also deal with logging
import NetworkManager
import time, copy, random # TODO change to numpy random
import BatchHandler
import numpy as np
import pandas as pd
import os
import tensorflow as tf


class TrainingManager:
    def __init__(self, cf_pool, test_pool, parameter_dict):
        self.cf_pool = cf_pool
        self.test_pool = test_pool
        self.parameter_dict = parameter_dict
        self.hyper_results_logfile = "hyper.csv"

        return


    def train_network(self,netManager,training_batch_handler,validation_batch_handler):
        fold_time = time.time()
        current_step = 0
        previous_losses = []
        step_time, loss = 0.0, 0.0
        steps_per_checkpoint = self.parameter_dict['steps_per_checkpoint']
        print "Starting Network training for:"
        print str(self.parameter_dict)
        while True:
            # The training loop!

            step_start_time = time.time()
            batch_frame = training_batch_handler.get_minibatch()
            # print "Time to get batch: " + str(time.time()-step_start_time)
            step_start_time = time.time()
            train_x, _, weights, train_y = training_batch_handler.format_minibatch_data(batch_frame['encoder_sample'],
                                                                                        batch_frame['dest_1_hot'],
                                                                                        batch_frame['padding'])
            accuracy, step_loss, _ = netManager.run_training_step(train_x, train_y, weights, True)
            # print "Time to step: " + str(time.time() - step_start_time)

            # Periodically, run without training for the summary logs
            # This will always run in the same loop as the checkpoint fn below.
            # Explicit check in case of rounding errors
            if current_step % (steps_per_checkpoint/10) == 0 or \
                current_step % steps_per_checkpoint == 0:
                train_acc, train_step_loss, _ = netManager.run_training_step(train_x, train_y, weights, False,
                                                                                summary_writer=netManager.train_writer)
                val_accuracy, val_step_loss, _ = netManager.run_validation(validation_batch_handler,
                                                                                             summary_writer=netManager.val_writer,quick=True)
            step_time += (time.time() - step_start_time) / steps_per_checkpoint
            step_time += (time.time() - step_start_time) / steps_per_checkpoint
            loss += step_loss / steps_per_checkpoint
            current_step += 1

            if current_step % steps_per_checkpoint == 0:

                # eval_accuracy, eval_step_loss, _ = netManager.run_validation(validation_batch_handler,
                #                                                              summary_writer=netManager.val_writer,quick=True)
                # graph_results = netManager.collect_graph_data(validation_batch_handler)
                # netManager.draw_graphs(graph_results)

                dist_results = netManager.compute_result_per_dis(validation_batch_handler, plot=False)
                metric_results = netManager.evaluate_metric(dist_results)
                metric_string = ""
                for value in metric_results:
                    metric_string += " %0.1f" % value

                print ("g_step %d lr %.6f step %.4fs av tr loss %.4f Acc %.3f v_acc %.3f p_dis"
                       % (netManager.get_global_step(),
                          netManager.get_learning_rate(),
                          step_time, loss, accuracy, train_acc) + metric_string)

                graphs = netManager.draw_png_graphs(dist_results)

                netManager.log_graphs_to_tensorboard(graphs)
                netManager.log_metric_to_tensorboard(metric_results)

                decrement_timestep = self.parameter_dict['decrement_steps']
                if (len(previous_losses) > decrement_timestep-1
                        and
                        loss > 0.99*(max(previous_losses))): #0.95 is float fudge factor
                    netManager.decay_learning_rate()
                    previous_losses = []

                if current_step % (steps_per_checkpoint*5) == 0:
                    # at 25M per checkpoint, don't do this too often
                    netManager.checkpoint_model()

                previous_losses.append(loss)
                previous_losses = previous_losses[-decrement_timestep:]
                step_time, loss = 0.0, 0.0

                # Training stop conditions:
                # Out of time
                # Out of learning rate
                now = time.time()
                if ((netManager.get_learning_rate()
                         <
                             self.parameter_dict['loss_decay_cutoff'] * self.parameter_dict['learning_rate'])
                    or
                    now - fold_time > 60 * self.parameter_dict['training_early_stop']):
                    break

        fold_results = copy.copy(self.parameter_dict)
        fold_results['input_columns'] = ",".join(fold_results['input_columns'])
        fold_results['eval_accuracy'] = train_acc
        fold_results['final_learning_rate'] = netManager.get_learning_rate()
        fold_results['training_accuracy'] = accuracy
        fold_results['training_loss'] = loss
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
        while True:

            #Select new hyperparameters
            if self.parameter_dict['hyper_learning_rate_args'] is not None:
                self.parameter_dict['learning_rate'] = \
                    10**self.parameter_dict['hyper_learning_rate_fn'](*self.parameter_dict['hyper_learning_rate_args'])
            if self.parameter_dict['hyper_rnn_size_args'] is not None:
                self.parameter_dict['rnn_size'] = \
                    int(self.parameter_dict['hyper_rnn_size_fn'](*self.parameter_dict['hyper_rnn_size_args']))

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
                training_batch_handler = BatchHandler.BatchHandler(train_pool, self.parameter_dict, True)
                validation_batch_handler = BatchHandler.BatchHandler(val_pool, self.parameter_dict, False)

                # Add input_size, num_classes
                self.parameter_dict['input_size'] = training_batch_handler.get_input_size()
                self.parameter_dict['num_classes'] = training_batch_handler.get_num_classes()

                netManager = NetworkManager.NetworkManager(self.parameter_dict, log_file_name)
                netManager.build_model()
                try:
                    cf_results = self.train_network(netManager,training_batch_handler,validation_batch_handler)
                except tf.errors.InvalidArgumentError:
                    print "**********************caugt error, probably gradients have exploded"
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

                #plot
                print "Drawing html graph"
                netManager.draw_html_graphs(netManager.compute_result_per_dis(validation_batch_handler))

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

    def long_train_network(self, params, train_pool, val_pool, test_pool):
        self.parameter_dict = params

        # Run for many minutes, or until loss decays significantly.
        self.parameter_dict['training_early_stop'] = self.parameter_dict['long_training_time']

        log_file_name = "best-" + str(time.time())

        training_batch_handler = BatchHandler.BatchHandler(train_pool, self.parameter_dict, True)
        validation_batch_handler = BatchHandler.BatchHandler(val_pool, self.parameter_dict, False)
        test_batch_handler = BatchHandler.BatchHandler(val_pool, self.parameter_dict, False)

        # Add input_size, num_classes
        self.parameter_dict['input_size'] = training_batch_handler.get_input_size()
        self.parameter_dict['num_classes'] = training_batch_handler.get_num_classes()

        netManager = NetworkManager.NetworkManager(self.parameter_dict, log_file_name)
        netManager.build_model()

        best_results = self.train_network(netManager,training_batch_handler,validation_batch_handler)
        best_results['test_accuracy'], best_results['test_loss'] = self.test_network(netManager,test_batch_handler)

        print "Drawing html graph"
        netManager.draw_html_graphs(netManager.compute_result_per_dis(test_batch_handler))

        # FIXME maybe this needs its own function?
        for key, value in best_results.iteritems():
            if (type(value) is list or
                        type(value) is np.ndarray or
                        type(value) is tuple):
                best_results[key] = pd.Series([value], dtype=object)
        best_results = pd.DataFrame(best_results,index=[0])
        best_results.to_csv(os.path.join(self.parameter_dict['master_dir'],"best.csv"))
        # Do it all again, but this time train with all data OR TODO return from best checkpoint
        # and test against that last test set
        # I guess this is where the HTML plots would be generated

        return best_results

