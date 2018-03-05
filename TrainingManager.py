# Class that manages the training loop. This class will also deal with logging
import NetworkManager
import time, copy, random # TODO change to numpy random
import BatchHandler
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import sys
import signal
import ReportWriter

class TrainingManager:
    def __init__(self, cf_pool, test_pool, encoder_means, encoder_stddev, parameter_dict):
        self.cf_pool = cf_pool
        self.test_pool = test_pool
        self.parameter_dict = parameter_dict
        self.hyper_results_logfile = "hyper.csv"
        self.encoder_means = encoder_means
        self.encoder_stddev = encoder_stddev
        self.sigint_caught = False
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

        # Define and register a SIGINT handler here
        original_sigint_handler = signal.getsignal(signal.SIGINT)


        def sigint_handler(signum, frame):
            print "TrainingManager caught SIGINT. Stopping training, writing report, and exiting."
            self.sigint_caught = True

        signal.signal(signal.SIGINT, sigint_handler)
        loss_a = []
        val_step_loss_a = []
        accuracy_a = []
        val_accuracy_a = []

        while True:

            #### TRAINING
            if not final_run:
                step_start_time = time.time()
                batch_frame = training_batch_handler.get_minibatch()
                # print "Time to get batch: " + str(time.time()-step_start_time)

                train_x, train_future, weights, train_labels, track_padded = \
                training_batch_handler.format_minibatch_data(
                    batch_frame['encoder_sample'],
                    batch_frame['dest_1_hot'] if self.parameter_dict['model_type'] == 'classifier' else
                    batch_frame['decoder_sample'] if self.parameter_dict['model_type'] == 'MDN' else exit(2),
                    batch_frame['batchwise_padding'],
                batch_frame['trackwise_padding'] if self.parameter_dict['track_padding'] else None)
                train_y = train_labels if self.parameter_dict['model_type'] == 'classifier' else \
                          train_future if self.parameter_dict['model_type'] == 'MDN' else exit(3)

                accuracy, step_loss, _, _, _ = netManager.run_training_step(train_x, train_y, weights,
                                                                         True, track_padded)
                # print "Time to step: " + str(time.time() - step_start_time)

                # Periodically, run without training for the summary logs
                # This will always run in the same loop as the checkpoint fn below.
                # Explicit check in case of rounding errors
                step_time += (time.time() - step_start_time) / steps_per_checkpoint
                loss += step_loss / steps_per_checkpoint
                current_step += 1
                netManager.decay_learning_rate()  # decay every step by 0.9999 as per sketchrnn


            #### TENSORBOARD LOGGING
            if current_step % (steps_per_checkpoint/2) == 0 or \
                current_step % steps_per_checkpoint == 0 or \
                final_run:
                train_acc, train_step_loss, _, _, _ = netManager.run_training_step(train_x, train_y, weights, False,
                                                                                track_padded,
                                                                                summary_writer=netManager.train_writer)
                #val_time = time.time()
                val_accuracy, val_step_loss, _, _ = netManager.run_validation(validation_batch_handler,
                                                                        summary_writer=netManager.val_writer,
                                                                           quick=(not final_run))
                loss_a.append(train_step_loss)
                val_step_loss_a.append(val_step_loss)
                accuracy_a.append(accuracy)
                val_accuracy_a.append(val_accuracy)
                sys.stdout.write("\rg_step %06d lr %.1e step %.4f avTL %.4f VL %.4f Acc %.3f v_acc %.3f "
                       % (netManager.get_global_step(),
                          netManager.get_learning_rate(),
                          step_time, np.mean(loss_a), np.mean(val_step_loss_a), np.mean(accuracy_a), np.mean(val_accuracy_a)))
                sys.stdout.flush()

                #print "valbatch Time: " + str(time.time()-val_time)

            #### EVALUATION / CHECKPOINTING
            sys.stdout.write("\rg_step %06d " % (current_step))
            sys.stdout.flush()
            if (current_step % steps_per_checkpoint == 0) or final_run:
                sys.stdout.write("\rg_step %06d lr %.1e step %.4f avTL %.4f VL %.4f Acc %.3f v_acc %.3f "
                       % (netManager.get_global_step(),
                          netManager.get_learning_rate(),
                          step_time, np.mean(loss_a), np.mean(val_step_loss_a), np.mean(accuracy_a), np.mean(val_accuracy_a)))
                sys.stdout.flush()

                # TODO make this run every n minutes, not a multiple of steps. Also add duration reporting to console
                if (((not self.parameter_dict['debug']) and current_step % (steps_per_checkpoint*10) == 0) or final_run)\
                        and self.parameter_dict['model_type'] == 'classifier':
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
                        graphs = netManager.draw_categorical_png_graphs_perf_dist(dist_results)
                        #netManager.log_graphs_to_tensorboard(graphs)
                    netManager.log_metric_to_tensorboard(metric_results)
                    sys.stdout.write("p_dis" + metric_string)
                elif (((not self.parameter_dict['debug']) and current_step % (steps_per_checkpoint*10) == 0) or final_run)\
                    and self.parameter_dict['model_type'] == 'MDN':
                    #print "Write PNG graphing functions here."
                    netManager.draw_generative_png_graphs(validation_batch_handler,multi_sample=1, final_run=final_run)
                    netManager.draw_generative_png_graphs(validation_batch_handler, multi_sample=20,
                                                          draw_prediction_track=False, final_run=final_run)
                    # I rarely use this, and now the multithreader cannot return a value if it is backgrounded.
                    #netManager.log_graphs_to_tensorboard(graphs)
                    metric_results = -999
                sys.stdout.write("\r\n")
                sys.stdout.flush()

                netManager.checkpoint_model()

                # Log all things
                results_dict = {'g_step':netManager.get_global_step(),
                                'training_loss':np.mean(loss_a),
                                'training_acc':np.mean(accuracy_a),
                                'validation_loss':np.mean(val_step_loss_a),
                                'validation_acc':np.mean(val_accuracy_a)}
                training_log_df = training_log_df.append(results_dict,ignore_index=True)

                ### Decay learning rate checks

                # if (len(previous_losses) > self.parameter_dict['decrement_steps']-1
                #         and
                #         loss > 0.99*(max(previous_losses))): #0.95 is float fudge factor
                #     netManager.decay_learning_rate()
                #     previous_losses = []
                # previous_losses.append(loss)
                # previous_losses = previous_losses[-self.parameter_dict['decrement_steps']:]
                previous_val_losses.append(val_step_loss)
                previous_val_losses = previous_val_losses[-self.parameter_dict['decrement_steps']:]

                ##### Training stop conditions:
                if final_run:
                    break
                # Check for significant divergence of val_loss and train_loss
                model_is_overfit = False
                if (loss < (val_step_loss)*0.9 and  # train / val have diverged
                    val_step_loss > 0.95*max(previous_val_losses) and
                    not self.parameter_dict['first_loss_only']):  # val is ~increasing
                    overfitting_steps += 1
                    print "Warning, overfitting detected. Will stop training if it continues"
                    if overfitting_steps > 20:
                        model_is_overfit = True
                else:
                    overfitting_steps = 0

                learning_rate_too_low = (netManager.get_learning_rate() <
                                         self.parameter_dict['loss_decay_cutoff'] *
                                         self.parameter_dict['learning_rate'])
                if learning_rate_too_low:
                    print "Stopping due to low learning rate"
                out_of_time = time.time() - fold_time > 60 * self.parameter_dict['training_early_stop']
                if out_of_time:
                    print "Stopping due to time cutoff"
                out_of_steps = (self.parameter_dict['long_training_steps'] is not None and
                                current_step > self.parameter_dict['long_training_steps'])
                if out_of_steps:
                    print "Stopping due to step cutoff"

                if learning_rate_too_low or out_of_time or model_is_overfit or out_of_steps or self.sigint_caught:
                    # Lookup best model based on val_step_loss
                    # Load best model.
                    # Run one more loop for final network scores
                    best_g_step = training_log_df.sort_values('validation_loss',ascending=True).iloc[0].g_step
                    print "FINAL RUN, Best model was at step: " + str(best_g_step)
                    netManager.load_from_checkpoint(best_g_step)
                    netManager.clean_checkpoint_dir(best_g_step)
                    final_run = True

                step_time, loss = 0.0, 0.0
                val_step_loss_a = []
                accuracy_a = []
                val_accuracy_a = []

        # Now restore old signal handler so that the sig capture function doesn't fall out of scope.
        signal.signal(signal.SIGINT, original_sigint_handler)

        fold_results = copy.copy(self.parameter_dict)
        fold_results['input_columns'] = ",".join(fold_results['input_columns'])
        fold_results['eval_accuracy'] = train_acc
        fold_results['final_learning_rate'] = netManager.get_learning_rate()
        fold_results['training_accuracy'] = accuracy
        fold_results['training_loss'] = train_step_loss
        fold_results['network_chkpt_dir'] = netManager.log_file_name
        fold_results['validation_accuracy'] = val_accuracy
        fold_results['validation_loss'] = val_step_loss

        if self.parameter_dict['model_type'] == 'classifier':
            for class_idx in range(len(metric_results)):
                key_str = 'perfect_distance_' + str(class_idx)
                fold_results[key_str] = metric_results[class_idx]

            fold_results['perfect_distance'] = np.max(metric_results) # worst distance
        else:
            fold_results['perfect_distance'] = 0
        return fold_results

    def test_network(self, netManager, test_batch_handler):
        # Function that takes the currently built network and runs the test data through it (each data point is run once
        #  and only once). Graphs are generated. Make it easy to generate many graphs as this will be helpful for the
        # sequence generation model
        # This section and its affiliate long_train_network is going to change a lot. There are several things I want to
        # do here.
        # 1. Add the d=0 constraint for the test data.
        # 2. Run graphs in their own test_results folder
        # 3. Redefine the metric used on the network here for scoring. Maybe even have multiple scoring types reported

        # I think I want to spin out all the graph drawers from NetworkManager, and for the final report here, I want
        # the data being fed to the graphs saved, such that it is easy to cross-compile graphs between methods

        test_accuracy, test_loss, report_df, _ = netManager.run_validation(test_batch_handler,
                                                                   summary_writer=netManager.test_writer,
                                                                   quick=False, report_writing=True)

        return test_accuracy, test_loss, report_df

    def run_hyperparameter_search(self):
        hyperparam_results_list = []
        hyper_time = time.time()
        self.parameter_dict['training_early_stop'] = self.parameter_dict['early_stop_cf']
        training_batch_handler_cache = {}
        validation_batch_handler_cache = {}

        def hyper_training_helper(hyper_learning_rate,
                                  hyper_rnn_size,
                                  hyper_reg_embedding_beta,
                                  hyper_reg_l2_beta,
                                  hyper_learning_rate_decay):
            """ 
            Function used to wrap the hyperparameters and settings such that it fits the format used by dlib.
            Some variables need to be side-loaded, mostly reporting values.
            """
            ############# SELECT NEW PARAMS
            self.parameter_dict['learning_rate'] = 10 ** hyper_learning_rate
            self.parameter_dict['rnn_size'] = hyper_rnn_size
            self.parameter_dict['reg_embedding_beta'] = 10 ** hyper_reg_embedding_beta
            self.parameter_dict['l2_reg_beta'] = 10 ** hyper_reg_l2_beta
            self.parameter_dict['learning_rate_decay_factor'] = hyper_learning_rate_decay
            self.parameter_dict['embedding_size'] = self.parameter_dict['rnn_size']

            # Update Cutoffs
            self.parameter_dict['long_training_time'] = self.parameter_dict['early_stop_cf']
            self.parameter_dict['long_training_steps'] = self.parameter_dict['hyper_search_step_cutoff']
            ######### / PARAMS
            print 'learning_rate              ' + str(10 ** hyper_learning_rate)
            print 'rnn_size                   ' + str(hyper_rnn_size)
            print 'reg_embedding_beta         ' + str(10 ** hyper_reg_embedding_beta)
            print 'l2_reg_beta                ' + str(10 ** hyper_reg_l2_beta)
            print 'learning_rate_decay_factor ' + str(hyper_learning_rate_decay)

            cf_fold = -1
            # I should call this outside the crossfold, so it occurs once
            # This way all the crossfolds for the same hyperparameters are adjacent in the checkpoint dirs
            log_file_time = str(time.time())
            cf_results_list = []

            for train_pool, val_pool in self.cf_pool:
                cf_fold += 1
                log_file_name = log_file_time + "-cf-" + str(cf_fold)

                print "Starting crossfold"
                # Collect batch_handlers, and check if they've been cached.
                try:
                    training_batch_handler = training_batch_handler_cache[hash(tuple(np.sort(train_pool.uniqueId.unique())))]
                except KeyError:
                    training_batch_handler = BatchHandler.BatchHandler(train_pool, self.parameter_dict, True)
                except AttributeError:
                    print 'This should not be attainable, as crossfold==2 is invalid'

                try:
                    validation_batch_handler = validation_batch_handler_cache[hash(tuple(np.sort(val_pool.uniqueId.unique())))]
                except KeyError:
                    validation_batch_handler = BatchHandler.BatchHandler(val_pool, self.parameter_dict, False)
                except AttributeError:
                    print 'This should not be attainable, as crossfold==2 is invalid'

                # Add input_size, num_classes
                self.parameter_dict['input_size'] = training_batch_handler.get_input_size()
                self.parameter_dict['num_classes'] = training_batch_handler.get_num_classes()

                netManager = NetworkManager.NetworkManager(self.parameter_dict, log_file_name)
                netManager.build_model(self.encoder_means,self.encoder_stddev)

                try:
                    cf_results = self.train_network(netManager,training_batch_handler,validation_batch_handler,hyper_search=True)
                except tf.errors.InvalidArgumentError:
                    print "**********************caught error, probably gradients have exploded"
                    return 99999999  # HUGE LOSS --> this was caused by bad init conditions, so it should be avoided.
                # Now assign the handlers to the cache IF AND ONLY IF the training was successful.
                # If it dies before the first pool sort in the training, the whole thing falls over.
                validation_batch_handler_cache[
                    hash(tuple(np.sort(val_pool.uniqueId.unique())))] = validation_batch_handler
                training_batch_handler_cache[
                    hash(tuple(np.sort(train_pool.uniqueId.unique())))] = training_batch_handler

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
                if self.parameter_dict['model_type'] == 'categorical':
                    netManager.draw_categorical_html_graphs(validation_batch_handler)
                else:
                    netManager.draw_generative_html_graphs(validation_batch_handler,multi_sample=1)
                    netManager.draw_generative_html_graphs(validation_batch_handler,multi_sample=20)

                # Here we have a fully trained model, but we are still in the cross fold.

                # FIXME Only do 1 fold per hyperparams. Its not neccessary to continue
                break

            cf_df = pd.concat(cf_results_list)
            # Condense results from cross fold (Average, best, worst, whatever selection method)
            hyperparam_results = copy.copy(self.parameter_dict)
            #hyperparam_results['input_columns'] = ",".join(hyperparam_results['input_columns'])
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
            return hyperparam_results['validation_loss'] # VALUE TO BE MINIMIZED.
################################

        import dlib
        #  http://blog.dlib.net/2017/12/a-global-optimization-algorithm-worth.html
        lowers = [
            min(self.parameter_dict['hyper_learning_rate_args']),
            min(self.parameter_dict['hyper_rnn_size_args']),
            min(self.parameter_dict['hyper_reg_embedding_beta_args']),
            min(self.parameter_dict['hyper_reg_l2_beta_args']),
            min(self.parameter_dict['hyper_learning_rate_decay_args'])
           ]
        uppers = [
            max(self.parameter_dict['hyper_learning_rate_args']),
            max(self.parameter_dict['hyper_rnn_size_args']),
            max(self.parameter_dict['hyper_reg_embedding_beta_args']),
            max(self.parameter_dict['hyper_reg_l2_beta_args']),
            max(self.parameter_dict['hyper_learning_rate_decay_args'])
           ]
        x,y = dlib.find_min_global(hyper_training_helper, lowers, uppers,
                                   [False, True, False, False, False],  # Is integer Variable
                                   self.parameter_dict['hyper_search_folds'])

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


        if not test_network_only:
            netManager.build_model(self.encoder_means, self.encoder_stddev)
            best_results = self.train_network(netManager,training_batch_handler,validation_batch_handler)
        else:
            # We are loading a network from a checkpoint
            netManager.build_model()
            best_results = {}
        best_results['test_accuracy'], best_results['test_loss'], report_df = self.test_network(netManager,
                                                                                                test_batch_handler)

        print "Drawing html graph"
        #netManager.draw_html_graphs(netManager.compute_result_per_dis(test_batch_handler))
        if self.parameter_dict['model_type'] == 'categorical':
            netManager.draw_categorical_html_graphs(test_batch_handler)
        else:
            netManager.draw_generative_html_graphs(test_batch_handler, multi_sample=1)
            netManager.draw_generative_html_graphs(test_batch_handler, multi_sample=20)

        # FIXME maybe this needs its own function?
        for key, value in best_results.iteritems():
            if (type(value) is list or
                        type(value) is np.ndarray or
                        type(value) is tuple):
                best_results[key] = pd.Series([value], dtype=object)
        best_results = pd.DataFrame(best_results,index=[0])
        if not test_network_only:
            best_results.to_csv(os.path.join(self.parameter_dict['master_dir'],"best.csv"))

        reports = ReportWriter.ReportWriter(training_batch_handler, validation_batch_handler, test_batch_handler,
                                            self.parameter_dict, report_df)

        return best_results

