# Class that manages the training loop. This class will also deal with logging
import time, copy, random # TODO change to numpy random
import BatchHandler
import NetworkManager
import numpy as np
import pandas as pd

class trainingManager:
    def __init__(self, cf_pool, test_pool, parameter_dict):
        self.cf_pool = cf_pool
        self.test_pool = test_pool
        self.parameter_dict = parameter_dict

        return

    def train_network(self,netManager,training_batch_handler,validation_batch_handler):
        fold_time = time.time()
        current_step = 0
        previous_losses = []
        step_time, loss = 0.0, 0.0
        steps_per_checkpoint = 40
        while True:
            # The training loop!

            step_start_time = time.time()
            batch_frame = training_batch_handler.get_minibatch()
            train_x, _, weights, train_y = training_batch_handler.format_minibatch_data(batch_frame['encoder_sample'],
                                                                                        batch_frame['dest_1_hot'],
                                                                                        batch_frame['padding'])
            accuracy, step_loss, _ = netManager.run_training_step(train_x, train_y, weights, True)

            # Periodically, run without training for the summary logs
            if current_step % 20 == 0:
                eval_accuracy, eval_step_loss, _ = netManager.run_training_step(train_x, train_y, weights, False,
                                                                                summary_writer=netManager.train_writer)
                # FIXME I feel this may break as it should be run once only for each global step for summary writer
                # eval_accuracy, eval_step_loss, _ = netManager.run_validation(validation_batch_handler, summary_writer=None)
            step_time += (time.time() - step_start_time) / steps_per_checkpoint
            step_time += (time.time() - step_start_time) / steps_per_checkpoint
            loss += step_loss / steps_per_checkpoint
            current_step += 1

            if current_step % steps_per_checkpoint == 0:

                eval_accuracy, eval_step_loss, _ = netManager.run_validation(validation_batch_handler,
                                                                             summary_writer=None)
                # graph_results = netManager.collect_graph_data(validation_batch_handler)
                # netManager.draw_graphs(graph_results)

                print ("g_step %d lr %.6f step-time %.4f Batch av tr loss %.4f Acc %.3f val acc %.3f"
                       % (netManager.model.global_step.eval(session=netManager.sess),
                          netManager.model.learning_rate.eval(session=netManager.sess),
                          step_time, loss, accuracy, eval_accuracy))

                previous_losses.append(loss)
                step_time, loss = 0.0, 0.0
                now = time.time()
                if now - fold_time > 60 * self.parameter_dict['early_stop_cf']:
                    break
                    # if (((perplexity < 0.0001 or model.learning_rate.eval() < 0.00001) and FLAGS.early_stop is 0) or
                    #     (FLAGS.early_stop is not 0) and (now - start_time > 60 * FLAGS.early_stop)):
                    #     #cross_train_accuracy.append(eval_accuracy)
                    #     #cross_train_loss.append(eval_step_loss)
                    #
                    #     # log_result_to_csv(model.global_step.eval(), perplexity, model.learning_rate.eval(),step_time)
                    #
                    #     break
        fold_results = copy.copy(self.parameter_dict)
        fold_results['input_columns'] = ",".join(fold_results['input_columns'])
        fold_results['eval_accuracy'] = eval_accuracy
        fold_results['final_learning_rate'] = netManager.model.learning_rate.eval(session=netManager.sess)
        fold_results['training_accuracy'] = accuracy
        fold_results['training_loss'] = loss

        return fold_results

    def run_hyperparamter_search(self):
        hyperparam_results_list = []
        hyper_time = time.time()
        first = True
        while True:

            #Select new hyperparameters
            learning_rate_range = [0.03, 0.01, 0.001, 0.003, 0.0001]

            rnn_size_range = np.arange(16,513,8)
            self.parameter_dict["rnn_size"] = random.choice(rnn_size_range)
            self.parameter_dict["learning_rate"] = random.choice(learning_rate_range)

            cf_fold = -1
            # I should call this outside the crossfold, so it occurs once
            # This way all the crossfolds for the same hyperparameters are adjacent in the checkpoint dirs
            log_file_time = str(time.time())
            cf_results_list = []

            for train_pool, val_pool in self.cf_pool:
                cf_fold += 1
                log_file_name = log_file_time + str(cf_fold)

                print "Starting crossfold"
                training_batch_handler = BatchHandler.BatchHandler(train_pool, self.parameter_dict, True)
                validation_batch_handler = BatchHandler.BatchHandler(val_pool, self.parameter_dict, False)

                # Add input_size, num_classes
                self.parameter_dict['input_size'] = training_batch_handler.get_input_size()
                self.parameter_dict['num_classes'] = training_batch_handler.get_num_classes()

                netManager = NetworkManager.NetworkManager(self.parameter_dict, log_file_name)
                netManager.build_model()

                cf_results = self.train_network(netManager,training_batch_handler,validation_batch_handler)
                cf_results['crossfold_number'] = cf_fold

                cf_results_list.append(pd.DataFrame(cf_results, index=[0]))

                #######
                # Here we have a fully trained model, but we are still in the cross fold.
                # So I would like to add the results into a cross-fold collection

                # TODO Add other evaluation metrics here
                # We still have the model in scope here, so I can probe it for whatever I want
                # eg
                #cf_results['d_100'] = netManager.run_d_thresh_metric(...)

            cf_df = pd.concat(cf_results_list)
            # Condense results from cross fold (Average, best, worst, whatever selection method)
            hyperparam_results = copy.copy(self.parameter_dict)
            hyperparam_results['input_columns'] = ",".join(hyperparam_results['input_columns'])
            # HACK save the condensed results under crossfold_number 0
            hyperparam_results['eval_accuracy'] = np.min(cf_df['eval_accuracy'])
            hyperparam_results['final_learning_rate'] = np.min(cf_df['final_learning_rate'])
            hyperparam_results['training_accuracy'] = np.min(cf_df['training_accuracy'])
            hyperparam_results['training_loss'] = np.average(cf_df['training_loss'])
            hyperparam_results['crossfold_number'] = 0
            hyperparam_results_list.append(pd.DataFrame(hyperparam_results, index=[0]))
            #Write results and hyperparams to hyperparameter_results_dataframe

            #Once cross folding has completed, select new hyperparameters, and re-run
            now = time.time()
            #if now - hyper_time > 60 * 60 * parameters.parameters['hyper_search_time']: # Stop the hyperparameter search after a set time
            if first:
                first = False
            else:
                break

        hyper_df = pd.concat(hyperparam_results_list)
        best_params = hyper_df.sort_values('eval_accuracy',ascending=False).iloc[[0]].to_dict('index')[0]

        return best_params

    def test_network(self):
        # Do it all again, but this time train with all data OR TODO return from best checkpoint
        # and test against that last test set
        # I guess this is where the HTML plots would be generated

        results = None

        return results