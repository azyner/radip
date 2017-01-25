# Main, the highest level instance.
# In here, hyperparameter selection, the cross fold section, and the final testing loops should be declared and run

# This file should hold the tf.session, inside the cross folder?

# read params
# read data
# instantiate sequence wrangler
#
# start cross fold loops
#   instance batch handler
#
import intersection_segments
import SequenceWrangler
import BatchHandler
import parameters
import NetworkManager
import time
import numpy as np
import pandas as pd
import copy
import random # TODO change to numpy random

# I want the logger and the crossfold here
# This is where the hyperparameter searcher goes

print "wrangling tracks"

Wrangler = SequenceWrangler.SequenceWrangler(parameters)

if not Wrangler.load_from_checkpoint():

    # This call copied from the dataset paper. It takes considerable time, so ensure it is run once only
    print "reading data"
    raw_sequences, raw_classes = intersection_segments.get_manouvre_sequences(parameters.parameters['input_columns'])
    Wrangler.generate_master_pool(raw_sequences, raw_classes)

Wrangler.split_into_evaluation_pools()
cf_pool, test_pool = Wrangler.get_pools()

# Here is where I want to spin off into another function, as the hyperparameter search should sit above
#HYPER SEARCH
hyperparam_results_list = []
hyper_time = time.time()
first = True
while True:

    #Select new hyperparameters
    learning_rate_range = [0.03, 0.01, 0.001, 0.003, 0.0001]

    rnn_size_range = np.arange(16,513,8)
    parameters.parameters["rnn_size"] = random.choice(rnn_size_range)
    parameters.parameters["learning_rate"] = random.choice(learning_rate_range)

    cf_fold = -1
    # I should call this outside the crossfold, so it occurs once
    # This way all the crossfolds for the same hyperparameters are adjacent in the checkpoint dirs
    log_file_time = str(time.time())
    cf_results_list = []

    for train_pool, val_pool in cf_pool:
        cf_fold += 1
        log_file_name = log_file_time + str(cf_fold)
        cf_time = time.time()
        print "Starting crossfold"
        training_batch_handler = BatchHandler.BatchHandler(train_pool, parameters.parameters, True)
        validation_batch_handler = BatchHandler.BatchHandler(val_pool, parameters.parameters, False)

        # Add input_size, num_classes
        parameters.parameters['input_size'] = training_batch_handler.get_input_size()
        parameters.parameters['num_classes'] = training_batch_handler.get_num_classes()

        netManager = NetworkManager.NetworkManager(parameters.parameters, log_file_name)
        netManager.build_model()

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
                #eval_accuracy, eval_step_loss, _ = netManager.run_validation(validation_batch_handler, summary_writer=None)
            step_time += (time.time() - step_start_time) / steps_per_checkpoint
            step_time += (time.time() - step_start_time) / steps_per_checkpoint
            loss += step_loss / steps_per_checkpoint
            current_step += 1

            if current_step % steps_per_checkpoint == 0:

                eval_accuracy, eval_step_loss, _ = netManager.run_validation(validation_batch_handler, summary_writer=None)
                #graph_results = netManager.collect_graph_data(validation_batch_handler)
                #netManager.draw_graphs(graph_results)

                print ("g_step %d lr %.6f step-time %.4f Batch av tr loss %.4f Acc %.3f val acc %.3f"
                       % (netManager.model.global_step.eval(session=netManager.sess),
                          netManager.model.learning_rate.eval(session=netManager.sess),
                          step_time, loss, accuracy, eval_accuracy))

                previous_losses.append(loss)
                step_time, loss = 0.0, 0.0
                now = time.time()
                if now - cf_time > 60 * parameters.parameters['early_stop_cf']:
                    break
                # if (((perplexity < 0.0001 or model.learning_rate.eval() < 0.00001) and FLAGS.early_stop is 0) or
                #     (FLAGS.early_stop is not 0) and (now - start_time > 60 * FLAGS.early_stop)):
                #     #cross_train_accuracy.append(eval_accuracy)
                #     #cross_train_loss.append(eval_step_loss)
                #
                #     # log_result_to_csv(model.global_step.eval(), perplexity, model.learning_rate.eval(),step_time)
                #
                #     break

        cf_results = copy.copy(parameters.parameters)
        cf_results['input_columns'] = ",".join(cf_results['input_columns'])
        cf_results['eval_accuracy'] = eval_accuracy
        cf_results['final_learning_rate'] = netManager.model.learning_rate.eval(session=netManager.sess)
        cf_results['training_accuracy'] = accuracy
        cf_results['training_loss'] = loss
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
    hyperparam_results = copy.copy(parameters.parameters)
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
i  = 0

best_params = hyper_df.sort_values('eval_accuracy',ascending=False).iloc[[0]]

# Select best model based on hyper parameters
# Train on all training/val data
# Run on test data
# Also run checkpointed model on test data