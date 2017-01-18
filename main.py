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
cf_fold = -1
# I should call this outside the crossfold, so it occurs once
log_file_time = str(time.time())

for train_pool, val_pool in cf_pool:
    cf_fold += 1
    log_file_name = log_file_time + str(cf_fold)

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
    steps_per_checkpoint = 200
    while True:
        # The training loop!

        # TODO:
        # The val_acc should not be jumping around this much.
        # As d_thresh has not been set, there should be significant validation samples, more than the 1024 batch size
        # This means I require a validation loop in the network manager, one that can exhaust all the validation samples
        # in a loop fashion, and report the overall acc. s.t. it is a one line call here. Step_val?
        # I will also need a function that does the evaluation at each d_thresh

        step_start_time = time.time()
        train_x, train_y, weights = training_batch_handler.get_minibatch()
        accuracy, step_loss, _ = netManager.run_training_step(train_x, train_y, weights, True)

        # Periodically, run without training for the summary logs
        if current_step % 200 == 0:
            eval_accuracy, eval_step_loss, _ = netManager.run_training_step(train_x, train_y, weights, False, summary_writer=None)
            eval_accuracy, eval_step_loss, _ = netManager.run_validation(validation_batch_handler, summary_writer=None)
        step_time += (time.time() - step_start_time) / steps_per_checkpoint
        step_time += (time.time() - step_start_time) / steps_per_checkpoint
        loss += step_loss / steps_per_checkpoint
        current_step += 1
        if current_step % steps_per_checkpoint == 0:

            eval_accuracy, eval_step_loss, _ = netManager.run_validation(validation_batch_handler, summary_writer=None)

            print ("g_step %d lr %.6f step-time %.4f Batch av tr loss %.4f Acc %.3f val acc %.3f"
                   % (netManager.model.global_step.eval(session=netManager.sess),
                                  netManager.model.learning_rate.eval(session=netManager.sess),
                                  step_time, loss, accuracy, eval_accuracy))

            previous_losses.append(loss)
            step_time, loss = 0.0, 0.0
            now = time.time()
            # if (((perplexity < 0.0001 or model.learning_rate.eval() < 0.00001) and FLAGS.early_stop is 0) or
            #     (FLAGS.early_stop is not 0) and (now - start_time > 60 * FLAGS.early_stop)):
            #     #cross_train_accuracy.append(eval_accuracy)
            #     #cross_train_loss.append(eval_step_loss)
            #
            #     # log_result_to_csv(model.global_step.eval(), perplexity, model.learning_rate.eval(),step_time)
            #
            #     break
