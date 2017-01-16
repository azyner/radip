#Main, the highest level instance.
# In here, hyperparam selection, the cross fold section, and the final testing loops should be declared and run

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

# I want the logger and the crossfold here
# This is where the hyperparameter searcher goes

print "wrangling tracks"

Wrangler = SequenceWrangler.SequenceWrangler(parameters)

if not Wrangler.load_from_checkpoint():

    #This call copied from the dataset paper. It takes considerable time, so ensure it is run once only
    print "reading data"
    raw_sequences, raw_classes = intersection_segments.get_manouvre_sequences(parameters.parameters['input_columns'])
    Wrangler.generate_master_pool(raw_sequences,raw_classes)

Wrangler.split_into_evaluation_pools()
cf_pool, test_pool = Wrangler.get_pools()

for train_pool, val_pool in cf_pool:
    #netManage = NetworkManager.NetworkManager()

    training_batch_handler = BatchHandler.BatchHandler(train_pool,17,True)
    validation_batch_handler = BatchHandler.BatchHandler(val_pool,17,False)

    print 'input_size'
    print training_batch_handler.get_input_size()
    print "num classes"
    print training_batch_handler.get_num_classes()

    train_x, train_y = training_batch_handler.get_minibatch()
