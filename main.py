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
import trainingManager

import parameters


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


trainingManager = trainingManager.trainingManager(cf_pool,test_pool,parameters.parameters)
best_params = trainingManager.run_hyperparamter_search()

ideas = None


# Select best model based on hyper parameters
# Train on all training/val data
# Run on test data
# Also run checkpointed model on test data