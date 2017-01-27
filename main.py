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
import datetime
import os
import pandas as pd

# I want the logger and the crossfold here
# This is where the hyperparameter searcher goes

print "wrangling tracks"
Wrangler = SequenceWrangler.SequenceWrangler(parameters,n_folds=parameters.parameters['n_folds'])
if not Wrangler.load_from_checkpoint():
    print "reading data and splitting into data pool, this will take some time (10? minutes). Grab a coffee"
    raw_sequences, raw_classes = intersection_segments.get_manouvre_sequences(parameters.parameters['input_columns'])
    Wrangler.generate_master_pool(raw_sequences, raw_classes)

Wrangler.split_into_evaluation_pools()
cf_pool, test_pool = Wrangler.get_pools()
full_cf_pool = pd.concat([cf_pool[0][0], cf_pool[0][1]])

results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
parameters.parameters['master_dir'] = os.path.join(results_dir,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(parameters.parameters['master_dir']):
    os.makedirs(parameters.parameters['master_dir'])

trainingManager = trainingManager.trainingManager(cf_pool,test_pool,parameters.parameters)
best_params = trainingManager.run_hyperparamter_search()

full_cf_pool = pd.concat([cf_pool[0][0], cf_pool[0][1]])
trainingManager.test_network(best_params,full_cf_pool,test_pool)
#Dumb statement for breakpoint before system finishes
ideas = None


# Select best model based on hyper parameters
# Train on all training/val data
# Run on test data
# Also run checkpointed model on test data