#!/usr/bin/env python
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
import TrainingManager
import intersection_segments
import SequenceWrangler
import parameters
import datetime
import os
import pandas as pd
import subprocess
import shutil
import ibeoCSVImporter
import pickle

# I want the logger and the crossfold here
# This is where the hyperparameter searcher goes

results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
parameters.parameters['master_dir'] = os.path.join(results_dir,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(parameters.parameters['master_dir']):
    os.makedirs(parameters.parameters['master_dir'])
shutil.copy("parameters.py",os.path.join(parameters.parameters['master_dir'],"parameters.py"))
print "results folder made, parameter file copied"

githash = subprocess.check_output(["git", "describe", "--always"])
with open(os.path.join(parameters.parameters['master_dir'], githash + ".githash"), "w") as outfile:
    outfile.write("Git hash:")
    outfile.write(githash)
    outfile.write("Git diff:")
    diff = subprocess.check_output(["git", "diff"])
    outfile.write(diff)

print "wrangling tracks"

ibeo = True

### TEST CODE ###
#ibeoCSV = ibeoCSVImporter.ibeoCSVImporter(parameters,'data/20170601-stationary-3-leith-croydon.csv')


# sourcename = '20170427-stationary-2-leith-croydon.csv'
sourcename = '20170601-stationary-3-leith-croydon.csv'
source_list = sourcename
source_list = ['split_20170601-stationary-3-leith-croydon_01.csv',
              'split_20170601-stationary-3-leith-croydon_02.csv',
              'split_20170601-stationary-3-leith-croydon_03.csv',
              'split_20170601-stationary-3-leith-croydon_04.csv',
              'split_20170601-stationary-3-leith-croydon_05.csv']
sourcename = source_list[0]

Wrangler = SequenceWrangler.SequenceWrangler(parameters,sourcename,n_folds=parameters.parameters['n_folds'])

if ibeo:
    if not Wrangler.load_from_checkpoint():
        print "reading data and splitting into data pool, this will take some time (10? minutes). Grab a coffee"
        ibeoCSV = ibeoCSVImporter.ibeoCSVImporter(parameters,source_list)
        Wrangler.generate_master_pool_ibeo(ibeoCSV.get_track_list())

else:
    if not Wrangler.load_from_checkpoint():
        print "reading data and splitting into data pool, this will take some time (10? minutes). Grab a coffee"
        raw_sequences, raw_classes = intersection_segments.get_manouvre_sequences(parameters.parameters['input_columns'])
        Wrangler.generate_master_pool_naturalistic_2015(raw_sequences, raw_classes)

Wrangler.split_into_evaluation_pools()
cf_pool, test_pool = Wrangler.get_pools()

trainingManager = TrainingManager.TrainingManager(cf_pool, test_pool, parameters.parameters)
if parameters.parameters['hyper_search_time'] > 0.001:
    best_params = trainingManager.run_hyperparameter_search()
else:
    best_params = parameters.parameters

print "Crossfolding finished, now training with the best parameters"

full_cf_pool = pd.concat([cf_pool[0][0], cf_pool[0][1]])
trainingManager.long_train_network(best_params,cf_pool[0][0], cf_pool[0][1], test_pool)
#Dumb statement for breakpoint before system finishes
ideas = None

#Anything else to pickle?
# I need the track idx's for test train split for the visualiser
to_pickle = {}
to_pickle['test_idxs'] = test_pool.track_idx.unique()
to_pickle['data_pool'] = Wrangler.get_pool_filename()

with open(os.path.join(parameters.parameters['master_dir'],'data.pkl'),'wb') as pkl_file:
    pickle.dump(to_pickle, pkl_file)

ideas = None
# Select best model based on hyper parameters
# Train on all training/val data
# Run on test data
# Also run checkpointed model on test data
