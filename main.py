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

import datetime
import os
import pandas as pd
import subprocess
import shutil
import ibeoCSVImporter
import dill as pickle

# I want the logger and the crossfold here
# This is where the hyperparameter searcher goes

# ibeoCSV = ibeoCSVImporter.ibeoCSVImporter(parameters,'data/20170601-stationary-3-leith-croydon.csv')
#checkpoint_dir = "network_plots/20170718-164817/train/best-1500360501.23"
#checkpoint_dir='network_plots/20170718-192827/train/best-1500370111.66'
#checkpoint_dir = None
#checkpoint_dir = 'results/20170814-121457/train/best-1502676987.0'
if 'checkpoint_dir' in locals():
    test_network_only = True
else:
    checkpoint_dir = None
    test_network_only = False


if checkpoint_dir is not None:
    print "Loading from checkpoint..."
    # The parameters file in the checkpoint master folder must be imported for things like obs length, etc
    master_dir = os.path.relpath(os.path.join(os.path.join(checkpoint_dir,os.pardir),os.pardir))

    def touch(fname):
        try:
            os.utime(fname, None)
        except OSError:
            open(fname, 'a').close()
    touch(os.path.join(master_dir, "__init__.py"))
    touch(os.path.join(os.path.join(master_dir,os.pardir), "__init__.py"))

    import importlib
    import_name = master_dir + ".parameters"
    parameters = importlib.import_module(import_name.replace('/','.'))
    parameters.parameters['master_dir'] = master_dir

if checkpoint_dir is None:
    import parameters
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



sourcename = '20170427-stationary-2-leith-croydon.csv'
sourcename = '20170601-stationary-3-leith-croydon.csv'
source_list = sourcename

source_list = ['split_20170601-stationary-3-leith-croydon_01.csv',
              'split_20170601-stationary-3-leith-croydon_02.csv',
              'split_20170601-stationary-3-leith-croydon_03.csv',
              'split_20170601-stationary-3-leith-croydon_04.csv',
              'split_20170601-stationary-3-leith-croydon_05.csv']
#source_list = []
for i in range(41):
    source_list.append("split_20170802-stationary-4-leith-croydon_%02d.csv" % (i+1))
for i in range(35):
    source_list.append("split_20170804-stationary-5-leith-croydon_%02d.csv" % (i+1))

sourcename = source_list[0]

Wrangler = SequenceWrangler.SequenceWrangler(parameters,source_list,n_folds=parameters.parameters['n_folds'])

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
if checkpoint_dir is not None:
    # Guarantee that the tracks used for testing are the same as during training
    with open(os.path.join(parameters.parameters['master_dir'], 'data.pkl'), 'rb') as pkl_file:
        from_pickle = pickle.load(pkl_file)
        Wrangler.split_into_evaluation_pools(test_idxs=from_pickle['test_idxs'], trainval_idxs=from_pickle['trainval_idxs'])
else:
    Wrangler.split_into_evaluation_pools()
cf_pool, test_pool = Wrangler.get_pools()

to_pickle = {}
to_pickle['test_idxs'] = Wrangler.test_idxs
to_pickle['trainval_idxs'] = Wrangler.trainval_idxs
to_pickle['data_pool'] = Wrangler.get_pool_filename()


trainingManager = TrainingManager.TrainingManager(cf_pool, test_pool, parameters.parameters)
if (parameters.parameters['hyper_search_time'] > 0.001) and not test_network_only:
    best_params = trainingManager.run_hyperparameter_search()
elif test_network_only:
    best_params = from_pickle['best_params']
else:
    best_params = parameters.parameters

to_pickle['best_params'] = best_params
with open(os.path.join(parameters.parameters['master_dir'],'data.pkl'),'wb') as pkl_file:
    pickle.dump(to_pickle, pkl_file)

print "Crossfolding finished, now training with the best parameters"

full_cf_pool = pd.concat([cf_pool[0][0], cf_pool[0][1]])
trainingManager.long_train_network(best_params, cf_pool[0][0], cf_pool[0][1], test_pool,checkpoint=checkpoint_dir,test_network_only=test_network_only)
#Dumb statement for breakpoint before system finishes
ideas = None

#Anything else to pickle?
# I need the track idx's for test train split for the visualiser

# Select best model based on hyper parameters
# Train on all training/val data
# Run on test data
# Also run checkpointed model on test data
