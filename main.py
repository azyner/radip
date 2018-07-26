#!/usr/bin/env python
# Main, the highest level instance.
# This is the function to call for training and running the RNN from a checkpoint
# It will either read the local parameters.py file or the parameters pkl inside the checkpoint folder.
# If
import TrainingManager
import SequenceWrangler
import datetime
import os
import pandas as pd
import subprocess
import shutil
import ibeoCSVImporter
import dill as pickle
import argparse

parser = argparse.ArgumentParser(description="The main program to run training of a RNN based tracjectory estimation "
                                             "algorithm. It may also load data from a checkpoint given here in "
                                             "arguments")
parser.add_argument('-c', '--checkpoint', action="store", dest='checkpoint_dir',  help="The directory containing the checkpoint files for a neural "
                                                          "network.")
args = parser.parse_args()
if args.checkpoint_dir is not None:
    checkpoint_dir = args.checkpoint_dir
    if '/' in checkpoint_dir[-1]:
        checkpoint_dir = checkpoint_dir[:-1]

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

# If we are not loading from checkpoint, create a results directory and copy the parameters file,
#  save the git hash and diff
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
    with open(os.path.join(parameters.parameters['master_dir'], githash[:-1] + ".githash"), "w") as outfile:
        outfile.write("Git hash:")
        outfile.write(githash)
        outfile.write("Git diff:")
        diff = subprocess.check_output(["git", "diff"])
        outfile.write(diff)

print "wrangling tracks"

ibeo = True

### TEST CODE ###

source_list = []

# sourcename = '20170427-stationary-2-leith-croydon.csv'
# sourcename = '20170601-stationary-3-leith-croydon.csv'
# source_list = sourcename
#
try:
    short_wrangle = parameters.parameters['short_wrangle']
except KeyError:
    short_wrangle = False

if short_wrangle:
    range_max = 1
else:
    range_max = 999

if 'leith-croydon' in parameters.parameters['data_list']:
    source_list = ['split_20170601-stationary-3-leith-croydon_01.csv']
    if not short_wrangle:
        source_list.extend([
                   'split_20170601-stationary-3-leith-croydon_02.csv',
                   'split_20170601-stationary-3-leith-croydon_03.csv',
                   'split_20170601-stationary-3-leith-croydon_04.csv',
                   'split_20170601-stationary-3-leith-croydon_05.csv'
        ])
    for i in range(min(range_max, 41)):
        source_list.append("split_20170802-stationary-4-leith-croydon_%02d.csv" % (i+1))
    for i in range(min(range_max, 35)):
        source_list.append("split_20170804-stationary-5-leith-croydon_%02d.csv" % (i+1))
if 'queen-hanks' in parameters.parameters['data_list']:
    for i in range(min(range_max, 31)):
        source_list.append("split_20180116-082129-urban-stationary-queen-hanks_%02d.csv" % (i+1))
if 'roslyn-crieff' in parameters.parameters['data_list']:
    for i in range(min(range_max, 24)):
        source_list.append("split_20180119-112135-urban-stationary-roslyn-crieff_%02d.csv" % (i + 1))
if 'oliver-wyndora' in parameters.parameters['data_list']:
    for i in range(min(range_max, 46)):
        source_list.append("split_20180123-072840-urban-stationary-oliver-wyndora_%02d.csv" % (i + 1))
if 'orchard-mitchell' in parameters.parameters['data_list']:
    for i in range(min(range_max, 21)):
        source_list.append("split_20180124-081438-urban-stationary-orchard-mitchell_%02d.csv" % (i + 1))
sourcename = source_list[0]


Wrangler = SequenceWrangler.SequenceWrangler(parameters,source_list,n_folds=parameters.parameters['n_folds'])

if ibeo:
    if not Wrangler.load_from_checkpoint():
        print "Reading raw data from csv's / checkpoint and splitting into data pool, this will take some time"
        ibeoCSV = ibeoCSVImporter.ibeoCSVImporter(parameters,source_list)
        Wrangler.generate_master_pool_ibeo(ibeoCSV.get_track_list())

else:
    if not Wrangler.load_from_checkpoint():
        print "deprecated. Please use ibeo=true in the parametes file"
        #"Reading data and splitting into data pool, this will take some time (3? hours)."
        #raw_sequences, raw_classes = intersection_segments.get_manouvre_sequences(parameters.parameters['input_columns'])
        #Wrangler.generate_master_pool_naturalistic_2015(raw_sequences, raw_classes)
if checkpoint_dir is not None:
    # Guarantee that the tracks used for testing are the same as during training
    with open(os.path.join(parameters.parameters['master_dir'], 'data.pkl'), 'rb') as pkl_file:
        from_pickle = pickle.load(pkl_file)
        Wrangler.split_into_evaluation_pools(test_idxs=from_pickle['test_idxs'], trainval_idxs=from_pickle['trainval_idxs'])
else:
    try:
        Wrangler.split_into_evaluation_pools(test_csv=parameters.parameters['test_csv'])
    except KeyError:
        Wrangler.split_into_evaluation_pools()
cf_pool, test_pool = Wrangler.get_pools()

to_pickle = {}
to_pickle['test_idxs'] = Wrangler.test_idxs
to_pickle['trainval_idxs'] = Wrangler.trainval_idxs
to_pickle['data_pool'] = Wrangler.get_pool_filename()


trainingManager = TrainingManager.TrainingManager(cf_pool, test_pool,Wrangler.encoder_means,Wrangler.encoder_stddev,
                                                  parameters.parameters)
if (parameters.parameters['hyper_search_folds'] > 0) and not test_network_only:
    best_params = trainingManager.run_hyperparameter_search()
elif test_network_only:
    best_params = from_pickle['best_params']
    best_params['master_dir'] = master_dir
    # TOTAL HACK - this is because the hyperparameter search will store the parameters as a dict within 'best_params'pkl
    # I don't want to directly load from checkpoint, as this will not find the parameters the hyper-search did
    # But if I want to do experiments based on the output graph, I need them to be placed into the parameter dict.
    # So I search for that here. In reality, the best solution should be: take RNN hyperparameters from the pkl, and
    # and all other values from the parameters dict.
    for key, value in parameters.parameters.iteritems():
        if 'cluster' in key:
            best_params[key] = value
else:
    best_params = parameters.parameters

to_pickle['best_params'] = best_params
with open(os.path.join(parameters.parameters['master_dir'],'data.pkl'),'wb') as pkl_file:
    pickle.dump(to_pickle, pkl_file)

print "Crossfolding finished, now training with the best parameters"

full_cf_pool = pd.concat([cf_pool[0][0], cf_pool[0][1]])
trainingManager.long_train_network(best_params, cf_pool[0][0], cf_pool[0][1], test_pool, checkpoint=checkpoint_dir,
                                   test_network_only=test_network_only)
#Dumb statement for breakpoint before system finishes
ideas = None
