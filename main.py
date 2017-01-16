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
import sys
import copy
import numpy as np
import pandas as pd
import SequenceWrangler
import BatchHandler
import parameters



print "wrangling tracks"

Wrangler = SequenceWrangler.SequenceWrangler(parameters)

if not Wrangler.load_from_checkpoint():
    input_columns = ['easting', 'northing', 'heading', 'speed']
    #This call copied from the dataset paper. It takes considerable time, so ensure it is run once only
    print "reading data"
    raw_sequences, raw_classes = intersection_segments.get_manouvre_sequences(input_columns)
    Wrangler.generate_master_pool(raw_sequences,raw_classes)

Wrangler.split_into_evaluation_pools()
cf_pool, test_pool = Wrangler.get_pools()

for train_pool, val_pool in cf_pool:
    training_batch_handler = BatchHandler.BatchHandler(train_pool,17,True)
    validation_batch_handler = BatchHandler.BatchHandler(val_pool,17,False)

    train_x, train_y = training_batch_handler.get_minibatch()

    if len(train_x) != 17:
        print "error"
    # Get number of unique tracks in val pool
    num_val_tracks = len(val_pool['track_idx'].unique())
    validation_batch_handler.update_minibatch_selection_parameters(d_thresh=22)
    complete = False
    pad_array = []
    while not complete:
        X, Y, pad, complete = validation_batch_handler.get_minibatch()
        pad_array.extend(pad)
    if (len(pad_array)-sum(np.array(pad_array).astype(int))) != num_val_tracks:
        print "error"