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


input_columns = ['easting', 'northing', 'heading', 'speed']
#This call copied from the dataset paper. It takes considerable time, so ensure it is run once only
print "reading data"
raw_sequences, raw_classes = intersection_segments.get_manouvre_sequences(input_columns)
print "wrangling tracks"
Wrangler = SequenceWrangler.SequenceWrangler(None,raw_sequences,raw_classes)

cf_pool, test_pool = Wrangler.get_pools()
print raw_sequences