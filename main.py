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


input_columns = ['easting', 'northing', 'heading', 'speed']
raw_sequences, raw_classes = intersection_segments.get_manouvre_sequences(input_columns)

print raw_sequences