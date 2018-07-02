#!/usr/bin/env python
import os
import subprocess
import time
import signal
import pandas as pd

# Script to run N number of main.py clients, and N tensorboard observers
# Starts at port 6006, and will increase by one per tensorboard instance
# Will complain in the following conditions:
#   results is not empty. Experiments should run in a clean results folder

import argparse

parser = argparse.ArgumentParser(description='Find the best results in a given checkpoint directory')
parser.add_argument('-c', '--checkpoint', action="store", dest='checkpoint_dir',  help="The root directory containing "
                                                                                       "all the checkpoint files")
parser.add_argument('-m', '--metric', action="store", dest='metric', default="MHD", help="The results metric to use for best")
parser.add_argument('-d', '--dataset', action="store", dest='dataset', default="right",
                    help="The turn style to use for best, [straight left right all]")
args = parser.parse_args()

models = os.listdir(args.checkpoint_dir)

score = 9999999999
best_model = None

for model_dir in models:
    if "__init__" in model_dir:
        continue
    metric_file_name = os.path.join(os.path.join(args.checkpoint_dir, model_dir), args.dataset + "-metrics.csv")
    df = pd.read_csv(metric_file_name)

    # Find it just in case I rename something later.
    for col in df.columns:
        if 'confident' not in col:
            continue
        c = col

    row = df['Unnamed: 0']
    row_name = args.metric + " worst 5%"
    r = row[row == row_name].index[0]

    model_score = df[c][r]
    print str(model_score) + " --- " + metric_file_name
    if model_score < score:
        best_model = metric_file_name
        score = model_score

print "The best model is:"
print str(score) + " --- " + best_model
