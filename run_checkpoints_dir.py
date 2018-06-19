#!/usr/bin/env python
import os
import subprocess
import time
import signal

# Script to run N number of main.py clients, and N tensorboard observers
# Starts at port 6006, and will increase by one per tensorboard instance
# Will complain in the following conditions:
#   results is not empty. Experiments should run in a clean results folder

import argparse

parser = argparse.ArgumentParser(description='Re-run all models in a checkpoint directory')
parser.add_argument('-c', '--checkpoint', action="store", dest='checkpoint_dir',  help="The root directory containing "
                                                                                       "all the checkpoint files")
args = parser.parse_args()

models = os.listdir(args.checkpoint_dir)
for model_dir in models:
    if "__init__" in model_dir:
        continue
    model_files = os.listdir(os.path.join(os.path.join(args.checkpoint_dir, model_dir), 'train'))
    for nn_model in model_files:
        if 'best' in nn_model:
            job_checkpoint_dir = os.path.join(os.path.join(
                os.path.join(args.checkpoint_dir, model_dir), 'train'), nn_model)
            p_child = subprocess.Popen(["/usr/bin/python2", "main.py", "-c", job_checkpoint_dir])

            # Define and register ctrl-c here.
            def sigint_handler(signum, frame):
                print "Caught SIGINT. Asking job to finish up. Will run another job if in queue."
                p_child.send_signal(signum)


            signal.signal(signal.SIGINT, sigint_handler)

            while p_child.poll() is None:
                time.sleep(1)