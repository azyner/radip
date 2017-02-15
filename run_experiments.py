import os
import subprocess
import time

# Script to run N number of main.py clients, and N tensorboard observers
# Starts at port 6006, and will increase by one per tensorboard instance
# Will complain in the following conditions:
#   results is not empty. Experiments should run in a clean results folder

# TODO read args from cmd line.

import argparse

parser = argparse.ArgumentParser(description='Spawn N TF workers (main.py) and N tensorboard programs.')
parser.add_argument('N', metavar='N', type=int, nargs='+',
                   help='Number of experiments')
args = parser.parse_args()
N = args.N[0]

if os.path.exists('results'):
    experments = os.listdir('results')
    if len(experments) is not 0:
        print "Non-empty results dir. Clean it before starting another experiment run"
        quit()

# run N Number of main.py
print "Spawning trainers"
for i in range(N):
    subprocess.Popen(["/usr/bin/python2","main.py"])
    time.sleep(2)
    print ["/usr/bin/python2","main.py"]

# Wait 10 seconds - count them off
print "waiting for all workers to begin"

while True:
    if os.path.exists('results'):
        experments = os.listdir('results')
        if len(experments) is N:
            break
    time.sleep(1)
    print '.'

print "Spawning tensorboard observers"

results_dir = os.listdir('results')

for i in range(N):
    tensorboard_string = os.path.join('results',os.path.join(results_dir[i],'tensorboard_logs'))
    port = 6006 + i
    tb_args =["tensorboard","--logdir",tensorboard_string,"--port",str(port)]
    subprocess.Popen(tb_args)
    print tb_args

print "experiment running."
try:
    while True:
        time.sleep(1)
        pass
except KeyboardInterrupt:
    quit()
