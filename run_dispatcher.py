#!/usr/bin/env python
import os
import subprocess
import time
import shutil
import signal

# Script to run N number of main.py clients, and N tensorboard observers
# Starts at port 6006, and will increase by one per tensorboard instance
# Will complain in the following conditions:
#   results is not empty. Experiments should run in a clean results folder
job_directory = 'job_queue'

if os.path.exists(job_directory):
    jobs = os.listdir(job_directory)
    if len(jobs) is 0:
        print "No jobs! Please populate job_queue directory with renamed parameter.py files"
        quit()
else:
    print "No jobs! Please make job_queue directory and populate with renamed parameter.py files"
    quit()

if 'parameters.py' in os.listdir('.'):
    print "Found parameters.py in current directory. This job will be run first, and then overwritten."
else:
    target_file = jobs[0]
    shutil.copyfile(os.path.join(job_directory, target_file), 'parameters.py')
    os.remove(os.path.join(job_directory, target_file))

while True:

    p_child = subprocess.Popen(["/usr/bin/python2", "main.py"])

    # Define and register ctrl-c here.
    def sigint_handler(signum, frame):
        print "Caught SIGINT. Asking job to finish up. Will run another job if in queue."
        p_child.send_signal(signum)

    signal.signal(signal.SIGINT, sigint_handler)

    while p_child.poll() is None:
        time.sleep(1)

    # Job has finished run another if exists
    jobs = os.listdir(job_directory)
    if len(jobs) is 0:
        print "Jobs done. Dispatcher exiting."
        quit()
    else:
        target_file = jobs[0]
        shutil.copyfile(os.path.join(job_directory, target_file), 'parameters.py')
        os.remove(os.path.join(job_directory, target_file))


