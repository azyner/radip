Recurrent Array for Driver Intention Prediciton (RADIP)

Code to acompany paper:
Naturalistic Driver Intention and Path Prediction using Recurrent Neural Networks
https://arxiv.org/abs/1807.09995

Installation deps

sudo apt-get install python-pip
sudo -H pip install --upgrade pip
sudo -H pip install tensorflow scipy pandas dill numpy bokeh matplotlib sklearn imblearn pathos GPy

Dataset:
This needs to be downloaded.
Either find it in http://its.acfr.usyd.edu.au/datasets/ or email the author if he hasn't uploaded it yet.
Place the dataset into radip/data/

Uses:
Inference
To perform inference, run the model loaded from a checkpoint with:

$ ./main.py -c journal_models/RNN-FF/train/best-1529969240.38/

This will load the network from the checkpoint file in the specified directory. It will load the parameters
from a pkl or csv file, except for the parameters used for clustering.
This will run all of the test tracks through the model, and then cluster them again using the clustering algorithm.
The results are then presented as a plot for each track in test data plots.
Plots of a sequence will also be output (05m, 0m, 5m, 10m, 20m) in the sequential_test_data_plots folder

Training
To train a model, copy parameters.example.py to parameters.py and change any desired params. Run main.py with no args.
The program will then create a directory named after the current date and time in results.
It will then copy the parameters file and a git log into this folder.
It will then scan for the input data either the original csvs or the preprocessed pkls.
This data will then be wrangled into snippets of the appropriate length, and this data will be saved in a checkpoint
This model will then train for the specified time or step count limit. Afterwards, it will perform inference
as above.

The training can be stopped early with Ctrl-C. It will then train to the next checkpoint and then run the above metrics
and plots.


Tools:
**run_dispatcher.py

To use this tool, place several parameters.py files (named as anything) into a radip/jobs folder, and optionally delete
the local parameters.py file, else it is run first. Then run run_dispatcher. For each job it will move the parameters
folder locally and run the system again. This allows for  several jobs to be queued over a week etc. Combining this with
several checkouts (one per GPU), symlinking the job folders between checkouts, and CUDA_VISIBLE_DEVICES=[0,1,2] allows
for jobs to be run simultaneously on different GPUs without duplicating jobs. This is a crude approximation of a server
job dispatcher. Ctrl-C stops training early and moves on to the next job. Use ctrl-z and kill -9 %1 to kill worker.


** run_checkpoints_dir.py -c folder/that/contains/several_results
Will run from checkpoint all saved results in a directory. Useful for edits to the output functions. Or just to run
all the output figures because they are not shared with the network checkpoints.

** csv_results_consolidate.py -c folder/that/contains/several_results
Find the best network as defined by the performance on a single metric. Change the metric with arguments.
Use -h for args list.

** python2 plot_all_tracks
Plots all vehicle tracks in the dataset, in an all_tracks_plot folder. Creates over 23000 images.
