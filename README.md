Recurrent Array for Driver Intention Prediciton (RADIP)

Licence to be added before public release. No warranties assumed or implied, yadda yadda

Installation deps

sudo apt-get install python-pip
sudo -H pip install --upgrade pip
sudo -H pip install tensorflow scipy pandas dill numpy bokeh matplotlib sklearn imblearn pathos GPy

Uses:
Inference
To perform inference, run the model loaded from a checkpoint with -c 'directory/to/results/train/best.../'
This will load the network from the checkpoint file in the specified directory. It will load the parameters
from a pkl file, except for the parameters used for clustering.
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
