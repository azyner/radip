# Example parameters function
# To be renamed as parameters.py in local only, git set to ignore. This is such that I do not have to push param
# changes to git, they should all exist in a log file anyway
import numpy as np
import random
import os

parameters = {}
##### GLOBAL
parameters['AAA'] = "Logfile Notes"
parameters['embedding_size'] = 256  # 64 for each input
parameters["num_layers"] = 3
parameters['augmentation_chance'] = 0.5
parameters["dropout_prob"] = 0.5
parameters["embedding_dropout"] = 0.5

parameters["batch_size"] = 1024
parameters["observation_steps"] = 5

parameters['input_mask'] = [1,1,1,1]  # Used to investigate the usefullness of an input parameter

parameters['RNN_cell'] = "LSTMCell"
parameters['peephole_connections'] = True
parameters['l2_recurrent_decay'] = False
parameters['l2_lstm_input_decay'] = False

##### HYPER SEARCH
parameters['early_stop_cf'] = 40  # Time in minutes for training one crossfold
parameters['hyper_search_time'] = 12  # Time in hours for hyper searching

parameters['loss_decay_cutoff'] = 1e-10
parameters['long_training_time'] = 3*60  # Final training is for this long (minutes)
parameters['hyper_rnn_size_fn'] = random.uniform
parameters['hyper_rnn_size_args'] = (16, 513)
parameters['hyper_learning_rate_fn'] = random.uniform
parameters['hyper_learning_rate_args'] = (-5, -1)  # or None
parameters['aug_function'] = random.uniform
parameters['aug_range'] = (-3, 3)  # or None
parameters['evaluation_metric_type'] = 'validation_loss'  # "perfect_distance" / validation_accuracy

parameters['hyper_reg_embedding_beta_fn'] = random.uniform
parameters['hyper_reg_embedding_beta_args'] = (-5, -1)  # 10^X # OR None
parameters['hyper_reg_l2_beta_fn'] = random.uniform
parameters['hyper_reg_l2_beta_args'] = None #(-5, -1)  # 10^X # OR None

##### SINGLE RUN
parameters["learning_rate"] = 0.01
parameters["rnn_size"] = 128
parameters["learning_rate_decay_factor"] = 0.1
parameters['reg_embedding_beta'] = 0
parameters['l2_reg_beta'] = 0.001

##### STATIC
parameters['device'] = 'gpu:0'
parameters["n_folds"] = 5
parameters["input_columns"] = ['easting', 'northing', 'heading', 'speed']
parameters["prediction_steps"] = 0
parameters["feed_future_data"] = False
parameters["max_gradient_norm"] = 10.0
parameters["random_bias"] = 0
parameters["subsample"] = 1
parameters["random_rotate"] = False
parameters["num_mixtures"] = 6
parameters["model_type"] = "classifier"
parameters['train_dir'] = 'train'
parameters['d_thresh_top_n'] = 5    # How many samples to take that exist immediately before d_thresh
parameters['steps_per_checkpoint'] = 200
parameters['decrement_steps'] = 15

parameters['debug'] = False  # Skip the metric computation to hasten looptime

# IBEO
parameters['ibeo_data_columns'] = ["Object_X","Object_Y","ObjBoxOrientation","AbsVelocity_X","AbsVelocity_Y","ObjectPredAge"]
parameters["data_format"] = "ibeo" # OR 'legacy'
parameters["use_scaling"] = True
#C hange this to 1 or zero to set the GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
