#Example parameters function
#To be renamed as parameters.py in local only, git set to ignore. This is such that I do not have to push param
# changes to git, they should all exist in a log file anyway
import numpy as np
import random

parameters = {}
parameters['device'] = 'gpu:0'

parameters["n_folds"] = 3
parameters["num_rnn_layers"] = 3
parameters["learning_rate"] = 0.001
parameters["observation_steps"] = 5
parameters["prediction_steps"] = 0
parameters["feed_future_data"] = False
parameters["batch_size"] = 1024
parameters["rnn_size"] = 120
parameters['embedding_size'] = 256  # 64 for each input
parameters["num_layers"] = 3
parameters["learning_rate_decay_factor"] = 0.1
parameters["max_gradient_norm"] = 10.0
parameters["dropout_prob"] = 0.5
parameters["embedding_dropout"] = 0.5
parameters["random_bias"] = 0
parameters["subsample"] = 1
parameters["random_rotate"] = False
parameters["num_mixtures"] = 6
parameters["model_type"] = "classifier"
parameters["input_columns"] = ['easting', 'northing', 'heading', 'speed']
parameters['input_mask'] = [1,1,1,1] # Used to investigate the usefullness of an input parameter
parameters['train_dir'] = 'train'
parameters['early_stop_cf'] = 30.0 # Time in minutes for training one crossfold
parameters['hyper_search_time'] = 12 # Time in hours for hyper searching
parameters['decrement_steps'] = 15
parameters['d_thresh_top_n'] = 1    #How many samples to take that exist immediately before d_thresh
parameters['steps_per_checkpoint'] = 200
parameters['loss_decay_cutoff'] = 1e-10
parameters['long_training_time'] = 180 # Final training is for this long (minutes)
parameters['hyper_rnn_size_fn'] = random.uniform
parameters['hyper_rnn_size_args'] = (16,257)
parameters['hyper_learning_rate_fn'] = random.uniform
parameters['hyper_learning_rate_args'] = (-5,-1)
parameters['augmentation_chance'] = 0.5
parameters['aug_function'] = random.uniform
parameters['aug_range'] = (-5,5)
parameters['evaluation_metric_type'] = 'validation_loss' # "perfect_distance" / validation_accuracy
parameters['reg_embedding_beta'] = 0
parameters['hyper_reg_embedding_beta_fn'] = random.uniform
parameters['hyper_reg_embedding_beta_args'] = (-5,-1) #10^X # OR None

