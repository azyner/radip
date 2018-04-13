# Example parameters function
# To be renamed as parameters.py in local only, git set to ignore. This is such that I do not have to push param
# changes to git, they should all exist in a log file anyway
import numpy as np
import random
import os

parameters = {}
##### GLOBAL
parameters['AAA'] = "Dispatcher Test 3"
parameters['data_list'] = [
    'queen-hanks',
    'leith-croydon',
    'roslyn-crieff',
    'oliver-wyndora',
    'orchard-mitchell'
]
parameters['test_csv'] = 'oliver-wyndora'

parameters['embedding_size'] =256 # 512  # 64 for each input
parameters["num_layers"] = 3
parameters['augmentation_chance'] = 0.0
parameters["dropout_prob"] = 0.0
parameters["embedding_dropout"] = 0.0

parameters["batch_size"] = 100
parameters["observation_steps"] = 15

#parameters['input_mask'] = [1,1,1,1]  # Used to investigate the usefullness of an input parameter

parameters['RNN_cell'] = "LSTMCell"
parameters['peephole_connections'] = True
parameters['l2_recurrent_decay'] = False
parameters['l2_lstm_input_decay'] = False

##### HYPER SEARCH
parameters['early_stop_cf'] = 6*60  # Time in minutes for training one crossfold
parameters['hyper_search_folds'] = 0 # Number of hyper searching attempts.
parameters['hyper_search_step_cutoff'] = 80000

parameters['loss_decay_cutoff'] = 1e-20
parameters['long_training_time'] = 48*60  # Final training is for this long (minutes)
parameters['long_training_steps'] = 2000

parameters['hyper_rnn_size_fn'] = random.uniform
parameters['hyper_rnn_size_args'] = (128, 1025)
parameters['hyper_learning_rate_fn'] = random.uniform
parameters['hyper_learning_rate_args'] = (-10, -3)
parameters['aug_function'] = random.uniform
parameters['aug_range'] = (-3, 3) 
parameters['evaluation_metric_type'] = 'validation_loss'  # "perfect_distance" / validation_accuracy

parameters['hyper_reg_embedding_beta_fn'] = random.uniform
parameters['hyper_reg_embedding_beta_args'] = (-10, -1)  # 10^X # OR None
parameters['hyper_reg_l2_beta_fn'] = random.uniform
parameters['hyper_reg_l2_beta_args'] = (-30, -8)  # 10^X # OR None
parameters['hyper_learning_rate_decay_args'] = (0.9, 1.0)
parameters['hyper_learning_rate_min_args'] = (-10, -2)

##### SINGLE RUN
parameters["learning_rate"] = 0.0005
parameters["learning_rate_min"] = 0.00001
parameters["rnn_size"] = 256
parameters["learning_rate_decay_factor"] = 0.9999
parameters['reg_embedding_beta'] = 0.0
parameters['l2_reg_beta'] = 0.0

##### STATIC
parameters['device'] = 'gpu:0'
parameters["n_folds"] = 5
parameters["input_columns"] = ['easting', 'northing', 'heading', 'speed']
parameters["prediction_steps"] = 25
parameters["feed_future_data"] = False
parameters["first_loss_only"] = False
parameters["max_gradient_norm"] = 10.0
parameters["random_bias"] = 0
parameters["subsample"] = 2
parameters["random_rotate"] = False
parameters["num_mixtures"] = 7
parameters["model_type"] = "MDN"
parameters['train_dir'] = 'train'
parameters['d_thresh_top_n'] = 1   # How many samples to take that exist immediately before d_thresh
parameters['steps_per_checkpoint'] = 500
parameters['decrement_steps'] = 1000

parameters['debug'] = False  # Skip the metric computation to hasten looptime

# IBEO
#parameters['ibeo_data_columns'] = ["Object_X","Object_Y","ObjBoxOrientation","AbsVelocity"]#_X","AbsVelocity_Y","ObjectPredAge"]
parameters['ibeo_data_columns'] = ["relative_x","relative_y","relative_angle","AbsVelocity"]#_X","AbsVelocity_Y","ObjectPredAge"]
parameters['input_mask'] = [1,1,1,1]  # Used to investigate the usefullness of an input parameter
parameters["data_format"] = "ibeo" # OR 'legacy'
parameters["use_scaling"] = True
parameters["velocity_threshold"] = 2.0
parameters["track_padding"] = True
parameters['sample_temperature'] = 1.0
parameters['padding_loss_weight'] = 0.0
parameters['no_feedforward'] = True
parameters['reject_stopped_vehicles_before_intersection_enable'] = True
parameters['reject_stopped_vehicles_before_intersection_speed'] = 1  # meters per second, 1 = 3.6kph, there is sensor noise meaning no true zero
parameters['reject_stopped_vehicles_before_intersection_duration'] = 1.0  # seconds

#C hange this to 1 or zero to set the GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
