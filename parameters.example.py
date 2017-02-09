# Example parameters function
# To be renamed as parameters.py in local only, git set to ignore. This is such that I do not have to push param
# changes to git, they should all exist in a log file anyway

parameters = {}
parameters['device'] = 'gpu:0'

parameters["n_folds"] = 3
parameters["num_rnn_layers"] = 3
parameters["learning_rate"] = 0.01
parameters["observation_steps"] = 5
parameters["prediction_steps"] = 0
parameters["feed_future_data"] = False
parameters["batch_size"] = 17
parameters["rnn_size"] = 128
parameters['embedding_size'] = 256  # 64 for each input
parameters["num_layers"] = 3
parameters["learning_rate"] = 0.01
parameters["learning_rate_decay_factor"] = 0.1
parameters["max_gradient_norm"] = 10.0
parameters["dropout_prob"] = 0.5
parameters["random_bias"] = 0
parameters["subsample"] = 1
parameters["random_rotate"] = False
parameters["num_mixtures"] = 6
parameters["model_type"] = "classifier"
parameters["input_columns"] = ['easting', 'northing', 'heading', 'speed']
parameters['train_dir'] = 'train'
parameters['early_stop_cf'] = 0.1 # Time in minutes for training one crossfold
parameters['hyper_search_time'] = 5.0/60 # Time in hours for hyper searching
parameters['decrement_steps'] = 15
parameters['d_thresh_top_n'] = 5    #How many samples to take that exist immediately before d_thresh
parameters['steps_per_checkpoint'] = 200
parameters['loss_decay_cutoff'] = 1e-10