#Example parameters function
#To be renamed as parameters.py in local only, git set to ignore. This is such that I do not have to push param
# changes to git, they should all exist in a log file anyway

parameters = {}
parameters["num_rnn_layers"] = 3
parameters["learning_rate"] = 0.01
parameters["observation_steps"] = 5
parameters["prediction_steps"] = 0
parameters["feed_future_data"] = False
parameters["batch_size"] = 17
parameters["rnn_size"] = 128
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