# Class that handles the batch sampling, all data etc
# In main this should be init'd three times, for each data pool
# I need to brainstorm how to implement the features in this class, such as batch selection bias
# Also data augmentation goes here.

import random
import pandas as pd
import numpy as np
from sklearn import preprocessing

class BatchHandler:
    def __init__(self, data_pool, batch_dim, training):
        self.data_pool = data_pool
        self.batch_dim = batch_dim
        self.training = training
        # Training is defined as the boolean flag of whether the data is for training or test
        # During training, the data is sampled from a pool
        # During test, the data is sampled sequentially, and exhaustively.
        # A vector needs to be given whether the data is padding data at the end of the dataset
        # A return state needs to be given to state if all test data is given.
        self.categorical = True

        # self.data_pool.destination_vec.unique()
        # dest_encoder = preprocessing.OneHotEncoder()
        # dest_encoder.fit(self.data_pool.destination_vec.unique())



        #make a 1 hot encoder for dest_labels here

        return

    def update_minibatch_selection_parameters(self,d_thresh=None):
        # TODO a function that changes some of the parameters used to select the mini-batch

        return


    def get_minibatch(self):
        if self.training:
            batch_idxs = np.random.choice(range(len(self.data_pool)),self.batch_dim) #make this more intelligent somehow

        # TODO Research
        # Bias sampling, importance sampling, weighted sampling
        X_data = list(self.data_pool.iloc[batch_idxs].encoder_sample)
        if self.categorical:
            Y_data = list(self.data_pool.iloc[batch_idxs].dest_1_hot)
        return X_data, Y_data
