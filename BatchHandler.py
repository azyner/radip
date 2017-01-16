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

        self.val_minibatch_idx = 0
        self.d_thresh = None

        return

    def update_minibatch_selection_parameters(self,d_thresh=None):
        # TODO a function that changes some of the parameters used to select the mini-batch

        return

    # This function collects the mini-batch for training
    # If the network is under test, it will sequentially feed the testing data in size minibatch
    # The last mini-batch for the dataset is padded with junk data (taken from the start of the sequence)
    # The batch_complete flag signals the last mini-batch for the batch, so the system should collate results
    # pad_vector is TRUE if the data is junk (padding data)
    def get_minibatch(self):
        # TODO Research
        # Bias sampling, importance sampling, weighted sampling

        if self.training:
            # TODO I can set a vector p of probabilities of each pick. Can use this for the biased sampler
            # Do I want this stratified?

            batch_idxs = np.random.choice(range(len(self.data_pool)),self.batch_dim,replace=False)

            X_data = list(self.data_pool.iloc[batch_idxs].encoder_sample)
            if self.categorical:
                Y_data = list(self.data_pool.iloc[batch_idxs].dest_1_hot)

        if not self.training:
            # Pick sequentially, compute padding vector
            if self.d_thresh == None:
                # We are in a validation step, not a graph generation
                if (self.val_minibatch_idx+self.batch_dim) > len(self.data_pool):
                    # Collect the remaining data
                    X_data = list(self.data_pool.iloc[self.val_minibatch_idx:].encoder_sample)
                    Y_data = list(self.data_pool.iloc[self.val_minibatch_idx:].dest_1_hot)

                    pad_length = self.batch_dim - (len(self.data_pool) - self.val_minibatch_idx)

                    X_data.extend(list(self.data_pool.iloc[:pad_length].encoder_sample))
                    Y_data.extend(list(self.data_pool.iloc[:pad_length].dest_1_hot))

                    pad_vector = np.ones(self.batch_dim,dtype=bool)
                    pad_vector[:self.batch_dim-pad_length] = False

                    batch_complete = True
                    self.val_minibatch_idx = 0
                else:
                    X_data = list(self.data_pool.iloc[self.val_minibatch_idx:self.val_minibatch_idx+self.batch_dim].encoder_sample)
                    Y_data = list(self.data_pool.iloc[self.val_minibatch_idx:self.val_minibatch_idx+self.batch_dim].dest_1_hot)

                    self.val_minibatch_idx += self.batch_dim
                    pad_vector = np.ones(self.batch_dim, dtype=bool)

                    #Did we get lucky and have no remainder?
                    if self.val_minibatch_idx == len(self.data_pool):
                        batch_complete = True
                        self.val_minibatch_idx = 0
                    else:
                        batch_complete = False

                return X_data, Y_data, pad_vector,batch_complete

        return X_data, Y_data
