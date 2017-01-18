# Class that handles the batch sampling, all data etc
# In main this should be init'd three times, for each data pool
# I need to brainstorm how to implement the features in this class, such as batch selection bias
# Also data augmentation goes here.

import random
import pandas as pd
import numpy as np
from sklearn import preprocessing

class BatchHandler:
    def __init__(self, data_pool, parameters, training):
        self.data_pool = data_pool
        self.parameters = parameters
        self.batch_size = parameters['batch_size']
        self.training = training
        # Training is defined as the boolean flag of whether the data is for training or test
        # During training, the data is sampled from a pool
        # During test, the data is sampled sequentially, and exhaustively.
        # A vector needs to be given whether the data is padding data at the end of the dataset
        # A return state needs to be given to state if all test data is given.
        self.categorical = True

        self.val_minibatch_idx = 0
        self.d_thresh = None
        self.reduced_pool = None

        return

    def get_input_size(self):
        return len(self.data_pool.iloc[0]['encoder_sample'][0])

    def get_num_classes(self):
        return len(self.data_pool['destination'].unique())

    def set_distance_threshold(self, d_thresh=None):
        # TODO a function that changes some of the parameters used to select the mini-batch
        self.d_thresh = d_thresh
        # for every track_idx, find the sample that is max(d<thresh)

        rp = []
        for track_idx in self.data_pool['track_idx'].unique():
            pool = self.data_pool[self.data_pool['track_idx']==track_idx]
            tp = pool[pool['distance']>d_thresh] # thresholded pool
            # Sort by distance, pick closest
            record = tp.sort_values('distance',ascending=False).iloc[[0]] #
            # Double list as it will return a Series of type object otherwise, ruining all labels, breaking the data
            # structure, and wasting an afternoon of my life.

            # Note, Why only the closest, not the closest n?
            rp.append(record)

        self.reduced_pool = pd.concat(rp)

        return

    #Function that gets the data as a list of sequences, (which are time length lists of features)
    # i.e. a list of length batch size, containing [time, input_size] elements
    # and converts it to a list of length time, containing [batch input_size] elements

    def format_minibatch_data(self, X, Y, pad_vector):
        batch_observation_inputs, batch_future_inputs, batch_weights, batch_labels = [], [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        # Need to re-index to make an encoder_steps long list of shape [batch input_size]
        # currently it is a list of length batch containing shape [timesteps input_size]
        for length_idx in xrange(self.parameters['observation_steps']):
            batch_observation_inputs.append(
                    np.array([X[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.float32))

        for length_idx in xrange(self.parameters['prediction_steps']):
            batch_future_inputs.append(
                    np.array([Y[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.float32))
            # TODO wrangle pad_vector here
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            batch_weights.append(batch_weight)

        # Encapsulate the label data in a list of size 1 to mimic a decoder seq of len 1
        if self.parameters['prediction_steps'] == 0:
            batch_labels = [Y]
            batch_weights = [np.logical_not(pad_vector)*np.ones(self.batch_size, dtype=np.float32)]

        # Batch_observation_inputs is now list of len encoder_steps, shape batch, input_size.
        #  Similarly with batch_future_inputs
        return batch_observation_inputs, batch_future_inputs, batch_weights, batch_labels

    # This function collects the mini-batch for training
    # If the network is under test, it will sequentially feed the testing data in size minibatch
    # The last mini-batch for the dataset is padded with junk data (taken from the start of the sequence)
    # The batch_complete flag signals the last mini-batch for the batch, so the system should collate results
    # pad_vector is TRUE if the data is junk (padding data)
    def get_minibatch(self):
        # TODO Research
        # Bias sampling, importance sampling, weighted sampling

        # TODO I can set a vector p of probabilities of each pick. Can use this for the biased sampler
        # Do I want this stratified?

        batch_idxs = np.random.choice(range(len(self.data_pool)), self.batch_size, replace=False)

        X_data = list(self.data_pool.iloc[batch_idxs].encoder_sample)
        if self.categorical:
            Y_data = list(self.data_pool.iloc[batch_idxs].dest_1_hot)

        # Nothing is padding, so np-zeros for pad vector
        batch_X, _, batch_weights, batch_Y = self.format_minibatch_data(X_data, Y_data, np.zeros(self.batch_size, dtype=bool))

        return batch_X, batch_Y, batch_weights

        # Testing / validating
    def get_sequential_minibatch(self):
            # Pick sequentially, compute padding vector
            if self.d_thresh is None:
                data_pool = self.data_pool
            else:
                data_pool = self.reduced_pool
            # if d_thresh is not none, I would reduce the dataset some way

            # If we do not have enough data remaining to fill a batch
            if (self.val_minibatch_idx+self.batch_size) > len(data_pool):
                # Collect the remaining data
                X_data = list(data_pool.iloc[self.val_minibatch_idx:].encoder_sample)
                Y_data = list(data_pool.iloc[self.val_minibatch_idx:].dest_1_hot)
                batch_frame = data_pool.iloc[self.val_minibatch_idx:].copy()
                batch_frame = batch_frame.assign(padding=np.zeros(len(batch_frame), dtype=bool))

                total_padding = self.batch_size - (len(X_data))
                pad_vector = np.zeros(self.batch_size, dtype=bool)
                #The last n are garbage
                pad_vector[-total_padding:] = True

                while len(X_data) < self.batch_size:
                    # Add garbage to the end, repeat if necessary
                    pad_length = self.batch_size - (len(X_data))
                    # This works because if pad_length > len(data_pool), it just returns the whole pool
                    X_data.extend(list(data_pool.iloc[0:pad_length].encoder_sample))
                    Y_data.extend(list(data_pool.iloc[0:pad_length].dest_1_hot))
                    padding_frame = data_pool.iloc[0:pad_length].copy()
                    padding_frame = padding_frame.assign(padding=np.ones(len(padding_frame), dtype=bool))
                    batch_frame = pd.concat([batch_frame, padding_frame])


                batch_complete = True
                self.val_minibatch_idx = 0
            else:
                X_data = list(data_pool.iloc[self.val_minibatch_idx:self.val_minibatch_idx+self.batch_size].encoder_sample)
                Y_data = list(data_pool.iloc[self.val_minibatch_idx:self.val_minibatch_idx+self.batch_size].dest_1_hot)
                batch_frame = data_pool.iloc[self.val_minibatch_idx:self.val_minibatch_idx + self.batch_size].copy()
                batch_frame = batch_frame.assign(padding=np.zeros(len(batch_frame), dtype=bool))
                self.val_minibatch_idx += self.batch_size
                pad_vector = np.zeros(self.batch_size, dtype=bool)

                # Did we get lucky and have no remainder?
                if self.val_minibatch_idx == len(data_pool):
                    batch_complete = True
                    self.val_minibatch_idx = 0
                else:
                    batch_complete = False

            frame_x, _, frame_weights, frame_Y = self.format_minibatch_data(list(batch_frame.encoder_sample),
                                                                            list(batch_frame.dest_1_hot),
                                                                            list(batch_frame.padding))
            batch_X, _, batch_weights, batch_Y = self.format_minibatch_data(X_data, Y_data, pad_vector)

            # FIXME frame_ and batch_ are now identical. I can now return the dataframe, which allows
            # the graph generator to pick out destination labels and group them
            
            return batch_X, batch_Y, batch_weights, pad_vector, batch_complete


