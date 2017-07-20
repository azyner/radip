# Class that handles the batch sampling, all data etc
# In main this should be init'd three times, for each data pool
# I need to brainstorm how to implement the features in this class, such as batch selection bias
# Also data augmentation goes here.

import random
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler
import sys

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
        self.d_thresh_range = None

        self.val_minibatch_idx = 0
        self.d_thresh = None
        self.reduced_pool = None
        self.distance_pool_cache = {}
        self.input_mask = pd.Series([np.tile(self.parameters['input_mask']
                                  ,
                                  (self.parameters['observation_steps'],1)
                                 )
                             for x in range(self.batch_size)
                             ],
                            dtype=object,index=([0]*self.batch_size))

        # Generate balanced index list
        ros = RandomOverSampler()
        selection_data = list(data_pool.track_class.as_matrix())
        le = preprocessing.LabelEncoder()
        le.fit(selection_data)
        indexed_classes = np.array(le.transform(selection_data))
        ros.fit(np.expand_dims(range(len(indexed_classes)),1),indexed_classes)
        balanced_idxs, balanced_classes = ros.sample(np.expand_dims(range(len(indexed_classes)),1),indexed_classes)
        self.balanced_idxs = np.squeeze(balanced_idxs)
        # bf = data_pool.iloc[balanced_idxs]
        # class_dict = {}
        # for class_t in data_pool.track_class.unique():
        #     class_dict[class_t] = len(bf[bf.track_class==class_t])/float(len(bf))
        return

    def get_input_size(self):
        return len(self.data_pool.iloc[0]['encoder_sample'][0])

    def get_num_classes(self):
        return len(self.data_pool['destination'].unique())

    def set_distance_threshold(self, d_thresh=None):
        # TODO Cache these.
        self.d_thresh = d_thresh
        # for every track_idx, find the sample that is max(d<thresh)

        if d_thresh is None:
            self.reduced_pool = None
            return

        rp = []
        for track_idx in self.data_pool['track_idx'].unique():
            pool = self.data_pool[self.data_pool['track_idx']==track_idx]
            tp = pool[pool['distance']<d_thresh] # thresholded pool - everything that came before d_thresh
            # Sort by distance, pick closest
            try:
                record = tp.sort_values('distance',ascending=False).iloc[range(self.parameters['d_thresh_top_n'])] #
            except IndexError:
                # TODO Change p_dis range values.
                continue

            # Double list as it will return a Series of type object otherwise, ruining all labels, breaking the data
            # structure, and wasting an afternoon of my life.

            # if the closest point is within 1m, it is valid data. Else we have no data at this threshold.
            if record.iloc[0].distance > (d_thresh -1):
                rp.append(record)

        self.reduced_pool = pd.concat(rp)

        return

    def set_distance_threshold_ranges(self, d_thresh_range):
        self.d_thresh_range = d_thresh_range
        return



    #Function that gets the data as a list of sequences, (which are time length lists of features)
    # i.e. a list of length batch size, containing [time, input_size] elements
    # and converts it to a list of length time, containing [batch input_size] elements

    def format_minibatch_data(self, X, Y, pad_vector):
        if type(X) is not list:
            X = list(X)
        if type(Y) is not list:
            Y = list(Y)
        if type(pad_vector) is not list:
            pad_vector = list(pad_vector)

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

        if self.training:
            # Select randomly such that there is a balance between classes. Over sampling is used for small classes
            batch_idxs = np.random.choice(self.balanced_idxs, self.batch_size, replace=False)
        else:
            # Select uniformly at random
            batch_idxs = np.random.choice(range(len(self.data_pool)), self.batch_size, replace=False)

        # class_dict = {}  # BALANCER VERIFICATION CODE
        # for class_t in data_pool.track_class.unique():
        #     class_dict[class_t] = len(bf[bf.track_class==class_t])/float(len(bf))
        # print class_dict

        batch_frame = self.data_pool.iloc[batch_idxs].copy()
        num_columns = batch_frame.encoder_sample.iloc[0].shape[1]

        if self.training and self.parameters['augmentation_chance'] > 0.001:
            # Generate same size matrix that contains the offsets
            # i.e. e,n,0,0 * REPMATRIX(encoder_length) for all samples
            # np.tile([randomx,randomy,0,0],(len_enc_samples,1))
            aug = pd.Series([
                            np.tile([
                                  self.parameters['aug_function'](*self.parameters['aug_range']),
                                  self.parameters['aug_function'](*self.parameters['aug_range'])
                                  ]+[0.0]*(num_columns-2)
                                  ,
                                  (self.parameters['observation_steps'],1)
                                 )
                             for x in range(self.batch_size)
                             ],
                            dtype=object,index=([0]*self.batch_size))
            aug_mask = pd.Series([
                                np.tile([np.random.choice([1.0,0.0],p=[self.parameters['augmentation_chance'],
                                                               1-self.parameters['augmentation_chance']])]
                                         * num_columns
                                        ,
                                        (self.parameters['observation_steps'], 1)
                                        )
                                for x in range(self.batch_size)
                                ],
                            dtype=object, index=([0] * self.batch_size))
            batch_frame.encoder_sample = batch_frame.encoder_sample + (aug*aug_mask)
        batch_frame.encoder_sample = batch_frame.encoder_sample*self.input_mask

        batch_frame = batch_frame.assign(padding=np.zeros(self.batch_size, dtype=bool))
        return batch_frame # batch_X, batch_Y, batch_weights

        # Testing / validating
    def get_sequential_minibatch(self):
            # Pick sequentially, compute padding vector
            if self.d_thresh is None:
                data_pool = self.data_pool
            else:
                data_pool = self.reduced_pool

            #data_pool = NEW FUNCTION THAT RUNS ALL THE SET_DIS_THRESHOLD AND ADDS D_THRESH TO THE FRAME
            # if d_thresh is not none, I would reduce the dataset some way
            if self.d_thresh_range is not None:
                data_pool = self.generate_distance_pool()

            # If we do not have enough data remaining to fill a batch
            if (self.val_minibatch_idx+self.batch_size) > len(data_pool):
                # Collect the remaining data
                batch_frame = data_pool.iloc[self.val_minibatch_idx:].copy()
                batch_frame = batch_frame.assign(padding=np.zeros(len(batch_frame), dtype=bool))

                total_padding = self.batch_size - (len(batch_frame))
                pad_vector = np.zeros(self.batch_size, dtype=bool)
                #The last n are garbage
                pad_vector[-total_padding:] = True

                while len(batch_frame) < self.batch_size:
                    # Add garbage to the end, repeat if necessary
                    pad_length = self.batch_size - (len(batch_frame))
                    # This works because if pad_length > len(data_pool), it just returns the whole pool
                    padding_frame = data_pool.iloc[0:pad_length].copy()
                    padding_frame = padding_frame.assign(padding=np.ones(len(padding_frame), dtype=bool))
                    batch_frame = pd.concat([batch_frame, padding_frame])

                batch_complete = True
                self.val_minibatch_idx = 0
            else:
                batch_frame = data_pool.iloc[self.val_minibatch_idx:self.val_minibatch_idx + self.batch_size].copy()
                batch_frame = batch_frame.assign(padding=np.zeros(len(batch_frame), dtype=bool))
                self.val_minibatch_idx += self.batch_size

                # Did we get lucky and have no remainder?
                if self.val_minibatch_idx == len(data_pool):
                    batch_complete = True
                    self.val_minibatch_idx = 0
                else:
                    batch_complete = False

            return batch_frame, batch_complete


    # Is this just the above function with a pool colleciton and partnered distance list?
    # Do I even need the partnered list, or just append d_thresh?
    def generate_distance_pool(self):
        pool_list = []
        try:
            return self.distance_pool_cache[tuple(self.d_thresh_range)]
        except KeyError:
            busy_indicator = ['.', 'o', 'O','@', '*']
            batch_counter = 0
            print ''
            for dis in self.d_thresh_range:
                sys.stdout.write("\rGenerating validation data pool cache...%s" % busy_indicator[batch_counter % len(busy_indicator)])
                sys.stdout.flush()
                self.set_distance_threshold(dis)
                local_pool = self.reduced_pool.copy()
                local_pool = local_pool.assign(d_thresh=np.repeat(dis, len(local_pool)))
                pool_list.append(local_pool)
                batch_counter+=1
            pool_df = pd.concat(pool_list)
            self.distance_pool_cache[tuple(self.d_thresh_range)] = pool_df
            print ''
            return pool_df


        # Unwind all pools into one big pool with partnered distance list.
