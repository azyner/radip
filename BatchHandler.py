# Class that handles the batch sampling, all data etc
# In main this should be init'd three times, for each data pool
# I need to brainstorm how to implement the features in this class, such as batch selection bias
# Also data augmentation goes here.

import random
import pandas as pd

class BatchHandler:
    def __init__(self, data_pool, batch_dim, training):
        self.data_pool = data_pool
        self.batch_dim = batch_dim
        self.training = training

        #make a 1 hot encoder for dest_labels here

        return

    def get_minibatch(self):
        batch_idxs = random.choice(range(len(self.data_pool))) #make this more intelligent somehow

        # TODO Research
        # Bias sampling, importance sampling, weighted sampling

        return
