#Class to take a list of continuous, contiguous data logs that need to be collated and split for the data feeder
#Is this different to the batch handler?
#The SequenceWrangler is aware of the three data pools, training, test and val
#

class SequenceWrangler:
    def __init__(self, data, training=0.8,val=0.1,test=0.1):
        return #pool pool pool


    def _sequence_splitter(self):
        # Start ripping code from last project.
        # It needs to use pandas, such that for each data sample, there is a wrapper class containing a list of properties
        return

    def _generate_pools(self):
        # The `go' button. This command writes the pool of data. The sequence length requirement is allowed to change during training or test time.
        # Split data track wise - how should this be done, randomly, pseudo random with an optional seed?
        # Call sequence splitter
        #
        train_pool, train_pool, val_pool = [],[],[]
        return train_pool, train_pool, val_pool
