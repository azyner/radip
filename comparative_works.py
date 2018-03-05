import numpy as np

# This module is the one to run comparative
# So the input will be important:
# - index numbers of tracks, or the actual train / test data
# - This may be a cache of the results from the network
# - Depending on the methods, I may want to cache this too
# - How about the graph writer? Should that become more global to allow it to be run with data from here?
#   -- Probably.
#
# Have a look at the handles `collate_graphs.py' gets. I may need to add more to the results folder.
# --> This only ran for categorical tests. I'll have to re-do the whole thing.

# In order for me to begin this, I need a standardized graph. I also need a standardized metric at a specific set of
# data points. Do that first, and do it within the standard networkmanager class. I'll spin it out later.

# Models I want:
# CV, CA, CTRA -- All of which should fail miserably at an intersection, really.
# There is a Trivedi HMM VGMM (variational Gaussian Mixture Models) which
# seems absolutely parallel to my current work, and so would be valuable to include

# Metrics I want:
# Horizon based metrics: 0.5, 1.0, 1.5, 2.0, 2.5 sec etc
#   Median and Mean Absolute Error (cartesian)
#   Worst 5% and worst 1%

# Track based metrics:
# Modified Hausdorff Distance
# Euclidean Summation?

# Implementing ALL of the above is more work than I've seen in a lot of journals.
# So it should be very thorough.

class comparative_works():
    def __init__(self):
        """
        Collection of works for comparison purposes.
        Each model should read in:
         "report_df"
         "training_batch_handler"
         "validation_batch_handler"
         "test_batch_handler"
         Even if said model does not use all of the above.
         
         and return:
         a copy of "report_df"
         
         also consider caching results based on hashing track specific parameters and track_idxs
        """
        return

    def CV_model(self,
                 training_batch_handler,
                 validation_batch_handler,
                 test_batch_handler,
                 parameters,
                 report_df):
        return

    def CTV_model(self,
                 training_batch_handler,
                 validation_batch_handler,
                 test_batch_handler,
                 parameters,
                 report_df):
        return

    def CTRA_model(self,
                   training_batch_handler,
                   validation_batch_handler,
                   test_batch_handler,
                   parameters,
                   report_df):
        return

    def GaussianProcesses(self,
                   training_batch_handler,
                   validation_batch_handler,
                   test_batch_handler,
                   parameters,
                   report_df):
        return

    def HMMGMM(self,
                   training_batch_handler,
                   validation_batch_handler,
                   test_batch_handler,
                   parameters,
                   report_df):
        import hmmlearn
        training_data = training_batch_handler.data_pool.encoder_sample.as_matrix()
        # This gives me an object dtype array of shape (len,) containing arrays of (sequence_len, n_params)
        # and there is no good way of getting to (len, sequence_len, n_params). Which is frustrating

        training_array = []
        training_lengths = []
        for data_element in training_data:
            training_array.extend(data_element)
            training_lengths.append(len(data_element))
        from hmmlearn import hmm
        hmm_instance = hmm.GaussianHMM(n_components=10).fit(training_array,training_lengths)
        ideas = None

        return


