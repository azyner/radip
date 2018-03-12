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

    def _pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    def CV_model(self,
             training_batch_handler,
             validation_batch_handler,
             test_batch_handler,
             parameters,
             report_df):
        return

    def CTRV_model(self,
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
        if 'angle' not in parameters['ibeo_data_columns'][2] or \
           'Velocity' not in parameters['ibeo_data_columns'][3]:
            raise ValueError('ibeo data columns need to contain speed and heading in positions 3 and 2')
        test_batch_handler.set_distance_threshold(0)
        test_batch_handler.reduced_pool
        CTRA_df = report_df.copy()
        CTRA_df = CTRA_df.drop(['outputs', 'mixtures'], axis=1)
        outputs = []
        p_steps = parameters['prediction_steps']
        for index, row in CTRA_df.iterrows():
            input_array = row.encoder_sample
            angle = input_array[:, 2]
            speed = input_array[:, 3]
            angle = [a if a > 0 else a + 2*np.pi for a in angle] # This moves the disjoint to
            # move angle back into continuous space
            #find turn rate
            d_angle_a = np.diff(angle)
            d_angle = np.mean(d_angle_a[-6:-1])
            #find accel
            accel_a = np.diff(speed)
            accel = np.mean(accel_a[-6:-1])

            #propagate n steps forward
            # Here its:
            # accel ==> velocity + C
            # velocity ==> delta_position
            # TR ==> yaws + C
            # That gives me a polar co-ord system for the position deltas and the yaws, so convert to cartesian.
            last_step = input_array[-1]
            last_pos = last_step[0:2]
            last_angle = angle[-1]
            last_speed = speed[-1]
            pred_angles = np.arange(last_angle, last_angle+((p_steps)*d_angle), d_angle)
            if last_speed < 0.0001:
                pred_speed = np.array([0.0]*p_steps)
            else:
                pred_speed = np.arange(last_speed, last_speed + ((p_steps) * accel), accel)
            try:
                x_d, y_d = self._pol2cart(pred_speed / (25.0 / parameters['subsample']), pred_angles + np.pi/2)
            except TypeError:
                ideas = None
            x_p = np.cumsum(x_d)
            y_p = np.cumsum(y_d)
            x_p += last_pos[0]
            y_p += last_pos[1]
            # Correct headings to be in -pi, pi

            prediction = np.array([x_p, y_p, pred_angles, pred_speed]).transpose()
            outputs.append(prediction)

        CTRA_df = CTRA_df.assign(outputs=outputs)
        return CTRA_df

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


