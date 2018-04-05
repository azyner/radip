import numpy as np
import pandas as pd

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
        return self.CTRA_model(training_batch_handler,
                               validation_batch_handler,
                               test_batch_handler,
                               parameters,
                               report_df, CV=True, C_TR=True)

    def CTRV_model(self,
                   training_batch_handler,
                   validation_batch_handler,
                   test_batch_handler,
                   parameters,
                   report_df):
        return self.CTRA_model(training_batch_handler,
                               validation_batch_handler,
                               test_batch_handler,
                               parameters,
                               report_df, C_TR=True)

    def CTRA_model(self,
                   training_batch_handler,
                   validation_batch_handler,
                   test_batch_handler,
                   parameters,
                   report_df, CV=False, C_TR=False):
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
            if C_TR:
                d_angle = 0
            else:
                d_angle_a = np.diff(angle)
                d_angle = np.mean(d_angle_a[-6:-1])
            #find accel
            if CV:
                accel = 0
            else:
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
            if abs(d_angle) < 0.0000001:
                pred_angles = np.array([last_angle] * p_steps)
            else:
                # An occasional rounding error occurs when subsample != 1, thus [:p_steps] was added to trim.
                pred_angles = np.arange(last_angle, last_angle + (p_steps * d_angle), d_angle)[:p_steps]
            if last_speed < 0.0001:
                pred_speed = np.array([0.0] * p_steps)
            elif abs(accel) < 0.000001:
                pred_speed = np.array([last_speed] * p_steps)
            else:
                pred_speed = np.arange(last_speed, last_speed + (p_steps * accel), accel)
            try:
                x_d, y_d = self._pol2cart(pred_speed / (25.0 / parameters['subsample']), pred_angles + np.pi/2)
            except:
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
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                                      ExpSineSquared, DotProduct,
                                                      ConstantKernel, WhiteKernel)
        training_encoder_data = training_batch_handler.data_pool.encoder_sample.as_matrix()
        training_decoder_data = training_batch_handler.data_pool.decoder_sample.as_matrix()
        # Now I want to reshape this such that n_samples is n_tracks, and features is unrolled track data
        X = []
        X_short = []
        for track in training_encoder_data:
            X_short.append(track[-2:].flatten())
            X.append(track.flatten())
        y = []
        y_short = []
        for track in training_decoder_data:
            y_short.append(track[0:2].flatten())
            y.append(track.flatten())
        X = np.array(X)
        y = np.array(y)
        X_short = np.array(X_short)
        y_short = np.array(y_short)

        import GPy
        n_samples = 8000 #X_short.shape[0]
        gp_succeeded = False
        while not gp_succeeded:
            try:
                sample_idxs = np.random.choice(X_short.shape[0], n_samples, replace=False)
                X_r = X[sample_idxs]
                Y_r = y[sample_idxs]
                X_sr = X_short[sample_idxs]
                Y_sr = y_short[sample_idxs]

                ker = GPy.kern.Matern52(2,ARD=True) + GPy.kern.White(2)
                m = GPy.models.GPRegression(X_r, Y_r)
                m.optimize(messages=True,max_f_eval = 1000)
                gp_succeeded = True
            except MemoryError:
                n_samples *= 0.9
                print "GP ran out of memory, number of samples reduced to: " + str(n_samples)


        # import pyGPs
        # model = pyGPs.GPR_FITC()
        # model.setData(X_short[0:1000], y_short[0:1000])
        # gp_kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(1e-1)
        # gpr = GaussianProcessRegressor()#kernel=gp_kernel)
        # gpr.fit(X[0:100], y[0:100])
        # t = gpr.sample_y(X[0:1]).reshape(20,4, order='a')
        outputs = []
        ideas = None
        test_batch_handler.set_distance_threshold(0)
        test_batch_handler.reduced_pool
        GP_df = report_df.copy()
        GP_df = GP_df.drop(['outputs', 'mixtures'], axis=1)
        for index, row in GP_df.iterrows():
            input_array = row.encoder_sample
            output, _ = m.predict(np.array([input_array.flatten()]))
            outputs.append(output[0].reshape(len(output[0])/4,4,order='a'))

        GP_df = GP_df.assign(outputs=outputs)
        return GP_df

    def classifierComboDistributionEstimator(self, training_batch_handler,
                                                   validation_batch_handler,
                                                   test_batch_handler,
                                                   parameters,
                                                   report_df):
        # This is a reproduction of the work of Nachiket Deo from UC San Diego
        # And "Probabilistic Trajectory Prediction with GMM's" Weist et. al.

        # The main flow of these papers is as follows:

        # Training:
        # Given the historical and future track data of each track snippet:
        # Step 1:
        #   Group the historicals using your chosen clustering algorithm (HMM is used above,
        #   realistically anything should work: kNN, GP, doesn't functionally matter)
        #   This is a parametric solution where the number of groups is chosen as a hyperparameter
        # Step 2:
        #   Group tracks in the training set via the above.
        #   For each track:
        #       Group each data point by time step
        #       Re:zero coordinates such that the last historical is at pos (0,0)
        #       train a new statistical fit model (GMM, VGMM etc) on each time step
        #       Use this to estimate future time
        #
        # Inference:
        # Step 1:
        #   Cluster track based on history
        # Step 2:
        #   Lookup appropriate GMM from library
        # Step 3:
        #   Rebase estimate based on last observed position

        ##########################################################################################
        # WARNING -- This assumes the order in encoder and decoder sample are the same (pandas says yes)
        training_encoder_data = training_batch_handler.data_pool.encoder_sample.as_matrix()
        training_decoder_data = training_batch_handler.data_pool.decoder_sample.as_matrix()

        # Data pre-processing
        # Here the data is flattened as most classifiers want a 1d vector, not a 2d vector
        # Now I want to reshape this such that n_samples is n_tracks, and features is unrolled track data
        X = []
        X_short = []
        for track in training_encoder_data:
            X_short.append(track[-2:].flatten())
            X.append(track.flatten())
        y = []
        y_short = []
        for track in training_decoder_data:
            y_short.append(track[0:2].flatten())
            y.append(track.flatten())
        X = np.array(X)
        y = np.array(y)
        X_short = np.array(X_short)
        y_short = np.array(y_short)

        # Some models, like GP's are memory intractible for 1,000,000 data points, so a random sample is taken
        n_samples = 1000  # X_short.shape[0]
        classifier_succeeded = False
        print "Training Classifier..."
        while not classifier_succeeded:
            try:
                ########### RANDOM SUBSAMPLE
                sample_idxs = np.random.choice(X_short.shape[0], n_samples, replace=False)
                X_r = X[sample_idxs]
                Y_r = y[sample_idxs]
                X_sr = X_short[sample_idxs]
                Y_sr = y_short[sample_idxs]

                ########### FIT MODEL
                n_groups = 20
                from sklearn import mixture
                m = mixture.GaussianMixture(n_components=n_groups)
                m.fit(X_r, Y_r)
                classifier_succeeded = True
            except MemoryError:
                n_samples *= 0.9
                print "GP ran out of memory, number of samples reduced to: " + str(n_samples)

        # Classifier model fit, now to group the training data by class
        rows_grouped_by_class = {}
        for index, row in training_batch_handler.data_pool.iterrows():
            ideas = None
            flattened_encoder = row.encoder_sample.flatten()
            label = m.predict([flattened_encoder])[0]
            try:
                rows_grouped_by_class[label].append(row)
            except KeyError:
                rows_grouped_by_class[label] = [row]

        df_dict = {}
        for label, collection in rows_grouped_by_class.iteritems():
            print "Concatenating dataframe for label " + str(label) + " of size " + str(len(collection))
            df_dict[label] = pd.concat(collection, axis=1).transpose()

        dist_estimator_df = {}
        for label, collection in df_dict.iteritems():
            ideas = None
            # From the decoder sample, group all data points by timestamps
            # Add to one big numpy array and slice?
            numpy_array = []
            for index, row in collection.iterrows():
                numpy_array.append(row.decoder_sample)
                ideas = None
            collection_data_as_matrix = np.array(numpy_array)
            speed_direction = collection_data_as_matrix[:, :, 2:4]
            data_isolated_by_time = np.swapaxes(speed_direction, 0, 1)
            estimator_list = []
            for i in range(data_isolated_by_time.shape[0]):
                print "Fitting estimator for time " + str(i)
                # Train a GMM on the data (n = 1???)
                # add to list
                m_t = mixture.GaussianMixture(n_components=1)
                m_t.fit(data_isolated_by_time[i, :, :])
                estimator_list.append(m_t)
            dist_estimator_df[label] = estimator_list

        # For the GMM, means and covariances can be accessed via estimator.means_ and .covariances_
        # See http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py
        outputs = []
        ideas = None
        test_batch_handler.set_distance_threshold(0)
        test_batch_handler.reduced_pool
        classEst_df = report_df.copy()
        classEst_df = classEst_df.drop(['outputs', 'mixtures'], axis=1)
        for index, row in classEst_df.iterrows():
            input_array = row.encoder_sample
            last_step = input_array[-1]
            last_pos = last_step[0:2]
            label = m.predict([input_array.flatten()])[0]
            estimator_set = dist_estimator_df[label]
            #TODO Do something smarter than picking means
            relative_predictions = np.array([est.means_[0] for est in estimator_set])
            pred_angles = relative_predictions[:, 0]
            pred_speed = relative_predictions[:, 1]
            x_d, y_d = self._pol2cart(pred_speed / (25.0 / parameters['subsample']), pred_angles + np.pi / 2)
            x_p = np.cumsum(x_d)
            y_p = np.cumsum(y_d)
            x_p += last_pos[0]
            y_p += last_pos[1]

            prediction = np.array([x_p, y_p, pred_angles, pred_speed]).transpose()
            outputs.append(prediction)

        classEst_df = classEst_df.assign(outputs=outputs)


        return classEst_df

    def VGMM(self, training_batch_handler,
                   validation_batch_handler,
                   test_batch_handler,
                   parameters,
                   report_df):
        training_encoder_data = training_batch_handler.data_pool.encoder_sample.as_matrix()
        training_decoder_data = training_batch_handler.data_pool.decoder_sample.as_matrix()
        # Now I want to reshape this such that n_samples is n_tracks, and features is unrolled track data
        X = []
        X_short = []
        for track in training_encoder_data:
            X_short.append(track[-2:].flatten())
            X.append(track.flatten())
        y = []
        y_short = []
        for track in training_decoder_data:
            y_short.append(track[0:2].flatten())
            y.append(track.flatten())
        X = np.array(X)
        y = np.array(y)
        X_short = np.array(X_short)
        y_short = np.array(y_short)

        n_samples = 1000  # X_short.shape[0]
        gp_succeeded = False
        while not gp_succeeded:
            try:
                sample_idxs = np.random.choice(X_short.shape[0], n_samples, replace=False)
                X_r = X[sample_idxs]
                Y_r = y[sample_idxs]
                X_sr = X_short[sample_idxs]
                Y_sr = y_short[sample_idxs]

                #ker = GPy.kern.Matern52(2, ARD=True) + GPy.kern.White(2)
                from sklearn import mixture
                m = mixture.GaussianMixture(n_components=20)
                m.fit(X_r, Y_r)
                gp_succeeded = True
            except MemoryError:
                n_samples *= 0.9
                print "GP ran out of memory, number of samples reduced to: " + str(n_samples)

        outputs = []
        ideas = None
        test_batch_handler.set_distance_threshold(0)
        test_batch_handler.reduced_pool
        VGMM_df = report_df.copy()
        VGMM_df = VGMM_df.drop(['outputs', 'mixtures'], axis=1)
        for index, row in VGMM_df.iterrows():
            input_array = row.encoder_sample
            output = m.predict(np.array([input_array.flatten()]))
            outputs.append(output[0].reshape(len(output[0]) / 4, 4, order='a'))

        VGMM_df = VGMM_df.assign(outputs=outputs)
        return VGMM_df

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


