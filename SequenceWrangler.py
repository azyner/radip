import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import preprocessing
import time
import sys
import os


#Class to take a list of continuous, contiguous data logs that need to be collated and split for the data feeder
#Is this different to the batch handler?
#The SequenceWrangler is aware of the three data pools, training, test and val
#


class SequenceWrangler:
    def __init__(self,parameters,sourcename, n_folds=5, training=0.55,val=0.2,test=0.25):
        self.n_folds = n_folds
        self.parameters = parameters.parameters
        #TODO Normalize the below splits
        self.training_split = training
        self.val_split = val
        self.test_split = test
        self.pool_dir = 'data_pool'
        self.sourcename = sourcename
        self.trainval_idxs = None
        self.test_idxs = None
        self.encoder_means = None
        self.encoder_vars = None
        self.encoder_stddev = None
        return

    def get_pool_filename(self):
        ibeo = True
        if ibeo:
            filename = "pool_ckpt_ibeo_" + \
                       ''.join([x[0] + x[-1] + '-' for x in self.parameters['ibeo_data_columns']]) + \
                       "obs-" + str(self.parameters["observation_steps"]) + \
                       "_pred-" + str(self.parameters["prediction_steps"]) + \
                        str(hash(tuple(self.sourcename))) + \
                       ".pkl"
        else:
            filename = "pool_ckpt_" +\
                        "obs-" + str(self.parameters["observation_steps"]) + \
                        "_pred-" + str(self.parameters["prediction_steps"]) + \
                       ".pkl"

        return filename

    def load_from_checkpoint(self,):
        #Function that returns True if data can be loaded, else false.

        if not os.path.exists(self.pool_dir):
            return False
        file_path = os.path.join(self.pool_dir,self.get_pool_filename())
        file_exists = os.path.isfile(file_path)
        if not file_exists:
            return False
        self.master_pool = pd.read_pickle(file_path)

        return True

    def split_into_evaluation_pools(self,trainval_idxs = None, test_idxs = None):
        # Consolidate with get_pools function?
        # self.master_pool should exist by now
        # TODO Here print normalization numbers.
        #I'll then add a get/set call for these numbers to be added to the network.
        seed = np.random.randint(4294967296)
        print "Using seed: " + str(seed) + " for test/train split"

        encoder_pool = []
        for encoder_data in self.master_pool.encoder_sample.iteritems():
            encoder_values = encoder_data[1]
            encoder_pool.append(encoder_values[0])
            last_encoder = encoder_values
        encoder_pool.extend(encoder_values[1:])
        encoder_pool = np.array(encoder_pool)

        #Compute averages here
        self.encoder_means = np.mean(encoder_pool, axis=0)
        self.encoder_vars = np.var(encoder_pool, axis=0)
        self.encoder_stddev = np.std(encoder_pool, axis=0)

        print "Encoder means: " + str(self.encoder_means)
        print "Encoder vars: " + str(self.encoder_vars)
        print "Encoder standard deviations: " + str(self.encoder_stddev)

        raw_indicies =self.master_pool.track_idx.unique()

        # origin_destination_class_list = self.master_pool.track_class.unique()

        # rebuild track_class vector
        raw_classes = []
        for raw_idx in raw_indicies:
            #Get the first results that matches the track_idx and return its destination class
            #by construction, this data is consistent across all sample values for that track
            track_class = self.master_pool[self.master_pool.track_idx==raw_idx]['track_class'].unique()
            raw_classes.append(track_class[0])

        st_encoder = preprocessing.LabelEncoder()
        st_encoder.fit(raw_classes)
        origin_destination_enc_classes = st_encoder.transform(raw_classes)

        if (trainval_idxs is None) and (test_idxs is None):
            self.trainval_idxs, self.test_idxs = train_test_split(raw_indicies,  # BREAK HERE
                                                    test_size=self.test_split,
                                                    stratify=origin_destination_enc_classes,
                                                    random_state=seed)
        else:
            self.trainval_idxs = trainval_idxs
            self.test_idxs = test_idxs

        crossfold_idx_lookup = np.array(self.trainval_idxs)

        #Now I need the class of each track in trainval_idx
        trainval_class = []
        for trainval_idx in self.trainval_idxs:
            track_class = self.master_pool[self.master_pool.track_idx==trainval_idx]['track_class'].unique()
            trainval_class.append(track_class[0])

        skf = StratifiedKFold(n_splits=self.n_folds,random_state=seed)
        crossfold_indicies = list(skf.split(self.trainval_idxs, trainval_class))
        crossfold_pool = [[[], []] for x in xrange(self.n_folds)]
        test_pool = []

        #Now iterate over each track, and dump it into the apropriate crossfold sub-pool or test pool
        for track_raw_idx in raw_indicies:
            # For each pool
            for fold_idx in range(len(crossfold_indicies)):
                # For train or validate in the pool
                for trainorval_pool_idx in range(len(crossfold_indicies[fold_idx])):
                    # If the crossfold_list index of the track matches
                    if track_raw_idx in crossfold_idx_lookup[crossfold_indicies[fold_idx][trainorval_pool_idx]]:

                        #Here, I want to append all data in the master pool that is from the track
                        crossfold_pool[fold_idx][trainorval_pool_idx].append(
                            self.master_pool[self.master_pool['track_idx']==track_raw_idx]
                        )
                        #print "Added track " + str(track_raw_idx) + " to cf pool " + str(fold_idx) + \
                        #      (" train" if trainorval_pool_idx is 0 else " test")
            # else it must exist in the test_pool
            if track_raw_idx in self.test_idxs:
                test_pool.append(
                        self.master_pool[self.master_pool['track_idx'] == track_raw_idx]
                )
                #print "Added track " + str(track_raw_idx) + " to test pool"

        print "concatenating pools"
        for fold_idx in range(len(crossfold_indicies)):
            for trainorval_pool_idx in range(len(crossfold_indicies[fold_idx])):
                crossfold_pool[fold_idx][trainorval_pool_idx] = pd.concat(crossfold_pool[fold_idx][trainorval_pool_idx])

        self.crossfold_pool = crossfold_pool
        self.test_pool = pd.concat(test_pool)


        return

    # This function will generate the data pool for the dataset from the natualistic driving data set.
    # Its input is a list of tracks, and a list of labels in the format "origin-destination"
    # The tracks are in [Data T], where Data is a list of floats of len 4, x,y, heading speed.

    def generate_master_pool_naturalistic_2015(self, raw_sequences=None, raw_classes=None):

        # Convert raw_classes into a list of indicies
        st_encoder = preprocessing.LabelEncoder()
        st_encoder.fit(raw_classes)
        origin_destintation_classes = st_encoder.transform(raw_classes)

        dest_raw_classes = [label[label.find('-') + 1:] for label in raw_classes]
        origin = [label[:label.find('-')] for label in raw_classes]
        des_encoder = preprocessing.LabelEncoder()
        des_encoder.fit(dest_raw_classes)
        self.des_classes = des_encoder.transform(dest_raw_classes)
        dest_1hot_enc = preprocessing.OneHotEncoder()
        dest_1hot_enc.fit(np.array(self.des_classes).reshape(-1,1))

        # Forces continuity b/w crossfold template and test template
        def _generate_template(track_idx, track_class,origin, destination, destination_vec):
            return pd.DataFrame({"track_idx": track_idx,
                                 "track_class": track_class,
                                 "origin":origin,
                                 "destination": destination,
                                 "destination_vec": destination_vec,
                                 "dest_1_hot":
                                     pd.Series([dest_1hot_enc.transform(destination_vec).toarray().astype(np.float32)[0]],
                                               dtype=object)
                                 }, index=[0])

        """
        The notionally correct way to validate the algorithm is as follows:
        --90/10 split for (train/val) and test
        --Within train/val, do a crossfold search
        So I'm going to wrap the crossvalidator in another test/train picker, so
        that both are picked with an even dataset.
        """

        master_pool = []

        # For all tracks
        for track_raw_idx in range(len(raw_sequences)):
            try:
                # if track_raw_idx > 10:
                #    break
                # Lookup the index in the original collection
                # Get data
                # print "Wrangling track: " + str(track_raw_idx)
                wrangle_time = time.time()
                single_track = raw_sequences[track_raw_idx]
                df_template = _generate_template(track_raw_idx, raw_classes[track_raw_idx],
                                                 origin[track_raw_idx],
                                                 dest_raw_classes[track_raw_idx],
                                                 self.des_classes[track_raw_idx])
                track_pool = self._track_slicer(single_track,
                                                self.parameters['observation_steps'],
                                                self.parameters['prediction_steps'],
                                                df_template,
                                                bbox=20)  # FIXME parameters.bbox)

                master_pool.append(track_pool)
            except ValueError:
                print "Warning, track discarded as it did not meet minimum length requirements"
                continue

        self.master_pool = pd.concat(master_pool)
        if not os.path.exists(self.pool_dir):
            os.makedirs(self.pool_dir)
        file_path = os.path.join(self.pool_dir, self.get_pool_filename())
        self.master_pool.to_pickle(file_path)

        return

    def _extract_ibeo_data_for_encoders(self,single_track):
        # Code that transforms the big dataframe into the input data list style for encoder/decoder
        # DOES NOT DO TRACK SPLITTING. Len output shoulbe be equal to len input

        '''
        'level_0', u'index', u'ObjectId', u'Flags',
        u'trackedByStationaryModel', u'mobile', u'motionModelValidated',
        u'ObjectAge', u'Timestamp', u'ObjectPredAge', u'Classification',
        u'ClassCertainty', u'ClassAge', u'ObjBoxCenter_X', u'ObjBoxCenter_Y',
        u'ObjBoxCenterSigma_X', u'ObjBoxCenterSigma_Y', u'ObjBoxSize_X',
        u'ObjBoxSize_Y', u'ObjCourseAngle', u'ObjCourseAngleSigma',
        u'ObjBoxOrientation', u'ObjBoxOrientationSigma', u'RelVelocity_X',
        u'RelVelocity_Y', u'RelVelocitySigma_X', u'RelVelocitySigma_Y',
        u'AbsVelocity_X', u'AbsVelocity_Y', u'AbsVelocitySigma_X',
        u'AbsVelocitySigma_Y', u'RefPointLocation', u'RefPointCoords_X',
        u'RefPointCoords_Y', u'RefPointCoordsSigma_X', u'RefPointCoordsSigma_Y',
        u'RefPointPosCorrCoeffs', u'ObjPriority', u'ObjExtMeasurement',
        u'EgoLatitude', u'EgoLongitude', u'EgoAltitude', u'EgoHeadingRad',
        u'EgoPosTimestamp', u'GPSFixStatus', u'ObjPrediction', u'Object_X',
        u'Object_Y', u'uniqueId', u'origin', u'destination', u'distance'],
        dtype = 'object')
        '''

        ibeo_data_columns = ["Object_X","Object_Y","ObjBoxOrientation","AbsVelocity_X","AbsVelocity_Y","ObjectPredAge"]

        output_df = single_track.loc[:,self.parameters["ibeo_data_columns"]].values.astype(np.float32)
        return output_df

    def generate_master_pool_ibeo(self, ibeo_track_list):

        # get the unique list of origins and destinations:
        # Add all the first rows of each track
        labelling_list = [track.iloc[0] for track in ibeo_track_list]
        labelling_df = pd.concat(labelling_list)
        destinations = labelling_df["destination"].unique()
        origins = labelling_df["origin"].unique()

        # Convert destination into a list of indicies
        des_encoder = preprocessing.LabelEncoder()
        des_encoder.fit(destinations)

        dest_1hot_enc = preprocessing.OneHotEncoder()
        dest_1hot_enc.fit(des_encoder.transform(destinations).reshape(-1, 1))

        # Forces continuity b/w crossfold template and test template
        def _generate_ibeo_template(track_idx, track_class, origin, destination, destination_vec):
            return pd.DataFrame({"track_idx": track_idx,
                                 "track_class": track_class,
                                 "origin": origin,
                                 "destination": destination,
                                 "destination_vec": destination_vec,
                                 "dest_1_hot":
                                     pd.Series([dest_1hot_enc.transform(destination_vec.reshape(-1, 1)
                                                                        ).astype(np.float32).toarray()[0]],
                                               dtype=object)
                                 }, index=[0])

        """
        The notionally correct way to validate the algorithm is as follows:
        --90/10 split for (train/val) and test
        --Within train/val, do a crossfold search
        So I'm going to wrap the crossvalidator in another test/train picker, so
        that both are picked with an even dataset.
        """

        #COMPUTE NORM PARAMS HERE
        # 1 - Collect all the encoder data. Ever.
        # 2 - computer normalization parameters.
        # 3 - save these parameters -- to be pickled for the batchhandler, or pre-computed here?
        #       Either way, they need to be saved somewhere, and coupled with the data_pool

        print "Computing batch normalization parameters."
        encoder_data = np.empty([0,len(self.parameters["ibeo_data_columns"])])
        for track_raw_idx in range(len(ibeo_track_list)):
            single_track = ibeo_track_list[track_raw_idx]
            data_for_encoders = self._extract_ibeo_data_for_encoders(single_track)
            encoder_data = np.append(encoder_data,data_for_encoders,axis=0)
        #encoder_data = pd.concat(encoder_data_list)
        encoder_means = np.mean(encoder_data, axis=0)
        encoder_vars = np.var(encoder_data, axis=0)
        encoder_stddev = np.std(encoder_data, axis=0)

        print "Encoder means: " + str(encoder_means)
        print "Encoder vars: " + str(encoder_vars)
        print "Encoder standard deviations: " + str(encoder_stddev)

        master_pool = []
        # Don't put everything in the pool, it takes forever.
        data_columns = ['index', 'ObjectId', 'Timestamp', 'ObjectPredAge', 'Classification',
          'ObjBoxOrientation',
          'Object_X', 'Object_Y', 'uniqueId', 'origin',
          'destination', 'AbsVelocity', 'distance', 'distance_to_exit']
        data_columns.extend(self.parameters['ibeo_data_columns'])
        data_columns = list(set(data_columns))

        # For all tracks
        for track_raw_idx in range(len(ibeo_track_list)):
            # Lookup the index in the original collection
            # Get data
            #rint "Wrangling track: " + str(track_raw_idx) + " of: " + str(len(ibeo_track_list))
            sys.stdout.write("\rWrangling track:  %04d of %04d " % (track_raw_idx, len(ibeo_track_list)))
            sys.stdout.flush()
            wrangle_time = time.time()
            single_track = ibeo_track_list[track_raw_idx]
            origin = single_track.iloc[0]['origin']
            destination = single_track.iloc[0]['destination']
            destination_vec = des_encoder.transform([destination])
            data_for_encoders = self._extract_ibeo_data_for_encoders(single_track)

            # Do not scale here. Scaling is to be done as the first network layer.
            df_template = _generate_ibeo_template(track_raw_idx, origin + "-" + destination, origin, destination,
                                                  destination_vec)
            # Instead, I am going to give the new track slicer a list for distance, as I have pre-computed it.
            track_pool = self._track_slicer(data_for_encoders,
                                            self.parameters['observation_steps'],
                                            self.parameters['prediction_steps'],
                                            df_template, # Metadata that is static across the whole track
                                            distance=single_track['distance'],#metadata that changes in the track.
                                            distance_to_exit=single_track['distance_to_exit'],
                                            additional_df=single_track[data_columns]) #Everything else. Useful for post network analysis


            master_pool.append(track_pool)
            #print "wrangle time: " + str(time.time()-wrangle_time)
        sys.stdout.write("\t\t\t\t%4s" % "[ OK ]")
        sys.stdout.write("\r\n")
        sys.stdout.flush()

        self.master_pool = pd.concat(master_pool)

        #TODO save master to pickle
        if not os.path.exists(self.pool_dir):
            os.makedirs(self.pool_dir)
        file_path = os.path.join(self.pool_dir, self.get_pool_filename())
        self.master_pool.to_pickle(file_path)

        return

    def get_pools(self):
        return self.crossfold_pool, self.test_pool

    # Unused?
    def _generate_classes(self,string_collection):
        # Takes in a list of strings, where the strings are the name of the classes
        # returns: class_dict - a dictionary to lookup class title to vector index
        # class_list - a list of one hot vectors for the input data

        class_key_list = []
        class_dictionary = {}
        class_one_hot = []
        class_vector_dictionary = {}

        # loop through all data once, collect all possible labels
        # add a 'None' class
        # Loop through list again generating a second list of one-hot vectors (ndarray)

        def get_class_from_str(string):
            # All classes are in the format 'origin-destination'
            # I only care about the destination
            dash_idx = string.find('-')
            return string[dash_idx + 1:]

        for i in range(len(string_collection)):
            class_dictionary[get_class_from_str(string_collection[i])] = '0'

        for key, value in class_dictionary.iteritems():
            class_key_list.append(get_class_from_str(key))
        #class_key_list.append('None')

        for i in range(len(string_collection)):
            new_vector = np.array([0] * len(class_key_list))
            new_vector[class_key_list.index(get_class_from_str(string_collection[i]))] = 1
            class_one_hot.append(new_vector)

        for key in class_key_list:
            new_vector = np.array([0] * len(class_key_list))
            new_vector[class_key_list.index(key)] = 1
            class_vector_dictionary[key] = new_vector

        return class_one_hot, class_vector_dictionary

    #This calculates the distance from the reference line for the length of the track. Useful for picking elements later
    #  or earlier in the sequence, depending on training style
    #ASSUMPTIONS
    # For this to work, the first two elements in the feature vector must be the co-ordinates in meters, centre normalized
    def _dis_from_ref_line(self,single_track,ref_line_dis):
        def is_inside_box(timestep_data, bounds):
            return (timestep_data[0] < bounds and
                    timestep_data[0] > -bounds and
                    timestep_data[1] < bounds and
                    timestep_data[1] > -bounds)

        ref_step = 0

        d = np.sqrt((single_track[1:, 0] - single_track[0:-1, 0]) ** 2 +
                    (single_track[1:, 1] - single_track[0:-1, 1]) ** 2)
        d = np.cumsum(np.append(0.0, d))

        for i in range(len(single_track)):
            if is_inside_box(single_track[i], ref_line_dis):
                ref_step = i
                break
        d -= d[ref_step]
        return d

    # So track slicer is only handling one track at a time. It should be passed a set of common parameters
    #   For example: destination label, or vehicle type etc.
    def _track_slicer(self, track, encoder_steps, decoder_steps, df_template,
                      bbox=None,distance=None,distance_to_exit=None, additional_df=None):
        """
        creates new data frame based on previous observation
          * example:
            l = [1, 2, 3, 4, 5, 6,7]
            encoder_steps = 2
            decoder_steps = 3
            -> encoder [[1, 2], [2, 3], [3, 4]]
            -> decoder [[3,4,5], [4,5,6], [5,6,7]]
        """
        if len(track) < encoder_steps + decoder_steps:
            raise ValueError("length of track is shorter than encoder_steps and decoder_steps.")

        # Do this once only.
        if bbox is not None:
            dis = self._dis_from_ref_line(track, bbox)

        sample_collection = []
        for i in range(len(track) - (encoder_steps+decoder_steps)+1):
            sample_dataframe = df_template.copy()
            sample_dataframe["encoder_sample"] = pd.Series([track[i: i + encoder_steps]], dtype=object)
            sample_dataframe["decoder_sample"] = pd.Series([track[i + encoder_steps:i + (encoder_steps + decoder_steps)]],
                                                           dtype=object)
            if bbox is not None:
                sample_dataframe["distance"] = dis[i+encoder_steps-1] #distance for the last element given to encoder
            if distance is not None:
                sample_dataframe["distance"] = distance[i+encoder_steps-1]
            if distance_to_exit is not None:
                sample_dataframe["distance_to_exit"] = distance_to_exit[i+encoder_steps-1]
            if additional_df is not None:
                for col_name in additional_df.columns:
                    if col_name is 'distance' or col_name is 'distance_to_exit':
                        continue
                    sample_dataframe[col_name] = additional_df[col_name][i+encoder_steps-1]
            sample_dataframe["track_time_idx"] = i

            sample_collection.append(pd.DataFrame(sample_dataframe))
        return pd.concat(sample_collection)

    # unused?
    def split_sequence_collection(self, collection, encoder_steps, decoder_steps, labels):

        def loop_through_collection(coll, encoder_steps, decoder_steps, labels):
            x_list = []
            y_list = []
            label_list = []
            for i in range(len(coll)):
                data = self._track_slicer(coll[i], encoder_steps, decoder_steps)
                x_list.extend(data[0])
                y_list.extend(data[1])
                for _ in range(len(data[0])):
                    label_list.append(labels[i])

            return x_list, y_list, label_list

        data_dx, data_dy, data_l = loop_through_collection(collection, encoder_steps, decoder_steps, labels)

        return np.array(data_dx), np.array(data_dy), np.array(data_l)

    # Unused?
    def generate_classes(self, string_collection):
        # Takes in a list of strings, where the strings are the name of the classes
        # returns: class_dict - a dictionary to lookup class title to vector index
        # class_list - a list of one hot vectors for the input data

        class_key_list = []
        class_dictionary = {}
        class_one_hot = []
        class_vector_dictionary = {}
        # loop through all data once, collect all possible labels
        # add a 'None' class
        # Loop through list again generating a second list of one-hot vectors (ndarray)

        def get_class_from_str(string):
            # All classes are in the format 'origin-destination'
            # I only care about the destination
            dash_idx = string.find('-')
            return string[dash_idx + 1:]

        for i in range(len(string_collection)):
            class_dictionary[get_class_from_str(string_collection[i])] = '0'

        for key, value in class_dictionary.iteritems():
            class_key_list.append(get_class_from_str(key))
        class_key_list.append('None')

        for i in range(len(string_collection)):
            new_vector = np.array([0]*len(class_key_list))
            new_vector[class_key_list.index(get_class_from_str(string_collection[i]))] = 1
            class_one_hot.append(new_vector)

        for key in class_key_list:
            new_vector = np.array([0]*len(class_key_list))
            new_vector[class_key_list.index(key)] = 1
            class_vector_dictionary[key] = new_vector

        return class_one_hot, class_vector_dictionary

    # FOR BATCH HANDLER?
    # Trim the seqence to the very start, or if bbox is defined (in meters from intersection centre),
    # Trim to the place of box entry
    # The bbox reference is actually a measure of distance travelled from the reference line, where
    # The reference line is a bounding box of 20meters from the intersection
    def trim_sequence(self, observation_steps, prediction_steps, data_collection, sample_dis=None):
        def is_inside_box(timestep_data,bounds):
            return (timestep_data[0] < bounds and
                    timestep_data[0] > -bounds and
                    timestep_data[1] < bounds and
                    timestep_data[1] > -bounds)
        trimmed_collection = []
        for member in data_collection:
            ref_step = 0
            start_step = 0

            if sample_dis is not None:
                d = np.sqrt((member[1:, 0] - member[0:-1, 0]) ** 2 +
                            (member[1:, 1] - member[0:-1, 1]) ** 2)
                d = np.cumsum(np.append(0.0, d))

                for i in range(len(member)):
                    if is_inside_box(member[i],20):
                        ref_step = i
                        break

                d -= d[ref_step]
                # Todo I'm sure there's a np optimized function for finding the first float past the post in an ordered list
                for i in range(len(member)):
                    if d[i] > sample_dis:
                        start_step = i
                        break

                #Data is left for the decoder in the classifier model as it is discarded later
                trimmed_collection.append(member[start_step - observation_steps - prediction_steps:start_step])
            else: #Just pick the first steps, as the class does not change
                trimmed_collection.append(member)#[0:0+observation_steps+prediction_steps])
        return trimmed_collection
