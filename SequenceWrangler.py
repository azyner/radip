import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import preprocessing
import time
import os
import pickle

#Class to take a list of continuous, contiguous data logs that need to be collated and split for the data feeder
#Is this different to the batch handler?
#The SequenceWrangler is aware of the three data pools, training, test and val
#


class SequenceWrangler:
    def __init__(self,parameters, n_folds=5, training=0.55,val=0.2,test=0.25):
        self.n_folds = n_folds
        self.parameters = parameters.parameters
        #TODO Normalize the below splits
        self.training_split = training
        self.val_split = val
        self.test_split = test
        self.pool_dir = 'data_pool'
        return

    def get_pool_filename(self):
        filename = "pool_ckpt_" +\
                    "obs-" + str(self.parameters["observation_steps"]) + \
                    "_pred-" + str(self.parameters["prediction_steps"]) + \
                   ".pkl"

        return filename

    def load_from_checkpoint(self):
        #Function that returns True if data can be loaded, else false.

        if not os.path.exists(self.pool_dir):
            return False
        file_path = os.path.join(self.pool_dir,self.get_pool_filename())
        file_exists = os.path.isfile(file_path)
        if not file_exists:
            return False
        self.master_pool = pd.read_pickle(file_path)

        return True

    def split_into_evaluation_pools(self):
        # Consolidate with get_pools function?
        # self.master_pool should exist by now

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

        trainval_idxs, test_idxs = train_test_split(raw_indicies,
                                                    test_size=self.test_split,
                                                    stratify=origin_destination_enc_classes)

        crossfold_idx_lookup = np.array(trainval_idxs)

        #Now I need the class of each track in trainval_idx
        trainval_class = []
        for trainval_idx in trainval_idxs:
            track_class = self.master_pool[self.master_pool.track_idx==trainval_idx]['track_class'].unique()
            trainval_class.append(track_class[0])


        skf = StratifiedKFold(n_splits=self.n_folds)
        crossfold_indicies = list(skf.split(trainval_idxs, trainval_class))
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
            if track_raw_idx in test_idxs:
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
                                            20)  # FIXME parameters.bbox)

            master_pool.append(track_pool)

        self.master_pool = pd.concat(master_pool)

        #TODO save master to pickle
        if not os.path.exists(self.pool_dir):
            os.makedirs(self.pool_dir)
        file_path = os.path.join(self.pool_dir, self.get_pool_filename())
        self.master_pool.to_pickle(file_path)

        return


    def generate_master_pool_ibeo(self, ibeo_df):

        # # Convert destination into a list of indicies
        # dest_raw_classes = [label[label.find('-') + 1:] for label in raw_classes]
        # origin = [label[:label.find('-')] for label in raw_classes]
        # des_encoder = preprocessing.LabelEncoder()
        # des_encoder.fit(ibeo_df["destination"])
        # self.des_classes = des_encoder.transform(dest_raw_classes)
        # dest_1hot_enc = preprocessing.OneHotEncoder()
        # dest_1hot_enc.fit(np.array(self.des_classes).reshape(-1,1))
        #
        # # Forces continuity b/w crossfold template and test template
        # def _generate_template(track_idx, track_class,origin, destination, destination_vec):
        #     return pd.DataFrame({"track_idx": track_idx,
        #                          "track_class": track_class,
        #                          "origin":origin,
        #                          "destination": destination,
        #                          "destination_vec": destination_vec,
        #                          "dest_1_hot":
        #                              pd.Series([dest_1hot_enc.transform(destination_vec).toarray().astype(np.float32)[0]],
        #                                        dtype=object)
        #                          }, index=[0])
        #
        # """
        # The notionally correct way to validate the algorithm is as follows:
        # --90/10 split for (train/val) and test
        # --Within train/val, do a crossfold search
        # So I'm going to wrap the crossvalidator in another test/train picker, so
        # that both are picked with an even dataset.
        # """
        #
        # master_pool = []
        #
        # # For all tracks
        # for track_raw_idx in range(len(raw_sequences)):
        #     # if track_raw_idx > 10:
        #     #    break
        #     # Lookup the index in the original collection
        #     # Get data
        #     # print "Wrangling track: " + str(track_raw_idx)
        #     wrangle_time = time.time()
        #     single_track = raw_sequences[track_raw_idx]
        #     df_template = _generate_template(track_raw_idx, raw_classes[track_raw_idx],
        #                                      origin[track_raw_idx],
        #                                      dest_raw_classes[track_raw_idx],
        #                                      self.des_classes[track_raw_idx])
        #     track_pool = self._track_slicer(single_track,
        #                                     self.parameters['observation_steps'],
        #                                     self.parameters['prediction_steps'],
        #                                     df_template,
        #                                     20)  # FIXME parameters.bbox)
        #
        #     master_pool.append(track_pool)
        #
        # self.master_pool = pd.concat(master_pool)
        #
        # #TODO save master to pickle
        # if not os.path.exists(self.pool_dir):
        #     os.makedirs(self.pool_dir)
        # file_path = os.path.join(self.pool_dir, self.get_pool_filename())
        # self.master_pool.to_pickle(file_path)

        return


    def get_pools(self):
        return self.crossfold_pool, self.test_pool

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
    def _track_slicer(self, track, encoder_steps, decoder_steps, df_template, bbox=None):
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

        if bbox is not None:
            dis = self._dis_from_ref_line(track,bbox)

        sample_collection = []
        for i in range(len(track) - (encoder_steps+decoder_steps)+1):
            sample_dataframe = df_template.copy()
            sample_dataframe["encoder_sample"] = pd.Series([track[i: i + encoder_steps]], dtype=object)
            sample_dataframe["decoder_sample"] = pd.Series([track[i + encoder_steps:i + (encoder_steps + decoder_steps)]],
                                                           dtype=object)
            if bbox is not None:
                sample_dataframe["distance"] = dis[i+encoder_steps-1] #distance for the last element given to encoder
            sample_dataframe["time_idx"] = i

            sample_collection.append(pd.DataFrame(sample_dataframe))
        return pd.concat(sample_collection)

    def split_sequence_collection(self,collection,encoder_steps,decoder_steps,labels):

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
