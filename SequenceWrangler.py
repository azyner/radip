import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import preprocessing

#Class to take a list of continuous, contiguous data logs that need to be collated and split for the data feeder
#Is this different to the batch handler?
#The SequenceWrangler is aware of the three data pools, training, test and val
#

class SequenceWrangler:
    def __init__(self,parameters, raw_sequences, raw_classes, n_folds=5, training=0.8,val=0.1,test=0.1):

        #Forces continuity b/w crossfold template and test template
        def _generate_template(track_idx,track_class,destination,destination_vec):
            return pd.DataFrame({"track_idx": track_idx,
                                        "class":track_class,
                                        "destination":destination,
                                        "destination_vec":destination_vec
                                        },index=[0])

        # Okay, so what is the format of data? Its both the raw tracks, and the raw vector classes?
        # The label --> one-hot converter should be in here.
        #This function needs to be persistent accross the n folds of the cross valdiation
        # In fact, it needs full persistance over the running program

        # The first thing I have to do is to convert the raw classes that are in format 'origin-destination'
        # , to something that the crossfold stratifier can digest. Indicies?
        # I'll want to crossfold against full indicies, so let's do that

        # Convert raw_classes into a list of indicies
        st_encoder = preprocessing.LabelEncoder()
        st_encoder.fit(raw_classes)
        origin_destintation_classes = st_encoder.transform(raw_classes)

        dest_raw_classes = [label[label.find('-') + 1:] for label in raw_classes]
        des_encoder = preprocessing.LabelEncoder()
        des_encoder.fit(dest_raw_classes)
        self.des_classes = des_encoder.transform(dest_raw_classes)

        """
        The notionally correct way to validate the algorithm is as follows:
        --90/10 split for (train/val) and test
        --Within train/val, do a crossfold search
        So I'm going to wrap the crossvalidator in another test/train picker, so
        that both are picked with an even dataset.
        """

        raw_indicies = range(len(raw_sequences))
        X_trainval, X_test, \
        Y_trainval, Y_test, \
        trainval_idxs, test_idxs = train_test_split(raw_sequences,raw_classes,raw_indicies,
                                                                  test_size=0.1,stratify=origin_destintation_classes)
        crossfold_idx_lookup = np.array(trainval_idxs)

        skf = StratifiedKFold(n_splits=n_folds)
        crossfold_indicies = list(skf.split(trainval_idxs,Y_trainval))
        crossfold_pool = [[[],[]] for x in xrange(n_folds)]
        test_pool = []

        # For all tracks
        for track_raw_idx in range(len(raw_sequences)):
            #if track_raw_idx > 10:
            #    break
            #Lookup the index in the original collection
            #Get data
            print "Wrangling track: " + str(track_raw_idx)
            single_track = raw_sequences[track_raw_idx]
            df_template = _generate_template(track_raw_idx,raw_classes[track_raw_idx],
                                             dest_raw_classes[track_raw_idx],
                                             self.des_classes[track_raw_idx])
            track_pool = self._track_slicer(single_track,
                                            5,#parameters.encoder_steps,
                                            0,#parameters.decoder_steps,
                                            df_template,
                                            20)#parameters.bbox)
            #Check if it exists in a crossfold pool
            for fold_idx in range(len(crossfold_indicies)):
                #For each pool, train pool or validation pool
                for trainorval_pool_idx in range(len(crossfold_indicies[fold_idx])):
                    # If the crossfold_list index of the track matches
                    if track_raw_idx in crossfold_idx_lookup[crossfold_indicies[fold_idx][trainorval_pool_idx]]:
                        crossfold_pool[fold_idx][trainorval_pool_idx].append(track_pool)
                        print "Added track " + str(track_raw_idx) + " to cf pool "+str(fold_idx) + \
                              (" train" if trainorval_pool_idx is 0 else " test")
            # else it must exist in the test_pool
            if track_raw_idx in test_idxs:
                test_pool.append(track_pool)
                print "Added track " + str(track_raw_idx) + " to test pool"

        print "concatenating pools"
        for fold_idx in range(len(crossfold_indicies)):
            for trainorval_pool_idx in range(len(crossfold_indicies[fold_idx])):
                crossfold_pool[fold_idx][trainorval_pool_idx] = pd.concat(crossfold_pool[fold_idx][trainorval_pool_idx])

        self.crossfold_pool = crossfold_pool
        self.test_pool = test_pool

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
