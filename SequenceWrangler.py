import numpy as np
import pandas as pd

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
    def _track_slicer(self, track, encoder_steps, decoder_steps):
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

        rnn_df_encoder = []
        rnn_df_decoder = []
        for i in range(len(track) - (encoder_steps+decoder_steps)+1):
            try:
                rnn_df_decoder.append(track[i + encoder_steps:i + (encoder_steps + decoder_steps)])
            except AttributeError:
                rnn_df_decoder.append(track[i + encoder_steps:i + (encoder_steps + decoder_steps)])
            data_ = track[i: i + encoder_steps]
            rnn_df_encoder.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])
        return np.array(rnn_df_encoder), np.array(rnn_df_decoder)



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

    def generate_sequence(regressor, test_sequence, seed_timesteps, prediction_length=None):
        if prediction_length > len(test_sequence)-seed_timesteps:
            raise AssertionError("Prediction length must be less than len(test_sequence)-seed_timesteps")
        if prediction_length == None:
            prediction_length = len(test_sequence)-seed_timesteps
        track = test_sequence[0:seed_timesteps]
        for i in range(prediction_length):
            packed =np.array([track])
            temp = regressor.predict(packed,axis=2)
            track = np.insert(track,track.shape[0],temp,axis=0) #Insert used (not append) to prevent array of shape (T,1)
                                                                # collapsing to a 1D array of (T,)
        return track



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
