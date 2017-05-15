import tensorflow as tf
import numpy as np
import random
import data_utils
import MDN
from tensorflow.python.ops import nn_ops
from TF_mods import basic_rnn_seq2seq_with_loop_function
#class that holds JUST THE MODEL
#Its not the trainer/runner. Just the model bit.

#The classification head may be simple enough to put in here, but it should be explicitly identified, such that it
# is obvious where the MDN head goes.

# The model should not be aware of the Ground Truth labels, so it should not do any loss computing, accuracy etc.
# That's the NetworkManager's job

#DEPRECATED

class RNN_model:
    def __init__(self, batch_size, num_classes, summary_writer=None):

        #move input params to self.param
        #run generate_model
        self.n_out = None
        self.model_type = None
        self.dtype = tf.float32
        self.batch_size = batch_size
        self.num_classes = num_classes
        input_set, output_set = self._generate_model()

        return input_set, output_set

    def _generate_model(self):

        # Internal Helper functions
        def _output_function(output):
            return nn_ops.xw_plus_b(output, output_projection[0], output_projection[1])

        # The loopback function needs to be a sampling function, it does not generate loss.
        def _simple_loop_function(prev, _):
            '''function that loops the data from the output of the LSTM to the input
            of the LSTM for the next timestep. So it needs to apply the output layers/function
            to generate the data at that timestep, and then'''
            if output_projection is not None:
                # Output layer
                prev = _output_function(prev)
            if self.model_type == 'MDN':
                # Sample to generate output
                prev = MDN.sample(prev)

            prev = nn_ops.xw_plus_b(prev, input_layer[0], input_layer[1])

            return prev

        def _seq2seq_f(encoder_inputs, decoder_inputs, feed_forward):
            if not feed_forward:  # feed last output as next input
                loopback_function = self._simple_loop_function
            else:
                loopback_function = None  # feed correct input
            return basic_rnn_seq2seq_with_loop_function(encoder_inputs, decoder_inputs, cell,
                                                        loop_function=loopback_function, dtype=self.dtype)

        with tf.variable_scope('output_proj'):
            o_w = tf.get_variable("proj_w", [self.rnn_size, self.n_out])
            o_b = tf.get_variable("proj_b", [self.n_out])
            output_projection = (o_w, o_b)
            #tf.histogram_summary("output_w",w)
            #tf.histogram_summary("output_b",b)
        with tf.variable_scope('input_layer'):
            i_w = tf.get_variable("in_w", [self.input_size, self.input_size])
            i_b = tf.get_variable("in_b", [self.input_size])
            input_layer = (i_w, i_b)

        # define layers here
        # input, linear RNN RNN linear etc
        single_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_size, state_is_tuple=True, use_peepholes=True)
        cell = single_cell
        if self.num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers, state_is_tuple=True)

        keep_prob = 1 - self.dropout_prob
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

        # Feeds for inputs.
        self.observation_inputs = []
        self.future_inputs = []
        self.target_weights = []
        targets = []
        targets_sparse = []

        for i in xrange(self.observation_steps):  # Last bucket is the biggest one.
            self.observation_inputs.append(tf.placeholder(tf.float32, shape=[batch_size, self.input_size],
                                                          name="observation{0}".format(i)))

        if self.model_type == 'MDN':
            for i in xrange(self.prediction_steps):
                self.future_inputs.append(tf.placeholder(tf.float32, shape=[batch_size, self.input_size],
                                                         name="prediction{0}".format(i)))
            for i in xrange(self.prediction_steps):
                self.target_weights.append(tf.placeholder(dtype, shape=[batch_size],
                                                        name="weight{0}".format(i)))
            #targets are just the future data
            targets = [self.future_inputs[i] for i in xrange(len(self.future_inputs))]

        if self.model_type == 'classifier':
            for i in xrange(self.prediction_steps):
                targets.append(tf.placeholder(tf.int32, shape=[batch_size, num_classes],
                                                         name="target{0}".format(i)))
            for target in targets:
                targets_sparse.append(tf.squeeze(tf.argmax(target,1)))
            for i in xrange(self.prediction_steps):
                self.target_weights.append(tf.ones([batch_size],name="weight{0}".format(i)))

        #Hook for the input_feed
        self.target_inputs = targets

        #Leave the last observation as the first input to the decoder
        #self.encoder_inputs = self.observation_inputs[0:-1]
        self.encoder_inputs = [nn_ops.xw_plus_b(input_timestep, input_layer[0], input_layer[1]) for
                               input_timestep in self.observation_inputs[0:-1]]

        #decoder inputs are the last observation and all but the last future
        self.decoder_inputs = [nn_ops.xw_plus_b(self.observation_inputs[-1], input_layer[0], input_layer[1])]
        # Todo should this have the input layer applied?
        self.decoder_inputs.extend([self.future_inputs[i] for i in xrange(len(self.future_inputs) - 1)])
        #for tensor in self.encoder_inputs:
        #    tf.histogram_summary(tensor.name, tensor)
        #for tensor in self.decoder_inputs:
        #    tf.histogram_summary(tensor.name,tensor)

        #if train: #Training
        self.LSTM_to_output_head, self.internal_states = _seq2seq_f(self.encoder_inputs, self.decoder_inputs, feed_future_data)
        #for tensor in self.LSTM_output_to_MDN:
        #    tf.histogram_summary(tensor.name,tensor)

        # self.outputs is a list of len(prediction_steps) containing [size batch x rnn_size]
        # The output projection below reduces this to:
        #                 a list of len(prediction_steps) containing [size batch x input_size]
        # BUG This is incorrect -- technically.
        # Because MDN.sample() is a random function, this sample is not the
        # sample being used in the loopback function.
        if output_projection is not None:
            self.network_output = [_output_function(output) for output in self.LSTM_to_output_head]
            if self.model_type == 'MDN':
                self.MDN_sample = [MDN.sample(x) for x in self.network_output]
        else:
            self.network_output = self.LSTM_to_output_head









        return input_set, output_set





    def run_test(self,batch_data):
        #generate plot, dump to tensorboard
        return

# This sounds complicated, but could I load two model instances, training and testing, so that they both use the exact
# same underlying weights in memory, but have different batch sizes, such that I can quickly run the test batch?
        # This is overly complicated, as I don't think the checkpointer is very costly.
# For more, I should probably read the exact functionality of the tensorflow checkpointer.

# Why?
#       If the test batch is smaller than the training batch, I can buff with zero data that is zero weighted.
#       Therefore I want weights in the loss function, as they are now very important.