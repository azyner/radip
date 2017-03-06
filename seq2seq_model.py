import tensorflow as tf
import numpy as np
import random
import MDN
from tensorflow.python.ops import nn_ops
from TF_mods import basic_rnn_seq2seq_with_loop_function
from tensorflow.python.ops import seq2seq
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops


class Seq2SeqModel(object):

    def __init__(self, parameters):
        #feed_future_data, train, num_observation_steps, num_prediction_steps, batch_size,
        #         rnn_size, num_layers, learning_rate, learning_rate_decay_factor, input_size, max_gradient_norm,
        #        dropout_prob,random_bias,subsample,random_rotate,num_mixtures,model_type):

        # feed_future_data: whether or not to feed the true data into the decoder instead of using a loopback
        #                function. If false, a loopback function is used, feeding the last generated output as the next
        #                decoder input.
        # train: train the model (or test)
        # Subsample: amount of subsampling. IMPORTANT If this is non-one, the input array must be n times longer than usual, as it only subsamples down
        # This is so that the track is not subsampled the same way each track sample.

        #######################################
        # The LSTM Model consists of:
        # Input Linear layer
        # N LSTM layers
        # a linear output layer to convert the LSTM output to MDN format
        #
        # MDN Format:
        # pi mu1 mu2 sigma1 sigma2 rho
        # (repeat for n mixtures)
        #
        # This MDN format is then either used for the loss, or is sampled to get a real value

        #TODO Tuesday: use model_type = 'MDN' | 'classifier' to split the two models.
        # Also try and fix how the loopback function sample is not being used as the output sample.

        #TODO Reorganise code using namespace for better readability

        self.max_gradient_norm = parameters['max_gradient_norm']
        self.rnn_size = parameters['rnn_size']
        self.num_layers = parameters['num_layers']
        dtype = tf.float32

        self.batch_size = parameters['batch_size']
        self.input_size = parameters['input_size']
        self.embedding_size = parameters['embedding_size']
        self.observation_steps = parameters['observation_steps']
        self.prediction_steps = parameters['prediction_steps']
        self.dropout_prob = parameters['dropout_prob']
        self.random_bias = parameters['random_bias']
        self.subsample = parameters['subsample']
        self.random_rotate = parameters['random_rotate']
        self.num_mixtures = parameters['num_mixtures']
        self.model_type = parameters['model_type']
        self.num_classes = parameters['num_classes']

        self.learning_rate = tf.Variable(float(parameters['learning_rate']), trainable=False, name="Learning_rate")
        self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * parameters['learning_rate_decay_factor'])
        self.global_step = tf.Variable(0, trainable=False,name="Global_step")
        self.network_summaries = []
        keep_prob = 1-self.dropout_prob

        # TODO Placeholder until I implement MDN
        feed_future_data = False


        if parameters['model_type'] == 'classifier' and self.prediction_steps > 1:
            raise Exception("Error. Classifier model can only have 1 prediction step")

        #if feed_future_data and not train:
        #    print "Warning, feeding the model future sequence data (feed_forward) is not recommended when the model is not training."

        # The output of the multiRNN is the size of rnn_size, and it needs to match the input size, or loopback makes
        #  no sense. Here a single layer without activation function is used, but it can be any number of
        #  non RNN layers / functions
        if self.model_type == 'MDN':
            n_out = 6*self.num_mixtures
        if self.model_type=='classifier':
            n_out = parameters['num_classes']

        ############## LAYERS

        # Layer is linear, just to re-scale the LSTM outputs [-1,1] to [-9999,9999]
        # If there is a regularizer, these weights should be excluded?

        with tf.variable_scope('output_proj'):
            o_w = tf.get_variable("proj_w", [self.rnn_size, n_out],
                                  initializer=tf.truncated_normal_initializer(stddev=1.0 / np.sqrt(self.embedding_size)))
            o_b = tf.get_variable("proj_b", [n_out],
                                  initializer=tf.constant_initializer(0.1))
            output_projection = (o_w, o_b)

        with tf.variable_scope('input_embedding_layer'):
            i_w = tf.get_variable("in_w", [self.input_size, self.embedding_size],
                                  initializer=tf.truncated_normal_initializer(stddev=1.0/np.sqrt(self.embedding_size)))
            i_b = tf.get_variable("in_b", [self.embedding_size],
                                  initializer=tf.constant_initializer(0.1))
            input_layer = (i_w, i_b)

        single_cell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(self.rnn_size,state_is_tuple=True,
                                                use_peepholes=parameters['peephole_connections'])
            ,output_keep_prob=keep_prob)
        RNN_layers = single_cell
        if self.num_layers > 1:
            RNN_layers = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers,state_is_tuple=True)

        # Don't double dropout
        #RNN_layers = tf.nn.rnn_cell.DropoutWrapper(RNN_layers,output_keep_prob=keep_prob)

        def output_function(output):
            return nn_ops.xw_plus_b(output, output_projection[0], output_projection[1],name="output_projection")

        #The loopback function needs to be a sampling function, it does not generate loss.
        def simple_loop_function(prev, _):
            '''function that loops the data from the output of the LSTM to the input
            of the LSTM for the next timestep. So it needs to apply the output layers/function
            to generate the data at that timestep, and then'''
            if output_projection is not None:
                #Output layer
                prev = output_function(prev)
            if self.model_type == 'MDN':
                # Sample to generate output
                prev = MDN.sample(prev)

            # Apply input layer
            prev = tf.nn.dropout(
                tf.nn.relu(nn_ops.xw_plus_b(prev, input_layer[0], input_layer[1]),name="Loopback_Input"),
                                 1-parameters['embedding_dropout'])

            return prev

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, feed_forward):
            if not feed_forward: #feed last output as next input
                loopback_function = simple_loop_function
            else:
                loopback_function = None #feed correct input
            #return basic_rnn_seq2seq_with_loop_function(encoder_inputs,decoder_inputs,cell,
            #                                                         loop_function=loopback_function,dtype=dtype)
            return seq2seq.tied_rnn_seq2seq(encoder_inputs,decoder_inputs,RNN_layers,
                                            loop_function=loopback_function,dtype=dtype)

        # Feeds for inputs.
        self.observation_inputs = []
        self.future_inputs = []
        self.target_weights = []
        targets = []
        targets_sparse = []

        for i in xrange(self.observation_steps):  # Last bucket is the biggest one.
            self.observation_inputs.append(tf.placeholder(tf.float32, shape=[self.batch_size, self.input_size],
                                                          name="observation{0}".format(i)))

        if self.model_type == 'MDN':
            for i in xrange(self.prediction_steps):
                self.future_inputs.append(tf.placeholder(tf.float32, shape=[self.batch_size, self.input_size],
                                                         name="prediction{0}".format(i)))
            for i in xrange(self.prediction_steps):
                self.target_weights.append(tf.placeholder(dtype, shape=[self.batch_size],
                                                        name="weight{0}".format(i)))
            #targets are just the future data
            targets = [self.future_inputs[i] for i in xrange(len(self.future_inputs))]

        if self.model_type == 'classifier':
            # Add a single target. Name is target0 for continuity
            target = tf.placeholder(tf.int32, shape=[self.batch_size, self.num_classes],
                                                         name="target".format(i))
            targets_sparse.append(tf.squeeze(tf.argmax(target,1),name="Sq_"+target.op.name))
            self.target_weights.append(tf.ones([self.batch_size],name="weight".format(i)))
            targets = [target]

        #Hook for the input_feed
        self.target_inputs = targets

        #Leave the last observation as the first input to the decoder
        #self.encoder_inputs = self.observation_inputs[0:-1]
        with tf.variable_scope('encoder_inputs'):
            self.encoder_inputs = [tf.nn.dropout(
                                        tf.nn.relu(
                                            nn_ops.xw_plus_b(
                                                input_timestep, input_layer[0], input_layer[1])),
                                        1-parameters['embedding_dropout'])
                                   for
                                   input_timestep in self.observation_inputs[0:-1]]

        #decoder inputs are the last observation and all but the last future
        with tf.variable_scope('decoder_inputs'):
            self.decoder_inputs = [tf.nn.dropout(
                                        tf.nn.relu(
                                            nn_ops.xw_plus_b(
                                                self.observation_inputs[-1], input_layer[0], input_layer[1])),
                                        1-parameters['embedding_dropout'])]

        # Todo should this have the input layer applied?
            self.decoder_inputs.extend([self.future_inputs[i] for i in xrange(len(self.future_inputs) - 1)])

        with tf.variable_scope('seq_rnn'):
            self.LSTM_output, self.internal_states = seq2seq_f(self.encoder_inputs, self.decoder_inputs, feed_future_data)

        # self.outputs is a list of len(prediction_steps) containing [size batch x rnn_size]
        # The output projection below reduces this to:
        #                 a list of len(prediction_steps) containing [size batch x input_size]
        # BUG This is incorrect -- technically.
        # Because MDN.sample() is a random function, this sample is not the
        # sample being used in the loopback function.
        if output_projection is not None:
            self.MDN_output = [output_function(output) for output in self.LSTM_output]
            if self.model_type == 'MDN':
                self.MDN_sample = [MDN.sample(x) for x in self.MDN_output]
        else:
            self.MDN_output = self.LSTM_output

        def mse(x, y):
            return tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, x))))

        # TODO There are several types of cost functions to compare tracks. Implement many
        # Mainly, average MSE over the whole track, or just at a horizon time (t+10 or something)
        # There's this corner alg that Social LSTM refernces, but I haven't looked into it.
        # NOTE - there is a good cost function for the MDN (MLE), this is different to the track accuracy metric (above)
        if self.model_type == 'MDN':
            self.losses = tf.nn.seq2seq.sequence_loss(self.MDN_output,targets, self.target_weights,
                                                  #softmax_loss_function=lambda x, y: mse(x,y))
                                                  softmax_loss_function=MDN.lossfunc_wrapper)
            self.losses = self.losses / self.batch_size
            self.accuracy = self.losses #TODO placeholder, use MSE or something visually intuitive
        if self.model_type == 'classifier':
            embedding_regularizer = tf.reduce_sum(tf.abs(i_w),name="Embedding_L1_reg") # Only regularize embedding layer
            # Don't forget that sequence loss uses sparse targets
            self.losses = (tf.nn.seq2seq.sequence_loss(self.MDN_output, targets_sparse, self.target_weights)
                           + parameters['reg_embedding_beta']*embedding_regularizer)
            #TODO I have to take into account padding here
            #squeeze away output to remove a single element list (It would be longer if classifier was allowed 2+ timesteps
            correct_prediction = tf.equal(tf.argmax(tf.squeeze(self.MDN_output), 1), targets_sparse,
                                          name="Correct_prediction")
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="Accuracy")


        # Gradients and SGD update operation for training the model.
        tvars = tf.trainable_variables()
        #if train:
        # I don't see the difference here, as during testing the updates are not run
        self.gradient_norms = []
        self.updates = []
        #opt = tf.train.AdadeltaOptimizer(self.learning_rate)
        opt = tf.train.RMSPropOptimizer(self.learning_rate)
        #opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        gradients = tf.gradients(self.losses, tvars)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

        self.gradient_norms.append(norm)

        gradients = zip(clipped_gradients, tvars)
        self.updates.append(opt.apply_gradients(
            gradients, global_step=self.global_step))

        for gradient, variable in gradients:  #plot the gradient of each trainable variable
            if variable.name.find("seq_rnn/combined_tied_rnn_seq2seq/tied_rnn_seq2seq/MultiRNNCell") == 0:
                var_log_name = variable.name[64:] #Make the thing readable in Tensorboard
            else:
                var_log_name = variable.name
            if isinstance(gradient, ops.IndexedSlices):
                grad_values = gradient.values
            else:
                grad_values = gradient
            self.network_summaries.append(
                tf.summary.histogram(var_log_name, variable))
            self.network_summaries.append(
                tf.summary.histogram(var_log_name + "/gradients", grad_values))
            self.network_summaries.append(
                tf.summary.histogram(var_log_name + "/gradient_norm", clip_ops.global_norm([grad_values])))

        self.network_summaries.append(tf.summary.scalar('Loss',self.losses))
        self.network_summaries.append(tf.summary.scalar('Learning Rate', self.learning_rate))

        self.summary_op = tf.summary.merge(self.network_summaries)

        self.saver = tf.train.Saver(var_list=None, max_to_keep=999)

        return

    def step(self, session, observation_inputs, future_inputs, target_weights, train_model, summary_writer=None):
        """Run a step of the model feeding the given inputs.
        Args:
          session: tensorflow session to use.
          observation_inputs: list of numpy int vectors to feed as encoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          train: whether to do the backward step or only forward.
        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        Raises:
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(self.observation_steps):
            input_feed[self.observation_inputs[l].name] = observation_inputs[l]
        if self.model_type == 'MDN':
            for l in xrange(self.prediction_steps):
                input_feed[self.future_inputs[l].name] = future_inputs[l]
                input_feed[self.target_weights[l].name] = target_weights[l]
        if self.model_type == 'classifier':
                input_feed[self.target_inputs[0].name] = future_inputs[0]
                input_feed[self.target_weights[0].name] = target_weights[0]


        # Output feed: depends on whether we do a backward step or not.
        if train_model:
            output_feed = (self.updates +  # Update Op that does SGD. #This is the learning flag
                         self.gradient_norms +  # Gradient norm.
                         [self.losses] +
                           [self.accuracy])  # Loss for this batch.
        else:
            output_feed = [self.accuracy, self.losses]# Loss for this batch.
            if self.model_type == 'MDN':
                for l in xrange(self.prediction_steps):  # Output logits.
                    output_feed.append(self.MDN_sample[l])
            if self.model_type == 'classifier':
                output_feed.append(self.MDN_output[0])

        outputs = session.run(output_feed, input_feed)
        if summary_writer is not None:

            summary_str = session.run(self.summary_op,input_feed)
            summary_writer.add_summary(summary_str, self.global_step.eval(session=session))
        if train_model:
            return outputs[3], outputs[2], None  # accuracy, loss, no outputs.
        else:
            return outputs[0], outputs[1], outputs[2:]  # accuracy, loss, outputs
