# Copyright (C) 2016 by Akira TAMAMORI
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Notice:
# This file is tested on TensorFlow v0.10.0 only.

# Commentary:
# TODO: implemation of another initializer for LSTM

import numpy as np
import tensorflow as tf

from tensorflow.python.ops.rnn_cell import RNNCell


# Thanks to 'initializers_enhanced.py' of Project RNN Enhancement:
# https://github.com/nicolas-ivanov/Seq2Seq_Upgrade_TensorFlow/blob/master/rnn_enhancement/initializers_enhanced.py
def orthogonal_initializer(scale=1.0,partition_info=None):
    def _initializer(shape, dtype=tf.float32,partition_info=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=dtype)
    return _initializer


# Thanks to https://github.com/OlavHN/bnlstm
def batch_norm(inputs, name_scope, is_training, epsilon=1e-3, decay=0.99):
    with tf.variable_scope(name_scope):
        size = inputs.get_shape().as_list()[1]

        scale = tf.get_variable(
            'scale', [size], initializer=tf.constant_initializer(0.1))
        offset = tf.get_variable('offset', [size])

        population_mean = tf.get_variable(
            'population_mean', [size],
            initializer=tf.zeros_initializer, trainable=False)
        population_var = tf.get_variable(
            'population_var', [size],
            initializer=tf.ones_initializer, trainable=False)
        batch_mean, batch_var = tf.nn.moments(inputs, [0])

        # The following part is based on the implementation of :
        # https://github.com/cooijmanstim/recurrent-batch-normalization
        train_mean_op = tf.assign(
            population_mean,
            population_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(
            population_var, population_var * decay + batch_var * (1 - decay))

        if is_training is True:
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(
                    inputs, batch_mean, batch_var, offset, scale, epsilon)
        else:
            return tf.nn.batch_normalization(
                inputs, population_mean, population_var, offset, scale,
                epsilon)


class BN_LSTMCell(RNNCell):
    """LSTM cell with Recurrent Batch Normalization.

    This implementation is based on:
         http://arxiv.org/abs/1603.09025

    This implementation is also based on:
         https://github.com/OlavHN/bnlstm
         https://github.com/nicolas-ivanov/Seq2Seq_Upgrade_TensorFlow

    """

    def __init__(self, num_units, is_training,
                 use_peepholes=False, cell_clip=None,
                 initializer=orthogonal_initializer(),
                 num_proj=None, proj_clip=None,
                 forget_bias=1.0,
                 activation=tf.tanh):
        """Initialize the parameters for an LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          is_training: bool, set True when training.
          use_peepholes: bool, set True to enable diagonal/peephole
            connections.
          cell_clip: (optional) A float value, if provided the cell state
            is clipped by this value prior to the cell output activation.
          initializer: (optional) The initializer to use for the weight
            matrices.
          num_proj: (optional) int, The output dimensionality for
            the projection matrices.  If None, no projection is performed.
          forget_bias: Biases of the forget gate are initialized by default
            to 1 in order to reduce the scale of forgetting at the beginning of
            the training.
          activation: Activation function of the inner states.
        """
        self.num_units = num_units
        self.is_training = is_training
        self.use_peepholes = use_peepholes
        self.cell_clip = cell_clip
        self.num_proj = num_proj
        self.proj_clip = proj_clip
        self.initializer = initializer
        self.forget_bias = forget_bias
        self.activation = activation

        if num_proj:
            self._state_size = num_units + num_proj
            self._output_size = num_proj
        else:
            self._state_size = 2 * num_units
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):

        num_proj = self.num_units if self.num_proj is None else self.num_proj

        c_prev = tf.slice(state, [0, 0], [-1, self.num_units])
        h_prev = tf.slice(state, [0, self.num_units], [-1, num_proj])

        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]

        with tf.variable_scope(scope or type(self).__name__):
            if input_size.value is None:
                raise ValueError(
                    "Could not infer input size from inputs.get_shape()[-1]")

            W_xh = tf.get_variable(
                'W_xh',
                [input_size, 4 * self.num_units],
                initializer=self.initializer)
            W_hh = tf.get_variable(
                'W_hh',
                [num_proj, 4 * self.num_units],
                initializer=self.initializer)
            bias = tf.get_variable('B', [4 * self.num_units])

            xh = tf.matmul(inputs, W_xh)
            hh = tf.matmul(h_prev, W_hh)

            bn_xh = batch_norm(xh, 'xh', self.is_training)
            bn_hh = batch_norm(hh, 'hh', self.is_training)

            # i:input gate, j:new input, f:forget gate, o:output gate
            lstm_matrix = tf.nn.bias_add(tf.add(bn_xh, bn_hh), bias)
            i, j, f, o = tf.split(1, 4, lstm_matrix)

            # Diagonal connections
            if self.use_peepholes:
                w_f_diag = tf.get_variable(
                    "W_F_diag", shape=[self.num_units], dtype=dtype)
                w_i_diag = tf.get_variable(
                    "W_I_diag", shape=[self.num_units], dtype=dtype)
                w_o_diag = tf.get_variable(
                    "W_O_diag", shape=[self.num_units], dtype=dtype)

            if self.use_peepholes:
                c = c_prev * tf.sigmoid(f + self.forget_bias +
                                        w_f_diag * c_prev) + \
                    tf.sigmoid(i + w_i_diag * c_prev) * self.activation(j)
            else:
                c = c_prev * tf.sigmoid(f + self.forget_bias) + \
                    tf.sigmoid(i) * self.activation(j)

            if self.cell_clip is not None:
                c = tf.clip_by_value(c, -self.cell_clip, self.cell_clip)

            bn_c = batch_norm(c, 'cell', self.is_training)

            if self.use_peepholes:
                h = tf.sigmoid(o + w_o_diag * c) * self.activation(bn_c)
            else:
                h = tf.sigmoid(o) * self.activation(bn_c)

            if self.num_proj is not None:
                w_proj = tf.get_variable(
                    "W_P", [self.num_units, num_proj], dtype=dtype)

                h = tf.matmul(h, w_proj)
                if self.proj_clip is not None:
                    h = tf.clip_by_value(h, -self.proj_clip, self.proj_clip)

            return h, tf.concat(1, [c, h])
