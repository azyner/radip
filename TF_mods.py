from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import rnn
from tensorflow.python.ops.seq2seq import rnn_decoder


def basic_rnn_seq2seq_with_loop_function(
        encoder_inputs, decoder_inputs, cell, dtype=dtypes.float32, loop_function=None, scope=None):
    """Basic RNN sequence-to-sequence model. Edited for a loopback function. Don't know why this isn't in the
    current library
    """
    with variable_scope.variable_scope(scope or "basic_rnn_seq2seq_with_loop_function"):
        _, enc_state = rnn.rnn(cell, encoder_inputs, dtype=dtype)
        return rnn_decoder(decoder_inputs, enc_state, cell, loop_function=loop_function)