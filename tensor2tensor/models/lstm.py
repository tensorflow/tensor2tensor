# Copyright 2017 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RNN LSTM models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports

from tensor2tensor.models import common_hparams
from tensor2tensor.models import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf
from tensorflow.python.util import nest


# Track Tuple of state and attention values
AttentionTuple = collections.namedtuple("AttentionTuple",
                                        ("state", "attention"))


class ExternalAttentionCellWrapper(tf.contrib.rnn.RNNCell):
  """Wrapper for external attention states for an encoder-decoder setup."""

  def __init__(self, cell, attn_states, attn_vec_size=None,
               input_size=None, state_is_tuple=True, reuse=None):
    """Create a cell with attention.

    Args:
      cell: an RNNCell, an attention is added to it.
      attn_states: External attention states typically the encoder output in the
        form [batch_size, time steps, hidden size]
      attn_vec_size: integer, the number of convolutional features calculated
        on attention state and a size of the hidden layer built from
        base cell state. Equal attn_size to by default.
      input_size: integer, the size of a hidden linear layer,
        built from inputs and attention. Derived from the input tensor
        by default.
      state_is_tuple: If True, accepted and returned states are n-tuples, where
        `n = len(cells)`.  Must be set to True else will raise an exception
        concatenated along the column axis.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if the flag `state_is_tuple` is `False` or if shape of
        `attn_states` is not 3 or if innermost dimension (hidden size) is None.
    """
    super(ExternalAttentionCellWrapper, self).__init__(_reuse=reuse)
    if not state_is_tuple:
      raise ValueError("Only tuple state is supported")

    self._cell = cell
    self._input_size = input_size

    # Validate attn_states shape.
    attn_shape = attn_states.get_shape()
    if not attn_shape or len(attn_shape) != 3:
      raise ValueError("attn_shape must be rank 3")

    self._attn_states = attn_states
    self._attn_size = attn_shape[2].value
    if self._attn_size is None:
      raise ValueError("Hidden size of attn_states cannot be None")

    self._attn_vec_size = attn_vec_size
    if self._attn_vec_size is None:
      self._attn_vec_size = self._attn_size

    self._reuse = reuse

  @property
  def state_size(self):
    return AttentionTuple(self._cell.state_size, self._attn_size)

  @property
  def output_size(self):
    return self._attn_size

  def combine_state(self, previous_state):
    """Combines previous state (from encoder) with internal attention values.

    You must use this function to derive the initial state passed into
    this cell as it expects a named tuple (AttentionTuple).

    Args:
      previous_state: State from another block that will be fed into this cell;
        Must have same structure as the state of the cell wrapped by this.
    Returns:
      Combined state (AttentionTuple).
    """
    batch_size = self._attn_states.get_shape()[0].value
    if batch_size is None:
      batch_size = tf.shape(self._attn_states)[0]
    zeroed_state = self.zero_state(batch_size, self._attn_states.dtype)
    return AttentionTuple(previous_state, zeroed_state.attention)

  def call(self, inputs, state):
    """Long short-term memory cell with attention (LSTMA)."""

    if not isinstance(state, AttentionTuple):
      raise TypeError("State must be of type AttentionTuple")

    state, attns = state
    attn_states = self._attn_states
    attn_length = attn_states.get_shape()[1].value
    if attn_length is None:
      attn_length = tf.shape(attn_states)[1]

    input_size = self._input_size
    if input_size is None:
      input_size = inputs.get_shape().as_list()[1]
    if attns is not None:
      inputs = tf.layers.dense(tf.concat([inputs, attns], axis=1), input_size)
    lstm_output, new_state = self._cell(inputs, state)

    new_state_cat = tf.concat(nest.flatten(new_state), 1)
    new_attns = self._attention(new_state_cat, attn_states, attn_length)

    with tf.variable_scope("attn_output_projection"):
      output = tf.layers.dense(tf.concat([lstm_output, new_attns], axis=1),
                               self._attn_size)

    new_state = AttentionTuple(new_state, new_attns)

    return output, new_state

  def _attention(self, query, attn_states, attn_length):
    conv2d = tf.nn.conv2d
    reduce_sum = tf.reduce_sum
    softmax = tf.nn.softmax
    tanh = tf.tanh

    with tf.variable_scope("attention"):
      k = tf.get_variable(
          "attn_w", [1, 1, self._attn_size, self._attn_vec_size])
      v = tf.get_variable("attn_v", [self._attn_vec_size, 1])
      hidden = tf.reshape(attn_states,
                          [-1, attn_length, 1, self._attn_size])
      hidden_features = conv2d(hidden, k, [1, 1, 1, 1], "SAME")
      y = tf.layers.dense(query, self._attn_vec_size)
      y = tf.reshape(y, [-1, 1, 1, self._attn_vec_size])
      s = reduce_sum(v * tanh(hidden_features + y), [2, 3])
      a = softmax(s)
      d = reduce_sum(
          tf.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
      new_attns = tf.reshape(d, [-1, self._attn_size])

      return new_attns


def lstm(inputs, hparams, train, name, initial_state=None):
  """Run LSTM cell on inputs, assuming they are [batch x time x size]."""

  def dropout_lstm_cell():
    return tf.contrib.rnn.DropoutWrapper(
        tf.contrib.rnn.BasicLSTMCell(hparams.hidden_size),
        input_keep_prob=1.0 - hparams.dropout * tf.to_float(train))

  layers = [dropout_lstm_cell() for _ in range(hparams.num_hidden_layers)]
  with tf.variable_scope(name):
    return tf.nn.dynamic_rnn(
        tf.contrib.rnn.MultiRNNCell(layers),
        inputs,
        initial_state=initial_state,
        dtype=tf.float32,
        time_major=False)


def lstm_attention_decoder(inputs, hparams, train, name,
                           initial_state, attn_states):
  """Run LSTM cell with attention on inputs of shape [batch x time x size]."""

  def dropout_lstm_cell():
    return tf.contrib.rnn.DropoutWrapper(
        tf.nn.rnn_cell.BasicLSTMCell(hparams.hidden_size),
        input_keep_prob=1.0 - hparams.dropout * tf.to_float(train))

  layers = [dropout_lstm_cell() for _ in range(hparams.num_hidden_layers)]
  cell = ExternalAttentionCellWrapper(tf.nn.rnn_cell.MultiRNNCell(layers),
                                      attn_states,
                                      attn_vec_size=hparams.attn_vec_size)
  initial_state = cell.combine_state(initial_state)
  with tf.variable_scope(name):
    return tf.nn.dynamic_rnn(
        cell,
        inputs,
        initial_state=initial_state,
        dtype=tf.float32,
        time_major=False)


def lstm_seq2seq_internal(inputs, targets, hparams, train):
  """The basic LSTM seq2seq model, main step used for training."""
  with tf.variable_scope("lstm_seq2seq"):
    # Flatten inputs.
    inputs = common_layers.flatten4d3d(inputs)
    # LSTM encoder.
    _, final_encoder_state = lstm(
        tf.reverse(inputs, axis=[1]), hparams, train, "encoder")
    # LSTM decoder.
    shifted_targets = common_layers.shift_left(targets)
    decoder_outputs, _ = lstm(
        common_layers.flatten4d3d(shifted_targets),
        hparams,
        train,
        "decoder",
        initial_state=final_encoder_state)
    return tf.expand_dims(decoder_outputs, axis=2)


def lstm_seq2seq_internal_attention(inputs, targets, hparams, train):
  """LSTM seq2seq model with attention, main step used for training."""
  with tf.variable_scope("lstm_seq2seq_attention"):
    # Flatten inputs.
    inputs = common_layers.flatten4d3d(inputs)
    # LSTM encoder.
    encoder_outputs, final_encoder_state = lstm(
        tf.reverse(inputs, axis=[1]), hparams, train, "encoder")
    # LSTM decoder with attention
    shifted_targets = common_layers.shift_left(targets)
    decoder_outputs, _ = lstm_attention_decoder(
        common_layers.flatten4d3d(shifted_targets),
        hparams,
        train,
        "decoder",
        final_encoder_state, encoder_outputs)
    return tf.expand_dims(decoder_outputs, axis=2)


@registry.register_model("baseline_lstm_seq2seq")
class LSTMSeq2Seq(t2t_model.T2TModel):

  def model_fn_body(self, features):
    train = self._hparams.mode == tf.contrib.learn.ModeKeys.TRAIN
    return lstm_seq2seq_internal(features["inputs"], features["targets"],
                                 self._hparams, train)


@registry.register_model("baseline_lstm_seq2seq_attention")
class LSTMSeq2SeqAttention(t2t_model.T2TModel):

  def model_fn_body(self, features):
    train = self._hparams.mode == tf.contrib.learn.ModeKeys.TRAIN
    return lstm_seq2seq_internal_attention(
        features["inputs"], features["targets"], self._hparams, train)


@registry.register_hparams
def lstm_attention():
  """hparams for LSTM with attention."""
  hparams = common_hparams.basic_params1()
  hparams.batch_size = 1024
  hparams.hidden_size = 128
  hparams.num_hidden_layers = 2

  # Attention
  hparams.add_hparam("attn_vec_size", hparams.hidden_size)
  return hparams
