# coding=utf-8
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

# Dependency imports

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


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


def lstm_attention_decoder(inputs, hparams, train, name, initial_state,
                           encoder_outputs):
  """Run LSTM cell with attention on inputs of shape [batch x time x size]."""

  def dropout_lstm_cell():
    return tf.contrib.rnn.DropoutWrapper(
        tf.nn.rnn_cell.BasicLSTMCell(hparams.hidden_size),
        input_keep_prob=1.0 - hparams.dropout * tf.to_float(train))

  layers = [dropout_lstm_cell() for _ in range(hparams.num_hidden_layers)]
  if hparams.attention_mechanism == "luong":
    attention_mechanism_class = tf.contrib.seq2seq.LuongAttention
  elif hparams.attention_mechanism == "bahdanau":
    attention_mechanism_class = tf.contrib.seq2seq.BahdanauAttention
  else:
    raise ValueError("Unknown hparams.attention_mechanism = %s, must be "
                     "luong or bahdanu." % hparams.attention_mechanism)
  attention_mechanism = attention_mechanism_class(
      hparams.hidden_size, encoder_outputs)

  cell = tf.contrib.seq2seq.AttentionWrapper(
      tf.nn.rnn_cell.MultiRNNCell(layers),
      [attention_mechanism]*hparams.num_heads,
      attention_layer_size=[hparams.attention_layer_size]*hparams.num_heads,
      output_attention=(hparams.output_attention == 1))

  batch_size = inputs.get_shape()[0].value
  if batch_size is None:
    batch_size = tf.shape(inputs)[0]

  initial_state = cell.zero_state(batch_size, tf.float32).clone(
      cell_state=initial_state)

  with tf.variable_scope(name):
    output, state = tf.nn.dynamic_rnn(
        cell,
        inputs,
        initial_state=initial_state,
        dtype=tf.float32,
        time_major=False)

    # For multi-head attention project output back to hidden size
    if hparams.output_attention == 1 and hparams.num_heads > 1:
      output = tf.layers.dense(output, hparams.hidden_size)

    return output, state


def lstm_seq2seq_internal(inputs, targets, hparams, train):
  """The basic LSTM seq2seq model, main step used for training."""
  with tf.variable_scope("lstm_seq2seq"):
    if inputs is not None:
      # Flatten inputs.
      inputs = common_layers.flatten4d3d(inputs)
      # LSTM encoder.
      _, final_encoder_state = lstm(
          tf.reverse(inputs, axis=[1]), hparams, train, "encoder")
    else:
      final_encoder_state = None
    # LSTM decoder.
    shifted_targets = common_layers.shift_right(targets)
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
    shifted_targets = common_layers.shift_right(targets)
    decoder_outputs, _ = lstm_attention_decoder(
        common_layers.flatten4d3d(shifted_targets), hparams, train, "decoder",
        final_encoder_state, encoder_outputs)
    return tf.expand_dims(decoder_outputs, axis=2)


@registry.register_model
class LSTMSeq2seq(t2t_model.T2TModel):

  def model_fn_body(self, features):
    # TODO(lukaszkaiser): investigate this issue and repair.
    if self._hparams.initializer == "orthogonal":
      raise ValueError("LSTM models fail with orthogonal initializer.")
    train = self._hparams.mode == tf.estimator.ModeKeys.TRAIN
    return lstm_seq2seq_internal(features.get("inputs"), features["targets"],
                                 self._hparams, train)


@registry.register_model
class LSTMSeq2seqAttention(t2t_model.T2TModel):

  def model_fn_body(self, features):
    # TODO(lukaszkaiser): investigate this issue and repair.
    if self._hparams.initializer == "orthogonal":
      raise ValueError("LSTM models fail with orthogonal initializer.")
    train = self._hparams.mode == tf.estimator.ModeKeys.TRAIN
    return lstm_seq2seq_internal_attention(
        features.get("inputs"), features["targets"], self._hparams, train)


@registry.register_hparams
def lstm_seq2seq():
  """hparams for LSTM."""
  hparams = common_hparams.basic_params1()
  hparams.batch_size = 1024
  hparams.hidden_size = 128
  hparams.num_hidden_layers = 2
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 0.0
  return hparams


def lstm_attention_base():
  """Base attention params."""
  hparams = lstm_seq2seq()
  hparams.add_hparam("attention_layer_size", hparams.hidden_size)
  hparams.add_hparam("output_attention", True)
  hparams.add_hparam("num_heads", 1)
  return hparams


@registry.register_hparams
def lstm_bahdanau_attention():
  """Hparams for LSTM with bahdanau attention."""
  hparams = lstm_attention_base()
  hparams.add_hparam("attention_mechanism", "bahdanau")
  return hparams


@registry.register_hparams
def lstm_luong_attention():
  """Hparams for LSTM with luong attention."""
  hparams = lstm_attention_base()
  hparams.add_hparam("attention_mechanism", "luong")
  return hparams


@registry.register_hparams
def lstm_attention():
  """For backwards compatibility, defaults to bahdanau."""
  return lstm_bahdanau_attention()


@registry.register_hparams
def lstm_bahdanau_attention_multi():
  """Multi-head Bahdanu attention."""
  hparams = lstm_bahdanau_attention()
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def lstm_luong_attention_multi():
  """Multi-head Luong attention."""
  hparams = lstm_luong_attention()
  hparams.num_heads = 4
  return hparams
