# Copyright 2017 Google Inc.
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

"""Baseline models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.models import common_layers
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


@registry.register_model("baseline_lstm_seq2seq")
class LSTMSeq2Seq(t2t_model.T2TModel):

  def model_fn_body(self, features, train):
    return lstm_seq2seq_internal(features["inputs"], features["targets"],
                                 self._hparams, train)
