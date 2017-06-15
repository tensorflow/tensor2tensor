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

"""The Neural GPU model and its variants."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.models import common_hparams
from tensor2tensor.models import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


def neural_gpu(inputs, hparams, train, name=None):
  """The core Neural GPU."""
  with tf.variable_scope(name, "neural_gpu"):

    def step(state, inp):  # pylint: disable=missing-docstring
      x = tf.nn.dropout(state, 1.0 - hparams.dropout * tf.to_float(train))
      for layer in xrange(hparams.num_hidden_layers):
        x = common_layers.conv_gru(
            x, (hparams.kernel_height, hparams.kernel_width),
            hparams.hidden_size,
            name="cgru_%d" % layer)
      # Padding input is zeroed-out in the modality, we check this by summing.
      padding_inp = tf.less(tf.reduce_sum(tf.abs(inp), axis=[1, 2]), 0.00001)
      new_state = tf.where(padding_inp, state, x)  # No-op where inp is padding.
      return new_state

    return tf.foldl(
        step,
        tf.transpose(inputs, [1, 0, 2, 3]),
        initializer=inputs,
        parallel_iterations=1,
        swap_memory=True)


@registry.register_model
class NeuralGPU(t2t_model.T2TModel):

  def model_fn_body(self, features, train):
    return neural_gpu(features["inputs"], self._hparams, train)


def diagonal_neural_gpu(inputs, hparams, train, name=None):
  """Improved Neural GPU as in https://arxiv.org/abs/1702.08727."""
  with tf.variable_scope(name, "diagonal_neural_gpu"):

    def step(state_tup, inp):
      """Single step of the improved Neural GPU."""
      state, _ = state_tup
      x = state
      for layer in xrange(hparams.num_hidden_layers):
        x, new_loss = common_layers.diagonal_conv_gru(
            x, (hparams.kernel_height, hparams.kernel_width),
            hparams.hidden_size,
            train,
            dropout=hparams.dropout,
            name="dcgru_%d" % layer)
      # Padding input is zeroed-out in the modality, we check this by summing.
      padding_inp = tf.less(tf.reduce_sum(tf.abs(inp), axis=[1, 2]), 0.00001)
      new_state = tf.where(padding_inp, state, x)  # No-op where inp is padding.
      return new_state, new_loss

    final_state, losses = tf.scan(
        step,
        tf.transpose(inputs, [1, 0, 2, 3]),
        initializer=(inputs, tf.constant(0.0)),
        parallel_iterations=1,
        swap_memory=True)
    return final_state[0, :, :, :, :], 2.0 * tf.reduce_mean(losses)


@registry.register_model
class DiagonalNeuralGPU(t2t_model.T2TModel):

  def model_fn_body(self, features, train):
    return diagonal_neural_gpu(features["inputs"], self._hparams, train)


@registry.register_hparams("neural_gpu1")
def neural_gpu_params1():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.batch_size = 1024
  hparams.num_hidden_layers = 1
  hparams.hidden_size = 256
  hparams.dropout = 0.1
  hparams.label_smoothing = 0.0
  hparams.clip_grad_norm = 10.0
  hparams.num_hidden_layers = 1
  hparams.kernel_height = 3
  hparams.kernel_width = 1
  hparams.learning_rate_decay_scheme = "exp50k"
  hparams.learning_rate = 0.02
  hparams.learning_rate_warmup_steps = 3000
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 0.0
  hparams.num_sampled_classes = 0
  hparams.sampling_method = "argmax"
  hparams.optimizer_adam_epsilon = 1e-6
  hparams.optimizer_adam_beta1 = 0.85
  hparams.optimizer_adam_beta2 = 0.997
  return hparams
