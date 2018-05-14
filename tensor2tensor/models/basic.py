# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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
"""Basic models for testing simple tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


@registry.register_model
class BasicFcRelu(t2t_model.T2TModel):
  """Basic fully-connected + ReLU model."""

  def body(self, features):
    hparams = self.hparams
    x = features["inputs"]
    shape = common_layers.shape_list(x)
    x = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])
    for i in range(hparams.num_hidden_layers):
      x = tf.layers.dense(x, hparams.hidden_size, name="layer_%d" % i)
      x = tf.nn.dropout(x, keep_prob=1.0 - hparams.dropout)
      x = tf.nn.relu(x)
    return tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)  # 4D For T2T.


@registry.register_model
class BasicAutoencoder(t2t_model.T2TModel):
  """A basic autoencoder, try with image_mnist_rev or image_cifar10_rev."""

  def __init__(self, *args, **kwargs):
    super(BasicAutoencoder, self).__init__(*args, **kwargs)
    self._cur_bottleneck_tensor = None
    self.is1d = None

  def bottleneck(self, x):
    with tf.variable_scope("bottleneck"):
      hparams = self.hparams
      x = tf.layers.dense(x, hparams.bottleneck_size, name="bottleneck")
      if hparams.mode == tf.estimator.ModeKeys.TRAIN:
        noise = 2.0 * tf.random_uniform(common_layers.shape_list(x)) - 1.0
        return tf.tanh(x) + noise * hparams.bottleneck_noise
      return tf.tanh(x)

  def unbottleneck(self, x, res_size):
    with tf.variable_scope("unbottleneck"):
      x = tf.layers.dense(x, res_size, name="dense")
      return x

  def bottleneck_loss(self, b):
    return 0.0

  def make_even_size(self, x):
    shape = [dim if dim is not None else -1 for dim in x.get_shape().as_list()]
    if shape[1] % 2 == 0 and shape[2] % 2 == 0:
      return x
    if shape[1] % 2 == 0 and self.is1d:
      return x
    x, _ = common_layers.pad_to_same_length(
        x, x, final_length_divisible_by=2, axis=1)
    if self.is1d:
      return x
    x, _ = common_layers.pad_to_same_length(
        x, x, final_length_divisible_by=2, axis=2)
    return x

  def encoder(self, x):
    with tf.variable_scope("encoder"):
      hparams = self.hparams
      kernel, strides = self._get_kernel_and_strides()
      # Down-convolutions.
      for i in range(hparams.num_hidden_layers):
        x = self.make_even_size(x)
        x = tf.layers.conv2d(
            x, hparams.hidden_size * 2**(i + 1), kernel, strides=strides,
            padding="SAME", activation=common_layers.belu, name="conv_%d" % i)
        x = common_layers.layer_norm(x)
      return x

  def decoder(self, x):
    with tf.variable_scope("decoder"):
      hparams = self.hparams
      kernel, strides = self._get_kernel_and_strides()
      # Up-convolutions.
      for i in range(hparams.num_hidden_layers):
        j = hparams.num_hidden_layers - i - 1
        x = tf.layers.conv2d_transpose(
            x, hparams.hidden_size * 2**j, kernel, strides=strides,
            padding="SAME", activation=common_layers.belu, name="deconv_%d" % j)
        x = common_layers.layer_norm(x)
      return x

  def body(self, features):
    hparams = self.hparams
    is_training = hparams.mode == tf.estimator.ModeKeys.TRAIN
    if hparams.mode != tf.estimator.ModeKeys.PREDICT:
      x = features["targets"]
      shape = common_layers.shape_list(x)
      is1d = shape[2] == 1
      self.is1d = is1d
      # Run encoder.
      x = self.encoder(x)
      # Bottleneck (mix during early training, not too important but stable).
      b = self.bottleneck(x)
      self._cur_bottleneck_tensor = b
      b_loss = self.bottleneck_loss(b)
      b = self.unbottleneck(b, common_layers.shape_list(x)[-1])
      b = common_layers.mix(b, x, hparams.bottleneck_warmup_steps, is_training)
      # With probability bottleneck_max_prob use the bottleneck, otherwise x.
      if hparams.bottleneck_max_prob < 1.0:
        x = tf.where(tf.less(tf.random_uniform([]),
                             hparams.bottleneck_max_prob), b, x)
      else:
        x = b
    else:
      if self._cur_bottleneck_tensor is None:
        b = self.sample()
      else:
        b = self._cur_bottleneck_tensor
      res_size = self.hparams.hidden_size * 2**self.hparams.num_hidden_layers
      res_size = min(res_size, hparams.max_hidden_size)
      x = self.unbottleneck(b, res_size)
    # Run decoder.
    x = self.decoder(x)
    if hparams.mode == tf.estimator.ModeKeys.PREDICT:
      return x, {"bottleneck_loss": 0.0}
    # Cut to the right size and mix before returning.
    res = x[:, :shape[1], :shape[2], :]
    res = common_layers.mix(res, features["targets"],
                            hparams.bottleneck_warmup_steps // 2, is_training)
    return res, {"bottleneck_loss": b_loss}

  def sample(self):
    hp = self.hparams
    div_x = 2**hp.num_hidden_layers
    div_y = 1 if self.is1d else 2**hp.num_hidden_layers
    size = [hp.batch_size, hp.sample_height // div_x, hp.sample_width // div_y,
            hp.bottleneck_size]
    # Sample in [-1, 1] as the bottleneck is under tanh.
    return 2.0 * tf.random_uniform(size) - 1.0

  def encode(self, x):
    """Auto-encode x and return the bottleneck."""
    features = {"targets": x}
    self(features)  # pylint: disable=not-callable
    res = tf.maximum(0.0, self._cur_bottleneck_tensor)  # Be 0/1 and not -1/1.
    self._cur_bottleneck_tensor = None
    return res

  def infer(self, features, *args, **kwargs):
    """Produce predictions from the model by sampling."""
    # Inputs and features preparation needed to handle edge cases.
    if not features:
      features = {}
    inputs_old = None
    if "inputs" in features and len(features["inputs"].shape) < 4:
      inputs_old = features["inputs"]
      features["inputs"] = tf.expand_dims(features["inputs"], 2)

    # Sample and decode.
    # TODO(lukaszkaiser): is this a universal enough way to get channels?
    try:
      num_channels = self.hparams.problem.num_channels
    except AttributeError:
      num_channels = 1
    if "targets" not in features:
      features["targets"] = tf.zeros(
          [self.hparams.batch_size, 1, 1, num_channels],
          dtype=tf.int32)
    logits, _ = self(features)  # pylint: disable=not-callable
    samples = tf.argmax(logits, axis=-1)

    # Restore inputs to not confuse Estimator in edge cases.
    if inputs_old is not None:
      features["inputs"] = inputs_old

    # Return samples.
    return samples

  def decode(self, bottleneck):
    """Auto-decode from the bottleneck and return the result."""
    # Get the shape from bottleneck and num channels.
    shape = common_layers.shape_list(bottleneck)
    try:
      num_channels = self.hparams.problem.num_channels
    except AttributeError:
      num_channels = 1
    dummy_targets = tf.zeros(shape[:-1] + [num_channels])
    # Set the bottleneck to decode.
    if len(shape) > 4:
      bottleneck = tf.squeeze(bottleneck, axis=[1])
    bottleneck = 2 * bottleneck - 1  # Be -1/1 instead of 0/1.
    self._cur_bottleneck_tensor = bottleneck
    # Run decoding.
    res = self.infer({"targets": dummy_targets})
    self._cur_bottleneck_tensor = None
    return res

  def _get_kernel_and_strides(self):
    hparams = self.hparams
    kernel = (hparams.kernel_height, hparams.kernel_width)
    kernel = (hparams.kernel_height, 1) if self.is1d else kernel
    strides = (2, 1) if self.is1d else (2, 2)
    return (kernel, strides)


@registry.register_hparams
def basic_fc_small():
  """Small fully connected model."""
  hparams = common_hparams.basic_params1()
  hparams.learning_rate = 0.1
  hparams.batch_size = 128
  hparams.hidden_size = 256
  hparams.num_hidden_layers = 2
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 0.0
  hparams.dropout = 0.0
  return hparams


@registry.register_hparams
def basic_autoencoder():
  """Basic autoencoder model."""
  hparams = common_hparams.basic_params1()
  hparams.optimizer = "Adam"
  hparams.learning_rate_constant = 0.0002
  hparams.learning_rate_warmup_steps = 500
  hparams.learning_rate_schedule = "constant * linear_warmup"
  hparams.label_smoothing = 0.0
  hparams.batch_size = 128
  hparams.hidden_size = 64
  hparams.num_hidden_layers = 5
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 0.0
  hparams.kernel_height = 4
  hparams.kernel_width = 4
  hparams.dropout = 0.1
  hparams.add_hparam("max_hidden_size", 1024)
  hparams.add_hparam("bottleneck_size", 128)
  hparams.add_hparam("bottleneck_noise", 0.1)
  hparams.add_hparam("bottleneck_warmup_steps", 3000)
  hparams.add_hparam("bottleneck_max_prob", 1.0)
  hparams.add_hparam("sample_height", 32)
  hparams.add_hparam("sample_width", 32)
  return hparams
