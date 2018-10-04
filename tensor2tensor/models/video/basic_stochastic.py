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

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_video

from tensor2tensor.models.video import base_vae
from tensor2tensor.models.video import basic_deterministic
from tensor2tensor.models.video import basic_deterministic_params

from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_model
class NextFrameBasicStochastic(
    basic_deterministic.NextFrameBasicDeterministic,
    base_vae.NextFrameBaseVae):
  """Stochastic version of basic next-frame model."""

  def inject_latent(self, layer, features, filters):
    """Inject a VAE-style latent."""
    # Latent for stochastic model
    input_frames = tf.to_float(features["inputs_raw"])
    target_frames = tf.to_float(features["targets_raw"])
    full_video = tf.concat([input_frames, target_frames], axis=1)
    latent_mean, latent_std = self.construct_latent_tower(
        full_video, time_axis=1)
    latent = common_video.get_gaussian_tensor(latent_mean, latent_std)
    latent = tf.layers.flatten(latent)
    latent = tf.expand_dims(latent, axis=1)
    latent = tf.expand_dims(latent, axis=1)
    latent_mask = tf.layers.dense(latent, filters, name="latent_mask")
    zeros_mask = tf.zeros(
        common_layers.shape_list(layer)[:-1] + [filters], dtype=tf.float32)
    layer = tf.concat([layer, latent_mask + zeros_mask], axis=-1)
    extra_loss = self.get_extra_loss(latent_mean, latent_std)
    return layer, extra_loss


@registry.register_model
class NextFrameBasicStochasticDiscrete(
    basic_deterministic.NextFrameBasicDeterministic):
  """Basic next-frame model with a tiny discrete latent."""

  def inject_latent(self, layer, features, filters):
    """Inject a deterministic latent based on the target frame."""
    del filters
    hparams = self.hparams
    final_filters = common_layers.shape_list(layer)[-1]
    filters = hparams.hidden_size
    kernel = (4, 4)

    def add_d(layer, d):
      z_mul = tf.layers.dense(d, final_filters, name="unbottleneck_mul")
      if not hparams.complex_addn:
        return layer + z_mul
      layer *= tf.nn.sigmoid(z_mul)
      z_add = tf.layers.dense(d, final_filters, name="unbottleneck_add")
      layer += z_add
      return layer

    if self.is_predicting:
      layer_shape = common_layers.shape_list(layer)
      if hparams.full_latent_tower:
        rand = tf.random_uniform(layer_shape[:-1] + [hparams.bottleneck_bits])
      else:
        rand = tf.random_uniform(layer_shape[:-3] + [
            1, 1, hparams.bottleneck_bits])
      d = 2.0 * tf.to_float(tf.less(0.5, rand)) - 1.0
      return add_d(layer, d), 0.0

    # Embed.
    frames = tf.concat(
        [features["cur_target_frame"], features["inputs"]], axis=-1)
    x = tf.layers.dense(
        frames, filters, name="latent_embed",
        bias_initializer=tf.random_normal_initializer(stddev=0.01))
    x = common_attention.add_timing_signal_nd(x)

    if hparams.full_latent_tower:
      for i in range(hparams.num_compress_steps):
        with tf.variable_scope("latent_downstride%d" % i):
          x = common_layers.make_even_size(x)
          if i < hparams.filter_double_steps:
            filters *= 2
          x = common_attention.add_timing_signal_nd(x)
          x = tf.layers.conv2d(x, filters, kernel,
                               activation=common_layers.belu,
                               strides=(2, 2), padding="SAME")
          x = common_layers.layer_norm(x)
    else:
      x = common_layers.double_discriminator(x)
      x = tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)
    x = tf.tanh(tf.layers.dense(x, hparams.bottleneck_bits, name="bottleneck"))
    d = x + tf.stop_gradient(2.0 * tf.to_float(tf.less(0.0, x)) - 1.0 - x)
    if hparams.mode == tf.estimator.ModeKeys.TRAIN:
      noise = tf.random_uniform(common_layers.shape_list(x))
      noise = 2.0 * tf.to_float(tf.less(hparams.bottleneck_noise, noise)) - 1.0
      x *= noise
      d = x + tf.stop_gradient(2.0 * tf.to_float(tf.less(0.0, x)) - 1.0 - x)
      p = common_layers.inverse_lin_decay(hparams.discrete_warmup_steps)
      d = tf.where(tf.less(tf.random_uniform([]), p), d, x)
    return add_d(layer, d), 0.0


@registry.register_hparams
def next_frame_basic_stochastic():
  """Basic 2-frame conv model with stochastic tower."""
  hparams = basic_deterministic_params.next_frame_basic_deterministic()
  hparams.stochastic_model = True
  hparams.add_hparam("latent_channels", 1)
  hparams.add_hparam("latent_std_min", -5.0)
  hparams.add_hparam("num_iterations_1st_stage", 15000)
  hparams.add_hparam("num_iterations_2nd_stage", 15000)
  hparams.add_hparam("latent_loss_multiplier", 1e-3)
  hparams.add_hparam("latent_loss_multiplier_dynamic", False)
  hparams.add_hparam("latent_loss_multiplier_alpha", 1e-5)
  hparams.add_hparam("latent_loss_multiplier_epsilon", 1.0)
  hparams.add_hparam("latent_loss_multiplier_schedule", "constant")
  hparams.add_hparam("latent_num_frames", 0)  # 0 means use all frames.
  hparams.add_hparam("anneal_end", 50000)
  hparams.add_hparam("information_capacity", 0.0)
  return hparams


@registry.register_hparams
def next_frame_sampling_stochastic():
  """Basic 2-frame conv model with stochastic tower."""
  hparams = basic_deterministic_params.next_frame_sampling()
  hparams.stochastic_model = True
  hparams.add_hparam("latent_channels", 1)
  hparams.add_hparam("latent_std_min", -5.0)
  hparams.add_hparam("num_iterations_1st_stage", 15000)
  hparams.add_hparam("num_iterations_2nd_stage", 15000)
  hparams.add_hparam("latent_loss_multiplier", 1e-3)
  hparams.add_hparam("latent_loss_multiplier_dynamic", False)
  hparams.add_hparam("latent_loss_multiplier_alpha", 1e-5)
  hparams.add_hparam("latent_loss_multiplier_epsilon", 1.0)
  hparams.add_hparam("latent_loss_multiplier_schedule", "constant")
  hparams.add_hparam("latent_num_frames", 0)  # 0 means use all frames.
  hparams.add_hparam("anneal_end", 40000)
  hparams.add_hparam("information_capacity", 0.0)
  return hparams


@registry.register_hparams
def next_frame_basic_stochastic_discrete():
  """Basic 2-frame conv model with stochastic discrete latent."""
  hparams = basic_deterministic_params.next_frame_sampling()
  hparams.add_hparam("bottleneck_bits", 32)
  hparams.add_hparam("bottleneck_noise", 0.05)
  hparams.add_hparam("discrete_warmup_steps", 4000)
  hparams.add_hparam("full_latent_tower", False)
  hparams.add_hparam("complex_addn", True)
  return hparams
