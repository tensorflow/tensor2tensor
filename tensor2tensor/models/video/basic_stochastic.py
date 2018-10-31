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
from tensor2tensor.layers import discretization

from tensor2tensor.models.video import base_vae
from tensor2tensor.models.video import basic_deterministic
from tensor2tensor.models.video import basic_deterministic_params

from tensor2tensor.utils import registry

import tensorflow as tf

tfl = tf.layers


@registry.register_model
class NextFrameBasicStochastic(
    basic_deterministic.NextFrameBasicDeterministic,
    base_vae.NextFrameBaseVae):
  """Stochastic version of basic next-frame model."""

  def inject_latent(self, layer, inputs, target):
    """Inject a VAE-style latent."""
    # Latent for stochastic model
    filters = 128
    full_video = tf.stack(inputs + [target], axis=1)
    latent_mean, latent_std = self.construct_latent_tower(
        full_video, time_axis=1)
    latent = common_video.get_gaussian_tensor(latent_mean, latent_std)
    latent = tfl.flatten(latent)
    latent = tf.expand_dims(latent, axis=1)
    latent = tf.expand_dims(latent, axis=1)
    latent_mask = tfl.dense(latent, filters, name="latent_mask")
    zeros_mask = tf.zeros(
        common_layers.shape_list(layer)[:-1] + [filters], dtype=tf.float32)
    layer = tf.concat([layer, latent_mask + zeros_mask], axis=-1)
    extra_loss = self.get_kl_loss([latent_mean], [latent_std])
    return layer, extra_loss


@registry.register_model
class NextFrameBasicStochasticDiscrete(
    basic_deterministic.NextFrameBasicDeterministic):
  """Basic next-frame model with a tiny discrete latent."""

  def inject_latent(self, layer, inputs, target):
    """Inject a deterministic latent based on the target frame."""
    hparams = self.hparams
    final_filters = common_layers.shape_list(layer)[-1]
    filters = hparams.hidden_size
    kernel = (4, 4)
    layer_shape = common_layers.shape_list(layer)

    def add_bits(layer, bits):
      z_mul = tfl.dense(bits, final_filters, name="unbottleneck_mul")
      if not hparams.complex_addn:
        return layer + z_mul
      layer *= tf.nn.sigmoid(z_mul)
      z_add = tfl.dense(bits, final_filters, name="unbottleneck_add")
      layer += z_add
      return layer

    if not self.is_training:
      if hparams.full_latent_tower:
        rand = tf.random_uniform(layer_shape[:-1] + [hparams.bottleneck_bits])
        bits = 2.0 * tf.to_float(tf.less(0.5, rand)) - 1.0
      else:
        bits, _ = discretization.predict_bits_with_lstm(
            layer, hparams.latent_predictor_state_size, hparams.bottleneck_bits,
            temperature=hparams.latent_predictor_temperature)
        bits = tf.expand_dims(tf.expand_dims(bits, axis=1), axis=2)
      return add_bits(layer, bits), 0.0

    # Embed.
    frames = tf.concat(inputs + [target], axis=-1)
    x = tfl.dense(
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
          x = tfl.conv2d(x, filters, kernel,
                         activation=common_layers.belu,
                         strides=(2, 2), padding="SAME")
          x = common_layers.layer_norm(x)
    else:
      x = common_layers.double_discriminator(x)
      x = tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)

    bits, bits_clean = discretization.tanh_discrete_bottleneck(
        x, hparams.bottleneck_bits, hparams.bottleneck_noise,
        hparams.discretize_warmup_steps, hparams.mode)
    if not hparams.full_latent_tower:
      _, pred_loss = discretization.predict_bits_with_lstm(
          layer, hparams.latent_predictor_state_size, hparams.bottleneck_bits,
          target_bits=bits_clean)

    return add_bits(layer, bits), pred_loss


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
  hparams.batch_size = 2
  hparams.video_num_target_frames = 16
  hparams.scheduled_sampling_mode = "prob_inverse_lin"
  hparams.scheduled_sampling_decay_steps = 40000
  hparams.scheduled_sampling_max_prob = 1.0
  hparams.dropout = 0.2
  hparams.filter_double_steps = 3
  hparams.hidden_size = 96
  hparams.learning_rate_constant = 0.005
  hparams.learning_rate_warmup_steps = 2000
  hparams.learning_rate_schedule = "linear_warmup * constant"
  hparams.add_hparam("bottleneck_bits", 256)
  hparams.add_hparam("bottleneck_noise", 0.15)
  hparams.add_hparam("discretize_warmup_steps", 40000)
  hparams.add_hparam("full_latent_tower", False)
  hparams.add_hparam("latent_predictor_state_size", 128)
  hparams.add_hparam("latent_predictor_temperature", 0.5)
  hparams.add_hparam("complex_addn", True)
  return hparams


@registry.register_ranged_hparams
def next_frame_stochastic_discrete_range(rhp):
  """Next frame stochastic discrete tuning grid."""
  rhp.set_float("learning_rate_constant", 0.001, 0.01)
  rhp.set_float("dropout", 0.2, 0.6)
  rhp.set_int("filter_double_steps", 3, 5)
  rhp.set_discrete("hidden_size", [64, 96, 128])
  rhp.set_discrete("bottleneck_bits", [32, 64, 128, 256])
  rhp.set_discrete("video_num_target_frames", [4])
  rhp.set_float("bottleneck_noise", 0.0, 0.2)
