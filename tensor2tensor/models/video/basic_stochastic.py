# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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
_MAX_BATCH = 128


@registry.register_model
class NextFrameBasicStochastic(
    basic_deterministic.NextFrameBasicDeterministic,
    base_vae.NextFrameBaseVae):
  """Stochastic version of basic next-frame model."""

  def inject_latent(self, layer, inputs, target, action):
    """Inject a VAE-style latent."""
    del action
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

  @property
  def is_recurrent_model(self):
    return True

  def init_internal_states(self):
    if not self.hparams.concat_internal_states:
      return None
    # Hardcoded frame shapes.
    max_batch_size = max(_MAX_BATCH, self.hparams.batch_size)
    shape = [max_batch_size] + self.hparams.problem.frame_shape[:-1] + [
        self.hparams.recurrent_state_size]
    with tf.variable_scope("clean_scope_for_internal_state"):
      v = tf.get_variable("state", shape, trainable=False,
                          initializer=tf.zeros_initializer())
    return [[v]]

  def reset_internal_states_ops(self):
    if not self.hparams.concat_internal_states:
      return [[tf.no_op()]]
    zeros = [[tf.zeros_like(s)] for s in self.internal_states[0]]
    return self.save_internal_states_ops(zeros)

  def load_internal_states_ops(self):
    if not self.hparams.concat_internal_states:
      return [[tf.no_op()]]
    ops = [[s.read_value()] for s in self.internal_states[0]]
    return ops

  def save_internal_states_ops(self, internal_states):
    if not self.hparams.concat_internal_states:
      return [[tf.no_op()]]
    ops = [[tf.assign(x, y)]
           for x, y in zip(self.internal_states[0], internal_states[0])]
    return ops

  def update_internal_states_early(self, internal_states, frames):
    """Update the internal states early in the network in GRU-like way."""
    batch_size = common_layers.shape_list(frames[0])[0]
    internal_state = internal_states[0][0][:batch_size, :, :, :]
    state_activation = tf.concat([internal_state, frames[0]], axis=-1)
    state_gate_candidate = tf.layers.conv2d(
        state_activation, 2 * self.hparams.recurrent_state_size,
        (3, 3), padding="SAME", name="state_conv")
    state_gate, state_candidate = tf.split(state_gate_candidate, 2, axis=-1)
    state_gate = tf.nn.sigmoid(state_gate)
    state_candidate = tf.tanh(state_candidate)
    internal_state = internal_state * state_gate
    internal_state += state_candidate * (1.0 - state_gate)
    max_batch_size = max(_MAX_BATCH, self.hparams.batch_size)
    diff_batch_size = max_batch_size - batch_size
    internal_state = tf.pad(
        internal_state, [[0, diff_batch_size], [0, 0], [0, 0], [0, 0]])
    return [[internal_state]]

  def inject_latent(self, layer, inputs, target, action):
    """Inject a deterministic latent based on the target frame."""
    hparams = self.hparams
    final_filters = common_layers.shape_list(layer)[-1]
    filters = hparams.hidden_size
    kernel = (4, 4)
    layer_shape = common_layers.shape_list(layer)
    activation_fn = common_layers.belu
    if hparams.activation_fn == "relu":
      activation_fn = tf.nn.relu

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

    # Add embedded action if present.
    if action is not None:
      x = common_video.inject_additional_input(
          x, action, "action_enc_latent", hparams.action_injection)

    if hparams.full_latent_tower:
      for i in range(hparams.num_compress_steps):
        with tf.variable_scope("latent_downstride%d" % i):
          x = common_layers.make_even_size(x)
          if i < hparams.filter_double_steps:
            filters *= 2
          x = common_attention.add_timing_signal_nd(x)
          x = tfl.conv2d(x, filters, kernel,
                         activation=activation_fn,
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
      # Mix bits from latent with predicted bits on forward pass as a noise.
      if hparams.latent_rnn_max_sampling > 0.0:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
          bits_pred, _ = discretization.predict_bits_with_lstm(
              layer, hparams.latent_predictor_state_size,
              hparams.bottleneck_bits,
              temperature=hparams.latent_predictor_temperature)
          bits_pred = tf.expand_dims(tf.expand_dims(bits_pred, axis=1), axis=2)
        # Be bits_pred on the forward pass but bits on the backward one.
        bits_pred = bits_clean + tf.stop_gradient(bits_pred - bits_clean)
        # Select which bits to take from pred sampling with bit_p probability.
        which_bit = tf.random_uniform(common_layers.shape_list(bits))
        bit_p = common_layers.inverse_lin_decay(hparams.latent_rnn_warmup_steps)
        bit_p *= hparams.latent_rnn_max_sampling
        bits = tf.where(which_bit < bit_p, bits_pred, bits)

    res = add_bits(layer, bits)
    # During training, sometimes skip the latent to help action-conditioning.
    res_p = common_layers.inverse_lin_decay(hparams.latent_rnn_warmup_steps / 2)
    res_p *= hparams.latent_use_max_probability
    res_rand = tf.random_uniform([layer_shape[0]])
    res = tf.where(res_rand < res_p, res, layer)
    return res, pred_loss


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
  hparams.batch_size = 4
  hparams.video_num_target_frames = 6
  hparams.scheduled_sampling_mode = "prob_inverse_lin"
  hparams.scheduled_sampling_decay_steps = 40000
  hparams.scheduled_sampling_max_prob = 1.0
  hparams.dropout = 0.15
  hparams.filter_double_steps = 3
  hparams.hidden_size = 96
  hparams.learning_rate_constant = 0.002
  hparams.learning_rate_warmup_steps = 2000
  hparams.learning_rate_schedule = "linear_warmup * constant"
  hparams.concat_internal_states = True
  hparams.video_modality_loss_cutoff = 0.03
  hparams.add_hparam("bottleneck_bits", 128)
  hparams.add_hparam("bottleneck_noise", 0.1)
  hparams.add_hparam("discretize_warmup_steps", 40000)
  hparams.add_hparam("latent_rnn_warmup_steps", 40000)
  hparams.add_hparam("latent_rnn_max_sampling", 0.5)
  hparams.add_hparam("latent_use_max_probability", 0.8)
  hparams.add_hparam("full_latent_tower", False)
  hparams.add_hparam("latent_predictor_state_size", 128)
  hparams.add_hparam("latent_predictor_temperature", 1.0)
  hparams.add_hparam("complex_addn", True)
  hparams.add_hparam("recurrent_state_size", 64)
  return hparams


@registry.register_hparams
def next_frame_basic_stochastic_discrete_long():
  """Conv model with stochastic discrete latent, long predictions."""
  hparams = next_frame_basic_stochastic_discrete()
  hparams.batch_size = 2
  hparams.video_num_target_frames = 16
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


@registry.register_ranged_hparams
def next_frame_stochastic_discrete_latent_range(rhp):
  rhp.set_float("latent_rnn_max_sampling", 0.1, 0.9)
  rhp.set_float("latent_predictor_temperature", 0.1, 1.2)
  rhp.set_float("latent_use_max_probability", 0.4, 1.0)
  rhp.set_float("dropout", 0.1, 0.4)
