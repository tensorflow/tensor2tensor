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

import six

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models.video import basic_deterministic_params  # pylint: disable=unused-import
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


tfl = tf.layers
tfcl = tf.contrib.layers


def inject_action(action, x):
  """Inject the action into x."""
  x_shape = common_layers.shape_list(x)
  filters = x_shape[-1]
  action_mask = tf.layers.dense(action, filters, name="action_mask")
  action_add = tf.layers.dense(action, filters, name="action_add")
  x *= tf.nn.sigmoid(action_mask)
  x += action_add
  return x


@registry.register_model
class NextFrameBasicDeterministic(t2t_model.T2TModel):
  """Basic next-frame model, may take actions and predict rewards too."""

  def inject_latent(self, layer, features, filters):
    """Do nothing for deterministic model."""
    del features, filters
    return layer, 0.0

  def body_single(self, features):
    hparams = self.hparams
    filters = hparams.hidden_size
    kernel1, kernel2 = (3, 3), (4, 4)

    # Embed the inputs.
    inputs_shape = common_layers.shape_list(features["inputs"])
    # Using non-zero bias initializer below for edge cases of uniform inputs.
    x = tf.layers.dense(
        features["inputs"], filters, name="inputs_embed",
        bias_initializer=tf.random_normal_initializer(stddev=0.01))
    x = common_attention.add_timing_signal_nd(x)

    # Down-stride.
    layer_inputs = [x]
    for i in range(hparams.num_compress_steps):
      with tf.variable_scope("downstride%d" % i):
        layer_inputs.append(x)
        x = common_layers.make_even_size(x)
        if i < hparams.filter_double_steps:
          filters *= 2
        x = common_attention.add_timing_signal_nd(x)
        x = tf.layers.conv2d(x, filters, kernel2, activation=common_layers.belu,
                             strides=(2, 2), padding="SAME")
        x = common_layers.layer_norm(x)

    # Add embedded action if present.
    if "input_action" in features:
      action = tf.reshape(features["input_action"][:, -1, :],
                          [-1, 1, 1, hparams.hidden_size])
      x = inject_action(action, x)

    x, extra_loss = self.inject_latent(x, features, filters)

    # Run a stack of convolutions.
    for i in range(hparams.num_hidden_layers):
      with tf.variable_scope("layer%d" % i):
        y = tf.nn.dropout(x, 1.0 - hparams.dropout)
        y = tf.layers.conv2d(y, filters, kernel1, activation=common_layers.belu,
                             strides=(1, 1), padding="SAME")
        if i == 0:
          x = y
        else:
          x = common_layers.layer_norm(x + y)

    # Up-convolve.
    layer_inputs = list(reversed(layer_inputs))
    for i in range(hparams.num_compress_steps):
      with tf.variable_scope("upstride%d" % i):
        if "input_action" in features:
          x = inject_action(action, x)
        if i >= hparams.num_compress_steps - hparams.filter_double_steps:
          filters //= 2
        x = tf.layers.conv2d_transpose(
            x, filters, kernel2, activation=common_layers.belu,
            strides=(2, 2), padding="SAME")
        y = layer_inputs[i]
        shape = common_layers.shape_list(y)
        x = x[:, :shape[1], :shape[2], :]
        x = common_layers.layer_norm(x + y)
        x = common_attention.add_timing_signal_nd(x)

    # Cut down to original size.
    x = x[:, :inputs_shape[1], :inputs_shape[2], :]
    x = tf.layers.dense(x, hparams.problem.num_channels * 256, name="logits")

    # Reward prediction if needed.
    if "target_reward" not in features:
      return x
    reward_pred = tf.expand_dims(  # Add a fake channels dim.
        tf.reduce_mean(x, axis=[1, 2], keepdims=True), axis=3)
    return {"targets": x, "target_reward": reward_pred}, extra_loss

  def body(self, features):
    hparams = self.hparams
    is_predicting = hparams.mode == tf.estimator.ModeKeys.PREDICT
    if hparams.video_num_target_frames < 2:
      res = self.body_single(features)
      return res

    # TODO(lukaszkaiser): the split axes and the argmax below heavily depend on
    # using the default (a bit strange) video modality - we should change that.

    # Split inputs and targets into lists.
    input_frames = list(tf.split(
        features["inputs"], hparams.video_num_input_frames, axis=-1))
    target_frames = list(tf.split(
        features["targets"], hparams.video_num_target_frames, axis=-1))
    all_frames = input_frames + target_frames
    if "input_action" in features:
      input_actions = list(tf.split(
          features["input_action"], hparams.video_num_input_frames, axis=1))
      target_actions = list(tf.split(
          features["target_action"], hparams.video_num_target_frames, axis=1))
      all_actions = input_actions + target_actions

    # Run a number of steps.
    res_frames = []
    if "target_reward" in features:
      res_rewards, extra_loss = [], 0.0
    sample_prob = common_layers.inverse_exp_decay(
        hparams.scheduled_sampling_warmup_steps)
    sample_prob *= hparams.scheduled_sampling_prob
    for i in range(hparams.video_num_target_frames):
      cur_frames = all_frames[i:i + hparams.video_num_input_frames]
      features["inputs"] = tf.concat(cur_frames, axis=-1)
      if "input_action" in features:
        cur_actions = all_actions[i:i + hparams.video_num_input_frames]
        features["input_action"] = tf.concat(cur_actions, axis=1)

      # Run model.
      with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
        if "target_reward" not in features:
          res_frames.append(self.body_single(features))
        else:
          res_dict, res_extra_loss = self.body_single(features)
          extra_loss += res_extra_loss
          res_frames.append(res_dict["targets"])
          res_rewards.append(res_dict["target_reward"])

      # When predicting, use the generated frame.
      orig_frame = all_frames[i + hparams.video_num_input_frames]
      shape = common_layers.shape_list(orig_frame)
      sampled_frame = tf.reshape(
          res_frames[-1], shape[:-1] + [hparams.problem.num_channels, 256])
      sampled_frame = tf.to_float(tf.argmax(sampled_frame, axis=-1))
      sampled_frame = common_layers.standardize_images(sampled_frame)
      if is_predicting:
        all_frames[i + hparams.video_num_input_frames] = sampled_frame

      # Scheduled sampling during training.
      if (hparams.scheduled_sampling_prob > 0.0 and self.is_training):
        do_sample = tf.less(tf.random_uniform([shape[0]]), sample_prob)
        sampled_frame = tf.where(do_sample, sampled_frame, orig_frame)
        all_frames[i + hparams.video_num_input_frames] = sampled_frame

    # Concatenate results and return them.
    frames = tf.concat(res_frames, axis=-1)
    if "target_reward" not in features:
      return frames
    rewards = tf.concat(res_rewards, axis=1)
    return {"targets": frames, "target_reward": rewards}, extra_loss

  def infer(self, features, *args, **kwargs):  # pylint: disable=arguments-differ
    """Produce predictions from the model by running it."""
    del args, kwargs
    # Inputs and features preparation needed to handle edge cases.
    if not features:
      features = {}
    inputs_old = None
    if "inputs" in features and len(features["inputs"].shape) < 4:
      inputs_old = features["inputs"]
      features["inputs"] = tf.expand_dims(features["inputs"], 2)

    def logits_to_samples(logits):
      """Get samples from logits."""
      # If the last dimension is 1 then we're using L1/L2 loss.
      if common_layers.shape_list(logits)[-1] == 1:
        return tf.to_int32(tf.squeeze(logits, axis=-1))
      # Argmax in TF doesn't handle more than 5 dimensions yet.
      logits_shape = common_layers.shape_list(logits)
      argmax = tf.argmax(tf.reshape(logits, [-1, logits_shape[-1]]), axis=-1)
      return tf.reshape(argmax, logits_shape[:-1])

    # Get predictions.
    try:
      num_channels = self.hparams.problem.num_channels
    except AttributeError:
      num_channels = 1
    if "inputs" in features:
      inputs_shape = common_layers.shape_list(features["inputs"])
      targets_shape = [inputs_shape[0], self.hparams.video_num_target_frames,
                       inputs_shape[2], inputs_shape[3], num_channels]
    else:
      tf.logging.warn("Guessing targets shape as no inputs are given.")
      targets_shape = [self.hparams.batch_size,
                       self.hparams.video_num_target_frames, 1, 1, num_channels]

    features["targets"] = tf.zeros(targets_shape, dtype=tf.int32)
    if "target_reward" in self.hparams.problem_hparams.target_modality:
      features["target_reward"] = tf.zeros(
          [targets_shape[0], 1, 1], dtype=tf.int32)
    logits, _ = self(features)  # pylint: disable=not-callable
    if isinstance(logits, dict):
      results = {}
      for k, v in six.iteritems(logits):
        results[k] = logits_to_samples(v)
        results["%s_logits" % k] = v
    else:
      results = logits_to_samples(logits)

    # Restore inputs to not confuse Estimator in edge cases.
    if inputs_old is not None:
      features["inputs"] = inputs_old

    # Return results.
    return results
