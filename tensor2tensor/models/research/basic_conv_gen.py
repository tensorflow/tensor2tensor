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
class BasicConvGen(t2t_model.T2TModel):

  def body(self, features):
    hparams = self.hparams
    filters = hparams.hidden_size
    kernel1, kernel2 = (3, 3), (4, 4)

    # Concat frames and down-stride.
    cur_frame = tf.to_float(features["inputs"])
    prev_frame = tf.to_float(features["inputs_prev"])
    x = tf.concat([cur_frame, prev_frame], axis=-1)
    for _ in range(hparams.num_compress_steps):
      x = tf.layers.conv2d(x, filters, kernel2, activation=common_layers.belu,
                           strides=(2, 2), padding="SAME")
      x = common_layers.layer_norm(x)
      filters *= 2
    # Add embedded action.
    action = tf.reshape(features["action"], [-1, 1, 1, hparams.hidden_size])
    zeros = tf.zeros(common_layers.shape_list(x)[:-1] + [hparams.hidden_size])
    x = tf.concat([x, action + zeros], axis=-1)

    # Run a stack of convolutions.
    for i in range(hparams.num_hidden_layers):
      with tf.variable_scope("layer%d" % i):
        y = tf.layers.conv2d(x, filters, kernel1, activation=common_layers.belu,
                             strides=(1, 1), padding="SAME")
        if i == 0:
          x = y
        else:
          x = common_layers.layer_norm(x + y)
    # Up-convolve.
    for _ in range(hparams.num_compress_steps):
      filters //= 2
      x = tf.layers.conv2d_transpose(
          x, filters, kernel2, activation=common_layers.belu,
          strides=(2, 2), padding="SAME")
      x = common_layers.layer_norm(x)

    # Reward prediction.
    reward_pred_h1 = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
    # Rewards are {-1, 0, 1} so we add 1 to the raw gold ones, predict 3.
    reward_pred = tf.layers.dense(reward_pred_h1, 3, name="reward")
    reward_gold = tf.expand_dims(tf.to_int32(features["reward_raw"]) + 1, 1)
    reward_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=reward_gold, logits=reward_pred, name="reward_loss")
    reward_loss = tf.reduce_mean(reward_loss)
    return x, {"reward": reward_loss}


@registry.register_hparams
def basic_conv():
  """Basic 2-frame conv model."""
  hparams = common_hparams.basic_params1()
  hparams.hidden_size = 64
  hparams.batch_size = 8
  hparams.num_hidden_layers = 3
  hparams.optimizer = "Adam"
  hparams.learning_rate_constant = 0.0002
  hparams.learning_rate_warmup_steps = 500
  hparams.learning_rate_schedule = "constant * linear_warmup"
  hparams.label_smoothing = 0.05
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 0.0
  hparams.add_hparam("num_compress_steps", 2)
  return hparams


@registry.register_hparams
def basic_conv_small():
  """Small conv model."""
  hparams = common_hparams.basic_params1()
  hparams.kernel_sizes = [(3,3), (5,5)]
  hparams.filter_numbers = [32, 3*256]
  hparams.batch_size = 2
  hparams.add_hparam("per_image_standardization", False)
  hparams.hidden_size = 32
  return hparams

@registry.register_hparams
def basic_conv_small_per_image_standardization():
  """Small conv model."""
  hparams = common_hparams.basic_params1()
  hparams.kernel_sizes = [(3,3), (5,5)]
  hparams.filter_numbers = [32, 3*256]
  hparams.batch_size = 2
  hparams.add_hparam("per_image_standardization", True)

  return hparams


@registry.register_hparams
def basic_conv_small_small_lr():
  """Small conv model."""
  hparams = common_hparams.basic_params1()
  hparams.batch_size = 2

  hparams.learning_rate = 0.0001
  return hparams


@registry.register_model
class StaticBasicConvGen(t2t_model.T2TModel):

  def body(self, features):
    filters = self.hparams.hidden_size
    cur_frame = features["inputs_0"]
    prev_frame = features["inputs_1"]
    if self.hparams.per_image_standardization:
      cur_frame = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), cur_frame)
      prev_frame = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), prev_frame)

    action = common_layers.embedding(tf.to_int64(features["action"]),
                                     10, filters)
    action = tf.reshape(action, [-1, 1, 1, filters])

    frames = tf.concat([cur_frame, prev_frame], axis=3)
    h1 = tf.layers.conv2d(frames, filters, kernel_size=(3, 3), padding="SAME")
    h2 = tf.layers.conv2d(tf.nn.relu(h1 + action), filters,
                          kernel_size=(5, 5), padding="SAME")
    res = tf.layers.conv2d(tf.nn.relu(h2 + action), 3 * 256,
                           kernel_size=(3, 3), padding="SAME")
    reward_pred_h1 = tf.reduce_mean(res, axis=[1, 2])
    reward_pred = tf.layers.dense(reward_pred_h1, 2, name="reward")
    # reward_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #   labels=tf.to_int32(features["reward"]), logits=reward_pred)
    # reward_loss = tf.reduce_mean(reward_loss)
    x = tf.layers.flatten(h2)
    # l = tf.shape(res)[1]
    # w = tf.shape(res)[2]
    l = 210
    w = 160
    res = tf.reshape(res, [-1, l, w, 768])
    return {"targets": res, "reward": x}

@registry.register_model
class ResidualBasicConvGen(t2t_model.T2TModel):

  def body(self, features):
    filters = 38
    num_hidden_layers = self.hparams.num_hidden_layers
    #TODO: possibly make embeding of inputs_0 and inputs_1
    cur_frame = features["inputs_0"]
    prev_frame = features["inputs_1"]

    if self.hparams.per_image_standardization:
      cur_frame = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), cur_frame)
      prev_frame = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), prev_frame)

    # prev_frame = tf.Print(prev_frame, [prev_frame], "prev frame = ", summarize=200)
    action_embedding_size = 32
    action_space_size = 10
    kernel = (3, 3)
    # Gather all inputs.
    action = common_layers.embedding(tf.to_int64(features["action"]),
                                     action_space_size, action_embedding_size)
    action = tf.reshape(action, [-1, 1, 1, action_embedding_size])
    #broadcast to the shape compatibile with pictures
    action += tf.expand_dims(tf.zeros_like(cur_frame[..., 0]), -1)
    frames = tf.concat([cur_frame, prev_frame, action], axis=3)
    # x = tf.layers.conv2d(frames, filters, kernel, activation=tf.nn.relu,
    #                      strides=(2, 2), padding="SAME")
    # Run a stack of convolutions.
    x = frames
    for _ in range(num_hidden_layers):
      y = tf.layers.conv2d(x, filters, kernel, activation=tf.nn.relu,
                           strides=(1, 1), padding="SAME")
      x = common_layers.layer_norm(x + y)
    # Up-convolve.
    # x = tf.layers.conv2d_transpose(
    #     frames, filters, kernel, activation=tf.nn.relu,
    #     strides=(1, 1), padding="SAME")
    # Output size is 3 * 256 for 3-channel color space.
    res = tf.layers.conv2d(x, 3 * 256, kernel, padding="SAME")
    x = tf.layers.flatten(x)

    # TODO: pm->pm: add done
    res_done = tf.layers.dense(x, 2)

    return {"targets":res, "reward": x}


@registry.register_model
class MichiganBasicConvGen(t2t_model.T2TModel):

  def body(self, features):
    from tensor2tensor.layers.common_layers import shape_list
    def standardize_images(x):
      """Image standardization on batches (tf.image.per_image_standardization)."""
      with tf.name_scope("standardize_images", [x]):
        x = tf.to_float(x)
        x_mean = tf.reduce_mean(x, axis=[1, 2, 3], keep_dims=True)
        x_variance = tf.reduce_mean(
          tf.square(x - x_mean), axis=[1, 2, 3], keep_dims=True)
        x_shape = shape_list(x)
        num_pixels = tf.to_float(x_shape[1] * x_shape[2] * 3)
        x = (x - x_mean) / tf.maximum(tf.sqrt(x_variance), tf.rsqrt(num_pixels))
        # TODO(lukaszkaiser): remove hack below, needed for greedy decoding for now.
        if x.shape and len(x.shape) == 4 and x.shape[3] == 1:
          x = tf.concat([x, x, x], axis=3)  # Not used, just a dead tf.cond branch.
        x.set_shape([None, None, None, 3])
        return x

    def deconv2d(cur, i, kernel_size, output_filters, activation=tf.nn.relu):
      from tensor2tensor.layers.common_layers import conv
      thicker = conv(
        cur,
        output_filters * 4, kernel_size,
        padding="SAME",
        activation=activation,
        name="deconv2d" + str(i))
      return tf.depth_to_space(thicker, 2)

    #
    # cur_frame = features["inputs_0"]
    # prev_frame = features["inputs_1"]

    cur_frame = standardize_images(features["inputs_0"])
    prev_frame = standardize_images(features["inputs_1"])
    # action = common_layers.embedding(tf.to_int64(features["action"]),
    #                                  10, filters)
    # action = tf.reshape(action, [-1, 1, 1, filters])

    frames = tf.concat([cur_frame, prev_frame], axis=3)
    frames = tf.reshape(frames, [-1, 210, 160, 6])
    # frames = tf.Print(frames, [tf.shape(frames)], "frames shape=")

    h1 = tf.layers.conv2d(frames, filters=64, strides=2, kernel_size=(8, 8), padding="SAME", activation=tf.nn.relu)
    h2 = tf.layers.conv2d(h1, filters=128, strides=2, kernel_size=(6, 6), padding="SAME", activation=tf.nn.relu)
    h3 = tf.layers.conv2d(h2, filters=128, strides=2, kernel_size=(6, 6), padding="SAME", activation=tf.nn.relu)
    h4 = tf.layers.conv2d(h3, filters=128, strides=2, kernel_size=(4, 4), padding="SAME", activation=tf.nn.relu)
    h45 = tf.reshape(h4, [-1, 14 * 10 * 128])
    h5 = tf.layers.dense(h45, 2048, activation=tf.nn.relu)
    h6 = tf.layers.dense(h5, 2048, activation=tf.nn.relu)
    h7 = tf.layers.dense(h6, 14 * 10 * 128, activation=tf.nn.relu)
    h8 = tf.reshape(h7, [-1, 14, 10, 128])

    h9 = deconv2d(h8, 1, (4, 4), 128, activation=tf.nn.relu)
    h9 = h9[:, :27, :, :]
    h10 = deconv2d(h9, 2, (6, 6), 128, activation=tf.nn.relu)
    h10 = h10[:, :53, :, :]
    h11 = deconv2d(h10, 3, (6, 6), 128, activation=tf.nn.relu)
    h11 = h11[:, :105, :, :]
    h12 = deconv2d(h11, 4, (8, 8), 3 * 256, activation=tf.identity)

    reward = tf.layers.flatten(h12)

    return {"targets": h12, "reward": reward}
