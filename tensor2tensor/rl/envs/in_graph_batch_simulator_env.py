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

"""Batch of environments inside the TensorFlow graph."""

# The code was based on Danijar Hafner's code from tf.agents:
# https://github.com/tensorflow/agents/blob/master/agents/tools/in_graph_batch_env.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensor2tensor.layers import common_layers
from tensor2tensor.models.research.basic_conv_gen import BasicConvGen
from tensor2tensor.utils import t2t_model, trainer_lib
#
#
# class GenModel(t2t_model.T2TModel):
#   def body(self, features):
#     filters = self.hparams.hidden_size
#     # cur_frame = common_layers.standardize_images(features["inputs"])
#     cur_frame = tf.to_float(features["inputs"])
#     # prev_frame =  common_layers.standardize_images(features["inputs_prev"])
#     prev_frame = tf.to_float(features["inputs_prev"])
#     action = common_layers.embedding(tf.to_int64(features["action"]),
#                                      10, filters)
#     action = tf.reshape(action, [-1, 1, 1, filters])
#
#     frames = tf.concat([cur_frame, prev_frame], axis=3)
#     h1 = tf.layers.conv2d(frames, filters, kernel_size=(3, 3), padding="SAME")
#     h2 = tf.layers.conv2d(tf.nn.relu(h1 + action), filters,
#                           kernel_size=(5, 5), padding="SAME")
#     res = tf.layers.conv2d(tf.nn.relu(h2 + action), 3 * 256,
#                            kernel_size=(3, 3), padding="SAME")
#     reward_pred_h1 = tf.reduce_mean(res, axis=[1, 2])
#     reward_pred = tf.layers.dense(reward_pred_h1, 2, name="reward")
#     # print(tf.shape(reward_pred))
#     # print(tf.shape(features["reward"]))
#     # reward_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
#     #   labels=tf.one_hot(tf.to_int32(features["reward"]), 2), logits=reward_pred)
#     # reward_loss = tf.reduce_mean(reward_loss)
#     # print(res.shape)
#     # print(features["targets"].shape)
#     l = tf.shape(res)[1]
#     w = tf.shape(res)[2]
#     res = tf.reshape(res, [-1, l, w, 3, 256])
#     res = tf.argmax(res, axis=-1)
#     return res #, {"reward": reward_loss}

class InGraphBatchSimulatorEnv(object):
  """Batch of environments inside the TensorFlow graph.

  The batch of environments will be stepped and reset inside of the graph using
  a tf.py_func(). The current batch of observations, actions, rewards, and done
  flags are held in according variables.
  """

  def __init__(self, len, observ_shape, observ_dtype, action_shape, action_dtype):
    """Batch of environments inside the TensorFlow graph.

    Args:
      batch_env: Batch environment.
    """

    self.length = len

    # hparams = trainer_lib.create_hparams("basic_1", data_dir=data_dir)
    hparams = trainer_lib.create_hparams("basic_1")
    hparams.hidden_size = 32
    # trainer_lib.add_problem_hparams(hparams, "env_problem")
    # self._model = GenModel(hparams, tf.estimator.ModeKeys.PREDICT)
    self._model = BasicConvGen(hparams, tf.estimator.ModeKeys.PREDICT)

    self.action_shape = action_shape
    self.action_dtype = action_dtype
    with tf.variable_scope('env_temporary'):
      self._observ = tf.Variable(
          tf.zeros((self.length,) + observ_shape, observ_dtype),
          name='observ', trainable=False)

  def __len__(self):
    """Number of combined environments."""
    return self.length

  def simulate(self, action):

    with tf.name_scope('environment/simulate'):
      action = tf.constant(0.0, tf.float32)
      input = {"inputs": self.observ, "inputs_prev":self.observ, "action": action}
      model_output = self._model(input)
      observ = model_output[0]
      observ = tf.argmax(observ, axis=-1)
      observ = tf.cast(observ, tf.float32)

      # observ = tf.Print(observ, [observ], "observ = ")


      # observ = tf.check_numerics(observ, 'observ')
      reward = tf.constant(0.0, tf.float32, shape=(self.length,))
      done = tf.constant(False, tf.bool, shape=(self.length,))


      with tf.control_dependencies([self._observ.assign(observ)]):
        return tf.identity(reward), tf.identity(done)

  def reset(self, indices=None):
    # return tf.cond(
    #     tf.cast(tf.shape(indices)[0], tf.bool),
    #     lambda: self._reset_non_empty(indices), lambda: 0.0)
    return tf.no_op("reset_to_be")


  @property
  def observ(self):
    """Access the variable holding the current observation."""
    return self._observ
