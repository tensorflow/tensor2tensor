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
from tensor2tensor.rl.envs.in_graph_batch_env import InGraphBatchEnv
from tensor2tensor.utils import t2t_model, trainer_lib


class SimulatedBatchEnv(InGraphBatchEnv):
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


    hparams = trainer_lib.create_hparams("basic_1", problem_name="gym_discrete_problem",
                                         data_dir="/home/piotr.milos/trash/loop_0309/data/0")

    hparams.hidden_size = 32
    from tensor2tensor.utils import registry

    hparams.force_full_predict = True
    self._model = registry.model("basic_conv_gen")(hparams, tf.estimator.ModeKeys.PREDICT)
    self.action_shape = action_shape
    self.action_dtype = action_dtype

    shape = (self.length,) + observ_shape
    self._observ = tf.Variable(tf.zeros(shape, observ_dtype), trainable=False)
    self._prev_observ = tf.Variable(tf.zeros(shape, observ_dtype), trainable=False)
    self._starting_observ = tf.Variable(tf.zeros(shape, observ_dtype), trainable=False)

    observ_dtype = tf.int64
    self._observ_not_sure_why_we_need_this = tf.Variable(
        tf.zeros((self.length,) + observ_shape, observ_dtype),
        name='observ_new', trainable=False)

    self._reward_not_sure_why_we_need_this = tf.Variable(tf.zeros((self.length,1), observ_dtype),
                                                         name='reward_new', trainable=False)


  @property
  def action_space(self):
    import gym
    return gym.make("PongNoFrameskip-v4").action_space

  def __len__(self):
    """Number of combined environments."""
    return self.length

  def simulate(self, action):

    with tf.name_scope('environment/simulate'):
      input = {"inputs_0": self._prev_observ, "inputs_1": self.observ,
               "action": action,
               "targets": self._observ_not_sure_why_we_need_this,
               "reward": self._reward_not_sure_why_we_need_this}
      model_output = self._model(input)
      observ_expaned = model_output[0]['targets']
      reward_expanded = model_output[0]['reward']
      observ = tf.cast(tf.argmax(observ_expaned, axis=-1), tf.float32)
      reward = tf.squeeze(tf.cast(tf.argmax(reward_expanded, axis=-1), tf.float32))

      done = tf.constant(False, tf.bool, shape=(self.length,))

      #TODO: move this ugly code bottom of basic_conv_gen.
      with tf.control_dependencies([self._prev_observ.assign(self.observ)]):
        with tf.control_dependencies([self._observ.assign(observ)]):
          return tf.identity(reward), tf.identity(done)

  def reset(self, indices=None):
    return tf.no_op("reset_to_be")
    # #TODO: pm->Błażej. Starting observations
    # with tf.control_dependencies([self._observ.assign(self._starting_observ)]):



  @property
  def observ(self):
    """Access the variable holding the current observation."""
    return self._observ
