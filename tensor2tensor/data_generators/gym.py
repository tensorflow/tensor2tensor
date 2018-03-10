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

"""Data generators for Gym environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# Dependency imports

import gym
import numpy as np

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.models.research import rl
from tensor2tensor.rl.envs import atari_wrappers
from tensor2tensor.utils import registry

import tensorflow as tf




flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "", "File with model for pong")


class GymDiscreteProblem(problem.Problem):
  """Gym environment with discrete actions and rewards."""

  def __init__(self, *args, **kwargs):
    super(GymDiscreteProblem, self).__init__(*args, **kwargs)
    self._env = None

  def example_reading_spec(self, label_repr=None):

    data_fields = {
        "inputs": tf.FixedLenFeature([210, 160, 3], tf.int64),
        "inputs_prev": tf.FixedLenFeature([210, 160, 3], tf.int64),
        "targets": tf.FixedLenFeature([210, 160, 3], tf.int64),
        "action": tf.FixedLenFeature([1], tf.int64)
    }

    return data_fields, None

  @property
  def env_name(self):
    # This is the name of the Gym environment for this problem.
    raise NotImplementedError()

  @property
  def env(self):
    if self._env is None:
      self._env = gym.make(self.env_name)
    return self._env

  @property
  def num_actions(self):
    raise NotImplementedError()

  @property
  def num_rewards(self):
    raise NotImplementedError()

  @property
  def num_steps(self):
    raise NotImplementedError()

  @property
  def num_shards(self):
    return 10

  @property
  def num_dev_shards(self):
    return 1

  def get_action(self, observation=None):
    return self.env.action_space.sample()

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.input_modality = {"inputs": ("image:identity", 256),
                        "inputs_prev": ("image:identity", 256),
                        "reward": ("symbol:identity", self.num_rewards),
                        "action": ("symbol:identity", self.num_actions)}
    p.target_modality = ("image:identity", 256)
    p.input_space_id = problem.SpaceID.IMAGE
    p.target_space_id = problem.SpaceID.IMAGE

  def generator(self, data_dir, tmp_dir):
    self.env.reset()
    action = self.get_action()
    prev_observation, observation = None, None
    for _ in range(self.num_steps):
      prev_prev_observation = prev_observation
      prev_observation = observation
      observation, reward, done, _ = self.env.step(action)
      action = self.get_action(observation)
      if done:
        self.env.reset()
      def flatten(nparray):
        flat1 = [x for sublist in nparray.tolist() for x in sublist]
        return [x for sublist in flat1 for x in sublist]
      if prev_prev_observation is not None:
        yield {"inputs_prev": flatten(prev_prev_observation),
               "inputs": flatten(prev_observation),
               "action": [action],
               "done": [done],
               "reward": [reward],
               "targets": flatten(observation)}

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    train_paths = self.training_filepaths(
        data_dir, self.num_shards, shuffled=False)
    dev_paths = self.dev_filepaths(
        data_dir, self.num_dev_shards, shuffled=False)
    all_paths = train_paths + dev_paths
    generator_utils.generate_files(
        self.generator(data_dir, tmp_dir), all_paths)
    generator_utils.shuffle_dataset(all_paths)


@registry.register_problem
class GymPongRandom5k(GymDiscreteProblem):
  """Pong game, random actions."""

  @property
  def env_name(self):
    return "PongNoFrameskip-v4"

  @property
  def num_actions(self):
    return 4

  @property
  def num_rewards(self):
    return 2

  @property
  def num_steps(self):
    return 5000


@registry.register_problem
class GymPongTrajectoriesFromPolicy(GymDiscreteProblem):
  """Pong game, loaded actions."""

  def __init__(self, *args, **kwargs):
    super(GymPongTrajectoriesFromPolicy, self).__init__(*args, **kwargs)
    self._env = None
    self._last_policy_op = None
    self._max_frame_pl = None
    self._last_action = self.env.action_space.sample()
    self._skip = 4
    self._skip_step = 0
    self._obs_buffer = np.zeros((2,) + self.env.observation_space.shape,
                                dtype=np.uint8)

  def generator(self, data_dir, tmp_dir):
    env_spec = lambda: atari_wrappers.wrap_atari(  # pylint: disable=g-long-lambda
        gym.make("PongNoFrameskip-v4"),
        warp=False,
        frame_skip=4,
        frame_stack=False)
    hparams = rl.atari_base()
    with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
      policy_lambda = hparams.network
      policy_factory = tf.make_template(
          "network",
          functools.partial(policy_lambda, env_spec().action_space, hparams))
      self._max_frame_pl = tf.placeholder(
          tf.float32, self.env.observation_space.shape)
      actor_critic = policy_factory(tf.expand_dims(tf.expand_dims(
          self._max_frame_pl, 0), 0))
      policy = actor_critic.policy
      self._last_policy_op = policy.mode()
      with tf.Session() as sess:
        model_saver = tf.train.Saver(
            tf.global_variables(".*network_parameters.*"))
        model_saver.restore(sess, FLAGS.model_path)
        for item in super(GymPongTrajectoriesFromPolicy,
                          self).generator(data_dir, tmp_dir):
          yield item

  # TODO(blazej0): For training of atari agents wrappers are usually used.
  # Below we have a hacky solution which is a workaround to be used together
  # with atari_wrappers.MaxAndSkipEnv.
  def get_action(self, observation=None):
    if self._skip_step == self._skip - 2: self._obs_buffer[0] = observation
    if self._skip_step == self._skip - 1: self._obs_buffer[1] = observation
    self._skip_step = (self._skip_step + 1) % self._skip
    if self._skip_step == 0:
      max_frame = self._obs_buffer.max(axis=0)
      self._last_action = int(tf.get_default_session().run(
          self._last_policy_op,
          feed_dict={self._max_frame_pl: max_frame})[0, 0])
    return self._last_action

  @property
  def env_name(self):
    return "PongNoFrameskip-v4"

  @property
  def num_actions(self):
    return 4

  @property
  def num_rewards(self):
    return 2

  @property
  def num_steps(self):
    return 5000
