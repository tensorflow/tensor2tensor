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

from collections import deque

import functools
# Dependency imports
import gym

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import video_utils

from tensor2tensor.models.research import rl
from tensor2tensor.rl import collect
from tensor2tensor.rl.envs import tf_atari_wrappers as atari
from tensor2tensor.rl.envs.utils import batch_env_factory

from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("agent_policy_path", "", "File with model for pong")


class GymDiscreteProblem(video_utils.VideoProblem):
  """Gym environment with discrete actions and rewards."""

  def __init__(self, *args, **kwargs):
    super(GymDiscreteProblem, self).__init__(*args, **kwargs)
    self._env = None

  @property
  def num_input_frames(self):
    """Number of frames to batch on one input."""
    return 2

  @property
  def num_target_frames(self):
    """Number of frames to batch on one target."""
    return 1

  def eval_metrics(self):
    eval_metrics = [
        metrics.Metrics.ACC, metrics.Metrics.ACC_PER_SEQ,
        metrics.Metrics.NEG_LOG_PERPLEXITY]
    return eval_metrics

  @property
  def extra_reading_spec(self):
    """Additional data fields to store on disk and their decoders."""
    data_fields = {
        "action": tf.FixedLenFeature([1], tf.int64),
        "reward": tf.FixedLenFeature([1], tf.int64)
    }
    decoders = {
        "action": tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="action"),
        "reward": tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="reward"),
    }
    return data_fields, decoders

  @property
  def is_generate_per_split(self):
    """Whether we have a train/test split or just hold out data."""
    return False  # Just hold out some generated data for evals.

  @property
  def env_name(self):
    """This is the name of the Gym environment for this problem."""
    raise NotImplementedError()

  @property
  def env(self):
    if self._env is None:
      self._env = gym.make(self.env_name)
    return self._env

  @property
  def num_actions(self):
    return self.env.action_space.n

  @property
  def num_rewards(self):
    raise NotImplementedError()

  @property
  def num_steps(self):
    raise NotImplementedError()

  @property
  def min_reward(self):
    raise NotImplementedError()

  def get_action(self, observation=None):
    return self.env.action_space.sample()

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.input_modality = {"inputs": ("video", 256),
                        "input_reward": ("symbol", self.num_rewards),
                        "input_action": ("symbol", self.num_actions)}
    p.target_modality = {"targets": ("video", 256),
                         "target_reward": ("symbol", self.num_rewards)}
    p.input_space_id = problem.SpaceID.IMAGE
    p.target_space_id = problem.SpaceID.IMAGE

  def generate_samples(self, data_dir, tmp_dir, unused_dataset_split):
    self.env.reset()
    action = self.get_action()
    for _ in range(self.num_steps):
      observation, reward, done, _ = self.env.step(action)
      action = self.get_action(observation)
      yield {"frame": observation,
             "action": [action],
             "done": [done],
             "reward": [int(reward - self.min_reward)]}


@registry.register_problem
class GymPongRandom5k(GymDiscreteProblem):
  """Pong game, random actions."""

  @property
  def env_name(self):
    return "PongDeterministic-v4"

  @property
  def frame_height(self):
    return 210

  @property
  def frame_width(self):
    return 160

  @property
  def min_reward(self):
    return -1

  @property
  def num_rewards(self):
    return 3

  @property
  def num_steps(self):
    return 5000


@registry.register_problem
class GymPongRandom50k(GymPongRandom5k):
  """Pong game, random actions."""

  @property
  def num_steps(self):
    return 50000


@registry.register_problem
class GymDiscreteProblemWithAgent(GymPongRandom5k):
  """Gym environment with discrete actions and rewards and an agent."""

  def __init__(self, *args, **kwargs):
    super(GymDiscreteProblemWithAgent, self).__init__(*args, **kwargs)
    self._env = None
    self.history_size = 2

    # defaults
    self.environment_spec = lambda: gym.make("PongDeterministic-v4")
    self.in_graph_wrappers = [(atari.MaxAndSkipWrapper, {"skip": 4})]
    self.collect_hparams = rl.atari_base()
    self.settable_num_steps = 1000
    self.simulated_environment = None
    self.warm_up = 70

  @property
  def num_steps(self):
    return self.settable_num_steps

  def _setup(self):
    in_graph_wrappers = [(atari.ShiftRewardWrapper, {"add_value": 2}),
                         (atari.MemoryWrapper, {})] + self.in_graph_wrappers
    env_hparams = tf.contrib.training.HParams(
        in_graph_wrappers=in_graph_wrappers,
        simulated_environment=self.simulated_environment)

    generator_batch_env = batch_env_factory(
        self.environment_spec, env_hparams, num_agents=1, xvfb=False)

    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
      policy_lambda = self.collect_hparams.network
      policy_factory = tf.make_template(
          "network",
          functools.partial(policy_lambda, self.environment_spec().action_space,
                            self.collect_hparams),
          create_scope_now_=True,
          unique_name_="network")

    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
      sample_policy = lambda policy: 0 * policy.sample()

      self.collect_hparams.epoch_length = 10
      _, self.collect_trigger_op = collect.define_collect(
          policy_factory, generator_batch_env, self.collect_hparams,
          eval_phase=False, policy_to_actions_lambda=sample_policy,
          scope="define_collect")

    self.avilable_data_size_op = atari.MemoryWrapper.singleton.speculum.size()
    self.data_get_op = atari.MemoryWrapper.singleton.speculum.dequeue()
    self.history_buffer = deque(maxlen=self.history_size+1)

  def restore_networks(self, sess):
    model_saver = tf.train.Saver(
        tf.global_variables(".*network_parameters.*"))
    if FLAGS.agent_policy_path:
      model_saver.restore(sess, FLAGS.agent_policy_path)

  def generate_encoded_samples(self, data_dir, tmp_dir, unused_dataset_split):
    self._setup()

    # When no agent_policy_path is set, just generate random samples.
    if not FLAGS.agent_policy_path:
      for sample in super(GymDiscreteProblemWithAgent,
                          self).generate_encoded_samples(
                              data_dir, tmp_dir, unused_dataset_split):
        yield sample
      return

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      self.restore_networks(sess)

      pieces_generated = 0
      while pieces_generated < self.num_steps + self.warm_up:
        avilable_data_size = sess.run(self.avilable_data_size_op)
        if avilable_data_size > 0:
          observ, reward, action, _ = sess.run(self.data_get_op)
          self.history_buffer.append(observ)

          if len(self.history_buffer) == self.history_size + 1:
            pieces_generated += 1
            ret_dict = {"image/encoded": [observ],
                        "image/format": ["png"],
                        "image/height": [self.frame_height],
                        "image/width": [self.frame_width],
                        "action": [int(action)],
                        "done": [int(False)],
                        "reward": [int(reward) - self.min_reward]}
            if pieces_generated > self.warm_up:
              yield ret_dict
        else:
          sess.run(self.collect_trigger_op)


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgent(GymDiscreteProblemWithAgent):
  """Simulated gym environment with discrete actions and rewards."""

  def __init__(self, *args, **kwargs):
    super(GymSimulatedDiscreteProblemWithAgent, self).__init__(*args, **kwargs)
    self.simulated_environment = True
    self.debug_dump_frames_path = "/tmp/t2t_debug_dump_frames"

  def restore_networks(self, sess):
    super(GymSimulatedDiscreteProblemWithAgent, self).restore_networks(sess)
    # TODO(blazej): adjust regexp for different models.
    env_model_loader = tf.train.Saver(tf.global_variables(".*basic_conv_gen.*"))
    sess = tf.get_default_session()

    ckpts = tf.train.get_checkpoint_state(FLAGS.output_dir)
    ckpt = ckpts.model_checkpoint_path
    env_model_loader.restore(sess, ckpt)
