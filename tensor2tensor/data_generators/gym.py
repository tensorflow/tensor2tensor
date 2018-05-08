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
import os

# Dependency imports

import gym

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import video_utils

from tensor2tensor.models.research import autoencoders
from tensor2tensor.models.research import rl
from tensor2tensor.rl import collect
from tensor2tensor.rl.envs import tf_atari_wrappers as atari
from tensor2tensor.rl.envs.utils import batch_env_factory

from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string("agent_policy_path", "", "File with model for agent.")
flags.DEFINE_string("autoencoder_path", "", "File with model for autoencoder.")


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
    eval_metrics = [metrics.Metrics.ACC, metrics.Metrics.ACC_PER_SEQ]
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
  def frame_height(self):
    return self.env.observation_space.shape[0]

  @property
  def frame_width(self):
    return self.env.observation_space.shape[1]

  @property
  def num_rewards(self):
    raise NotImplementedError()

  @property
  def num_steps(self):
    raise NotImplementedError()

  @property
  def total_number_of_frames(self):
    return self.num_steps

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
    next_observation = self.env.reset()
    for _ in range(self.num_steps):
      observation = next_observation
      action = self.get_action(observation)
      next_observation, reward, done, _ = self.env.step(action)
      if done:
        next_observation = self.env.reset()
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
class GymFreewayRandom5k(GymDiscreteProblem):
  """Freeway game, random actions."""

  @property
  def env_name(self):
    return "FreewayDeterministic-v4"

  @property
  def min_reward(self):
    return 0

  @property
  def num_rewards(self):
    return 2

  @property
  def num_steps(self):
    return 5000


@registry.register_problem
class GymFreewayRandom50k(GymFreewayRandom5k):
  """Freeway game, random actions."""

  @property
  def num_steps(self):
    return 50000


class GymDiscreteProblemWithAgent(GymDiscreteProblem):
  """Gym environment with discrete actions and rewards and an agent."""

  def __init__(self, *args, **kwargs):
    super(GymDiscreteProblemWithAgent, self).__init__(*args, **kwargs)
    self._env = None
    self.debug_dump_frames_path = "debug_frames_env"

    # defaults
    self.environment_spec = lambda: gym.make(self.env_name)
    self.in_graph_wrappers = []
    self.collect_hparams = rl.atari_base()
    self.settable_num_steps = 20000
    self.simulated_environment = None
    self.warm_up = 10

  @property
  def num_steps(self):
    return self.settable_num_steps

  def _setup(self):
    in_graph_wrappers = [(atari.MemoryWrapper, {})] + self.in_graph_wrappers
    env_hparams = tf.contrib.training.HParams(
        in_graph_wrappers=in_graph_wrappers,
        simulated_environment=self.simulated_environment)

    generator_batch_env = batch_env_factory(
        self.environment_spec, env_hparams, num_agents=1, xvfb=False)

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      if FLAGS.agent_policy_path:
        policy_lambda = self.collect_hparams.network
      else:
        # When no agent_policy_path is set, just generate random samples.
        policy_lambda = rl.random_policy_fun
      policy_factory = tf.make_template(
          "network",
          functools.partial(policy_lambda, self.environment_spec().action_space,
                            self.collect_hparams),
          create_scope_now_=True,
          unique_name_="network")

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      self.collect_hparams.epoch_length = 10
      _, self.collect_trigger_op = collect.define_collect(
          policy_factory, generator_batch_env, self.collect_hparams,
          eval_phase=False, scope="define_collect")

    self.avilable_data_size_op = atari.MemoryWrapper.singleton.speculum.size()
    self.data_get_op = atari.MemoryWrapper.singleton.speculum.dequeue()

  def restore_networks(self, sess):
    if FLAGS.agent_policy_path:
      model_saver = tf.train.Saver(
          tf.global_variables(".*network_parameters.*"))
      model_saver.restore(sess, FLAGS.agent_policy_path)

  def autoencode(self, image, sess):
    with tf.Graph().as_default():
      hparams = autoencoders.autoencoder_discrete_pong()
      hparams.data_dir = "unused"
      hparams.problem_hparams = self.get_hparams(hparams)
      hparams.problem = self
      model = autoencoders.AutoencoderOrderedDiscrete(
          hparams, tf.estimator.ModeKeys.EVAL)
      img = tf.constant(image)
      img = tf.to_int32(tf.reshape(
          img, [1, 1, self.frame_height, self.frame_width, self.num_channels]))
      encoded = model.encode(img)
      model_saver = tf.train.Saver(tf.global_variables())
      model_saver.restore(sess, FLAGS.autoencoder_path)
      return sess.run(encoded)

  def generate_encoded_samples(self, data_dir, tmp_dir, unused_dataset_split):
    self._setup()
    self.debug_dump_frames_path = os.path.join(
        data_dir, self.debug_dump_frames_path)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      self.restore_networks(sess)
      pieces_generated = 0
      while pieces_generated < self.num_steps + self.warm_up:
        avilable_data_size = sess.run(self.avilable_data_size_op)
        if avilable_data_size < 1:
          sess.run(self.collect_trigger_op)
        observ, reward, action, _, img = sess.run(self.data_get_op)
        if FLAGS.autoencoder_path:
          observ = self.autoencode(img, sess)
        yield {"image/encoded": [observ],
               "image/format": ["png"],
               "image/height": [self.frame_height],
               "image/width": [self.frame_width],
               "action": [int(action)],
               "done": [int(False)],
               "reward": [int(reward) - self.min_reward]}
        pieces_generated += 1


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgent(GymDiscreteProblemWithAgent):
  """Simulated gym environment with discrete actions and rewards."""

  def __init__(self, *args, **kwargs):
    super(GymSimulatedDiscreteProblemWithAgent, self).__init__(*args, **kwargs)
    self.simulated_environment = True
    self.debug_dump_frames_path = "debug_frames_sim"

  def restore_networks(self, sess):
    super(GymSimulatedDiscreteProblemWithAgent, self).restore_networks(sess)
    # TODO(blazej): adjust regexp for different models.
    env_model_loader = tf.train.Saver(tf.global_variables(".*basic_conv_gen.*"))
    sess = tf.get_default_session()

    ckpts = tf.train.get_checkpoint_state(FLAGS.output_dir)
    ckpt = ckpts.model_checkpoint_path
    env_model_loader.restore(sess, ckpt)


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgentOnPong(
    GymSimulatedDiscreteProblemWithAgent, GymPongRandom5k):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnPong(
    GymDiscreteProblemWithAgent, GymPongRandom5k):
  pass


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgentOnFreeway(
    GymSimulatedDiscreteProblemWithAgent, GymFreewayRandom5k):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnFreeway(
    GymDiscreteProblemWithAgent, GymFreewayRandom5k):
  pass
