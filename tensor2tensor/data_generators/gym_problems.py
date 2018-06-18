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

import math
import os
import gym
import numpy as np

# We need gym_utils for the game environments defined there.
from tensor2tensor.data_generators import gym_utils  # pylint: disable=unused-import
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import video_utils
from tensor2tensor.layers import discretization
from tensor2tensor.models.research import autoencoders
from tensor2tensor.models.research import rl
from tensor2tensor.models.research.rl import standard_atari_env_spec
from tensor2tensor.rl import collect
from tensor2tensor.rl.envs import tf_atari_wrappers as atari
from tensor2tensor.rl.envs.tf_atari_wrappers import StackAndSkipWrapper
from tensor2tensor.rl.envs.tf_atari_wrappers import TimeLimitWrapper
from tensor2tensor.rl.envs.utils import batch_env_factory


from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string("agent_policy_path", None, "File with model for agent.")
flags.DEFINE_string("autoencoder_path", None,
                    "File with model for autoencoder.")
flags.DEFINE_boolean("only_use_ae_for_policy", False,
                     "Whether to only use the autoencoder for the policy and "
                     "still write out full-resolution frames.")


class GymDiscreteProblem(video_utils.VideoProblem):
  """Gym environment with discrete actions and rewards."""

  def __init__(self, *args, **kwargs):
    super(GymDiscreteProblem, self).__init__(*args, **kwargs)
    self._env = None
    self._env = None
    self.debug_dump_frames_path = "debug_frames_env"
    self.settable_num_steps = 5000

    self.environment_spec = self.get_environment_spec()
    self.eval_phase = False

    # Debug info
    self.make_extra_debug_info = True
    self.dones = 0
    self.real_reward = 0
    self.total_sim_reward = 0.0
    self.total_real_reward = 0.0
    self.sum_of_rewards = 0.0
    self.successful_episode_reward_predictions = 0
    self.report_reward_statistics_every = 10

  def _setup(self):
    collect_hparams = rl.ppo_pong_base()
    collect_hparams.add_hparam("environment_spec", self.environment_spec)

    if not FLAGS.agent_policy_path:
      collect_hparams.policy_network = rl.random_policy_fun

    self._internal_memory_size = 10
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      collect_hparams.epoch_length = self._internal_memory_size
      collect_hparams.num_agents = 1 #TODO (piotrmilos). it is possible to set more
      self.collect_memory, self.collect_trigger_op \
        = collect.define_collect(collect_hparams, scope="gym_problems",
                                 collect_level=0, eval_phase=self.eval_phase)

  def generate_samples(self, data_dir, tmp_dir, unused_dataset_split):
    self._setup()
    self.debug_dump_frames_path = os.path.join(
        data_dir, self.debug_dump_frames_path)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      self.restore_networks(sess)
      pieces_generated = 0
      memory_index = 0
      memory = None
      while pieces_generated < self.num_steps:
        if memory is None or memory_index>=self._internal_memory_size:
          sess.run(self.collect_trigger_op)
          memory = sess.run(self.collect_memory)
          memory_index = 0
        data = [memory[i][memory_index][0] for i in range(4)]
        memory_index += 1
        observ, reward, done, action = data
        observ = observ.astype(np.uint8) # TODO(piotrmilos). This should be probably done in collect
        debug_im = None
        if self.make_extra_debug_info:
          debug_im = self.get_debug_image(data)

        ret_dict = {"frame": observ,
                    "image/format": ["png"],
                    "image/height": [self.frame_height],
                    "image/width": [self.frame_width],
                    "action": [int(action)],
                    "done": [int(False)],
                    "reward": [int(reward) - self.min_reward]}
        if self.make_extra_debug_info:
          ret_dict["image/debug"] = debug_im
        yield ret_dict
        pieces_generated += 1

  def get_debug_image(self, data):
    raise NotImplemented()

  def restore_networks(self, sess):
    if FLAGS.agent_policy_path:
      model_saver = tf.train.Saver(
          tf.global_variables(".*network_parameters.*"))
      ckpts = tf.train.get_checkpoint_state(FLAGS.agent_policy_path)
      ckpt = ckpts.model_checkpoint_path
      model_saver.restore(sess, ckpt)


  @property
  def num_input_frames(self):
    """Number of frames on input for real environment."""
    # TODO(lukaszkaiser): This must be equal to hparams.video_num_input_frames,
    # we should automate this to avoid bug in the future.
    return 4

  def eval_metrics(self):
    eval_metrics = [metrics.Metrics.ACC, metrics.Metrics.ACC_PER_SEQ,
                    metrics.Metrics.IMAGE_RMSE]
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

  def get_environment_spec(self):
    raise NotImplementedError()

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
    return self.settable_num_steps

  @property
  def total_number_of_frames(self):
    return self.num_steps

  @property
  def min_reward(self):
    raise NotImplementedError()

  @property
  def num_testing_steps(self):
    return None

  def get_action(self, observation=None):
    del observation
    return self.env.action_space.sample()

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.input_modality = {
        "inputs": ("video", 256),
        "input_reward": ("symbol:weights_all", self.num_rewards),
        "input_action": ("symbol:weights_all", self.num_actions)
    }
    p.target_modality = {
        "targets": ("video", 256),
        "target_reward": ("symbol:weights_all", self.num_rewards),
        "target_action": ("symbol:weights_all", self.num_actions)
    }
    p.input_space_id = problem.SpaceID.IMAGE
    p.target_space_id = problem.SpaceID.IMAGE


@registry.register_problem
class GymPongRandom(GymDiscreteProblem):
  """Pong game, random actions."""

  # Hard-coding num_actions, frame_height, frame_width to avoid loading
  # libale.so file.
  @property
  def num_actions(self):
    return 6

  @property
  def frame_height(self):
    return 210

  @property
  def frame_width(self):
    return 160

  @property
  def env_name(self):
    return "PongDeterministic-v4"

  @property
  def min_reward(self):
    return -1

  @property
  def num_rewards(self):
    return 3


@registry.register_problem
class GymWrappedPongRandom(GymDiscreteProblem):
  """Pong game, random actions."""

  @property
  def env_name(self):
    return "T2TPongWarmUp20RewSkip200Steps-v1"

  @property
  def min_reward(self):
    return -1

  @property
  def num_rewards(self):
    return 3


@registry.register_problem
class GymWrappedLongPongRandom(GymDiscreteProblem):
  """Pong game, random actions."""

  @property
  def env_name(self):
    return "T2TPongWarmUp20RewSkip2000Steps-v1"

  @property
  def min_reward(self):
    return -1

  @property
  def num_rewards(self):
    return 3

  @property
  def num_testing_steps(self):
    return 100


@registry.register_problem
class GymWrappedBreakoutRandom(GymDiscreteProblem):
  """Pong game, random actions."""

  @property
  def env_name(self):
    return "T2TBreakoutWarmUp20RewSkip500Steps-v1"

  @property
  def min_reward(self):
    return -1

  @property
  def num_rewards(self):
    return 3


@registry.register_problem
class GymFreewayRandom(GymDiscreteProblem):
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


class GymRealDiscreteProblem(GymDiscreteProblem):

  def __init__(self, *args, **kwargs):
    super(GymRealDiscreteProblem, self).__init__(*args, **kwargs)
    self.make_extra_debug_info = False

  def get_debug_image(self):
    #TODO(piotrmilos): possibly change this
    raise NotImplemented()

  def get_environment_spec(self):
    return standard_atari_env_spec(self.env_name)


class GymAEDiscreteProblem(GymDiscreteProblem):
  pass

@registry.register_problem
class GymSimulatedDiscreteProblem(GymDiscreteProblem):
  """Simulated gym environment with discrete actions and rewards."""

  def __init__(self, *args, **kwargs):
    self.simulated_environment = True
    self.make_extra_debug_info = True
    self.debug_dump_frames_path = "debug_frames_sim"
    self.intrinsic_reward_scale = 0.0
    self.simulation_random_starts = True
    super(GymSimulatedDiscreteProblem, self).__init__(*args, **kwargs)

  def _setup(self):
    super(GymSimulatedDiscreteProblem, self)._setup()
    if self.make_extra_debug_info:
      self.report_reward_statistics_every = 10
      self.dones = 0
      self.real_reward = 0
      # Slight weirdness to make sim env and real env aligned
      self.real_env.reset()
      for _ in range(self.num_input_frames):
        self.real_ob, _, _, _ = self.real_env.step(0)
      self.total_sim_reward, self.total_real_reward = 0.0, 0.0
      self.sum_of_rewards = 0.0
      self.successful_episode_reward_predictions = 0

  def get_debug_image(self, data):
    observ, reward, done, action = data
    self.total_sim_reward += reward
    err = np.ndarray.astype(np.maximum(np.abs(
      self.real_ob - observ, dtype=np.int) - 10, 0),
                            np.uint8)
    debug_im = np.concatenate([observ, self.real_ob, err], axis=1)

    if done:
      self.dones += 1
      self.sum_of_rewards += self.real_reward
      if self.total_real_reward == self.total_sim_reward:
        self.successful_episode_reward_predictions += 1

      self.total_real_reward = 0.0
      self.total_sim_reward = 0.0
      self.real_reward = 0
      # Slight weirdness to make sim env and real env aligned
      for _ in range(self.num_input_frames):
        self.real_ob, _, _, _ = self.real_env.step(0)
    else:
      self.real_ob, self.real_reward, _, _ = self.real_env.step(action)
      self.total_real_reward += self.real_reward
      self.sum_of_rewards += self.real_reward

    return debug_im

  @property
  def real_env(self):
    """Lazy caching environment construction."""
    if self._real_env is None:
      self._real_env = self.environment_spec()
      if self.num_testing_steps is not None:
        timelimit = self.num_testing_steps
      else:
        try:
          # We assume that the real env is wrapped with TimeLimit.
          history = self.num_input_frames
          timelimit = self.real_env._max_episode_steps - history  # pylint: disable=protected-access
        except:  # pylint: disable=bare-except
          # If not, set some reasonable default.
          timelimit = 100
      self.in_graph_wrappers.append(
          (TimeLimitWrapper, {"timelimit": timelimit}))
    return self._real_env

  def get_environment_spec(self):
    env_spec = standard_atari_env_spec(self.env_name)
    env_spec.simulated_env = True
    env_spec.add_hparam("simulation_random_starts",
                           self.simulation_random_starts)
    env_spec.add_hparam("intrinsic_reward_scale",
                           self.intrinsic_reward_scale)
    return env_spec

  def restore_networks(self, sess):
    super(GymSimulatedDiscreteProblem, self).restore_networks(sess)
    # TODO(blazej): adjust regexp for different models.
    env_model_loader = tf.train.Saver(tf.global_variables("next_frame*"))
    sess = tf.get_default_session()

    ckpts = tf.train.get_checkpoint_state(FLAGS.output_dir)
    ckpt = ckpts.model_checkpoint_path
    env_model_loader.restore(sess, ckpt)


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgentOnPong(
    GymSimulatedDiscreteProblem, GymPongRandom):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnPong(
    GymRealDiscreteProblem, GymPongRandom):
  pass


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgentOnWrappedPong(
  GymSimulatedDiscreteProblem, GymWrappedPongRandom):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedLongPong(
  GymRealDiscreteProblem, GymWrappedLongPongRandom):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedLongPongAe(  # with autoencoder
    GymDiscreteProblemWithAgentOnWrappedLongPong):
  pass


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgentOnWrappedLongPong(
    GymSimulatedDiscreteProblem, GymWrappedLongPongRandom):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedBreakout(
  GymRealDiscreteProblem, GymWrappedBreakoutRandom):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedBreakoutAe(
    GymDiscreteProblemWithAgentOnWrappedBreakout):
  pass


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgentOnWrappedBreakout(
    GymSimulatedDiscreteProblem, GymWrappedBreakoutRandom):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedPong(
  GymRealDiscreteProblem, GymWrappedPongRandom):
  """GymDiscreteProblemWithAgentOnWrappedPong."""

  # Hard-coding num_actions, frame_height, frame_width to avoid loading
  # libale.so file.
  @property
  def num_actions(self):
    return 6

  @property
  def frame_height(self):
    if not FLAGS.autoencoder_path:
      return 210
    return int(math.ceil(210 / self.autoencoder_factor))

  @property
  def frame_width(self):
    if not FLAGS.autoencoder_path:
      return 160
    return int(math.ceil(160 / self.autoencoder_factor))


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedPongAe(  # With autoencoder.
    GymDiscreteProblemWithAgentOnWrappedPong):
  pass


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgentOnFreeway(
    GymSimulatedDiscreteProblem, GymFreewayRandom):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnFreeway(
  GymRealDiscreteProblem, GymFreewayRandom):
  """Freeway with agent."""

  # Hard-coding num_actions, frame_height, frame_width to avoid loading
  # libale.so file.
  @property
  def num_actions(self):
    return 3

  @property
  def frame_height(self):
    if not FLAGS.autoencoder_path:
      return 210
    return int(math.ceil(210 / self.autoencoder_factor))

  @property
  def frame_width(self):
    if not FLAGS.autoencoder_path:
      return 160
    return int(math.ceil(160 / self.autoencoder_factor))


@registry.register_problem
class GymDiscreteProblemWithAgentOnFreewayAe(  # with autoencoder
    GymDiscreteProblemWithAgentOnFreeway):
  pass
