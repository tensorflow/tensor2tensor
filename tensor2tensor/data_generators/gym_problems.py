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
from tensor2tensor.models.research import rl
from tensor2tensor.rl import collect
from tensor2tensor.rl.envs import tf_atari_wrappers
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


def standard_atari_env_spec(env):
  """Parameters of environment specification."""
  standard_wrappers = [[tf_atari_wrappers.MaxAndSkipWrapper, {"skip": 4}]]
  env_lambda = None
  if isinstance(env, str):
    env_lambda = lambda: gym.make(env)
  if callable(env):
    env_lambda = env
  assert env is not None, "Unknown specification of environment"

  return tf.contrib.training.HParams(env_lambda=env_lambda,
                                     wrappers=standard_wrappers,
                                     simulated_env=False)


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
    self.sum_of_rewards = 0.0
    self.dones = 0

  def _setup(self):
    collect_hparams = rl.ppo_pong_base()
    collect_hparams.add_hparam("environment_spec", self.environment_spec)

    if not FLAGS.agent_policy_path:
      collect_hparams.policy_network = rl.random_policy_fun

    self._internal_memory_size = 10
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      collect_hparams.epoch_length = self._internal_memory_size
      # TODO(piotrmilos). it is possible to set more than 1.
      collect_hparams.num_agents = 1
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
        if memory is None or memory_index >= self._internal_memory_size:
          sess.run(self.collect_trigger_op)
          memory = sess.run(self.collect_memory)
          memory_index = 0
        data = [memory[i][memory_index][0] for i in range(4)]
        memory_index += 1
        observ, reward, done, action = data
        observ = observ.astype(np.uint8)

        self.sum_of_rewards += reward
        self.dones += int(done)

        ret_dict = {"frame": observ,
                    "image/format": ["png"],
                    "image/height": [self.frame_height],
                    "image/width": [self.frame_width],
                    "action": [int(action)],
                    "done": [int(False)],
                    "reward": [int(reward) - self.min_reward]}

        yield ret_dict
        pieces_generated += 1

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
    return standard_atari_env_spec(self.env_name)

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


class GymAEDiscreteProblem(GymDiscreteProblem):
  pass


class GymRealDiscreteProblem(GymDiscreteProblem):

  def __init__(self, *args, **kwargs):
    super(GymRealDiscreteProblem, self).__init__(*args, **kwargs)
    self.make_extra_debug_info = False


class GymSimulatedDiscreteProblem(GymDiscreteProblem):
  """Simulated gym environment with discrete actions and rewards."""

  def __init__(self, *args, **kwargs):
    self.simulated_environment = True
    self.debug_dump_frames_path = "debug_frames_sim"
    self.intrinsic_reward_scale = 0.0
    self.simulation_random_starts = False
    super(GymSimulatedDiscreteProblem, self).__init__(*args, **kwargs)

  @property
  def initial_frames_problem(self):
    raise NotImplementedError()

  def get_environment_spec(self):
    env_spec = standard_atari_env_spec(self.env_name)

    # Set reasonable time limit (as we do not simulate done).
    real_env = env_spec.env_lambda()
    if self.num_testing_steps is not None:
      timelimit = self.num_testing_steps
    else:
      try:
        # We assume that the real env is wrapped with TimeLimit.
        history = self.num_input_frames
        timelimit = real_env._max_episode_steps - history  # pylint: disable=protected-access
      except:  # pylint: disable=bare-except
        # If not, set some reasonable default.
        timelimit = 100

    env_spec.simulated_env = True
    env_spec.add_hparam("simulation_random_starts",
                        self.simulation_random_starts)
    env_spec.add_hparam("intrinsic_reward_scale",
                        self.intrinsic_reward_scale)
    initial_frames_problem = registry.problem(self.initial_frames_problem)
    env_spec.add_hparam("initial_frames_problem", initial_frames_problem)
    env_spec.wrappers.append(
        [tf_atari_wrappers.TimeLimitWrapper, {"timelimit": timelimit}])

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
class GymSimulatedDiscreteProblemWithAgentOnPong(
    GymSimulatedDiscreteProblem, GymPongRandom):

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_pong"


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


@registry.register_problem
class GymDiscreteProblemWithAgentOnPong(
    GymRealDiscreteProblem, GymPongRandom):
  pass


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgentOnWrappedPong(
    GymSimulatedDiscreteProblem, GymWrappedPongRandom):

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_wrapped_pong"


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

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_wrapped_long_pong"


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

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_wrapped_breakout"


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

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_freeway"


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
