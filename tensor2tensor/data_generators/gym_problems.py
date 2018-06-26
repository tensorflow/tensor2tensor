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

from tensorflow.contrib.training import HParams


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
    self.debug_dump_frames_path = "debug_frames_env"
    self.settable_num_steps = 5000

    self.environment_spec = self.get_environment_spec()
    self.eval_phase = False

    self._internal_memory_size = 20
    self._internal_memory_force_beginning_resets = False
    self._session = None

  def _setup(self):
    collect_hparams = rl.ppo_pong_base()
    collect_hparams.add_hparam("environment_spec", self.environment_spec)
    collect_hparams.add_hparam("force_beginning_resets",
                               self._internal_memory_force_beginning_resets)
    collect_hparams.epoch_length = self._internal_memory_size
    collect_hparams.num_agents = 1

    if not FLAGS.agent_policy_path:
      collect_hparams.policy_network = rl.random_policy_fun

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      self.collect_memory, self.collect_trigger_op, collect_init \
        = collect.define_collect(collect_hparams, scope="gym_problems",
                                 collect_level=0, eval_phase=self.eval_phase)

    self._session = tf.Session()
    collect_init(self._session)
    self._session.run(tf.global_variables_initializer())

  @property
  def random_skip(self):
    return False

  def generate_samples(self, data_dir, tmp_dir, unused_dataset_split):
    self._setup()
    self.debug_dump_frames_path = os.path.join(
        data_dir, self.debug_dump_frames_path)

    with self._session as sess:
      self.restore_networks(sess)
      pieces_generated = 0
      memory_index = 0
      memory = None
      while pieces_generated < self.num_steps:
        if memory is None or memory_index >= self._internal_memory_size:
          memory = sess.run(self.collect_memory)
          memory_index = 0
        data = [memory[i][memory_index][0] for i in range(4)]
        memory_index += 1
        observation, reward, _, action = data
        observation = observation.astype(np.uint8)

        debug_image = self.collect_statistics_and_generate_debug_image(
            pieces_generated, *data)
        ret_dict = {
            "frame": observation,
            "frame_number": [int(pieces_generated)],
            "image/format": ["png"],
            "image/height": [self.frame_height],
            "image/width": [self.frame_width],
            "action": [int(action)],
            "done": [int(False)],
            "reward": [int(reward) - self.min_reward]
        }

        if debug_image is not None:
          ret_dict["image/debug"] = debug_image

        yield ret_dict
        pieces_generated += 1

  def restore_networks(self, sess):
    if FLAGS.agent_policy_path:
      model_saver = tf.train.Saver(
          tf.global_variables(".*network_parameters.*"))
      ckpts = tf.train.get_checkpoint_state(FLAGS.agent_policy_path)
      ckpt = ckpts.model_checkpoint_path
      model_saver.restore(sess, ckpt)

  def eval_metrics(self):
    eval_metrics = [metrics.Metrics.ACC, metrics.Metrics.ACC_PER_SEQ,
                    metrics.Metrics.IMAGE_RMSE]
    return eval_metrics

  @property
  def extra_reading_spec(self):
    """Additional data fields to store on disk and their decoders."""
    data_fields = {
        "frame_number": tf.FixedLenFeature([1], tf.int64),
        "action": tf.FixedLenFeature([1], tf.int64),
        "reward": tf.FixedLenFeature([1], tf.int64)
    }
    decoders = {
        "frame_number": tf.contrib.slim.tfexample_decoder.Tensor(
            tensor_key="frame_number"),
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

  def collect_statistics_and_generate_debug_image(self, index,
                                                  observation,
                                                  reward,
                                                  done,
                                                  action):
    """This generates extra statistics and debug images."""
    raise NotImplementedError()

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

  @property
  def only_keep_videos_from_0th_frame(self):
    return False

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


class BasicStatistics(object):
  """Keeps basic statistics to calculate mean reward """

  def __init__(self):
    self.sum_of_rewards = 0.0
    self.number_of_dones = 0


class GymRealDiscreteProblem(GymDiscreteProblem):
  """Discrete problem."""

  def __init__(self, *args, **kwargs):
    super(GymRealDiscreteProblem, self).__init__(*args, **kwargs)
    self.statistics = BasicStatistics()

    self.make_extra_debug_info = False

  def collect_statistics_and_generate_debug_image(self, index, observation,
                                                  reward, done, action):
    """Collects info required to calculate mean reward."""

    self.statistics.sum_of_rewards += reward
    self.statistics.number_of_dones += int(done)

    debug_image = None

    return debug_image


class RewardPerSequenceStatistics(BasicStatistics):
  """This encapsulates all pieces required to calculate
  the correctness of rewards per sequence metric
  """

  def __init__(self):
    super(RewardPerSequenceStatistics, self).__init__()

    # data to calculate
    # correctness of rewards per sequence metric
    self.episode_sim_reward = 0.0
    self.episode_real_reward = 0.0,
    self.successful_episode_reward_predictions = 0
    self.report_reward_statistics_every = 10

    # auxiliary objects
    self.real_env = None
    self.real_ob = None


class GymSimulatedDiscreteProblem(GymDiscreteProblem):
  """Simulated gym environment with discrete actions and rewards."""

  def __init__(self, *args, **kwargs):
    self.simulated_environment = True
    self.debug_dump_frames_path = "debug_frames_sim"
    self.intrinsic_reward_scale = 0.0
    self.simulation_random_starts = False
    self.statistics = RewardPerSequenceStatistics()
    super(GymSimulatedDiscreteProblem, self).__init__(*args, **kwargs)

    # This is hackish way of introducing resets every
    # self.num_testing_steps. It cannot be done easily
    # using other ways as we do not control
    # the amount of skips induced but wrappers
    self._internal_memory_size = self.num_testing_steps
    self._internal_memory_force_beginning_resets = True
    env_spec = standard_atari_env_spec(self.env_name)
    real_env = env_spec.env_lambda()
    self.statistics.real_env = real_env

  def _setup(self):
    super(GymSimulatedDiscreteProblem, self)._setup()

    environment_spec = self.environment_spec
    hparams = HParams(video_num_input_frames=
                      environment_spec.video_num_input_frames,
                      video_num_target_frames=
                      environment_spec.video_num_target_frames,
                      environment_spec=environment_spec)

    initial_frames_problem = environment_spec.initial_frames_problem
    # initial_frames_problem.random_skip = False
    dataset = initial_frames_problem.dataset(
        tf.estimator.ModeKeys.TRAIN, FLAGS.data_dir,
        shuffle_files=False, hparams=hparams)
    dataset = dataset.map(lambda x: x["input_action"]).take(1)
    input_data_iterator = (
        dataset.batch(1).make_initializable_iterator())
    self._session.run(input_data_iterator.initializer)

    res = self._session.run(input_data_iterator.get_next())
    self._initial_action = res[0, :, 0]
    self._reset_real_env()

  @property
  def initial_frames_problem(self):
    raise NotImplementedError()

  @property
  def num_input_frames(self):
    """Number of frames on input for real environment."""
    # TODO(lukaszkaiser): This must be equal to hparams.video_num_input_frames,
    # we should automate this to avoid bug in the future.
    return 4

  @property
  def video_num_target_frames(self):
    """Number of frames on input for real environment."""
    # TODO(piotrmilos): This must be equal to hparams.video_num_target_frames,
    # we should automate this to avoid bug in the future.
    return 1

  @property
  def num_testing_steps(self):
    return None

  def get_environment_spec(self):
    env_spec = standard_atari_env_spec(self.env_name)
    env_spec.simulated_env = True
    env_spec.add_hparam("simulation_random_starts",
                        self.simulation_random_starts)

    env_spec.add_hparam("intrinsic_reward_scale",
                        self.intrinsic_reward_scale)
    initial_frames_problem = registry.problem(self.initial_frames_problem)
    env_spec.add_hparam("initial_frames_problem", initial_frames_problem)
    env_spec.add_hparam("video_num_input_frames", self.num_input_frames)
    env_spec.add_hparam("video_num_target_frames", self.video_num_target_frames)

    return env_spec

  def _reset_real_env(self):
    stat = self.statistics
    stat.real_env.reset()
    for a in self._initial_action:
      stat.real_ob, _, _, _ = stat.real_env.step(a)

    stat.episode_sim_reward = 0.0
    stat.episode_real_reward = 0.0

  def collect_statistics_and_generate_debug_image(self, index,
                                                  observation,
                                                  reward, done, action):
    stat = self.statistics

    stat.sum_of_rewards += reward
    stat.number_of_dones += int(done)
    stat.episode_sim_reward += reward

    ob = np.ndarray.astype(observation, np.int)
    err = np.ndarray.astype(np.maximum(np.abs(
        stat.real_ob - ob, dtype=np.int) - 10, 0), np.uint8)
    debug_im = np.concatenate([observation, stat.real_ob, err], axis=1)

    assert (self._internal_memory_size == self.num_testing_steps and
            self._internal_memory_force_beginning_resets), (
                "The collect memory should be set in force_beginning_resets "
                "mode for the code below to work properly.")

    if index % self._internal_memory_size == 0:
      if stat.episode_sim_reward == stat.episode_real_reward:
        stat.successful_episode_reward_predictions += 1
      self._reset_real_env()
    else:
      stat.real_ob, real_reward, _, _ = stat.real_env.step(action)
      stat.episode_real_reward += real_reward

    return debug_im

  def restore_networks(self, sess):
    super(GymSimulatedDiscreteProblem, self).restore_networks(sess)
    # TODO(blazej): adjust regexp for different models.
    # TODO(piotrmilos): move restoring networks to SimulatedBatchEnv.initialize
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
  """Simulated pong."""

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_pong"

  @property
  def num_testing_steps(self):
    return 100


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
  """Similated pong."""

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_wrapped_pong"

  @property
  def num_testing_steps(self):
    return 100


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
  """Similated pong."""

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_wrapped_long_pong"

  @property
  def num_testing_steps(self):
    return 100


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
  """Similated breakout."""

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_wrapped_breakout"

  @property
  def num_testing_steps(self):
    return 100


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
  """Similated freeway."""

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_freeway"

  @property
  def num_testing_steps(self):
    return 100


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
