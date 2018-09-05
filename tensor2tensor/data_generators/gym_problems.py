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

import json
import math
import os
import gym
import numpy as np

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import video_utils
from tensor2tensor.models.research import autoencoders
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


def standard_atari_env_spec(env):
  """Parameters of environment specification."""
  standard_wrappers = [[tf_atari_wrappers.RewardClippingWrapper, {}],
                       [tf_atari_wrappers.StackWrapper, {"history": 4}]]
  env_lambda = None
  if isinstance(env, str):
    env_lambda = lambda: gym.make(env)
  if callable(env):
    env_lambda = env
  assert env_lambda is not None, "Unknown specification of environment"

  return tf.contrib.training.HParams(
      env_lambda=env_lambda, wrappers=standard_wrappers, simulated_env=False)


def standard_atari_ae_env_spec(env):
  """Parameters of environment specification."""
  standard_wrappers = [[tf_atari_wrappers.StackWrapper, {"history": 4}],
                       [tf_atari_wrappers.AutoencoderWrapper, {}]]
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
    # TODO(piotrmilos): Check if self._env is used.
    self._env = None
    self.debug_dump_frames_path = "debug_frames_env"
    self.settable_num_steps = 5000

    self.environment_spec = self.get_environment_spec()
    self.settable_eval_phase = False

    self._internal_memory_size = 20
    self._internal_memory_force_beginning_resets = False
    self._session = None
    self.statistics = BasicStatistics()

  def _setup(self):
    # TODO(piotrmilos):this should be consistent with
    # ppo_params in model_rl_experiment
    collect_hparams = rl.ppo_pong_base()
    collect_hparams.add_hparam("environment_spec", self.environment_spec)
    collect_hparams.add_hparam("force_beginning_resets",
                               self._internal_memory_force_beginning_resets)
    collect_hparams.epoch_length = self._internal_memory_size
    collect_hparams.num_agents = 1

    if not FLAGS.agent_policy_path:
      collect_hparams.policy_network = rl.random_policy_fun

    policy_to_actions_lambda = None
    if self.settable_eval_phase:
      policy_to_actions_lambda = lambda policy: policy.mode()

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      self.collect_memory, self.collect_trigger_op, collect_init = (
          collect.define_collect(
              collect_hparams,
              scope="gym_problems",
              eval_phase=False,
              collect_level=0,
              policy_to_actions_lambda=policy_to_actions_lambda))

    self._session = tf.Session()
    collect_init(self._session)
    self._session.run(tf.global_variables_initializer())
    self.restore_networks(self._session)

  @property
  def random_skip(self):
    return False

  def generate_samples(self, data_dir, tmp_dir, unused_dataset_split):
    self._setup()

    # We only want to save frames for eval and simulated experience, not the
    # frames used for world model training.
    base_dir = os.path.basename(os.path.dirname(data_dir + "/"))
    if (base_dir == "eval" or self.debug_dump_frames_path in [
        "debug_frames_sim_eval", "debug_frames_sim"
    ]):
      self.debug_dump_frames_path = os.path.join(data_dir,
                                                 self.debug_dump_frames_path)
    else:
      # Disable frame saving
      self.debug_dump_frames_path = ""

    with self._session as sess:
      frame_counter = 0
      memory_index = 0
      memory = None
      pieces_generated = 0
      prev_reward = 0
      prev_done = False

      # TODO(piotrmilos): self.settable_eval_phase possibly violates sematics
      # of VideoProblem
      while pieces_generated < self.num_steps or self.settable_eval_phase:
        if memory is None or memory_index >= self._internal_memory_size:
          memory = sess.run(self.collect_memory)
          memory_index = 0
        data = [memory[i][memory_index][0] for i in range(4)]
        memory_index += 1
        observation, reward, done, action = data

        debug_image = self.collect_statistics_and_generate_debug_image(
            pieces_generated, *data)
        ret_dict = {
            "frame": observation,
            "frame_number": [int(frame_counter)],
            "image/format": ["png"],
            "image/height": [self.frame_height],
            "image/width": [self.frame_width],
            "action": [int(action)],
            "done": [int(prev_done)],
            "reward": [int(prev_reward - self.min_reward)]
        }

        if debug_image is not None:
          ret_dict["image/debug"] = debug_image

        yield ret_dict

        if done and self.settable_eval_phase:
          return

        prev_done, prev_reward = done, reward

        pieces_generated += 1
        frame_counter += 1
        if done:
          frame_counter = 0

  def restore_networks(self, sess):
    if FLAGS.agent_policy_path:
      model_saver = tf.train.Saver(
          tf.global_variables(".*network_parameters.*"))
      ckpts = tf.train.get_checkpoint_state(FLAGS.agent_policy_path)
      ckpt = ckpts.model_checkpoint_path
      model_saver.restore(sess, ckpt)

  def eval_metrics(self):
    eval_metrics = [
        metrics.Metrics.ACC, metrics.Metrics.ACC_PER_SEQ,
        metrics.Metrics.IMAGE_RMSE
    ]
    return eval_metrics

  @property
  def extra_reading_spec(self):
    """Additional data fields to store on disk and their decoders."""

    # TODO(piotrmilos): shouldn't done be included here?
    data_fields = {
        "frame_number": tf.FixedLenFeature([1], tf.int64),
        "action": tf.FixedLenFeature([1], tf.int64),
        "reward": tf.FixedLenFeature([1], tf.int64)
    }
    decoders = {
        "frame_number":
            tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="frame_number"),
        "action":
            tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="action"),
        "reward":
            tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="reward"),
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
    # TODO(piotrmilos): possibly remove
    if self._env is None:
      self._env = gym.make(self.env_name)
    return self._env

  @property
  def num_actions(self):
    return self.env.action_space.n

  # pylint: disable=unused-argument
  def collect_statistics_and_generate_debug_image(self, index, observation,
                                                  reward, done, action):
    """This generates extra statistics and debug images."""
    return None
  # pylint: enable=unused-argument

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

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    super(GymDiscreteProblem, self).generate_data(data_dir, tmp_dir, task_id)
    # Save stats to file, or restore if data was already generated.
    stats_file = os.path.join(data_dir,
                              "%s.stats.json" % self.dataset_filename())
    if tf.gfile.Exists(stats_file):
      self.statistics.update_from_file(stats_file)
    else:
      self.statistics.save_to_file(stats_file)


class BasicStatistics(object):
  """Keeps basic statistics to calculate mean reward """

  def __init__(self):
    self.sum_of_rewards = 0.0
    self.number_of_dones = 0
    self.sum_of_rewards_current_episode = 0.0
    self.last_done = False

  def update_from_dict(self, stats_dict):
    keys = set(self.to_dict().keys())
    for k, v in stats_dict.items():
      if k not in keys:
        raise ValueError("Key %s not a property of %s" %
                         (k, type(self).__name__))
      setattr(self, k, v)
    return self

  def to_dict(self):
    # Cast the values to base types as some are numpy types.
    keys_and_types = [
        ("sum_of_rewards", float),
        ("number_of_dones", int),
        ("sum_of_rewards_current_episode", float),
        ("last_done", bool),
    ]
    stats_dict = dict([(k, t(getattr(self, k))) for k, t in keys_and_types])
    return stats_dict

  def save_to_file(self, fname):
    with tf.gfile.Open(fname, "w") as f:
      f.write(json.dumps(self.to_dict()))

  def update_from_file(self, fname):
    with tf.gfile.Open(fname) as f:
      self.update_from_dict(json.loads(f.read()))
      return self


# TODO(piotrmilos): merge with the superclass
class GymRealDiscreteProblem(GymDiscreteProblem):
  """Discrete problem."""

  def __init__(self, *args, **kwargs):
    super(GymRealDiscreteProblem, self).__init__(*args, **kwargs)
    self.statistics = BasicStatistics()

    self.make_extra_debug_info = False

  def collect_statistics_and_generate_debug_image(self, index, observation,
                                                  reward, done, action):
    """Collects info required to calculate mean reward."""

    self.statistics.sum_of_rewards_current_episode += reward
    # we ignore consecutive dones as they are artefacts of skip wrappers
    if done and not self.statistics.last_done:
      self.statistics.number_of_dones += int(done)
      self.statistics.sum_of_rewards += (
          self.statistics.sum_of_rewards_current_episode)
      self.statistics.sum_of_rewards_current_episode = 0.0

    self.statistics.last_done = done

    debug_image = None
    return debug_image


class GymDiscreteProblemWithAutoencoder(GymRealDiscreteProblem):
  """Gym discrete problem with autoencoder."""

  def get_environment_spec(self):
    return standard_atari_ae_env_spec(self.env_name)

  def restore_networks(self, sess):
    super(GymDiscreteProblemWithAutoencoder, self).restore_networks(sess)
    if FLAGS.autoencoder_path:
      autoencoder_saver = tf.train.Saver(
          tf.global_variables("autoencoder.*"))
      ckpts = tf.train.get_checkpoint_state(FLAGS.autoencoder_path)
      ckpt = ckpts.model_checkpoint_path
      autoencoder_saver.restore(sess, ckpt)


class GymDiscreteProblemAutoencoded(GymRealDiscreteProblem):
  """Gym discrete problem with frames already autoencoded."""

  def generate_samples(self, data_dir, tmp_dir, unused_dataset_split):
    raise RuntimeError("GymDiscreteProblemAutoencoded can be used only"
                       " for reading encoded frames")

  def get_environment_spec(self):
    return standard_atari_ae_env_spec(self.env_name)

  @property
  def autoencoder_factor(self):
    """By how much to divide sizes when using autoencoders."""
    hparams = autoencoders.autoencoder_discrete_pong()
    return 2**hparams.num_hidden_layers

  @property
  def frame_height(self):
    height = self.env.observation_space.shape[0]
    ae_height = int(math.ceil(height / self.autoencoder_factor))
    return ae_height

  @property
  def frame_width(self):
    width = self.env.observation_space.shape[1]
    return int(math.ceil(width / self.autoencoder_factor))


class RewardPerSequenceStatistics(BasicStatistics):
  """This encapsulates all pieces required to calculate
  the correctness of rewards per sequence metric
  """

  def __init__(self):
    super(RewardPerSequenceStatistics, self).__init__()

    # data to calculate
    # correctness of rewards per sequence metric
    self.episode_sim_reward = 0.0
    self.episode_real_reward = 0.0
    self.successful_episode_reward_predictions = 0
    self.report_reward_statistics_every = 10

    # auxiliary objects
    self.real_env = None
    self.real_ob = None

  def to_dict(self):
    stats_dict = super(RewardPerSequenceStatistics, self).to_dict()
    keys_and_types = [
        ("episode_sim_reward", float),
        ("episode_real_reward", float),
        ("successful_episode_reward_predictions", int),
        ("report_reward_statistics_every", int),
    ]
    additional = dict([(k, t(getattr(self, k))) for k, t in keys_and_types])
    stats_dict.update(additional)
    return stats_dict


class GymSimulatedDiscreteProblem(GymDiscreteProblem):
  """Simulated gym environment with discrete actions and rewards."""

  def __init__(self, *args, **kwargs):
    super(GymSimulatedDiscreteProblem, self).__init__(*args, **kwargs)
    self.debug_dump_frames_path = "debug_frames_sim"

    # This is hackish way of introducing resets every
    # self.num_testing_steps. It cannot be done easily
    # using other ways as we do not control
    # the amount of skips induced but wrappers
    self._internal_memory_size = self.num_testing_steps
    self._internal_memory_force_beginning_resets = True
    env_spec = standard_atari_env_spec(self.env_name)
    real_env = env_spec.env_lambda()

    self.statistics = RewardPerSequenceStatistics()
    self.statistics.real_env = real_env

  def _setup(self):
    super(GymSimulatedDiscreteProblem, self)._setup()

    environment_spec = self.environment_spec
    hparams = HParams(
        video_num_input_frames=environment_spec.video_num_input_frames,
        video_num_target_frames=environment_spec.video_num_target_frames,
        environment_spec=environment_spec)

    initial_frames_problem = environment_spec.initial_frames_problem
    dataset = initial_frames_problem.dataset(
        tf.estimator.ModeKeys.TRAIN,
        FLAGS.data_dir,
        shuffle_files=False,
        hparams=hparams)
    dataset = dataset.map(lambda x: x["input_action"]).take(1)
    input_data_iterator = (dataset.batch(1).make_initializable_iterator())
    self._session.run(input_data_iterator.initializer)

    res = self._session.run(input_data_iterator.get_next())
    self._initial_actions = res[0, :, 0][:-1]
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
    env_spec.add_hparam("simulation_random_starts", False)
    env_spec.add_hparam("simulation_flip_first_random_for_beginning", False)
    env_spec.add_hparam("intrinsic_reward_scale", 0.0)
    initial_frames_problem = registry.problem(self.initial_frames_problem)
    env_spec.add_hparam("initial_frames_problem", initial_frames_problem)
    env_spec.add_hparam("video_num_input_frames", self.num_input_frames)
    env_spec.add_hparam("video_num_target_frames", self.video_num_target_frames)

    return env_spec

  def _reset_real_env(self):
    stat = self.statistics
    stat.real_env.reset()
    for a in self._initial_actions:
      stat.real_ob, _, _, _ = stat.real_env.step(a)

  def collect_statistics_and_generate_debug_image(self, index,
                                                  observation,
                                                  reward, done, action):
    stat = self.statistics

    # TODO(piotrmilos): possibly make the same behaviour as
    # in the BasicStatistics
    stat.sum_of_rewards += reward
    stat.episode_sim_reward += reward

    ob = np.ndarray.astype(observation, np.int)
    err = np.ndarray.astype(
        np.maximum(np.abs(stat.real_ob - ob, dtype=np.int) - 10, 0), np.uint8)
    debug_im = np.concatenate([observation, stat.real_ob, err], axis=1)

    assert (self._internal_memory_size == self.num_testing_steps and
            self._internal_memory_force_beginning_resets), (
                "The collect memory should be set in force_beginning_resets "
                "mode for the code below to work properly.")

    if (index+1) % self._internal_memory_size == 0:

      if stat.episode_sim_reward == stat.episode_real_reward:
        stat.successful_episode_reward_predictions += 1
        stat.episode_sim_reward = 0.0
        stat.episode_real_reward = 0.0

      stat.number_of_dones += 1
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
    ckpts = tf.train.get_checkpoint_state(FLAGS.output_dir)
    ckpt = ckpts.model_checkpoint_path
    env_model_loader.restore(sess, ckpt)


class GymSimulatedDiscreteProblemAutoencoded(GymSimulatedDiscreteProblem):
  """Gym simulated discrete problem with frames already autoencoded."""

  def get_environment_spec(self):
    env_spec = standard_atari_env_spec(self.env_name)
    env_spec.wrappers = [[tf_atari_wrappers.IntToBitWrapper, {}]]
    env_spec.simulated_env = True
    env_spec.add_hparam("simulation_random_starts", False)

    env_spec.add_hparam("intrinsic_reward_scale", 0.0)
    initial_frames_problem = registry.problem(self.initial_frames_problem)
    env_spec.add_hparam("initial_frames_problem", initial_frames_problem)
    env_spec.add_hparam("video_num_input_frames", self.num_input_frames)
    env_spec.add_hparam("video_num_target_frames", self.video_num_target_frames)

    return env_spec

  @property
  def autoencoder_factor(self):
    """By how much to divide sizes when using autoencoders."""
    hparams = autoencoders.autoencoder_discrete_pong()
    return 2**hparams.num_hidden_layers

  @property
  def frame_height(self):
    height = self.env.observation_space.shape[0]
    ae_height = int(math.ceil(height / self.autoencoder_factor))
    return ae_height

  @property
  def frame_width(self):
    width = self.env.observation_space.shape[1]
    return int(math.ceil(width / self.autoencoder_factor))
