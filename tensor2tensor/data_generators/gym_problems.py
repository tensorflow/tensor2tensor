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

import collections
import json
import math
import os
import gym
import numpy as np
import six

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import video_utils
from tensor2tensor.models.research import autoencoders
from tensor2tensor.models.research import rl
from tensor2tensor.rl import collect
from tensor2tensor.rl.envs import tf_atari_wrappers
from tensor2tensor.rl.envs.utils import InitialFrameChooser
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string("agent_policy_path", None, "File with model for agent.")

flags.DEFINE_string("autoencoder_path", None,
                    "File with model for autoencoder.")


def standard_atari_env_spec(env, simulated=False,
                            resize_height_factor=1, resize_width_factor=1):
  """Parameters of environment specification."""
  standard_wrappers = [
      [tf_atari_wrappers.ResizeWrapper,
       {"height_factor": resize_height_factor,
        "width_factor": resize_width_factor}],
      [tf_atari_wrappers.RewardClippingWrapper, {}],
      [tf_atari_wrappers.StackWrapper, {"history": 4}],
  ]
  if simulated:  # No resizing on simulated environments.
    standard_wrappers = standard_wrappers[1:]
  env_lambda = None
  if isinstance(env, str):
    env_lambda = lambda: gym.make(env)
  if callable(env):
    env_lambda = env
  assert env_lambda is not None, "Unknown specification of environment"

  return tf.contrib.training.HParams(
      env_lambda=env_lambda,
      wrappers=standard_wrappers,
      simulated_env=simulated)

def standard_atari_env_eval_spec(env, simulated=False,
                            resize_height_factor=1, resize_width_factor=1):
  """Parameters of environment specification."""
  standard_wrappers = [
      [tf_atari_wrappers.ResizeWrapper,
       {"height_factor": resize_height_factor,
        "width_factor": resize_width_factor}],
      [tf_atari_wrappers.StackWrapper, {"history": 4}],
  ]
  if simulated:  # No resizing on simulated environments.
    standard_wrappers = standard_wrappers[1:]
  env_lambda = None
  if isinstance(env, str):
    env_lambda = lambda: gym.make(env)
  if callable(env):
    env_lambda = env
  assert env_lambda is not None, "Unknown specification of environment"

  return tf.contrib.training.HParams(
      env_lambda=env_lambda,
      wrappers=standard_wrappers,
      simulated_env=simulated)


def standard_atari_ae_env_spec(env):
  """Parameters of environment specification."""
  standard_wrappers = [[tf_atari_wrappers.AutoencoderWrapper, {}],
                       [tf_atari_wrappers.StackWrapper, {"history": 4}]]
  env_lambda = None
  if isinstance(env, str):
    env_lambda = lambda: gym.make(env)
  if callable(env):
    env_lambda = env
  assert env is not None, "Unknown specification of environment"

  return tf.contrib.training.HParams(env_lambda=env_lambda,
                                     wrappers=standard_wrappers,
                                     simulated_env=False)


frame_dumper_use_disk = False  # Whether to use memory or disk to dump frames.
frame_dumper = {}


class GymDiscreteProblem(video_utils.VideoProblem):
  """Gym environment with discrete actions and rewards."""

  def __init__(self, *args, **kwargs):
    super(GymDiscreteProblem, self).__init__(*args, **kwargs)
    # TODO(piotrmilos): Check if self._env is used.
    self._env = None

    self.debug_dump_frames_path = "debug_frames_env"
    self.settable_num_steps = 5000

    self._environment_spec = None
    self.settable_eval_phase = False

    self._internal_memory_size = 20
    self._internal_memory_force_beginning_resets = False
    self._session = None
    self.statistics = BasicStatistics()
    self._use_dumper_data = False
    self._dumper_data_index = 0
    self._forced_collect_level = None

  @property
  def resize_height_factor(self):
    return 1

  @property
  def resize_width_factor(self):
    return 1

  def _setup(self, data_dir, extra_collect_hparams=None,
             override_collect_hparams=None):
    dumper_path = os.path.join(data_dir, "dumper")
    dumper_exists = tf.gfile.Exists(dumper_path)
    tf.logging.info("Dumper path %s." % dumper_path)
    if dumper_exists and not self.settable_eval_phase:
      tf.logging.info("Using dumper data.")
      self._use_dumper_data = True
      self._dumper_data_index = 0
      self._dumper_path = dumper_path
    else:
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

      if extra_collect_hparams is not None:
        for (key, value) in six.iteritems(extra_collect_hparams):
          collect_hparams.add_hparam(key, value)

      if override_collect_hparams is not None:
        # Override hparams manually - HParams.override_from_dict does not work
        # with functions.
        for (key, value) in six.iteritems(override_collect_hparams):
          setattr(collect_hparams, key, value)

      policy_to_actions_lambda = None
      if self.settable_eval_phase:
        policy_to_actions_lambda = lambda policy: policy.mode()

      collect_level = 2  # After Resize and RewardClipping.
      if collect_hparams.environment_spec.simulated_env:
        collect_level = 1  # We still have reward clipping.
      if self._forced_collect_level is not None:  # For autoencoders.
        collect_level = self._forced_collect_level

      with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        self.collect_memory, self.collect_trigger_op, collect_init = (
            collect.define_collect(
                collect_hparams,
                scope="gym_problems",
                eval_phase=False,
                collect_level=collect_level,
                policy_to_actions_lambda=policy_to_actions_lambda))

      self._session = tf.Session()
      collect_init(self._session)
      self._session.run(tf.global_variables_initializer())
      self.restore_networks(self._session)
      self.memory_index = 0
      self.memory = None

  @property
  def random_skip(self):
    return False

  def _get_data(self):
    if self._use_dumper_data:
      file_path = os.path.join(self._dumper_path,
                               "frame_{}.npz".format(self._dumper_data_index))
      if frame_dumper_use_disk:
        with tf.gfile.Open(file_path) as gfile:
          data = np.load(gfile)
      else:
        data = frame_dumper.pop(file_path)
      self._dumper_data_index += 1
      return (data["observ"][0, ...], data["reward"][0], data["done"][0],
              data["action"][0])
    else:
      if self.memory is None or self.memory_index >= self._internal_memory_size:
        self.memory = self._session.run(self.collect_memory)
        self.memory_index = 0
      data = [self.memory[i][self.memory_index][0] for i in range(4)]
      self.memory_index += 1

      return data

  def generate_samples(self, data_dir, tmp_dir, unused_dataset_split):
    self._setup(data_dir)

    self.debug_dump_frames_path = os.path.join(
        data_dir, self.debug_dump_frames_path)

    frame_counter = 0
    pieces_generated = 0
    prev_reward = 0
    prev_done = False

    # TODO(piotrmilos): self.settable_eval_phase possibly violates sematics
    # of VideoProblem
    while pieces_generated < self.num_steps or self.settable_eval_phase:
      data = self._get_data()
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
    return standard_atari_env_spec(
        self.env_name,
        resize_height_factor=self.resize_height_factor,
        resize_width_factor=self.resize_width_factor)

  @property
  def environment_spec(self):
    if self._environment_spec is None:
      self._environment_spec = self.get_environment_spec()
    return self._environment_spec

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
    return self.env.observation_space.shape[0] // self.resize_height_factor

  @property
  def frame_width(self):
    return self.env.observation_space.shape[1] // self.resize_width_factor

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

  def __init__(self, *args, **kwargs):
    super(GymDiscreteProblemWithAutoencoder, self).__init__(*args, **kwargs)
    self._forced_collect_level = 0

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

  def hparams(self, defaults, unused_model_hparams):
    """Overrides VideoProblem.hparams to work on images instead of videos."""
    p = defaults
    p.input_modality = {
        "inputs": ("image", 256),
    }
    p.target_modality = ("image", 256)
    p.input_space_id = problem.SpaceID.IMAGE
    p.target_space_id = problem.SpaceID.IMAGE

  def preprocess(self, dataset, mode, hparams, interleave=True):
    """Overrides VideoProblem.preprocess to work on images instead of videos."""
    def set_targets(example):
      example["targets"] = example["frame"]
      return example
    return dataset.map(set_targets)


class GymDiscreteProblemAutoencoded(GymRealDiscreteProblem):
  """Gym discrete problem with frames already autoencoded."""

  def __init__(self, *args, **kwargs):
    super(GymDiscreteProblemAutoencoded, self).__init__(*args, **kwargs)
    self._forced_collect_level = 0

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

  def __init__(self, rollout_fractions):
    super(RewardPerSequenceStatistics, self).__init__()

    # data to calculate
    # correctness of rewards per sequence metric
    self.episode_sim_reward = 0.0
    self.episode_real_reward = 0.0
    self.successful_episode_reward_predictions = collections.OrderedDict([
        (frac, 0) for frac in rollout_fractions
    ])
    self.report_reward_statistics_every = 10
    # auxiliary objects
    self.real_obs = None
    self.real_rewards = None

  def to_dict(self):
    stats_dict = super(RewardPerSequenceStatistics, self).to_dict()
    keys_and_types = [
        ("episode_sim_reward", float),
        ("episode_real_reward", float),
        ("successful_episode_reward_predictions", collections.OrderedDict),
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

    self.statistics = BasicStatistics()
    self._initial_frame_chooser = None

  def _setup(self, data_dir, extra_collect_hparams=None,
             override_collect_hparams=None):
    if extra_collect_hparams is None:
      extra_collect_hparams = {}

    if self._initial_frame_chooser is None:
      self._initial_frame_chooser = InitialFrameChooser(
          self.environment_spec, mode=tf.estimator.ModeKeys.EVAL
      )
    extra_collect_hparams["initial_frame_chooser"] = self._initial_frame_chooser

    super(GymSimulatedDiscreteProblem, self)._setup(
        data_dir, extra_collect_hparams, override_collect_hparams
    )

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
    env_spec = standard_atari_env_spec(
        self.env_name,
        simulated=True,
        resize_height_factor=self.resize_height_factor,
        resize_width_factor=self.resize_width_factor)
    env_spec.add_hparam("simulation_random_starts", True)
    env_spec.add_hparam("simulation_flip_first_random_for_beginning", True)
    env_spec.add_hparam("intrinsic_reward_scale", 0.0)
    initial_frames_problem = registry.problem(self.initial_frames_problem)
    env_spec.add_hparam("initial_frames_problem", initial_frames_problem)
    env_spec.add_hparam("video_num_input_frames", self.num_input_frames)
    env_spec.add_hparam("video_num_target_frames", self.video_num_target_frames)

    return env_spec

  def restore_networks(self, sess):
    super(GymSimulatedDiscreteProblem, self).restore_networks(sess)
    # TODO(blazej): adjust regexp for different models.
    # TODO(piotrmilos): move restoring networks to SimulatedBatchEnv.initialize
    env_model_loader = tf.train.Saver(tf.global_variables("next_frame*"))
    ckpts = tf.train.get_checkpoint_state(FLAGS.output_dir)
    ckpt = ckpts.model_checkpoint_path
    env_model_loader.restore(sess, ckpt)


class GymSimulatedDiscreteProblemForWorldModelEval(GymSimulatedDiscreteProblem):
  """Simulated gym environment for evaluating world model."""

  def __init__(self, *args, **kwargs):
    super(GymSimulatedDiscreteProblemForWorldModelEval, self).__init__(
        *args, **kwargs
    )
    self.settable_rollout_fractions = [1]
    self.statistics = RewardPerSequenceStatistics(
        self.settable_rollout_fractions
    )

  def get_environment_spec(self):
    env_spec = super(
        GymSimulatedDiscreteProblemForWorldModelEval, self
    ).get_environment_spec()
    env_spec.simulation_flip_first_random_for_beginning = False
    return env_spec

  def _setup(self, data_dir):
    trajectory_length = self.num_testing_steps
    if self.num_steps < 1200:
      # Decrease the trajectory length for tiny experiments, otherwise we don't
      # have enough data to run the evaluation.
      trajectory_length = 2
    self._initial_frame_chooser = InitialFrameChooser(
        self.environment_spec, mode=tf.estimator.ModeKeys.EVAL,
        trajectory_length=trajectory_length
    )

    frame_index = tf.Variable(0, trainable=False)

    def fixed_action_policy_fun(action_space, unused_config, observations):
      """Policy which replays actions from a trajectory."""
      action = self._initial_frame_chooser.trajectory["action"].read_value()[
          :, frame_index.read_value(), :
      ]
      inc_frame_index = frame_index.assign(
          (frame_index.read_value() + 1) % trajectory_length
      )
      with tf.control_dependencies([inc_frame_index]):
        action = tf.identity(action)

      obs_shape = observations.shape.as_list()
      with tf.variable_scope("network_parameters"):
        probs = tf.one_hot(
            tf.transpose(action), depth=action_space.n
        )
        policy = tf.distributions.Categorical(probs=probs)
        value = tf.zeros(obs_shape[:2])
      return rl.NetworkOutput(policy, value, lambda a: a)

    super(GymSimulatedDiscreteProblemForWorldModelEval, self)._setup(
        data_dir, override_collect_hparams={
            "policy_network": fixed_action_policy_fun
        }
    )

  def collect_statistics_and_generate_debug_image(self, index,
                                                  observation,
                                                  reward, done, action):
    stat = self.statistics

    # TODO(piotrmilos): possibly make the same behaviour as
    # in the BasicStatistics
    stat.sum_of_rewards += reward
    stat.episode_sim_reward += reward

    if index % self._internal_memory_size == 0:
      real_frame_tensor = {
          key: var.read_value()[0, ...]
          for (key, var) in six.iteritems(
              self._initial_frame_chooser.trajectory
          )
      }
      (stat.real_obs, stat.real_rewards) = self._session.run((
          real_frame_tensor["inputs"], real_frame_tensor["reward"]
      ))
      stat.real_rewards += self.min_reward

    real_ob = stat.real_obs[index % stat.real_obs.shape[0], ...]
    debug_im = self._generate_debug_image(real_ob, observation)

    assert (self._internal_memory_size == self.num_testing_steps and
            self._internal_memory_force_beginning_resets), (
                "The collect memory should be set in force_beginning_resets "
                "mode for the code below to work properly.")

    index_in_rollout = index % self._internal_memory_size + 1

    if stat.episode_sim_reward == stat.episode_real_reward:
      for frac in stat.successful_episode_reward_predictions:
        if index_in_rollout == int(self._internal_memory_size * frac):
          stat.successful_episode_reward_predictions[frac] += 1

    if index_in_rollout == self._internal_memory_size:
      stat.episode_sim_reward = 0.0
      stat.episode_real_reward = 0.0
      stat.number_of_dones += 1
    else:
      real_reward = stat.real_rewards[index % stat.real_rewards.shape[0], 0]
      stat.episode_real_reward += real_reward

    return debug_im

  def _generate_debug_image(self, real_ob, sim_ob):
    ob = np.ndarray.astype(sim_ob, np.int)
    if ob.shape == real_ob.shape:
      err = np.ndarray.astype(
          np.maximum(np.abs(real_ob - ob, dtype=np.int) - 10, 0), np.uint8)
      debug_im = np.concatenate([sim_ob, real_ob, err], axis=1)
    else:
      # Real env does not get the ResizeWrapper and we don't have it in python,
      # so we skip the debug image here and just output observations.
      debug_im = sim_ob
    return debug_im


class GymSimulatedDiscreteProblemAutoencoded(GymSimulatedDiscreteProblem):
  """Gym simulated discrete problem with frames already autoencoded."""

  def __init__(self, *args, **kwargs):
    super(GymSimulatedDiscreteProblemAutoencoded, self).__init__(
        *args, **kwargs)
    self._forced_collect_level = 0

  def get_environment_spec(self):
    env_spec = standard_atari_env_spec(self.env_name)
    env_spec.wrappers = [
        [tf_atari_wrappers.IntToBitWrapper, {}],
        [tf_atari_wrappers.StackWrapper, {"history": 4}]
    ]
    env_spec.simulated_env = True
    env_spec.add_hparam("simulation_random_starts", True)
    env_spec.add_hparam("simulation_flip_first_random_for_beginning", True)
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


class GymSimulatedDiscreteProblemForWorldModelEvalAutoencoded(
    GymSimulatedDiscreteProblemForWorldModelEval,
    GymSimulatedDiscreteProblemAutoencoded):
  """TODO(owner): Write a small docstring."""

  def _generate_debug_image(self, real_ob, sim_ob):
    def unpack(x):
      return np.ndarray.astype(np.unpackbits(x, axis=2), np.int)
    real_ob_unpacked = unpack(real_ob)
    sim_ob_unpacked = unpack(sim_ob)
    # Hamming distance on binary latent codes, seen as a grayscale image.
    err = np.ndarray.astype(
        np.transpose(
            np.broadcast_to(
                np.sum(np.abs(real_ob_unpacked - sim_ob_unpacked), axis=2) /
                24.0 * 255,
                # Channels first to satisfy numpy broadcasting rules.
                shape=((real_ob.shape[2],) + real_ob.shape[:2])),
            (1, 2, 0)),
        np.uint8)
    return np.concatenate([sim_ob, real_ob, err], axis=1)


@registry.register_problem
class DummyAutoencoderProblem(GymDiscreteProblemWithAutoencoder):
  """Dummy problem for running the autoencoder inside AutoencoderWrapper."""

  @property
  def env_name(self):
    return "DummyAutoencoder"
