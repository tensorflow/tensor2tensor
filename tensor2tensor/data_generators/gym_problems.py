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

# Dependency imports

import gym
import numpy as np

from tensor2tensor.data_generators import gym_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import video_utils
from tensor2tensor.layers import discretization
from tensor2tensor.models.research import autoencoders
from tensor2tensor.models.research import rl
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

  @property
  def num_input_frames(self):
    """Number of frames to batch on one input."""
    return 4

  @property
  def num_target_frames(self):
    """Number of frames to batch on one target."""
    return 1

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

  @property
  def num_testing_steps(self):
    return None

  def get_action(self, observation=None):
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
        "target_reward": ("symbol:weights_all", self.num_rewards)
    }
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
class GymWrappedPongRandom5k(GymDiscreteProblem):
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

  @property
  def num_steps(self):
    return 5000


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
  def num_steps(self):
    return 5000

  @property
  def num_testing_steps(self):
    return 100


@registry.register_problem
class GymWrappedBreakoutRandom5k(GymDiscreteProblem):
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

  @property
  def num_steps(self):
    return 5000


@registry.register_problem
class GymWrappedPongRandom50k(GymPongRandom5k):
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
    self.make_extra_debug_info = True
    self.autoencoder_model = None

    # Defaults.
    self.environment_spec = lambda: gym.make(self.env_name)
    self._real_env = None
    self.real_env_problem = None
    self.in_graph_wrappers = []
    self.collect_hparams = rl.ppo_pong_base()
    if FLAGS.autoencoder_path:
      self.collect_hparams = rl.ppo_pong_ae_base()
    self.settable_num_steps = 50000
    self.simulated_environment = None
    self.eval_phase = False
    self.warm_up = 10  # TODO(piotrm): This should be probably removed.

    # Debug info.
    self.dones = 0
    self.real_reward = 0
    self.total_sim_reward, self.total_real_reward = 0.0, 0.0
    self.sum_of_rewards = 0.0
    self.successful_episode_reward_predictions = 0

  @property
  def real_env(self):
    """Lazy caching environment construction."""
    if self._real_env is None:
      self._real_env = self.environment_spec()
    return self._real_env

  @property
  def num_steps(self):
    return self.settable_num_steps

  @property
  def raw_frame_height(self):
    return self.env.observation_space.shape[0]

  @property
  def frame_height(self):
    if FLAGS.autoencoder_path:
      # TODO(lukaszkaiser): remove hard-coded autoencoder params.
      return int(math.ceil(self.raw_frame_height / self.autoencoder_factor))
    return self.raw_frame_height

  @property
  def autoencoder_factor(self):
    """By how much to divide sizes when using autoencoders."""
    hparams = autoencoders.autoencoder_discrete_pong()
    return 2**hparams.num_hidden_layers

  @property
  def raw_frame_width(self):
    return self.env.observation_space.shape[1]

  @property
  def frame_width(self):
    if FLAGS.autoencoder_path:
      # TODO(lukaszkaiser): remove hard-coded autoencoder params.
      return int(math.ceil(self.raw_frame_width / self.autoencoder_factor))
    return self.raw_frame_width

  def setup_autoencoder(self):
    if self.autoencoder_model is not None:
      return
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      autoencoder_hparams = autoencoders.autoencoder_discrete_pong()
      autoencoder_hparams.data_dir = "unused"
      autoencoder_hparams.problem_hparams = self.get_hparams(
          autoencoder_hparams)
      autoencoder_hparams.problem = self
      self.autoencoder_model = autoencoders.AutoencoderOrderedDiscrete(
          autoencoder_hparams, tf.estimator.ModeKeys.EVAL)

  def autoencode_tensor(self, x, batch_size=1):
    if self.autoencoder_model is None:
      return x
    shape = [self.raw_frame_height, self.raw_frame_width, self.num_channels]
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      self.autoencoder_model.set_mode(tf.estimator.ModeKeys.EVAL)
      autoencoded = self.autoencoder_model.encode(
          tf.reshape(x, [batch_size, 1] + shape))
    autoencoded = tf.reshape(
        autoencoded, [batch_size, self.frame_height, self.frame_width,
                      self.num_channels, 8])  # 8-bit groups.
    if batch_size == 1:
      autoencoded = tf.squeeze(autoencoded, axis=0)
    return discretization.bit_to_int(autoencoded, 8)

  def _setup(self):
    if self.make_extra_debug_info:
      self.report_reward_statistics_every = 10
      self.dones = 0
      self.real_reward = 0
      # Slight weirdness to make sim env and real env aligned
      if self.simulated_environment:
        self.real_env.reset()
        for _ in range(self.num_input_frames):
          self.real_ob, _, _, _ = self.real_env.step(0)
      self.total_sim_reward, self.total_real_reward = 0.0, 0.0
      self.sum_of_rewards = 0.0
      self.successful_episode_reward_predictions = 0

    in_graph_wrappers = self.in_graph_wrappers + [
        (atari.MemoryWrapper, {}), (StackAndSkipWrapper, {"skip": 4})]
    env_hparams = tf.contrib.training.HParams(
        in_graph_wrappers=in_graph_wrappers,
        problem=self.real_env_problem if self.real_env_problem else self,
        simulated_environment=self.simulated_environment)
    if self.simulated_environment:
      env_hparams.add_hparam("simulation_random_starts",
                             self.simulation_random_starts)
      env_hparams.add_hparam("intrinsic_reward_scale",
                             self.intrinsic_reward_scale)

    generator_batch_env = batch_env_factory(
        self.environment_spec, env_hparams, num_agents=1, xvfb=False)

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      if FLAGS.agent_policy_path:
        policy_lambda = self.collect_hparams.network
      else:
        # When no agent_policy_path is set, just generate random samples.
        policy_lambda = rl.random_policy_fun

    if FLAGS.autoencoder_path:
      # TODO(lukaszkaiser): remove hard-coded autoencoder params.
      with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        self.setup_autoencoder()
        autoencoder_model = self.autoencoder_model
        # Feeds for autoencoding.
        shape = [self.raw_frame_height, self.raw_frame_width, self.num_channels]
        self.autoencoder_feed = tf.placeholder(tf.int32, shape=shape)
        self.autoencoder_result = self.autoencode_tensor(self.autoencoder_feed)
        # Now for autodecoding.
        shape = [self.frame_height, self.frame_width, self.num_channels]
        self.autodecoder_feed = tf.placeholder(tf.int32, shape=shape)
        bottleneck = tf.reshape(
            discretization.int_to_bit(self.autodecoder_feed, 8),
            [1, 1, self.frame_height, self.frame_width, self.num_channels * 8])
        autoencoder_model.set_mode(tf.estimator.ModeKeys.PREDICT)
        self.autodecoder_result = autoencoder_model.decode(bottleneck)

    def preprocess_fn(x):
      shape = [self.raw_frame_height, self.raw_frame_width, self.num_channels]
      # TODO(lukaszkaiser): we assume x comes from StackAndSkipWrapper skip=4.
      xs = [tf.reshape(t, [1] + shape) for t in tf.split(x, 4, axis=-1)]
      autoencoded = self.autoencode_tensor(tf.concat(xs, axis=0), batch_size=4)
      encs = [tf.squeeze(t, axis=[0]) for t in tf.split(autoencoded, 4, axis=0)]
      res = tf.to_float(tf.concat(encs, axis=-1))
      return tf.expand_dims(res, axis=0)

    # TODO(lukaszkaiser): x is from StackAndSkipWrapper thus 4*num_channels.
    shape = [1, self.frame_height, self.frame_width, 4 * self.num_channels]
    do_preprocess = (self.autoencoder_model is not None and
                     not self.simulated_environment)
    preprocess = (preprocess_fn, shape) if do_preprocess else None

    def policy(x):
      return policy_lambda(self.environment_spec().action_space,
                           self.collect_hparams, x)

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      self.collect_hparams.epoch_length = 10
      _, self.collect_trigger_op = collect.define_collect(
          policy, generator_batch_env, self.collect_hparams,
          eval_phase=self.eval_phase,
          scope="define_collect", preprocess=preprocess)

    self.avilable_data_size_op = atari.MemoryWrapper.singleton.speculum.size()
    self.data_get_op = atari.MemoryWrapper.singleton.speculum.dequeue()

  def restore_networks(self, sess):
    if FLAGS.agent_policy_path:
      model_saver = tf.train.Saver(
          tf.global_variables(".*network_parameters.*"))
      ckpts = tf.train.get_checkpoint_state(FLAGS.agent_policy_path)
      ckpt = ckpts.model_checkpoint_path
      model_saver.restore(sess, ckpt)
    if FLAGS.autoencoder_path:
      autoencoder_saver = tf.train.Saver(
          tf.global_variables("autoencoder.*"))
      ckpts = tf.train.get_checkpoint_state(FLAGS.autoencoder_path)
      ckpt = ckpts.model_checkpoint_path
      autoencoder_saver.restore(sess, ckpt)

  def autoencode(self, image, sess):
    return sess.run(self.autoencoder_result, {self.autoencoder_feed: image})

  def autodecode(self, encoded, sess):
    res = sess.run(self.autodecoder_result, {self.autodecoder_feed: encoded})
    return res[0, 0, :self.raw_frame_height, :self.raw_frame_width, :]

  def generate_samples(self, data_dir, tmp_dir, unused_dataset_split):
    self._setup()
    self.debug_dump_frames_path = os.path.join(
        data_dir, self.debug_dump_frames_path)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      self.restore_networks(sess)
      if FLAGS.only_use_ae_for_policy:
        # If only the policy should use the autoencoder, then reset the flag so
        # that other components here act as though there is no autoencoder and
        # so write out full-resolution images. The policy graph was already
        # built and self.collect_trigger_op is all that's used from it.
        FLAGS.autoencoder_path = None
      pieces_generated = 0
      while pieces_generated < self.num_steps + self.warm_up:
        avilable_data_size = sess.run(self.avilable_data_size_op)
        if avilable_data_size < 1:
          sess.run(self.collect_trigger_op)
        observ, reward, action, done = sess.run(self.data_get_op)
        debug_im = None
        if self.make_extra_debug_info:
          self.total_sim_reward += reward
          if not self.simulated_environment:
            self.real_ob = observ
            self.real_reward = reward
          if not FLAGS.autoencoder_path:
            err = np.ndarray.astype(np.maximum(np.abs(
                self.real_ob - observ, dtype=np.int) - 10, 0),
                                    np.uint8)
            debug_im_np = np.concatenate([observ, self.real_ob, err], axis=1)
            debug_im = gym_utils.encode_image_to_png(debug_im_np)
          if done:
            self.dones += 1
            self.sum_of_rewards += self.real_reward
            if self.total_real_reward == self.total_sim_reward:
              self.successful_episode_reward_predictions += 1

            self.total_real_reward = 0.0
            self.total_sim_reward = 0.0
            self.real_reward = 0
            if self.simulated_environment:
              self.real_env.reset()
              # Slight weirdness to make sim env and real env aligned
              for _ in range(self.num_input_frames):
                self.real_ob, _, _, _ = self.real_env.step(0)
          else:
            if self.simulated_environment:
              self.real_ob, self.real_reward, _, _ = self.real_env.step(action)
            self.total_real_reward += self.real_reward
            self.sum_of_rewards += self.real_reward
        if FLAGS.autoencoder_path:
          if self.simulated_environment:
            debug_im = gym_utils.encode_image_to_png(
                self.autodecode(observ, sess))
          else:
            orig_observ = observ
            observ = self.autoencode(observ, sess)
            debug_im_np = np.concatenate([self.autodecode(observ, sess),
                                          orig_observ], axis=1)
            debug_im = gym_utils.encode_image_to_png(debug_im_np)
        ret_dict = {"frame": observ,
                    "image/format": ["png"],
                    "image/height": [self.frame_height],
                    "image/width": [self.frame_width],
                    "action": [int(action)],
                    "done": [int(False)],
                    "reward": [int(reward) - self.min_reward]}
        if self.make_extra_debug_info:
          ret_dict["image/encoded_debug"] = [debug_im]
        yield ret_dict
        pieces_generated += 1


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgent(GymDiscreteProblemWithAgent):
  """Simulated gym environment with discrete actions and rewards."""

  def __init__(self, *args, **kwargs):
    super(GymSimulatedDiscreteProblemWithAgent, self).__init__(*args, **kwargs)
    self.simulated_environment = True
    self.make_extra_debug_info = True
    self.debug_dump_frames_path = "debug_frames_sim"

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

  def restore_networks(self, sess):
    super(GymSimulatedDiscreteProblemWithAgent, self).restore_networks(sess)
    # TODO(blazej): adjust regexp for different models.
    env_model_loader = tf.train.Saver(tf.global_variables("basic_conv_gen.*"))
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
class GymSimulatedDiscreteProblemWithAgentOnWrappedPong(
    GymSimulatedDiscreteProblemWithAgent, GymWrappedPongRandom5k):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedLongPong(
    GymDiscreteProblemWithAgent, GymWrappedLongPongRandom):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedLongPongAe(  # with autoencoder
    GymDiscreteProblemWithAgentOnWrappedLongPong):
  pass


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgentOnWrappedLongPong(
    GymSimulatedDiscreteProblemWithAgent, GymWrappedLongPongRandom):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedBreakout(
    GymDiscreteProblemWithAgent, GymWrappedBreakoutRandom5k):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedBreakoutAe(
    GymDiscreteProblemWithAgentOnWrappedBreakout):
  pass


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgentOnWrappedBreakout(
    GymSimulatedDiscreteProblemWithAgent, GymWrappedBreakoutRandom5k):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedPong(
    GymDiscreteProblemWithAgent, GymWrappedPongRandom5k):
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
    GymSimulatedDiscreteProblemWithAgent, GymFreewayRandom5k):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnFreeway(
    GymDiscreteProblemWithAgent, GymFreewayRandom5k):
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

  @property
  def raw_frame_height(self):
    return self.frame_height

  @property
  def raw_frame_width(self):
    return self.frame_width


@registry.register_problem
class GymDiscreteProblemWithAgentOnFreewayAe(  # with autoencoder
    GymDiscreteProblemWithAgentOnFreeway):
  pass
