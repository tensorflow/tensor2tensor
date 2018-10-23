from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np
import tensorflow as tf

from absl import flags
from gym.core import Env, Wrapper
from gym.spaces import Box
from gym.spaces import Discrete
import gin

from dopamine.agents.dqn import dqn_agent
from dopamine.atari import run_experiment
from tensor2tensor.models.research.rl import get_policy
from tensor2tensor.rl.envs.simulated_batch_env import SimulatedBatchEnv

#Remove if possibe
flags.DEFINE_bool('debug_mode', False,
                  'If set to true, the agent will output in-episode statistics '
                  'to Tensorboard. Disabled by default as this results in '
                  'slower training.')
flags.DEFINE_string('agent_name', None,
                    'Name of the agent. Must be one of '
                    '(dqn, rainbow, implicit_quantile)')
flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
    '"dopamine/agents/dqn/dqn.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')

FLAGS = flags.FLAGS


class FlatBatchEnv:
  def __init__(self, batch_env):
    if batch_env.batch_size != 1:
      raise ValueError("Number of environments in batch must be equal to one")
    self.batch_env = batch_env
    self.action_space = self.batch_env.action_space
    # TODO(KC): replace this when removing _augment_observation()
    # self.observation_space = self.batch_env.observation_space
    self.observation_space = Box(low=0, high=255, shape=(84, 84, 3),
                                 dtype=np.uint8)
    self.game_over = False  # Dopamine needs it?

  def step(self, action):
    obs, rewards, dones = self.batch_env.step([action])
    return self._augment_observation(obs[0]), rewards[0], dones[0], {}

  def reset(self):
    ob = self.batch_env.reset()[0]
    return self._augment_observation(ob)

  def _augment_observation(self, ob):
    # TODO(KC): remove this
    dopamine_ob = np.zeros(shape=(84, 84, 3),
                         dtype=np.uint8)
    dopamine_ob[:80, :80, :] = ob[:80, :80, :]
    return dopamine_ob


class SimulatedBatchGymEnv:
  """
  Environment wrapping SimulatedBatchEnv in a Gym-like interface (but for batch
  of environments)
  """

  # TODO: Make it singleton, or use with tf.Graph().as_default()
  # Put it into a separate file or put with SimulatedBatchEnv
  # rename this file as "dompamine_connector"

  def __init__(self, hparams, batch_size, timesteps_limit=100, sess=None):
    # TODO(KC): pass ars explicitly without hparams (optionally add static
    # method for hparams initialization)
    self.batch_size = batch_size
    self.timesteps_limit = timesteps_limit

    self.action_space = Discrete(2)
    # TODO: check sizes
    # self.observation_space = self._batch_env.observ_space
    self.observation_space = Box(
        low=0, high=255, shape=(84, 84, 3),
        dtype=np.uint8)
    self.res = None
    self.game_over = False

    with tf.Graph().as_default():
      self._batch_env = SimulatedBatchEnv(hparams.environment_spec,
                                          self.batch_size)

      self.action_space = self._batch_env.action_space

      self._sess = sess if sess is not None else tf.Session()
      self._actions_t = tf.placeholder(shape=(1,), dtype=tf.int32)
      self._rewards_t, self._dones_t = self._batch_env.simulate(self._actions_t)
      self._obs_t = self._batch_env.observ
      self._reset_op = self._batch_env.reset(tf.constant([0], dtype=tf.int32))

      environment_wrappers = hparams.environment_spec.wrappers
      wrappers = copy.copy(environment_wrappers) if environment_wrappers else []

      self._to_initialize = [self._batch_env]
      for w in wrappers:
        self._batch_env = w[0](self._batch_env, **w[1])
        self._to_initialize.append(self._batch_env)

      self._sess_initialized = False
      self._step_num = 0

  def _initialize_sess(self):
    self._sess.run(tf.global_variables_initializer())
    for _batch_env in self._to_initialize:
      _batch_env.initialize(self._sess)
    self._sess_initialized = True

  def render(self, mode="human"):
    raise NotImplementedError()

  def reset(self, indicies=None):
    if indicies:
      raise NotImplementedError()
    if not self._sess_initialized:
      self._initialize_sess()
    obs = self._sess.run(self._reset_op)
    # TODO(pmilos): remove if possible
    obs[:, 0, 0, 0] = 0
    obs[:, 0, 0, 1] = 255
    return obs

  def step(self, actions):
    self._step_num += 1
    obs, rewards, dones = self._sess.run(
      [self._obs_t, self._rewards_t, self._dones_t],
      feed_dict={self._actions_t: [actions]})

    if self._step_num >= 100:
      dones = [True] * self.batch_size

    return obs, rewards, dones


def create_agent(sess, environment, summary_writer=None):
  """Creates a DQN agent.

  Simplified version of `dopamine.atari.train.create_agent`
  """
  if not FLAGS.debug_mode:
    summary_writer = None
  return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n,
                            summary_writer=summary_writer,
                            tf_device='/cpu:*')  # TODO:


def get_create_env_simulated_fun(hparams):
  def create_env_fun(game_name, sticky_actions=True):
    # Possibly use wrappers as used by atari training in dopamine
    return FlatBatchEnv(SimulatedBatchGymEnv(hparams, 1))
  return create_env_fun


def get_create_env_real_fun(hparams):
  env = hparams.environment_spec.env

  def create_env_fun(_1, _2):
    return FlatBatchEnv(env)

  return create_env_fun



def create_runner(hparams):
  """ Simplified version of `dopamine.atari.train.create_runner` """

  # TODO: pass and clean up hparams
  steps_to_make = 1000
  if hparams.environment_spec.simulated_env:
    get_create_env_fun = get_create_env_simulated_fun
    num_iterations = np.ceil(steps_to_make / 1000)
    training_steps = 200
  else:
    get_create_env_fun = get_create_env_real_fun
    num_iterations = 1
    training_steps = 100
  # TODO: this must be 0 for real_env (to not generate addtionall data for
  # world-model, but maybe we can use it in simulated env?
  evaluation_steps = 0

  with gin.unlock_config():
    # This is slight wierdness
    run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)

  with tf.Graph().as_default():
    runner = run_experiment.Runner(FLAGS.base_dir, create_agent,
                                   create_environment_fn=get_create_env_fun(hparams),
                                   num_iterations=1,
                                   training_steps=training_steps,
                                   evaluation_steps=evaluation_steps,
                                   max_steps_per_episode=20  # TODO: remove this
                                   )

    runner.run_experiment()
