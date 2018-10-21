from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np
import tensorflow as tf

from absl import flags
from gym.core import Env
from gym.spaces import Box
from gym.spaces import Discrete

from dopamine.agents.dqn import dqn_agent
from dopamine.atari import run_experiment
from tensor2tensor.models.research.rl import get_policy
from tensor2tensor.rl.envs.simulated_batch_env import SimulatedBatchEnv

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


class DopamineBatchGymEnvWrapper(Env):
  """
  Environment wrapping SimulatedBatchEnv in a Gym
  (and Dopamine) compatible environment.

  Dopamine agent uses only the following attributes:
  - reset
  - step
  - game_over
  """

  def __init__(self, hparams, sess=None):
    # TODO
    self.action_space = Discrete(2)
    self.observation_space = Box(
        low=0, high=255, shape=(84, 84, 3),
        dtype=np.uint8)
    self.res = None
    self.game_over = False
    self.sess = sess if sess is not None else tf.Session()
    self._prepare_networks(hparams, self.sess)

  def _prepare_networks(self, hparams, sess):
    self.action = tf.placeholder(shape=(1,), dtype=tf.int32)
    batch_env = SimulatedBatchEnv(hparams.environment_spec, hparams.num_agents)
    self.reward, self.done = batch_env.simulate(self.action)
    self.observation = batch_env.observ
    self.reset_op = batch_env.reset(tf.constant([0], dtype=tf.int32))

    environment_wrappers = hparams.environment_spec.wrappers
    wrappers = copy.copy(environment_wrappers) if environment_wrappers else []

    to_initialize = [batch_env]
    for w in wrappers:
      batch_env = w[0](batch_env, **w[1])
      to_initialize.append(batch_env)

    def initialization_lambda():
      sess.run(tf.global_variables_initializer())
      for batch_env in to_initialize:
        batch_env.initialize(sess)
      self.initialized = True

    self.initialized = False
    self.initialize = initialization_lambda

    obs_copy = batch_env.observ + 0

    actor_critic = get_policy(tf.expand_dims(obs_copy, 0), hparams)
    self.policy_probs = actor_critic.policy.probs[0, 0, :]
    self.value = actor_critic.value[0, :]

  def render(self, mode="human"):
    raise NotImplementedError()

  def _reset_env(self):
    # TODO
    observ = self.sess.run(self.reset_op)[0, ...]
    observ[0, 0, 0] = 0
    observ[0, 0, 1] = 255
    # TODO(pmilos): put correct numbers
    self.res = (observ, 0, False, [0.1, 0.5, 0.5], 1.1)

  def reset(self):
    if not self.initialized:
      self.initialize()
    self._reset_env()
    observ = self._augment_observation()
    return observ

  def _step_env(self, action):
    observ, rew, done, probs, vf = self.sess.\
      run([self.observation, self.reward, self.done, self.policy_probs,
           self.value],
          feed_dict={self.action: [action]})

    return observ[0, ...], rew[0, ...], done[0, ...], probs, vf

  def _augment_observation(self):
    # TODO
    observ, rew, _, probs, vf = self.res
    dopamine_observ = np.zeros(shape=(84, 84, 3),
                         dtype=np.uint8)
    dopamine_observ[:80, :80] = observ[:80, :80]
    return dopamine_observ

  def step(self, action):
    observ, rew, done, probs, vf = self._step_env(action)
    self.res = (observ, rew, done, probs, vf)

    observ = self._augment_observation()
    return observ, rew, done, {"probs": probs, "vf": vf}


def create_agent(sess, environment, summary_writer=None):
  """Creates a DQN agent.

  Simplified version of `dopamine.atari.train.create_agent`
  """
  if not FLAGS.debug_mode:
    summary_writer = None
  return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n,
                            summary_writer=summary_writer,
                            tf_device='/cpu:*')  # TODO:


def get_create_env_fun(hparams):
  def create_env_fun(game_name, sticky_actions=True):
    return DopamineBatchGymEnvWrapper(hparams)

  return create_env_fun


def create_runner(hparams):
  """ Simplified version of `dopamine.atari.train.create_runner` """
  run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
  runner = run_experiment.Runner(FLAGS.base_dir, create_agent, create_environment_fn=get_create_env_fun(hparams))
  return runner
