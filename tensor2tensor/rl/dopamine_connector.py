# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

"""Connects dopamine to as the another rl traning framework."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import random
import sys

from dopamine.agents.dqn import dqn_agent
from dopamine.replay_memory import circular_replay_buffer
from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer
from dopamine.replay_memory.circular_replay_buffer import ReplayElement
import gym
from gym import spaces
from gym import Wrapper
from gym.wrappers import TimeLimit
import numpy as np
from tensor2tensor.rl.envs.simulated_batch_gym_env import FlatBatchEnv
from tensor2tensor.rl.policy_learner import PolicyLearner
import tensorflow as tf

# pylint: disable=g-import-not-at-top
try:
  import cv2
except ImportError:
  cv2 = None
try:
  from dopamine.discrete_domains import run_experiment
except ImportError:
  run_experiment = None
# pylint: enable=g-import-not-at-top


class ResizeObservation(gym.ObservationWrapper):
  """TODO(konradczechowski): Add doc-string."""

  def __init__(self, env, size=84):
    """Based on WarpFrame from openai baselines atari_wrappers.py.

    Dopamine also uses cv2.resize(..., interpolation=cv2.INTER_AREA).

    Args:
      env: TODO(konradczechowski): Add doc-string.
      size: TODO(konradczechowski): Add doc-string.
    """
    gym.ObservationWrapper.__init__(self, env)
    self.width = size
    self.height = size
    assert env.observation_space.dtype == np.uint8
    self.observation_space = spaces.Box(
        low=0,
        high=255,
        shape=(self.height, self.width, env.observation_space.shape[2]),
        dtype=np.uint8)

  def observation(self, frame):
    if not cv2:
      return frame
    return cv2.resize(
        frame, (self.width, self.height), interpolation=cv2.INTER_AREA)


class GameOverOnDone(Wrapper):
  """TODO(konradczechowski): Add doc-string."""

  def __init__(self, env):
    Wrapper.__init__(self, env)
    self.game_over = False

  def reset(self, **kwargs):
    self.game_over = False
    return self.env.reset(**kwargs)

  def step(self, action):
    ob, reward, done, info = self.env.step(action)
    self.game_over = done
    return ob, reward, done, info


class _DQNAgent(dqn_agent.DQNAgent):
  """Modify dopamine DQNAgent to match our needs.

  Allow passing batch_size and replay_capacity to ReplayBuffer, allow not using
  (some of) terminal episode transitions in training.
  """

  def __init__(self, replay_capacity, buffer_batch_size,
               generates_trainable_dones, **kwargs):
    self._replay_capacity = replay_capacity
    self._buffer_batch_size = buffer_batch_size
    self._generates_trainable_dones = generates_trainable_dones
    super(_DQNAgent, self).__init__(**kwargs)

  def _build_replay_buffer(self, use_staging):
    """Build WrappedReplayBuffer with custom OutOfGraphReplayBuffer."""
    replay_buffer_kwargs = dict(
        observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
        stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
        replay_capacity=self._replay_capacity,
        batch_size=self._buffer_batch_size,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        extra_storage_types=None,
        observation_dtype=np.uint8,
    )
    replay_memory = _OutOfGraphReplayBuffer(
        artificial_done=not self._generates_trainable_dones,
        **replay_buffer_kwargs)

    return circular_replay_buffer.WrappedReplayBuffer(
        wrapped_memory=replay_memory,
        use_staging=use_staging,
        **replay_buffer_kwargs)


class BatchDQNAgent(_DQNAgent):
  """
  Episodes are stored on done.

  Assumes that all rollouts in batch would end at the same moment.
  """

  def __init__(self, env_batch_size, *args, **kwargs):
    super(BatchDQNAgent, self).__init__(*args, **kwargs)
    self.env_batch_size = env_batch_size
    obs_size = NATURE_DQN_OBSERVATION_SHAPE
    state_shape = [self.env_batch_size, obs_size, obs_size,
                   NATURE_DQN_STACK_SIZE]
    self.state_batch = np.zeros(state_shape)
    self.state = None  # assure it will be not used
    self._observation = None  # assure it will be not used
    self._current_rollouts = [[] for _ in range(self.env_batch_size)]

  def _record_observation(self, observation_batch):
    # Set current observation. Represents an (batch_size x 84 x 84 x 1) image
    # frame.
    observation_batch = np.array(observation_batch)
    self._observation_batch = observation_batch[:, :, :, 0]
    # Swap out the oldest frames with the current frames.
    self.state_batch = np.roll(self.state_batch, -1, axis=3)
    self.state_batch[:, :, :, -1] = self._observation_batch

  def _reset_state(self):
    self.state_batch.fill(0)

  def begin_episode(self, observation):
    self._reset_state()
    self._record_observation(observation)

    if not self.eval_mode:
      self._train_step()

    self.action = self._select_action()
    return self.action

  def _update_current_rollouts(self, last_observation, action, reward,
                              are_terminal):
    transitions = zip(last_observation, action, reward, are_terminal)
    for transition, rollout in zip(transitions, self._current_rollouts):
      rollout.append(transition)

  def _store_current_rollouts(self):
    for rollout in self._current_rollouts:
      for transition in rollout:
        self._store_transition(*transition)
    self._current_rollouts = [] * self.env_batch_size

  def step(self, reward, observation):
    self._last_observation = self._observation_batch
    self._record_observation(observation)

    if not self.eval_mode:
      self._update_current_rollouts(self._last_observation, self.action, reward,
                                    [False] * self.env_batch_size)
      # We want to have the same train_step:env_step ratio not depending on
      # batch size.
      for _ in range(self.env_batch_size):
        self._train_step()

    self.action = self._select_action()
    return self.action

  def end_episode(self, reward):
    if not self.eval_mode:
      self._update_current_rollouts(self._observation_batch, self.action, reward,
                                    [True] * self.env_batch_size)
      self._store_current_rollouts()

  def _select_action(self):
    epsilon = self.epsilon_eval if self.eval_mode else self.epsilon_fn(
        self.epsilon_decay_period,
        self.training_steps,
        self.min_replay_history,
        self.epsilon_train)

    def choose_action(ix):
      if random.random() <= epsilon:
        # Choose a random action with probability epsilon.
        return random.randint(0, self.num_actions - 1)
      else:
        # Choose the action with highest Q-value at the current state.
        return self._sess.run(self._q_argmax,
                              {self.state_ph: self.state_batch[ix:ix+1]})

    return np.array([choose_action(ix) for ix in range(self.env_batch_size)])


class BatchRunner(run_experiment.Runner):
  """

  Assumes that all environments would end at the same moment.
  """
  def __init__(self, base_dir, create_agent_fn, batch_size, **kwargs):
    super(BatchRunner, self).__init__(base_dir, create_agent_fn, **kwargs)
    self.batch_size = batch_size

  def _run_one_episode(self):
    # This assumes that everything inside _run_one_episode works on batches,
    # which is risky for future.
    steps_number, total_rewards = super(BatchRunner, self)._run_one_episode()
    return steps_number * self.batch_size, total_rewards

  def _run_one_phase(self, min_steps, statistics, run_mode_str):
    # Mostly copy of parent method.
    step_count = 0
    num_episodes = 0
    sum_returns = 0.

    while step_count < min_steps:
      num_steps, episode_returns = self._run_one_episode()
      for  episode_return in episode_returns:
        statistics.append({
            '{}_episode_lengths'.format(run_mode_str):
                num_steps / self.batch_size,
            '{}_episode_returns'.format(run_mode_str): episode_return
        })
      step_count += num_steps
      sum_returns += sum(episode_returns)
      num_episodes += self.batch_size
      # We use sys.stdout.write instead of tf.logging so as to flush frequently
      # without generating a line break.
      sys.stdout.write('Steps executed: {} '.format(step_count) +
                       'Batch episodes steps: {} '.format(num_steps) +
                       'Returns: {}\r'.format(episode_returns))
      sys.stdout.flush()
    return step_count, sum_returns, num_episodes


class _OutOfGraphReplayBuffer(OutOfGraphReplayBuffer):
  """Replay not sampling artificial_terminal transition.

  Adds to stored tuples 'artificial_done' field (as last ReplayElement).
  When sampling, ignores tuples for which artificial_done is True.

  When adding new attributes check if there are loaded from disk, when using
  load() method.

  Attributes:
      are_terminal_valid: A boolean indicating if newly added terminal
        transitions should be marked as artificially done. Replay data loaded
        from disk will not be overridden.
  """

  def __init__(self, artificial_done, **kwargs):
    extra_storage_types = kwargs.pop("extra_storage_types", None) or []
    extra_storage_types.append(ReplayElement("artificial_done", (), np.uint8))
    super(_OutOfGraphReplayBuffer, self).__init__(
        extra_storage_types=extra_storage_types, **kwargs)
    self._artificial_done = artificial_done

  def is_valid_transition(self, index):
    valid = super(_OutOfGraphReplayBuffer, self).is_valid_transition(index)
    valid &= not self.get_artificial_done_stack(index).any()
    return valid

  def get_artificial_done_stack(self, index):
    return self.get_range(self._store["artificial_done"],
                          index - self._stack_size + 1, index + 1)

  def add(self, observation, action, reward, terminal, *args):
    """Append artificial_done to *args and run parent method."""
    # If this will be a problem for maintenance, we could probably override
    # DQNAgent.add() method instead.
    artificial_done = self._artificial_done and terminal
    args = list(args)
    args.append(artificial_done)
    return super(_OutOfGraphReplayBuffer, self).add(observation, action, reward,
                                                    terminal, *args)

  def load(self, *args, **kwargs):
    # Check that appropriate attributes are not overridden
    are_terminal_valid = self._artificial_done
    super(_OutOfGraphReplayBuffer, self).load(*args, **kwargs)
    assert self._artificial_done == are_terminal_valid


def get_create_agent(agent_kwargs):
  """TODO(): Document."""

  def create_agent(sess, environment, summary_writer=None):
    """Creates a DQN agent.

    Simplified version of `dopamine.discrete_domains.train.create_agent`

    Args:
      sess: a session
      environment: an environment
      summary_writer: a summary writer.

    Returns:
      a DQN agent.
    """
    return _DQNAgent(
        sess=sess,
        num_actions=environment.action_space.n,
        summary_writer=summary_writer,
        tf_device="/gpu:*",
        **agent_kwargs)

  return create_agent


def get_create_env_fun(batch_env_fn, time_limit):
  """TODO(konradczechowski): Add doc-string."""

  def create_env_fun(game_name, sticky_actions=True):
    del game_name, sticky_actions
    batch_env = batch_env_fn(in_graph=False)
    env = FlatBatchEnv(batch_env)
    env = TimeLimit(env, max_episode_steps=time_limit)
    env = ResizeObservation(env)  # pylint: disable=redefined-variable-type
    env = GameOverOnDone(env)
    return env

  return create_env_fun


def _parse_hparams(hparams):
  """TODO(konradczechowski): Add doc-string."""
  prefixes = ["agent_", "optimizer_", "runner_", "replay_buffer_"]
  ret = []

  for prefix in prefixes:
    ret_dict = {}
    for key in hparams.values():
      if prefix in key:
        par_name = key[len(prefix):]
        ret_dict[par_name] = hparams.get(key)
    ret.append(ret_dict)

  return ret


def _get_optimizer(params):
  assert params["class"] == "RMSProp", "RMSProp is the only one supported"
  params.pop("class")
  return tf.train.RMSPropOptimizer(**params)


class DQNLearner(PolicyLearner):
  """Interface for learning dqn implemented in dopamine."""

  def __init__(self, frame_stack_size, base_event_dir, agent_model_dir):
    super(DQNLearner, self).__init__(frame_stack_size, base_event_dir,
                                     agent_model_dir)
    self.completed_iterations = 0

  def _target_iteractions_and_steps(self, num_env_steps, save_continuously,
                                    save_every_steps):

    if save_continuously:
      training_steps_per_iteration = min(num_env_steps, save_every_steps)
      num_iterations_to_do = num_env_steps // training_steps_per_iteration
    else:
      num_iterations_to_do = 1
      training_steps_per_iteration = num_env_steps
    target_iterations = self.completed_iterations + num_iterations_to_do
    return target_iterations, training_steps_per_iteration

  def create_runner(self, env_fn, hparams, target_iterations,
                    training_steps_per_iteration):
    # pylint: disable=unbalanced-tuple-unpacking
    agent_params, optimizer_params, \
    runner_params, replay_buffer_params = _parse_hparams(hparams)
    # pylint: enable=unbalanced-tuple-unpacking
    optimizer = _get_optimizer(optimizer_params)
    agent_params["optimizer"] = optimizer
    agent_params.update(replay_buffer_params)
    create_agent_fn = get_create_agent(agent_params)
    runner = run_experiment.Runner(
        base_dir=self.agent_model_dir,
        create_agent_fn=create_agent_fn,
        create_environment_fn=get_create_env_fun(
            env_fn, time_limit=hparams.time_limit),
        evaluation_steps=0,
        num_iterations=target_iterations,
        training_steps=training_steps_per_iteration,
        **runner_params)
    return runner

  def train(self,
            env_fn,
            hparams,
            simulated,
            save_continuously,
            epoch,
            sampling_temp=1.0,
            num_env_steps=None,
            env_step_multiplier=1,
            eval_env_fn=None,
            report_fn=None):
    # TODO(konradczechowski): evaluation during training (with eval_env_fun)
    del epoch, eval_env_fn, simulated, report_fn
    if num_env_steps is None:
      num_env_steps = hparams.num_frames

    hparams = copy.copy(hparams)
    hparams.set_hparam(
        "agent_epsilon_eval", min(hparams.agent_epsilon_eval * sampling_temp, 1)
    )

    target_iterations, training_steps_per_iteration = \
      self._target_iteractions_and_steps(
          num_env_steps=num_env_steps * env_step_multiplier,
          save_continuously=save_continuously,
          save_every_steps=hparams.save_every_steps,)

    with tf.Graph().as_default():
      runner = self.create_runner(env_fn, hparams, target_iterations,
                                  training_steps_per_iteration)
      runner.run_experiment()

    self.completed_iterations = target_iterations

  def evaluate(self, env_fn, hparams, sampling_temp):
    target_iterations = 0
    training_steps_per_iteration = 0

    hparams = copy.copy(hparams)
    hparams.set_hparam(
        "agent_epsilon_eval", min(hparams.agent_epsilon_eval * sampling_temp, 1)
    )

    create_environment_fn = get_create_env_fun(
        env_fn, time_limit=hparams.time_limit)
    env = create_environment_fn(
        game_name="unused_arg", sticky_actions="unused_arg")

    with tf.Graph().as_default():
      runner = self.create_runner(env_fn, hparams, target_iterations,
                                  training_steps_per_iteration)
      agent = runner._agent  # pylint: disable=protected-access
      del runner
      agent.eval = True

      # TODO(konradczechowski): correct number of episodes, when this will
      # be hparam
      for _ in range(30):
        # Run single episode
        ob = env.reset()
        action = agent.begin_episode(ob)
        done = False
        while not done:
          ob, reward, done, _ = env.step(action)
          action = agent.step(reward, ob)
