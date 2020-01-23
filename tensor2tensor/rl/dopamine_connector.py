# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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
from dopamine.agents.rainbow import rainbow_agent
from dopamine.replay_memory import circular_replay_buffer
from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer
from dopamine.replay_memory.circular_replay_buffer import ReplayElement
from dopamine.replay_memory.prioritized_replay_buffer import OutOfGraphPrioritizedReplayBuffer
from dopamine.replay_memory.prioritized_replay_buffer import WrappedPrioritizedReplayBuffer
import numpy as np

from tensor2tensor.rl.policy_learner import PolicyLearner
import tensorflow.compat.v1 as tf

# pylint: disable=g-import-not-at-top
# pylint: disable=ungrouped-imports
try:
  import cv2
except ImportError:
  cv2 = None

try:
  from dopamine.discrete_domains import run_experiment
except ImportError:
  run_experiment = None

# pylint: enable=g-import-not-at-top
# pylint: enable=ungrouped-imports

# TODO(rlmb): Vanilla DQN and Rainbow have a lot of common code. We will want
#  to remove Vanilla DQN and only have Rainbow. To do so one needs to remove
#  following:
#    * _DQNAgent
#    * BatchDQNAgent
#    * _OutOfGraphReplayBuffer
#    * "if" clause in create_agent()
#    * parameter "agent_type" from dqn_atari_base() hparams and possibly other
#      rlmb dqn hparams sets
#  If we want to keep both Vanilla DQN and Rainbow, larger refactor is required.


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
  """Batch agent for DQN.

  Episodes are stored on done.

  Assumes that all rollouts in batch would end at the same moment.
  """

  def __init__(self, env_batch_size, *args, **kwargs):
    super(BatchDQNAgent, self).__init__(*args, **kwargs)
    self.env_batch_size = env_batch_size
    obs_size = dqn_agent.NATURE_DQN_OBSERVATION_SHAPE
    state_shape = [self.env_batch_size, obs_size[0], obs_size[1],
                   dqn_agent.NATURE_DQN_STACK_SIZE]
    self.state_batch = np.zeros(state_shape)
    self.state = None  # assure it will be not used
    self._observation = None  # assure it will be not used
    self.reset_current_rollouts()

  def reset_current_rollouts(self):
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
    self.reset_current_rollouts()

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
      self._update_current_rollouts(
          self._observation_batch, self.action, reward,
          [True] * self.env_batch_size)
      self._store_current_rollouts()

  def _select_action(self):
    epsilon = self.epsilon_eval
    if not self.eval_mode:
      epsilon = self.epsilon_fn(
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


class _OutOfGraphReplayBuffer(OutOfGraphReplayBuffer):
  """Replay not sampling artificial_terminal transition.

  Adds to stored tuples "artificial_done" field (as last ReplayElement).
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


class _WrappedPrioritizedReplayBuffer(WrappedPrioritizedReplayBuffer):
  """Allows to pass out-of-graph-replay-buffer via wrapped_memory."""

  def __init__(self, wrapped_memory, batch_size, use_staging):
    self.batch_size = batch_size
    self.memory = wrapped_memory
    self.create_sampling_ops(use_staging)


class _RainbowAgent(rainbow_agent.RainbowAgent):
  """Modify dopamine DQNAgent to match our needs.

  Allow passing batch_size and replay_capacity to ReplayBuffer, allow not using
  (some of) terminal episode transitions in training.
  """

  def __init__(self, replay_capacity, buffer_batch_size,
               generates_trainable_dones, **kwargs):
    self._replay_capacity = replay_capacity
    self._buffer_batch_size = buffer_batch_size
    self._generates_trainable_dones = generates_trainable_dones
    super(_RainbowAgent, self).__init__(**kwargs)

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

    replay_memory = _OutOfGraphPrioritizedReplayBuffer(
        artificial_done=not self._generates_trainable_dones,
        **replay_buffer_kwargs)

    return _WrappedPrioritizedReplayBuffer(
        wrapped_memory=replay_memory,
        use_staging=use_staging, batch_size=self._buffer_batch_size)
    # **replay_buffer_kwargs)


class BatchRainbowAgent(_RainbowAgent):
  """Batch agent for DQN.

  Episodes are stored on done.

  Assumes that all rollouts in batch would end at the same moment.
  """

  def __init__(self, env_batch_size, *args, **kwargs):
    super(BatchRainbowAgent, self).__init__(*args, **kwargs)
    self.env_batch_size = env_batch_size
    obs_size = dqn_agent.NATURE_DQN_OBSERVATION_SHAPE
    state_shape = [self.env_batch_size, obs_size[0], obs_size[1],
                   dqn_agent.NATURE_DQN_STACK_SIZE]
    self.state_batch = np.zeros(state_shape)
    self.state = None  # assure it will be not used
    self._observation = None  # assure it will be not used
    self.reset_current_rollouts()

  def reset_current_rollouts(self):
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
    self.reset_current_rollouts()

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
      self._update_current_rollouts(
          self._observation_batch, self.action, reward,
          [True] * self.env_batch_size)
      self._store_current_rollouts()

  def _select_action(self):
    epsilon = self.epsilon_eval
    if not self.eval_mode:
      epsilon = self.epsilon_fn(
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
  """Run a batch of environments.

  Assumes that all environments would end at the same moment.
  """

  def __init__(self, base_dir, create_agent_fn, **kwargs):
    super(BatchRunner, self).__init__(base_dir, create_agent_fn, **kwargs)
    self.batch_size = self._environment.batch_size

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
      for episode_return in episode_returns:
        statistics.append({
            "{}_episode_lengths".format(run_mode_str):
                num_steps / self.batch_size,
            "{}_episode_returns".format(run_mode_str): episode_return
        })
      step_count += num_steps
      sum_returns += sum(episode_returns)
      num_episodes += self.batch_size
      # We use sys.stdout.write instead of tf.logging so as to flush frequently
      # without generating a line break.
      sys.stdout.write("Steps executed: {} ".format(step_count) +
                       "Batch episodes steps: {} ".format(num_steps) +
                       "Returns: {}\r".format(episode_returns))
      sys.stdout.flush()
    return step_count, sum_returns, num_episodes

  def close(self):
    self._environment.close()


class _OutOfGraphPrioritizedReplayBuffer(OutOfGraphPrioritizedReplayBuffer):
  """Replay not sampling artificial_terminal transition.

  Adds to stored tuples "artificial_done" field (as last ReplayElement).
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
    msg = "Other extra_storage_types aren't currently supported for this class."
    assert not extra_storage_types, msg
    extra_storage_types.append(ReplayElement("artificial_done", (), np.uint8))
    super(_OutOfGraphPrioritizedReplayBuffer, self).__init__(
        extra_storage_types=extra_storage_types, **kwargs)
    self._artificial_done = artificial_done

  def is_valid_transition(self, index):
    valid = super(_OutOfGraphPrioritizedReplayBuffer,
                  self).is_valid_transition(index)
    if valid:
      valid = not self.get_artificial_done_stack(index).any()
    return valid

  def get_artificial_done_stack(self, index):
    return self.get_range(self._store["artificial_done"],
                          index - self._stack_size + 1, index + 1)

  def add(self, observation, action, reward, terminal, priority):
    """Infer artificial_done and call parent method."""
    # If this will be a problem for maintenance, we could probably override
    # DQNAgent.add() method instead.
    if not isinstance(priority, (float, np.floating)):
      raise ValueError("priority should be float, got type {}"
                       .format(type(priority)))
    artificial_done = self._artificial_done and terminal
    return super(_OutOfGraphPrioritizedReplayBuffer, self).add(
        observation, action, reward, terminal, artificial_done, priority
    )

  def load(self, *args, **kwargs):
    # Check that appropriate attributes are not overridden
    are_terminal_valid = self._artificial_done
    super(_OutOfGraphPrioritizedReplayBuffer, self).load(*args, **kwargs)
    assert self._artificial_done == are_terminal_valid


def get_create_agent(agent_kwargs):
  """Factory for dopamine agent initialization.

  Args:
    agent_kwargs: dict of BatchDQNAgent parameters

  Returns:
    Function(sess, environment, summary_writer) -> BatchDQNAgent instance.
  """
  agent_kwargs = copy.deepcopy(agent_kwargs)
  agent_type = agent_kwargs.pop("type")

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
    if agent_type == "Rainbow":
      return BatchRainbowAgent(
          env_batch_size=environment.batch_size,
          sess=sess,
          num_actions=environment.action_space.n,
          summary_writer=summary_writer,
          tf_device="/gpu:*",
          **agent_kwargs)
    elif agent_type == "VanillaDQN":
      return BatchDQNAgent(
          env_batch_size=environment.batch_size,
          sess=sess,
          num_actions=environment.action_space.n,
          summary_writer=summary_writer,
          tf_device="/gpu:*",
          **agent_kwargs)
    else:
      raise ValueError("Unknown agent_type {}".format(agent_type))

  return create_agent


class ResizeBatchObservation(object):
  """Wrapper resizing observations for batched environment.

  Dopamine also uses cv2.resize(..., interpolation=cv2.INTER_AREA).

  Attributes:
    batch_env: batched environment
    batch_size: batch size
    action_space: the action space
    size: size of width and height for returned observations
  """

  def __init__(self, batch_env, size=84):
    self.size = size
    self.batch_env = batch_env

  def observation(self, frames):
    if not cv2:
      return frames
    return np.array([cv2.resize(
        frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
                     for frame in frames])

  def step(self, actions):
    obs, rewards, dones = self.batch_env.step(actions)
    obs = self.observation(obs)
    return obs, rewards, dones

  def reset(self, *args, **kwargs):
    return self.observation(self.batch_env.reset(*args, **kwargs))

  @property
  def action_space(self):
    return self.batch_env.action_space

  @property
  def batch_size(self):
    return self.batch_env.batch_size

  def close(self):
    self.batch_env.close()


class DopamineBatchEnv(object):
  """Batch of environments.

  Assumes that all given environments finishes at the same time.

  Observations and rewards are returned as batches (arrays). Done is returned
  as single boolean.
  """

  def __init__(self, batch_env, max_episode_steps):
    self.batch_env = batch_env
    self._max_episode_steps = max_episode_steps
    self.game_over = None
    self._elapsed_steps = 0

  def reset(self):
    self.game_over = False
    self._elapsed_steps = 0
    return np.array(self.batch_env.reset())

  def step(self, actions):
    """Step."""
    self._elapsed_steps += 1
    obs, rewards, dones = \
        [np.array(r) for r in self.batch_env.step(actions)]
    if self._elapsed_steps > self._max_episode_steps:
      done = True
      if self._elapsed_steps > self._max_episode_steps + 1:
        rewards.fill(0)
    else:
      done = dones[0]
      assert np.all(done == dones), ("Current modifications of Dopamine "
                                     "require same number of steps for each "
                                     "environment in batch")
      del dones

    self.game_over = done
    return obs, rewards, done, {}

  def render(self, mode):
    pass

  def close(self):
    self.batch_env.close()

  @property
  def action_space(self):
    return self.batch_env.action_space

  @property
  def batch_size(self):
    return self.batch_env.batch_size


class PaddedTrajectoriesEnv(DopamineBatchEnv):
  """Pad finished episodes with zeros.

  Allow episodes in batch to end on different timesteps, return zero
  observations and rewards for finished ones. Return done=True when all
  episodes are finished.

  Note that output of this class might be misleading - the agent/evaluator
  which uses this environment gets false information about when episodes have
  ended. This class is used for informal check of Batched dopamine
  implementation in model-free pipeline.
  """

  def reset(self):
    self.done_envs = [False] * self.batch_size
    self.game_over = False
    self._elapsed_steps = 0
    return np.array(self.batch_env.reset())

  def step(self, actions):
    if any(self.done_envs):
      print("Warning, some environments already ended, using mocked data.")

    self._elapsed_steps += 1
    obs, rewards, dones = \
        [np.array(r) for r in self.batch_env.step(actions)]
    for i, ignore in enumerate(self.done_envs):
      if ignore:
        obs[i] = np.zeros(obs[i].shape, dtype=obs.dtype)
        rewards[i] = 0
      if dones[i]:
        self.batch_env.reset([i])
        self.done_envs[i] = True

    all_done = all(self.done_envs)

    if self._elapsed_steps > self._max_episode_steps:
      all_done = True
      if self._elapsed_steps > self._max_episode_steps + 1:
        rewards.fill(0)

    self.game_over = all_done
    return obs, rewards, all_done, {}


def get_create_batch_env_fun(batch_env_fn, time_limit):
  """Factory for dopamine environment initialization function.

  Args:
    batch_env_fn: function(in_graph: bool) -> batch environment.
    time_limit: time steps limit for environment.

  Returns:
    function (with optional, unused parameters) initializing environment.
  """

  def create_env_fun(game_name=None, sticky_actions=None):
    del game_name, sticky_actions
    batch_env = batch_env_fn(in_graph=False)
    batch_env = ResizeBatchObservation(batch_env)  # pylint: disable=redefined-variable-type
    batch_env = DopamineBatchEnv(batch_env, max_episode_steps=time_limit)
    return batch_env

  return create_env_fun


def _parse_hparams(hparams):
  """Split hparams, based on key prefixes.

  Args:
    hparams: hyperparameters

  Returns:
    Tuple of hparams for respectably: agent, optimizer, runner, replay_buffer.
  """
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

  def __init__(self, frame_stack_size, base_event_dir, agent_model_dir,
               total_num_epochs, **kwargs):
    super(DQNLearner, self).__init__(
        frame_stack_size, base_event_dir, agent_model_dir, total_num_epochs)
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
    runner = BatchRunner(
        base_dir=self.agent_model_dir,
        create_agent_fn=create_agent_fn,
        create_environment_fn=get_create_batch_env_fun(
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
            report_fn=None,
            model_save_fn=None):
    # TODO(konradczechowski): evaluation during training (with eval_env_fun)
    del epoch, eval_env_fn, simulated, report_fn, model_save_fn
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
          save_every_steps=hparams.save_every_steps)

    with tf.Graph().as_default():
      runner = self.create_runner(env_fn, hparams, target_iterations,
                                  training_steps_per_iteration)
      runner.run_experiment()
      runner.close()
    self.completed_iterations = target_iterations

  def evaluate(self, env_fn, hparams, sampling_temp):
    target_iterations = 0
    training_steps_per_iteration = 0

    hparams = copy.copy(hparams)
    hparams.set_hparam(
        "agent_epsilon_eval", min(hparams.agent_epsilon_eval * sampling_temp, 1)
    )

    create_environment_fn = get_create_batch_env_fun(
        env_fn, time_limit=hparams.time_limit)
    env = create_environment_fn(
        game_name="unused_arg", sticky_actions="unused_arg")

    with tf.Graph().as_default():
      runner = self.create_runner(env_fn, hparams, target_iterations,
                                  training_steps_per_iteration)
      assert runner.batch_size == 1
      agent = runner._agent  # pylint: disable=protected-access
      runner.close()
      del runner
      agent.eval_mode = True

      for _ in range(hparams.eval_episodes_num):
        # Run single episode
        ob = env.reset()
        action = agent.begin_episode(ob)
        done = False
        while not done:
          ob, reward, done, _ = env.step(action)
          action = agent.step(reward, ob)
