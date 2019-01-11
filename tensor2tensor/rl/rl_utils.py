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

"""Utilities for RL training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from gym.spaces import Box
import numpy as np
import six

from tensor2tensor.data_generators.gym_env import T2TGymEnv
from tensor2tensor.layers import common_layers
from tensor2tensor.models.research import rl
from tensor2tensor.rl.dopamine_connector import DQNLearner
from tensor2tensor.rl.envs.simulated_batch_gym_env import SimulatedBatchGymEnv
from tensor2tensor.rl.ppo_learner import PPOLearner
from tensor2tensor.utils import misc_utils
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


def compute_mean_reward(rollouts, clipped):
  """Calculate mean rewards from given epoch."""
  reward_name = "reward" if clipped else "unclipped_reward"
  rewards = []
  for rollout in rollouts:
    if rollout[-1].done:
      rollout_reward = sum(getattr(frame, reward_name) for frame in rollout)
      rewards.append(rollout_reward)
  if rewards:
    mean_rewards = np.mean(rewards)
  else:
    mean_rewards = 0
  return mean_rewards


def get_metric_name(sampling_temp, max_num_noops, clipped):
  return "mean_reward/eval/sampling_temp_{}_max_noops_{}_{}".format(
      sampling_temp, max_num_noops, "clipped" if clipped else "unclipped"
  )


def _eval_fn_with_learner(
    env, hparams, policy_hparams, policy_dir, sampling_temp
):
  env_fn = rl.make_real_env_fn(env)
  learner = LEARNERS[hparams.base_algo](
      hparams.frame_stack_size, base_event_dir=None,
      agent_model_dir=policy_dir, total_num_epochs=1
  )
  learner.evaluate(env_fn, policy_hparams, sampling_temp)


def evaluate_single_config(
    hparams, sampling_temp, max_num_noops, agent_model_dir,
    eval_fn=_eval_fn_with_learner
):
  """Evaluate the PPO agent in the real environment."""
  eval_hparams = trainer_lib.create_hparams(hparams.base_algo_params)
  env = setup_env(
      hparams, batch_size=hparams.eval_batch_size, max_num_noops=max_num_noops,
      rl_env_max_episode_steps=hparams.eval_rl_env_max_episode_steps
  )
  env.start_new_epoch(0)
  eval_fn(env, hparams, eval_hparams, agent_model_dir, sampling_temp)
  rollouts = env.current_epoch_rollouts()
  env.close()

  return tuple(
      compute_mean_reward(rollouts, clipped) for clipped in (True, False)
  )


def evaluate_all_configs(
    hparams, agent_model_dir, eval_fn=_eval_fn_with_learner
):
  """Evaluate the agent with multiple eval configurations."""
  metrics = {}
  # Iterate over all combinations of sampling temperatures and whether to do
  # initial no-ops.
  for sampling_temp in hparams.eval_sampling_temps:
    for max_num_noops in (hparams.eval_max_num_noops, 0):
      scores = evaluate_single_config(
          hparams, sampling_temp, max_num_noops, agent_model_dir, eval_fn
      )
      for (score, clipped) in zip(scores, (True, False)):
        metric_name = get_metric_name(sampling_temp, max_num_noops, clipped)
        metrics[metric_name] = score

  return metrics


def summarize_metrics(eval_metrics_writer, metrics, epoch):
  """Write metrics to summary."""
  for (name, value) in six.iteritems(metrics):
    summary = tf.Summary()
    summary.value.add(tag=name, simple_value=value)
    eval_metrics_writer.add_summary(summary, epoch)
  eval_metrics_writer.flush()


LEARNERS = {
    "ppo": PPOLearner,
    "dqn": DQNLearner,
}


ATARI_GAME_MODE = "NoFrameskip-v4"


def full_game_name(short_name):
  """CamelCase game name with mode suffix.

  Args:
    short_name: snake_case name without mode e.g "crazy_climber"

  Returns:
    full game name e.g. "CrazyClimberNoFrameskip-v4"
  """
  camel_game_name = misc_utils.snakecase_to_camelcase(short_name)
  full_name = camel_game_name + ATARI_GAME_MODE
  return full_name


def setup_env(hparams, batch_size, max_num_noops, rl_env_max_episode_steps=-1):
  """Setup."""
  env_name = full_game_name(hparams.game)

  env = T2TGymEnv(base_env_name=env_name,
                  batch_size=batch_size,
                  grayscale=hparams.grayscale,
                  resize_width_factor=hparams.resize_width_factor,
                  resize_height_factor=hparams.resize_height_factor,
                  rl_env_max_episode_steps=rl_env_max_episode_steps,
                  max_num_noops=max_num_noops, maxskip_envs=True)
  return env


def update_hparams_from_hparams(target_hparams, source_hparams, prefix):
  """Copy a subset of hparams to target_hparams."""
  for (param_name, param_value) in six.iteritems(source_hparams.values()):
    if param_name.startswith(prefix):
      target_hparams.set_hparam(param_name[len(prefix):], param_value)


def random_rollout_subsequences(rollouts, num_subsequences, subsequence_length):
  """Chooses a random frame sequence of given length from a set of rollouts."""
  def choose_subsequence():
    # TODO(koz4k): Weigh rollouts by their lengths so sampling is uniform over
    # frames and not rollouts.
    rollout = random.choice(rollouts)
    try:
      from_index = random.randrange(len(rollout) - subsequence_length + 1)
    except ValueError:
      # Rollout too short; repeat.
      return choose_subsequence()
    return rollout[from_index:(from_index + subsequence_length)]

  return [choose_subsequence() for _ in range(num_subsequences)]


def make_initial_frame_chooser(real_env, frame_stack_size,
                               simulation_random_starts,
                               simulation_flip_first_random_for_beginning):
  """Make frame chooser."""
  initial_frame_rollouts = real_env.current_epoch_rollouts(
      split=tf.estimator.ModeKeys.TRAIN,
      minimal_rollout_frames=frame_stack_size,
  )
  def initial_frame_chooser(batch_size):
    """Frame chooser."""

    deterministic_initial_frames =\
        initial_frame_rollouts[0][:frame_stack_size]
    if not simulation_random_starts:
      # Deterministic starts: repeat first frames from the first rollout.
      initial_frames = [deterministic_initial_frames] * batch_size
    else:
      # Random starts: choose random initial frames from random rollouts.
      initial_frames = random_rollout_subsequences(
          initial_frame_rollouts, batch_size, frame_stack_size
      )
      if simulation_flip_first_random_for_beginning:
        # Flip first entry in the batch for deterministic initial frames.
        initial_frames[0] = deterministic_initial_frames

    return np.stack([
        [frame.observation.decode() for frame in initial_frame_stack]
        for initial_frame_stack in initial_frames
    ])
  return initial_frame_chooser


def absolute_hinge_difference(arr1, arr2, min_diff=10, dtype=np.uint8):
  """Point-wise, hinge loss-like, difference between arrays.

  Args:
    arr1: integer array to compare.
    arr2: integer array to compare.
    min_diff: minimal difference taken into consideration.
    dtype: dtype of returned array.

  Returns:
    array
  """
  diff = np.abs(arr1.astype(np.int) - arr2, dtype=np.int)
  return np.maximum(diff - min_diff, 0).astype(dtype)


def run_rollouts(
    env, agent, initial_observations, step_limit=None, discount_factor=1.0
):
  """Runs a batch of rollouts from given initial observations."""
  num_dones = 0
  first_dones = [False] * env.batch_size
  observations = initial_observations
  step_index = 0
  cum_rewards = 0

  def proceed():
    if step_limit is not None:
      return step_index < step_limit
    else:
      return num_dones < env.batch_size

  while proceed():
    actions = agent.act(observations)
    (observations, rewards, dones) = env.step(actions)
    observations = list(observations)
    now_done_indices = []
    for (i, done) in enumerate(dones):
      if done and not first_dones[i]:
        now_done_indices.append(i)
        first_dones[i] = True
        num_dones += 1
    if now_done_indices:
      # Reset only envs done the first time in this timestep to ensure that
      # we collect exactly 1 rollout from each env.
      reset_observations = env.reset(now_done_indices)
      for (i, observation) in zip(now_done_indices, reset_observations):
        observations[i] = observation
    observations = np.array(observations)
    cum_rewards = cum_rewards * discount_factor + rewards
    step_index += 1

  return (observations, cum_rewards)


class BatchAgent(object):
  """Python API for agents.

  Runs a batch of parallel agents. Operates on Numpy arrays.
  """

  def __init__(self, batch_size, observation_space, action_space):
    self.batch_size = batch_size
    self.observation_space = observation_space
    self.action_space = action_space

  def act(self, observations):
    """Picks actions based on observations.

    Args:
      observations: A batch of observations.

    Returns:
      A batch of actions.
    """
    raise NotImplementedError

  def estimate_value(self, observations):
    """Estimates values of states based on observations.

    Used for temporal-difference planning.

    Args:
      observations: A batch of observations.

    Returns:
      A batch of values.
    """
    raise NotImplementedError


class RandomAgent(BatchAgent):
  """Random agent, sampling actions from the uniform distribution."""

  def act(self, observations):
    return np.array([
        self.action_space.sample() for _ in range(observations.shape[0])
    ])

  def estimate_value(self, observations):
    return np.zeros(observations.shape[0])


class PolicyAgent(BatchAgent):
  """Agent based on a policy network."""

  def __init__(
      self, batch_size, observation_space, action_space, policy_hparams,
      policy_dir, sampling_temp
  ):
    super(PolicyAgent, self).__init__(
        batch_size, observation_space, action_space
    )
    self._sampling_temp = sampling_temp
    with tf.Graph().as_default():
      self._observations_t = tf.placeholder(
          shape=((batch_size,) + self.observation_space.shape),
          dtype=self.observation_space.dtype
      )
      (logits, self._values_t) = rl.get_policy(
          self._observations_t, policy_hparams, self.action_space
      )
      actions = common_layers.sample_with_temperature(logits, sampling_temp)
      self._actions_t = tf.cast(actions, tf.int32)
      model_saver = tf.train.Saver(
          tf.global_variables(policy_hparams.policy_network + "/.*")  # pylint: disable=unexpected-keyword-arg
      )
      self._sess = tf.Session()
      self._sess.run(tf.global_variables_initializer())
      trainer_lib.restore_checkpoint(policy_dir, model_saver, self._sess)

  def _run(self, observations):
    return self._sess.run(
        [self._actions_t, self._values_t],
        feed_dict={self._observations_t: observations}
    )

  def act(self, observations):
    (actions, _) = self._run(observations)
    return actions

  def estimate_value(self, observations):
    (_, values) = self._run(observations)
    return values


class PlannerAgent(BatchAgent):
  """Agent based on temporal difference planning."""

  def __init__(
      self, batch_size, rollout_agent, sim_env, wrapper_fn, planning_horizon,
      discount_factor=1.0
  ):
    super(PlannerAgent, self).__init__(
        batch_size, rollout_agent.observation_space, rollout_agent.action_space
    )
    self._rollout_agent = rollout_agent
    self._sim_env = sim_env
    self._wrapped_env = wrapper_fn(sim_env)
    self._discount_factor = discount_factor
    self._planning_horizon = planning_horizon

  def act(self, observations):
    def run_batch_from(observation, action):
      self._sim_env.initial_frames = np.array(
          [observation] * self._sim_env.batch_size
      )
      self._wrapped_env.reset()
      (initial_observations, initial_rewards, _) = self._wrapped_env.step(
          np.array([action] * self._wrapped_env.batch_size)
      )
      (final_observations, cum_rewards) = run_rollouts(
          self._wrapped_env, self._rollout_agent, initial_observations,
          discount_factor=self._discount_factor,
          step_limit=self._planning_horizon
      )
      values = self._rollout_agent.estimate_value(final_observations)
      total_values = (
          initial_rewards + self._discount_factor * cum_rewards +
          self._discount_factor ** (self._planning_horizon + 1) * values
      )
      return total_values.mean()

    def choose_best_action(observation):
      return max(
          range(self.action_space.n),
          key=(lambda action: run_batch_from(observation, action))
      )

    return np.array(list(map(choose_best_action, observations)))


# TODO(koz4k): Unify interfaces of batch envs.
class BatchWrapper(object):
  """Base class for batch env wrappers."""

  def __init__(self, env):
    self.env = env
    self.batch_size = env.batch_size
    self.observation_space = env.observation_space
    self.action_space = env.action_space
    self.reward_range = env.reward_range

  def reset(self, indices=None):
    return self.env.reset(indices)

  def step(self, actions):
    return self.env.step(actions)

  def close(self):
    self.env.close()


class BatchStackWrapper(BatchWrapper):
  """Out-of-graph batch stack wrapper."""

  def __init__(self, env, stack_size):
    super(BatchStackWrapper, self).__init__(env)
    self.stack_size = stack_size
    inner_space = env.observation_space
    self.observation_space = Box(
        low=np.array([inner_space.low] * self.stack_size),
        high=np.array([inner_space.high] * self.stack_size),
        dtype=inner_space.dtype,
    )
    self._history_buffer = np.zeros(
        (self.batch_size,) + self.observation_space.shape,
        dtype=inner_space.dtype
    )

  def reset(self, indices=None):
    if indices is None:
      indices = range(self.batch_size)

    observations = self.env.reset(indices)
    for (index, observation) in zip(indices, observations):
      self._history_buffer[index, ...] = [observation] * self.stack_size
    return self._history_buffer

  def step(self, actions):
    (observations, rewards, dones) = self.env.step(actions)
    self._history_buffer = np.roll(self._history_buffer, shift=-1, axis=1)
    self._history_buffer[:, -1, ...] = observations
    return (self._history_buffer, rewards, dones)


class SimulatedBatchGymEnvWithFixedInitialFrames(BatchWrapper):
  """Wrapper for SimulatedBatchGymEnv that allows to fix initial frames."""

  def __init__(self, *args, **kwargs):
    self.initial_frames = None
    def initial_frame_chooser(batch_size):
      assert batch_size == self.initial_frames.shape[0]
      return self.initial_frames
    env = SimulatedBatchGymEnv(
        *args, initial_frame_chooser=initial_frame_chooser, **kwargs
    )
    super(SimulatedBatchGymEnvWithFixedInitialFrames, self).__init__(env)
