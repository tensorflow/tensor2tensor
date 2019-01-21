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

import copy
import math
import random

from gym.spaces import Box
import numpy as np
import six

from tensor2tensor.data_generators.gym_env import T2TGymEnv
from tensor2tensor.layers import common_layers
from tensor2tensor.models.research import rl
from tensor2tensor.rl.dopamine_connector import DQNLearner
from tensor2tensor.rl.envs.simulated_batch_env import PIL_Image
from tensor2tensor.rl.envs.simulated_batch_env import PIL_ImageDraw
from tensor2tensor.rl.envs.simulated_batch_gym_env import SimulatedBatchGymEnv
from tensor2tensor.rl.ppo_learner import PPOLearner
from tensor2tensor.utils import misc_utils
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


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
  tf.logging.info("Evaluating metric %s", get_metric_name(
      sampling_temp, max_num_noops, clipped=False
  ))
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
    # Iterate over a set so if eval_max_num_noops == 0 then it's 1 iteration.
    for max_num_noops in set([hparams.eval_max_num_noops, 0]):
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


# TODO(koz4k): Use this function in player and all debug videos.
def augment_observation(
    observation, reward, cum_reward, frame_index, bar_color=None,
    header_height=27
):
  """Augments an observation with debug info."""
  img = PIL_Image().new(
      "RGB", (observation.shape[1], header_height,)
  )
  draw = PIL_ImageDraw().Draw(img)
  draw.text(
      (1, 0), "c:{:3}, r:{:3}".format(int(cum_reward), int(reward)),
      fill=(255, 0, 0)
  )
  draw.text(
      (1, 15), "f:{:3}".format(int(frame_index)),
      fill=(255, 0, 0)
  )
  header = np.asarray(img)
  del img
  header.setflags(write=1)
  if bar_color is not None:
    header[0, :, :] = bar_color
  return np.concatenate([header, observation], axis=0)


def run_rollouts(
    env, agent, initial_observations, step_limit=None, discount_factor=1.0,
    log_every_steps=None, video_writer=None
):
  """Runs a batch of rollouts from given initial observations."""
  num_dones = 0
  first_dones = [False] * env.batch_size
  observations = initial_observations
  step_index = 0
  cum_rewards = 0

  if video_writer is not None:
    obs_stack = initial_observations[0]
    for (i, ob) in enumerate(obs_stack):
      debug_frame = augment_observation(
          ob, reward=0, cum_reward=0, frame_index=(-len(obs_stack) + i + 1),
          bar_color=(0, 255, 0)
      )
      video_writer.write(debug_frame)

  def proceed():
    if step_limit is not None:
      return step_index < step_limit
    else:
      return num_dones < env.batch_size

  while proceed():
    act_kwargs = {}
    if agent.needs_env_state:
      act_kwargs["env_state"] = env.state
    actions = agent.act(observations, **act_kwargs)
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

    if video_writer is not None:
      ob = observations[0, -1]
      debug_frame = augment_observation(
          ob, reward=rewards[0], cum_reward=cum_rewards[0],
          frame_index=step_index, bar_color=(255, 0, 0)
      )
      video_writer.write(debug_frame)

    # TODO(afrozm): Clean this up with tf.logging.log_every_n
    if log_every_steps is not None and step_index % log_every_steps == 0:
      tf.logging.info("Step %d, mean_score: %f", step_index, cum_rewards.mean())

  return (observations, cum_rewards)


class BatchAgent(object):
  """Python API for agents.

  Runs a batch of parallel agents. Operates on Numpy arrays.
  """

  needs_env_state = False

  def __init__(self, batch_size, observation_space, action_space):
    self.batch_size = batch_size
    self.observation_space = observation_space
    self.action_space = action_space

  def act(self, observations, env_state=None):
    """Picks actions based on observations.

    Args:
      observations: A batch of observations.
      env_state: State.

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

  def action_distribution(self, observations):
    """Calculates action distribution based on observations.

    Used for temporal-difference planning.

    Args:
      observations: A batch of observations.

    Returns:
      A batch of action probabilities.
    """
    raise NotImplementedError


class RandomAgent(BatchAgent):
  """Random agent, sampling actions from the uniform distribution."""

  def act(self, observations, env_state=None):
    del env_state
    return np.array([
        self.action_space.sample() for _ in range(observations.shape[0])
    ])

  def estimate_value(self, observations):
    return np.zeros(observations.shape[0])

  def action_distribution(self, observations):
    return np.full(
        (observations.shape[0], self.action_space.n), 1.0 / self.action_space.n
    )


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
      self._probs_t = tf.nn.softmax(logits / sampling_temp)
      self._actions_t = tf.cast(actions, tf.int32)
      model_saver = tf.train.Saver(
          tf.global_variables(policy_hparams.policy_network + "/.*")  # pylint: disable=unexpected-keyword-arg
      )
      self._sess = tf.Session()
      self._sess.run(tf.global_variables_initializer())
      trainer_lib.restore_checkpoint(policy_dir, model_saver, self._sess)

  def _run(self, observations):
    return self._sess.run(
        [self._actions_t, self._values_t, self._probs_t],
        feed_dict={self._observations_t: observations}
    )

  def act(self, observations, env_state=None):
    del env_state
    (actions, _, _) = self._run(observations)
    return actions

  def estimate_value(self, observations):
    (_, values, _) = self._run(observations)
    return values

  def action_distribution(self, observations):
    (_, _, probs) = self._run(observations)
    return probs


class PlannerAgent(BatchAgent):
  """Agent based on temporal difference planning."""

  needs_env_state = True

  def __init__(
      self, batch_size, rollout_agent, sim_env, wrapper_fn, num_rollouts,
      planning_horizon, discount_factor=1.0, uct_const=0,
      uniform_first_action=True, video_writer=None
  ):
    super(PlannerAgent, self).__init__(
        batch_size, rollout_agent.observation_space, rollout_agent.action_space
    )
    self._rollout_agent = rollout_agent
    self._sim_env = sim_env
    self._wrapped_env = wrapper_fn(sim_env)
    self._num_rollouts = num_rollouts
    self._num_batches = num_rollouts // rollout_agent.batch_size
    self._discount_factor = discount_factor
    self._planning_horizon = planning_horizon
    self._uct_const = uct_const
    self._uniform_first_action = uniform_first_action
    self._video_writer = video_writer

  def act(self, observations, env_state=None):
    def run_batch_from(observation, planner_index, batch_index):
      """Run a batch of actions."""
      repeated_observation = np.array(
          [observation] * self._wrapped_env.batch_size
      )
      actions = self._get_first_actions(repeated_observation)
      self._wrapped_env.set_initial_state(
          initial_state=[
              copy.deepcopy(env_state[planner_index])
              for _ in range(self._sim_env.batch_size)
          ],
          initial_frames=repeated_observation
      )
      self._wrapped_env.reset()
      (initial_observations, initial_rewards, _) = self._wrapped_env.step(
          actions
      )
      writer = None
      if planner_index == 0 and batch_index == 0:
        writer = self._video_writer
      (final_observations, cum_rewards) = run_rollouts(
          self._wrapped_env, self._rollout_agent, initial_observations,
          discount_factor=self._discount_factor,
          step_limit=self._planning_horizon,
          video_writer=writer)
      values = self._rollout_agent.estimate_value(final_observations)
      total_values = (
          initial_rewards + self._discount_factor * cum_rewards +
          self._discount_factor ** (self._planning_horizon + 1) * values
      )
      return list(zip(actions, total_values))

    def run_batches_from(observation, planner_index):
      sums = {a: 0 for a in range(self.action_space.n)}
      counts = copy.copy(sums)
      for i in range(self._num_batches):
        for (action, total_value) in run_batch_from(
            observation, planner_index, i
        ):
          sums[action] += total_value
          counts[action] += 1
      return {a: (sums[a], counts[a]) for a in sums}

    def choose_best_action(observation, planner_index):
      """Choose the best action."""
      action_probs = self._rollout_agent.action_distribution(
          np.array([observation] * self._rollout_agent.batch_size)
      )[0, :]
      sums_and_counts = run_batches_from(observation, planner_index)

      def uct(action):
        (value_sum, count) = sums_and_counts[action]
        if count > 0:
          mean_value = value_sum / count
        else:
          mean_value = 0
        return mean_value + self._uct_bonus(
            count, action_probs[action]
        )

      return max(range(self.action_space.n), key=uct)

    return np.array([
        choose_best_action(observation, i)
        for (i, observation) in enumerate(observations)
    ])

  def _uct_bonus(self, count, prob):
    return self._uct_const * prob * math.sqrt(
        math.log(self._num_rollouts) / (1 + count)
    )

  def _get_first_actions(self, observations):
    if self._uniform_first_action:
      return np.array([
          int(x) for x in np.linspace(
              0, self.action_space.n, self._num_rollouts + 1
          )
      ])[:self._num_rollouts]
    else:
      return list(sorted(self._rollout_agent.act(observations)))


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
  """Out-of-graph batch stack wrapper.

  Its behavior is consistent with tf_atari_wrappers.StackWrapper.
  """

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
    self._initial_frames = None

  @property
  def state(self):
    """Gets the current state."""
    return self.env.state

  def set_initial_state(self, initial_state, initial_frames):
    """Sets the state that will be used on next reset."""
    self.env.set_initial_state(initial_state, initial_frames)
    self._initial_frames = initial_frames

  def reset(self, indices=None):
    if indices is None:
      indices = range(self.batch_size)

    observations = self.env.reset(indices)
    try:
      # If we wrap the simulated env, take the initial frames from there.
      assert self.env.initial_frames.shape[1] == self.stack_size
      self._history_buffer[...] = self.env.initial_frames
    except AttributeError:
      # Otherwise, check if set_initial_state was called and we can take the
      # frames from there.
      if self._initial_frames is not None:
        for (index, observation) in zip(indices, observations):
          assert (self._initial_frames[index, -1, ...] == observation).all()
          self._history_buffer[index, ...] = self._initial_frames[index, ...]
      else:
        # Otherwise, repeat the first observation stack_size times.
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

  @property
  def state(self):
    """Gets the current state."""
    return [None] * self.batch_size

  def set_initial_state(self, initial_state, initial_frames):
    """Sets the state that will be used on next reset."""
    del initial_state
    self.initial_frames = initial_frames
