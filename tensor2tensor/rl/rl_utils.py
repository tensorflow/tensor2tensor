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
from tensor2tensor.layers import common_video
from tensor2tensor.models.research import rl
from tensor2tensor.rl.dopamine_connector import DQNLearner
from tensor2tensor.rl.envs.simulated_batch_env import PIL_Image
from tensor2tensor.rl.envs.simulated_batch_env import PIL_ImageDraw
from tensor2tensor.rl.envs.simulated_batch_gym_env import SimulatedBatchGymEnv
from tensor2tensor.rl.ppo_learner import PPOLearner
from tensor2tensor.utils import misc_utils
from tensor2tensor.utils import trainer_lib

import tensorflow.compat.v1 as tf


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
      rl_env_max_episode_steps=hparams.eval_rl_env_max_episode_steps,
      env_name=hparams.rl_env_name)
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


def evaluate_world_model(
    real_env, hparams, world_model_dir, debug_video_path,
    split=tf.estimator.ModeKeys.EVAL,
):
  """Evaluate the world model (reward accuracy)."""
  frame_stack_size = hparams.frame_stack_size
  rollout_subsequences = []
  def initial_frame_chooser(batch_size):
    assert batch_size == len(rollout_subsequences)
    return np.stack([
        [frame.observation.decode() for frame in subsequence[:frame_stack_size]]    # pylint: disable=g-complex-comprehension
        for subsequence in rollout_subsequences
    ])

  env_fn = rl.make_simulated_env_fn_from_hparams(
      real_env, hparams, batch_size=hparams.wm_eval_batch_size,
      initial_frame_chooser=initial_frame_chooser, model_dir=world_model_dir
  )
  sim_env = env_fn(in_graph=False)
  subsequence_length = int(
      max(hparams.wm_eval_rollout_ratios) * hparams.simulated_rollout_length
  )
  rollouts = real_env.current_epoch_rollouts(
      split=split,
      minimal_rollout_frames=(subsequence_length + frame_stack_size)
  )

  video_writer = common_video.WholeVideoWriter(
      fps=10, output_path=debug_video_path, file_format="avi"
  )

  reward_accuracies_by_length = {
      int(ratio * hparams.simulated_rollout_length): []
      for ratio in hparams.wm_eval_rollout_ratios
  }
  for _ in range(hparams.wm_eval_num_batches):
    rollout_subsequences[:] = random_rollout_subsequences(
        rollouts, hparams.wm_eval_batch_size,
        subsequence_length + frame_stack_size
    )

    eval_subsequences = [
        subsequence[(frame_stack_size - 1):]
        for subsequence in rollout_subsequences
    ]

    # Check that the initial observation is the same in the real and simulated
    # rollout.
    sim_init_obs = sim_env.reset()
    def decode_real_obs(index):
      return np.stack([
          subsequence[index].observation.decode()
          for subsequence in eval_subsequences  # pylint: disable=cell-var-from-loop
      ])
    real_init_obs = decode_real_obs(0)
    assert np.all(sim_init_obs == real_init_obs)

    debug_frame_batches = []
    def append_debug_frame_batch(sim_obs, real_obs, sim_cum_rews,
                                 real_cum_rews, sim_rews, real_rews):
      """Add a debug frame."""
      rews = [[sim_cum_rews, sim_rews], [real_cum_rews, real_rews]]
      headers = []
      for j in range(len(sim_obs)):
        local_nps = []
        for i in range(2):
          img = PIL_Image().new("RGB", (sim_obs.shape[-2], 11),)
          draw = PIL_ImageDraw().Draw(img)
          draw.text((0, 0), "c:{:3}, r:{:3}".format(int(rews[i][0][j]),
                                                    int(rews[i][1][j])),
                    fill=(255, 0, 0))
          local_nps.append(np.asarray(img))
        local_nps.append(np.zeros_like(local_nps[0]))
        headers.append(np.concatenate(local_nps, axis=1))
      errs = absolute_hinge_difference(sim_obs, real_obs)
      headers = np.stack(headers)
      debug_frame_batches.append(  # pylint: disable=cell-var-from-loop
          np.concatenate([headers,
                          np.concatenate([sim_obs, real_obs, errs], axis=2)],
                         axis=1)
      )
    append_debug_frame_batch(sim_init_obs, real_init_obs,
                             np.zeros(hparams.wm_eval_batch_size),
                             np.zeros(hparams.wm_eval_batch_size),
                             np.zeros(hparams.wm_eval_batch_size),
                             np.zeros(hparams.wm_eval_batch_size))

    (sim_cum_rewards, real_cum_rewards) = (
        np.zeros(hparams.wm_eval_batch_size) for _ in range(2)
    )
    for i in range(subsequence_length):
      actions = [subsequence[i].action for subsequence in eval_subsequences]
      (sim_obs, sim_rewards, _) = sim_env.step(actions)
      sim_cum_rewards += sim_rewards

      real_rewards = np.array([
          subsequence[i + 1].reward for subsequence in eval_subsequences
      ])
      real_cum_rewards += real_rewards
      for (length, reward_accuracies) in six.iteritems(
          reward_accuracies_by_length
      ):
        if i + 1 == length:
          reward_accuracies.append(
              np.sum(sim_cum_rewards == real_cum_rewards) /
              len(real_cum_rewards)
          )

      real_obs = decode_real_obs(i + 1)
      append_debug_frame_batch(sim_obs, real_obs, sim_cum_rewards,
                               real_cum_rewards, sim_rewards, real_rewards)

    for debug_frames in np.stack(debug_frame_batches, axis=1):
      debug_frame = None
      for debug_frame in debug_frames:
        video_writer.write(debug_frame)

      if debug_frame is not None:
        # Append two black frames for aesthetics.
        for _ in range(2):
          video_writer.write(np.zeros_like(debug_frame))

  video_writer.finish_to_disk()

  return {
      "reward_accuracy/at_{}".format(length): np.mean(reward_accuracies)
      for (length, reward_accuracies) in six.iteritems(
          reward_accuracies_by_length
      )
  }


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


def should_apply_max_and_skip_env(hparams):
  """MaxAndSkipEnv doesn't make sense for some games, so omit it if needed."""
  return hparams.game != "tictactoe"


def setup_env(hparams,
              batch_size,
              max_num_noops,
              rl_env_max_episode_steps=-1,
              env_name=None):
  """Setup."""
  if not env_name:
    env_name = full_game_name(hparams.game)

  maxskip_envs = should_apply_max_and_skip_env(hparams)

  env = T2TGymEnv(
      base_env_name=env_name,
      batch_size=batch_size,
      grayscale=hparams.grayscale,
      should_derive_observation_space=hparams
      .rl_should_derive_observation_space,
      resize_width_factor=hparams.resize_width_factor,
      resize_height_factor=hparams.resize_height_factor,
      rl_env_max_episode_steps=rl_env_max_episode_steps,
      max_num_noops=max_num_noops,
      maxskip_envs=maxskip_envs,
      sticky_actions=hparams.sticky_actions
  )
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


def make_initial_frame_chooser(
    real_env, frame_stack_size, simulation_random_starts,
    simulation_flip_first_random_for_beginning,
    split=tf.estimator.ModeKeys.TRAIN,
):
  """Make frame chooser.

  Args:
    real_env: T2TEnv to take initial frames from.
    frame_stack_size (int): Number of consecutive frames to extract.
    simulation_random_starts (bool): Whether to choose frames at random.
    simulation_flip_first_random_for_beginning (bool): Whether to flip the first
      frame stack in every batch for the frames at the beginning.
    split (tf.estimator.ModeKeys or None): Data split to take the frames from,
      None means use all frames.

  Returns:
    Function batch_size -> initial_frames.
  """
  initial_frame_rollouts = real_env.current_epoch_rollouts(
      split=split, minimal_rollout_frames=frame_stack_size,
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
        [frame.observation.decode() for frame in initial_frame_stack]  # pylint: disable=g-complex-comprehension
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
  header = np.copy(np.asarray(img))
  del img
  if bar_color is not None:
    header[0, :, :] = bar_color
  return np.concatenate([header, observation], axis=0)


def run_rollouts(
    env, agent, initial_observations, step_limit=None, discount_factor=1.0,
    log_every_steps=None, video_writers=(), color_bar=False,
    many_rollouts_from_each_env=False
):
  """Runs a batch of rollouts from given initial observations."""
  assert step_limit is not None or not many_rollouts_from_each_env, (
      "When collecting many rollouts from each environment, time limit must "
      "be set."
  )

  num_dones = 0
  first_dones = np.array([False] * env.batch_size)
  observations = initial_observations
  step_index = 0
  cum_rewards = np.zeros(env.batch_size)

  for (video_writer, obs_stack) in zip(video_writers, initial_observations):
    for (i, ob) in enumerate(obs_stack):
      debug_frame = augment_observation(
          ob, reward=0, cum_reward=0, frame_index=(-len(obs_stack) + i + 1),
          bar_color=((0, 255, 0) if color_bar else None)
      )
      video_writer.write(debug_frame)

  def proceed():
    if step_index < step_limit:
      return num_dones < env.batch_size or many_rollouts_from_each_env
    else:
      return False

  while proceed():
    act_kwargs = {}
    if agent.needs_env_state:
      act_kwargs["env_state"] = env.state
    actions = agent.act(observations, **act_kwargs)
    (observations, rewards, dones) = env.step(actions)
    observations = list(observations)
    now_done_indices = []
    for (i, done) in enumerate(dones):
      if done and (not first_dones[i] or many_rollouts_from_each_env):
        now_done_indices.append(i)
        first_dones[i] = True
        num_dones += 1
    if now_done_indices:
      # Unless many_rollouts_from_each_env, reset only envs done the first time
      # in this timestep to ensure that we collect exactly 1 rollout from each
      # env.
      reset_observations = env.reset(now_done_indices)
      for (i, observation) in zip(now_done_indices, reset_observations):
        observations[i] = observation
    observations = np.array(observations)
    cum_rewards[~first_dones] = (
        cum_rewards[~first_dones] * discount_factor + rewards[~first_dones]
    )
    step_index += 1

    for (video_writer, obs_stack, reward, cum_reward, done) in zip(
        video_writers, observations, rewards, cum_rewards, first_dones
    ):
      if done:
        continue
      ob = obs_stack[-1]
      debug_frame = augment_observation(
          ob, reward=reward, cum_reward=cum_reward,
          frame_index=step_index, bar_color=((255, 0, 0) if color_bar else None)
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
  records_own_videos = False

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
  records_own_videos = True

  def __init__(
      self,
      batch_size,
      rollout_agent,
      sim_env,
      wrapper_fn,
      num_rollouts,
      planning_horizon,
      discount_factor=1.0,
      uct_const=0,
      uniform_first_action=True,
      normalizer_window_size=30,
      normalizer_epsilon=0.001,
      video_writers=(),
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
    self._normalizer_window_size = normalizer_window_size
    self._normalizer_epsilon = normalizer_epsilon
    self._video_writers = video_writers
    self._best_mc_values = [[] for _ in range(self.batch_size)]

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
      video_writers = ()
      if planner_index < len(self._video_writers) and batch_index == 0:
        video_writers = (self._video_writers[planner_index],)
      (final_observations, cum_rewards) = run_rollouts(
          self._wrapped_env, self._rollout_agent, initial_observations,
          discount_factor=self._discount_factor,
          step_limit=self._planning_horizon,
          video_writers=video_writers, color_bar=True)
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
      """Choose the best action, update best Monte Carlo values."""
      best_mc_values = self._best_mc_values[planner_index]
      action_probs = self._rollout_agent.action_distribution(
          np.array([observation] * self._rollout_agent.batch_size)
      )[0, :]
      sums_and_counts = run_batches_from(observation, planner_index)

      def monte_carlo_value(action):
        (value_sum, count) = sums_and_counts[action]
        if count > 0:
          mean_value = value_sum / count
        else:
          mean_value = -np.inf
        return mean_value

      mc_values = np.array(
          [monte_carlo_value(action) for action in range(self.action_space.n)]
      )
      best_mc_values.append(mc_values.max())

      normalizer = max(
          np.std(best_mc_values[-self._normalizer_window_size:]),
          self._normalizer_epsilon
      )
      normalized_mc_values = mc_values / normalizer

      uct_bonuses = np.array(
          [self._uct_bonus(sums_and_counts[action][1], action_probs[action])
           for action in range(self.action_space.n)]
      )
      values = normalized_mc_values + uct_bonuses
      return np.argmax(values)

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
              0, self.action_space.n, self._rollout_agent.batch_size + 1
          )
      ])[:self._rollout_agent.batch_size]
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
