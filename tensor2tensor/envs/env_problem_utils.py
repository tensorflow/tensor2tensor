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

"""Utilities to deal with EnvProblem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import time
import numpy as np

from tensor2tensor.envs import gym_env_problem
from tensor2tensor.envs import rendered_env_problem
from tensor2tensor.rl import gym_utils


def done_indices(dones):
  """Calculates the indices where dones has True."""
  return np.argwhere(dones).squeeze(axis=1)


def play_env_problem_randomly(env_problem, num_steps):
  """Plays the env problem by randomly sampling actions for `num_steps`."""
  # Reset all environments.
  env_problem.reset()

  # Play all environments, sampling random actions each time.
  for _ in range(num_steps):
    # Sample batch_size actions from the action space and stack them.
    actions = np.stack([
        env_problem.action_space.sample() for _ in range(env_problem.batch_size)
    ])

    # Execute actions, observations are stored in `env_problem`.
    _, _, dones, _ = env_problem.step(actions)

    # Get the indices where we are done and reset those.
    env_problem.reset(indices=done_indices(dones))


def play_env_problem_with_policy(env,
                                 policy_fun,
                                 num_trajectories=1,
                                 max_timestep=None,
                                 reset=True,
                                 state=None,
                                 rng=None,
                                 temperature=1.0,
                                 boundary=32,
                                 len_history_for_policy=32,
                                 num_to_keep=1):
  """Plays the given env with the policy function to collect trajectories.

  Args:
    env: environment object, should be a subclass of env_problem.EnvProblem.
    policy_fun: callable, taking in observations((B, T) + OBS) and returning
      back log-probabilities (B, T, A).
    num_trajectories: int, number of trajectories to collect.
    max_timestep: int or None, if not None or a negative number, we cut any
      trajectory that exceeds this time put it in the completed bin, and *dont*
      reset the env.
    reset: bool, true if we want to reset the envs. The envs are also reset if
      max_max_timestep is None or < 0.
    state: the state for `policy_fn`.
    rng: jax rng, splittable.
    temperature: float, temperature used in Gumbel sampling.
    boundary: int, pad the sequences to the multiples of this number.
    len_history_for_policy: int or None, the maximum history to keep for
      applying the policy on. If None, use the whole history.
    num_to_keep: int, while truncating trajectory how many time-steps to keep.

  Returns:
    A tuple, (trajectories, number of completed trajectories). Where
    trajectories is a list of triples of (observation, action, reward) ndarrays.
  """

  def gumbel_sample(log_probs):
    """Gumbel sampling."""
    u = np.random.uniform(low=1e-6, high=1.0 - 1e-6, size=log_probs.shape)
    g = -np.log(-np.log(u))
    return np.argmax((log_probs / temperature) + g, axis=1)

  # We need to reset all environments, if we're coming here the first time.
  if reset or max_timestep is None or max_timestep <= 0:
    env.reset()
  else:
    # Clear completed trajectories held internally.
    env.trajectories.clear_completed_trajectories()

  num_done_trajectories = 0

  policy_application_total_time = 0
  env_actions_total_time = 0
  bare_env_run_time = 0
  while env.trajectories.num_completed_trajectories < num_trajectories:
    # Get all the observations for all the active trajectories.
    # Shape is (B, T) + OBS
    # Bucket on whatever length is needed.
    padded_observations, lengths = env.trajectories.observations_np(
        boundary=boundary,
        len_history_for_policy=len_history_for_policy)

    B, T = padded_observations.shape[:2]  # pylint: disable=invalid-name

    assert B == env.batch_size
    assert (B,) == lengths.shape

    t1 = time.time()
    log_prob_actions, value_predictions, state, rng = policy_fun(
        padded_observations, state=state, rng=rng)
    policy_application_total_time += (time.time() - t1)

    assert (B, T) == log_prob_actions.shape[:2]
    A = log_prob_actions.shape[2]  # pylint: disable=invalid-name

    # We need the log_probs of those actions that correspond to the last actual
    # time-step.
    index = lengths - 1  # Since we want to index using lengths.
    log_probs = log_prob_actions[np.arange(B)[:, None], index[:, None],
                                 np.arange(A)]
    value_preds = value_predictions[np.arange(B)[:, None], index[:, None],
                                    np.arange(1)]
    assert (B, A) == log_probs.shape, \
        "B=%d, A=%d, log_probs.shape=%s" % (B, A, log_probs.shape)
    assert (B, 1) == value_preds.shape, \
        "B=%d, value_preds.shape=%s" % (B, value_preds.shape)

    actions = gumbel_sample(log_probs)

    # Step through the env.
    t1 = time.time()
    _, _, dones, env_infos = env.step(
        actions,
        infos={
            "log_prob_actions": log_probs,
            "value_predictions": value_preds
        })
    env_actions_total_time += (time.time() - t1)
    bare_env_run_time += sum(
        info["__bare_env_run_time__"] for info in env_infos)

    # Count the number of done trajectories, the others could just have been
    # truncated.
    num_done_trajectories += np.sum(dones)

    # Get the indices where we are done ...
    done_idxs = done_indices(dones)

    # ... and reset those.
    t1 = time.time()
    if done_idxs.size:
      env.reset(indices=done_idxs)
    env_actions_total_time += (time.time() - t1)

    if max_timestep is None or max_timestep < 1:
      continue

    # Are there any trajectories that have exceeded the time-limit we want.
    lengths = env.trajectories.trajectory_lengths
    exceeded_time_limit_idxs = done_indices(lengths > max_timestep)

    # If so, reset these as well.
    t1 = time.time()
    if exceeded_time_limit_idxs.size:
      # This just cuts the trajectory, doesn't reset the env, so it continues
      # from where it left off.
      env.truncate(indices=exceeded_time_limit_idxs, num_to_keep=num_to_keep)
    env_actions_total_time += (time.time() - t1)

  # We have the trajectories we need, return a list of triples:
  # (observations, actions, rewards)
  completed_trajectories = []
  for trajectory in env.trajectories.completed_trajectories[:num_trajectories]:
    completed_trajectories.append(trajectory.as_numpy)

  timing_info = {
      "trajectory_collection/policy_application": policy_application_total_time,
      "trajectory_collection/env_actions": env_actions_total_time,
      "trajectory_collection/env_actions/bare_env": bare_env_run_time,
  }
  timing_info = {k: round(1000 * v, 2) for k, v in timing_info.items()}

  return completed_trajectories, num_done_trajectories, timing_info, state


def make_env(batch_size=1,
             env_problem_name="",
             resize=True,
             resized_height=105,
             resized_width=80,
             max_timestep="None",
             clip_rewards=True,
             parallelism=1,
             use_tpu=False,
             **env_kwargs):
  """Creates the env."""

  if clip_rewards:
    env_kwargs.update({"reward_range": (-1, 1), "discrete_rewards": True})
  else:
    env_kwargs.update({"discrete_rewards": False})

  # No resizing needed, so let's be on the normal EnvProblem.
  if not resize:  # None or False
    return gym_env_problem.GymEnvProblem(
        base_env_name=env_problem_name,
        batch_size=batch_size,
        parallelism=parallelism,
        **env_kwargs)

  try:
    max_timestep = int(max_timestep)
  except Exception:  # pylint: disable=broad-except
    max_timestep = None

  wrapper_fn = functools.partial(
      gym_utils.gym_env_wrapper, **{
          "rl_env_max_episode_steps": max_timestep,
          "maxskip_env": True,
          "rendered_env": True,
          "rendered_env_resize_to": (resized_height, resized_width),
          "sticky_actions": False,
          "output_dtype": np.int32 if use_tpu else None,
      })

  return rendered_env_problem.RenderedEnvProblem(
      base_env_name=env_problem_name,
      batch_size=batch_size,
      parallelism=parallelism,
      env_wrapper_fn=wrapper_fn,
      **env_kwargs)
