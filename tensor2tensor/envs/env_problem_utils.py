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

import time
import numpy as np

CATEGORICAL_SAMPLING = "categorical"
EPSILON_GREEDY = "epsilon-greedy"
GUMBEL_SAMPLING = "gumbel"


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
                                 rng=None,
                                 policy_sampling=CATEGORICAL_SAMPLING,
                                 temperature=0.5,
                                 eps=0.1,
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
      max_max_timestep is None or < 0
    rng: jax rng, splittable.
    policy_sampling: string, how to select an action given a policy, one of:
      CATEGORICAL_SAMPLING, GREEDY, GUMBEL_SAMPLING
    temperature: float, temperature used in gumbel sampling.
    eps: float, epsilon to use in epsilon greedy.
    len_history_for_policy: int, the maximum history to keep for applying the
      policy on. We also bucket observations on this number.
    num_to_keep: int, while truncating trajectory how many time-steps to keep.

  Returns:
    A tuple, (trajectories, number of completed trajectories). Where
    trajectories is a list of triples of (observation, action, reward) ndarrays.
  """
  t0 = time.time()

  def categorical_sample(log_probs):
    """Categorical sampling."""

    def multinomial_sample(probs):
      """Sample from this vector of probabilities.

      Args:
        probs: numpy array of shape (A,) where A is the number of actions, these
          must sum up to 1.0

      Returns:
        an integer of which action to pick.
      """

      return int(np.argwhere(np.random.multinomial(1, probs) == 1))

    # Convert to probs, since we need to do categorical sampling.
    probs = np.exp(log_probs)

    # Let's cast up to float64, because that's what numpy does when sampling
    # and it leads to the sum(pvals[:-1]) > 1.0 error.
    #
    # We also re-normalize when we do this.
    probs = np.float64(probs)
    probs /= np.sum(probs, axis=1, keepdims=True)

    # Now pick actions from this probs array.
    return np.apply_along_axis(multinomial_sample, 1, probs)

  def gumbel_sample(log_probs):
    """Gumbel sampling."""
    u = np.random.uniform(low=1e-6, high=1.0 - 1e-6, size=log_probs.shape)
    g = -np.log(-np.log(u))
    return np.argmax((log_probs / temperature) + g, axis=1)

  def epsilon_greedy(log_probs):
    """Epsilon greedy sampling."""
    _, A = log_probs.shape  # pylint: disable=invalid-name
    actions = []
    for log_prob in log_probs:
      # Pick the argmax action.
      action = np.argmax(log_prob)
      if np.random.uniform() < eps:
        # Pick an action at random.
        action = np.random.choice(range(A))
      actions.append(action)
    return np.stack(actions)

  # We need to reset all environments, if we're coming here the first time.
  if reset or max_timestep is None or max_timestep <= 0:
    env.reset()
  else:
    # Clear completed trajectories held internally.
    env.trajectories.clear_completed_trajectories()

  num_done_trajectories = 0

  policy_application_total_time = 0
  env_actions_total_time = 0
  while env.trajectories.num_completed_trajectories < num_trajectories:
    # Get all the observations for all the active trajectories.
    # Shape is (B, T) + OBS
    # Bucket on whatever length is needed.
    padded_observations, lengths = env.trajectories.observations_np(
        boundary=len_history_for_policy,
        len_history_for_policy=len_history_for_policy)

    B, T = padded_observations.shape[:2]  # pylint: disable=invalid-name

    assert B == env.batch_size
    assert (B,) == lengths.shape

    t1 = time.time()
    log_prob_actions, _, rng = policy_fun(padded_observations, rng=rng)
    policy_application_total_time += (time.time() - t1)

    assert (B, T) == log_prob_actions.shape[:2]
    A = log_prob_actions.shape[2]  # pylint: disable=invalid-name

    # We need the log_probs of those actions that correspond to the last actual
    # time-step.
    index = lengths - 1  # Since we want to index using lengths.
    log_probs = log_prob_actions[np.arange(B)[:, None], index[:, None],
                                 np.arange(A)]
    assert (B, A) == log_probs.shape, \
        "B=%d, A=%d, log_probs.shape=%s" % (B, A, log_probs.shape)

    actions = None
    if policy_sampling == CATEGORICAL_SAMPLING:
      actions = categorical_sample(log_probs)
    elif policy_sampling == GUMBEL_SAMPLING:
      actions = gumbel_sample(log_probs)
    elif policy_sampling == EPSILON_GREEDY:
      actions = epsilon_greedy(log_probs)
    else:
      raise ValueError("Unknown sampling policy [%s]" % policy_sampling)

    # Step through the env.
    t1 = time.time()
    _, _, dones, _ = env.step(actions)
    env_actions_total_time += (time.time() - t1)

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

  misc_time = (time.time() - t0) - policy_application_total_time
  timing_info = {
      "trajectory_collection/policy_application": policy_application_total_time,
      "trajectory_collection/misc": misc_time,
      "trajectory_collection/env_actions": env_actions_total_time,
  }
  timing_info = {k: round(1000 * v, 2) for k, v in timing_info.items()}

  return completed_trajectories, num_done_trajectories, timing_info
