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

import numpy as np


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
                                 boundary=20):
  """Plays the given env with the policy function to collect trajectories.

  Args:
    env: environment object, should be a subclass of env_problem.EnvProblem.
    policy_fun: callable, taking in observations((B, T) + OBS) and returning
        back log-probabilities (B, T, A).
    num_trajectories: int, number of trajectories to collect.
    max_timestep: int or None, if not None or a negative number, we cut any
        trajectory that exceeds this time and mark that as completed by
        resetting that trajectory.
    boundary: this is the bucket length, we pad the observations to integer
        multiples of this + 1 and then feed the padded observations to the
        policy_fun.

  Returns:
    Completed trajectories that is a list of triples of (observation, action,
    reward) ndarrays.
  """

  def multinomial_sample(probs):
    """Sample from this vector of probabilities.

    Args:
      probs: numpy array of shape (A,) where A is the number of actions, these
        must sum up to 1.0

    Returns:
      an integer of which action to pick.
    """
    return int(np.argwhere(np.random.multinomial(1, probs) == 1))

  # We need to reset all environments.
  env.reset()

  while True:
    # Get all the observations for all the active trajectories.
    # Shape is (B, T) + OBS
    padded_observations = env.trajectories.observations_np(boundary=boundary)
    lengths = env.trajectories.trajectory_lengths

    B, T = padded_observations.shape[:2]  # pylint: disable=invalid-name

    assert B == env.batch_size
    assert (B,) == lengths.shape

    log_prob_actions = policy_fun(padded_observations)
    assert (B, T) == log_prob_actions.shape[:2]
    A = log_prob_actions.shape[2]  # pylint: disable=invalid-name

    # We need the log_probs of those actions that correspond to the last actual
    # time-step.
    index = lengths - 1  # Since we want to index using lengths.
    log_probs = log_prob_actions[np.arange(B)[:, None],
                                 index[:, None],
                                 np.arange(A)]
    assert (B, A) == log_probs.shape, \
        "B=%d, A=%d, log_probs.shape=%s" % (B, A, log_probs.shape)

    # Convert to probs, since we need to do categorical sampling.
    probs = np.exp(log_probs)

    # Sometimes log_probs contains a 0, it shouldn't. This makes the
    # probabilities sum up to more than 1, since the addition happens
    # in float64, so just add and subtract 1.0 to zero those probabilites
    # out.
    #
    # Also testing for this is brittle.
    probs += 1
    probs -= 1

    # For some reason, sometimes, this isn't the case.
    probs_sum = np.sum(probs, axis=1, keepdims=True)
    if not all(probs_sum == 1.0):
      probs = probs / probs_sum

    # Now pick actions from this probs array.
    actions = np.apply_along_axis(multinomial_sample, 1, probs)

    # Step through the env.
    _, _, dones, _ = env.step(actions)

    # Get the indices where we are done ...
    done_idxs = done_indices(dones)

    # ... and reset those.
    if done_idxs.size:
      env.reset(indices=done_idxs)

    # Do we have enough trajectories right now?
    if env.trajectories.num_completed_trajectories >= num_trajectories:
      break

    if max_timestep is None or max_timestep < 1:
      continue

    # Are there any trajectories that have exceeded the time-limit we want.
    lengths = env.trajectories.trajectory_lengths
    exceeded_time_limit_idxs = done_indices(lengths > max_timestep)

    # If so, reset these as well.
    if exceeded_time_limit_idxs.size:
      env.reset(indices=exceeded_time_limit_idxs)
    # Do we have enough trajectories right now?
    if env.trajectories.num_completed_trajectories >= num_trajectories:
      break

  # We have the trajectories we need, return a list of triples:
  # (observations, actions, rewards)
  completed_trajectories = []
  for trajectory in env.trajectories.completed_trajectories[:num_trajectories]:
    completed_trajectories.append(trajectory.as_numpy)

  return completed_trajectories
