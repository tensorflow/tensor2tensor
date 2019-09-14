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

"""Uitlity functions for the async trajectory collector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import time

from absl import logging
from tensor2tensor.envs import trajectory
from tensor2tensor.trax.rl import ppo
from tensor2tensor.trax.rl import trainers as rl_trainers
from tensorflow.io import gfile

LARGE_MAX_TRIES_FOR_POLICY_FILE = 100


# TODO(afrozm): Is there a better way to poll for a file on CNS?
def get_newer_policy_model_file(output_dir,
                                min_epoch=-1,
                                sleep_time_secs=0.1,
                                max_sleep_time_secs=1.0,
                                max_tries=1,
                                wait_forever=False,):
  """Gets a policy model file subject to availability and wait time."""

  while max_tries or wait_forever:
    max_tries -= 1
    policy_files = ppo.get_policy_model_files(output_dir)

    def do_wait(t):
      time.sleep(t)
      t *= 2
      return min(t, max_sleep_time_secs)

    # No policy files at all.
    if not policy_files:
      logging.info("There are no policy files in [%s], waiting for %s secs.",
                   output_dir, sleep_time_secs)
      sleep_time_secs = do_wait(sleep_time_secs)
      continue

    # Check if we have a newer epoch.
    policy_file = policy_files[0]
    epoch = ppo.get_epoch_from_policy_model_file(policy_file)

    # We don't - wait.
    if epoch <= min_epoch:
      logging.info("epoch [%s] <= min_epoch [%s], waiting for %s secs.", epoch,
                   min_epoch, sleep_time_secs)
      sleep_time_secs = do_wait(sleep_time_secs)
      continue

    # We do have a new file, return it.
    policy_file = policy_files[0]
    epoch = ppo.get_epoch_from_policy_model_file(policy_file)
    logging.info("Found epoch [%s] and policy file [%s]", epoch, policy_file)
    return policy_file, epoch

  # Exhausted our waiting limit.
  return None


def dump_trajectory(output_dir, epoch, env_id, temperature, random_string,
                    trajs):
  """Write the trajectory to disk."""

  assert 1 == len(trajs)
  traj = trajs[0]

  trajectory_file_name = trajectory.TRAJECTORY_FILE_FORMAT.format(
      epoch=epoch, env_id=env_id, temperature=temperature, r=random_string)

  with gfile.GFile(os.path.join(output_dir, trajectory_file_name), "w") as f:
    trajectory.get_pickle_module().dump(traj, f)


def continuously_collect_trajectories(output_dir,
                                      train_env,
                                      eval_env,
                                      trajectory_dump_dir=None,
                                      env_id=None,
                                      max_trajectories_to_collect=None,
                                      try_abort=True):
  """Instantiates a PPO trainer and collects trajectories."""

  # Make the PPO trainer.
  ppo_trainer = rl_trainers.PPO(
      output_dir=output_dir,
      train_env=train_env,
      eval_env=eval_env,
      trajectory_dump_dir=trajectory_dump_dir,
  )

  # TODO(afrozm): Update base_trainer interface to support SimPLe as well.
  assert isinstance(ppo_trainer, rl_trainers.PPO)

  assert env_id is not None

  # Get an initial policy and wait a forever to get it if needed.
  policy_and_epoch = get_newer_policy_model_file(output_dir, wait_forever=True)
  assert policy_and_epoch
  policy_file, epoch = policy_and_epoch
  logging.info("Read initial policy for epoch [%s] -> [%s]", epoch, policy_file)

  # Returns immediately if there is a newer epoch available.
  def is_newer_policy_file_available(epoch_, sleep_time_secs_=0.1):
    return get_newer_policy_model_file(
        output_dir, min_epoch=epoch_, sleep_time_secs=sleep_time_secs_)

  assert 1 == train_env.batch_size
  assert 1 == eval_env.batch_size

  temperature = 1.0

  trajectories_collected = 0

  train_env_trajectory_dump_dir = os.path.join(output_dir, "trajectories/train")
  eval_env_trajectory_dump_dir = os.path.join(output_dir, "trajectories/eval")

  gfile.makedirs(train_env_trajectory_dump_dir)
  gfile.makedirs(eval_env_trajectory_dump_dir)

  while max_trajectories_to_collect is None or trajectories_collected < int(
      max_trajectories_to_collect):
    logging.info("Collecting a trajectory, trajectories_collected = %s",
                 trajectories_collected)

    # Abort function -- if something newever is available, then abort the
    # current computation and reload.

    # Useful if env.step is long.
    def long_abort_fn():
      # We want this to be as quick as possible.
      return is_newer_policy_file_available(epoch, 0) is not None

    abort_fn = long_abort_fn if try_abort else None

    # Collect a training trajectory.
    trajs, n_done, unused_timing_info, unused_model_state = (
        ppo_trainer.collect_trajectories(train=True,
                                         temperature=temperature,
                                         abort_fn=abort_fn,
                                         raw_trajectory=True))

    if trajs and n_done > 0:
      assert 1 == n_done
      trajectories_collected += n_done

      # Write the trajectory down.
      logging.info(
          "Dumping the collected trajectory, trajectories_collected = %s",
          trajectories_collected)
      dump_trajectory(train_env_trajectory_dump_dir, epoch, env_id, temperature,
                      str(random.randint(0, 2**31 - 1)), trajs)
    else:
      logging.info("Computation was aborted, a new policy is available.")

    # This maybe useless, since `abort_fn` will take care of it. We might want
    # to have this here if abort_fn is False always.
    # Do we have a newer policy?
    policy_file_and_epoch = is_newer_policy_file_available(epoch)
    if policy_file_and_epoch is None:
      # Continue churning out these policies.
      logging.info("We don't have a newer policy, continuing with the old one.")
      continue

    # We have a newer policy, read it and update the parameters.
    policy_file, epoch = policy_file_and_epoch
    logging.info(
        "We have a newer policy epoch [%s], file [%s], updating parameters.",
        epoch, policy_file)
    ppo_trainer.update_optimization_state(
        output_dir, policy_and_value_opt_state=None)
    logging.info("Parameters of PPOTrainer updated.")

    # Check that the epochs match.
    assert epoch == ppo_trainer.epoch
