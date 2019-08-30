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

"""PPO trainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time

from absl import logging
import cloudpickle as pickle
import gym
from jax import jit
from jax import numpy as np
from jax import random as jax_random
from tensor2tensor.trax import jaxboard
from tensor2tensor.trax import models as trax_models
from tensor2tensor.trax import optimizers as trax_opt
from tensor2tensor.trax import trax
from tensor2tensor.trax.rl import base_trainer
from tensor2tensor.trax.rl import ppo
from tensorflow.io import gfile


DEBUG_LOGGING = False
GAMMA = 0.99
LAMBDA = 0.95
EPSILON = 0.1
EPOCHS = 50  # 100
N_OPTIMIZER_STEPS = 100
PRINT_EVERY_OPTIMIZER_STEP = 20
BATCH_TRAJECTORIES = 32


class PPO(base_trainer.BaseTrainer):
  """PPO trainer."""

  def __init__(
      self,
      train_env,
      eval_env,
      output_dir,
      policy_and_value_model=trax_models.FrameStackMLP,
      policy_and_value_optimizer=functools.partial(
          trax_opt.Adam, learning_rate=1e-3),
      policy_and_value_two_towers=False,
      n_optimizer_steps=N_OPTIMIZER_STEPS,
      print_every_optimizer_steps=PRINT_EVERY_OPTIMIZER_STEP,
      target_kl=0.01,
      boundary=20,
      max_timestep=None,
      max_timestep_eval=20000,
      random_seed=None,
      gamma=GAMMA,
      lambda_=LAMBDA,
      c1=1.0,
      c2=0.01,
      eval_every_n=1000,
      done_frac_for_policy_save=0.5,
      n_evals=1,
      len_history_for_policy=4,
      eval_temperatures=(1.0, 0.5),
  ):
    """Creates the PPO trainer.

    Args:
      train_env: gym.Env to use for training.
      eval_env: gym.Env to use for evaluation.
      output_dir: Output dir.
      policy_and_value_model: Function defining the policy and value network,
        without the policy and value heads.
      policy_and_value_optimizer: Function defining the optimizer.
      policy_and_value_two_towers: Whether to use two separate models as the
        policy and value networks. If False, share their parameters.
      n_optimizer_steps: Number of optimizer steps.
      print_every_optimizer_steps: How often to log during the policy
        optimization process.
      target_kl: Policy iteration early stopping. Set to infinity to disable
        early stopping.
      boundary: We pad trajectories at integer multiples of this number.
      max_timestep: If set to an integer, maximum number of time-steps in
        a trajectory. Used in the collect procedure.
      max_timestep_eval: If set to an integer, maximum number of time-steps in
        an evaluation trajectory. Used in the collect procedure.
      random_seed: Random seed.
      gamma: Reward discount factor.
      lambda_: N-step TD-error discount factor in GAE.
      c1: Value loss coefficient.
      c2: Entropy loss coefficient.
      eval_every_n: How frequently to eval the policy.
      done_frac_for_policy_save: Fraction of the trajectories that should be
        done to checkpoint the policy.
      n_evals: Number of times to evaluate.
      len_history_for_policy: How much of history to give to the policy.
      eval_temperatures: Sequence of temperatures to try for categorical
        sampling during evaluation.
    """
    # Set in base class constructor.
    self._train_env = None
    self._should_reset = None

    super(PPO, self).__init__(train_env, eval_env, output_dir)

    self._n_optimizer_steps = n_optimizer_steps
    self._print_every_optimizer_steps = print_every_optimizer_steps
    self._target_kl = target_kl
    self._boundary = boundary
    self._max_timestep = max_timestep
    self._max_timestep_eval = max_timestep_eval
    self._gamma = gamma
    self._lambda_ = lambda_
    self._c1 = c1
    self._c2 = c2
    self._eval_every_n = eval_every_n
    self._done_frac_for_policy_save = done_frac_for_policy_save
    self._n_evals = n_evals
    self._len_history_for_policy = len_history_for_policy
    self._eval_temperatures = eval_temperatures

    assert isinstance(self.train_env.action_space, gym.spaces.Discrete)
    n_actions = self.train_env.action_space.n

    # Batch Observations Shape = [1, 1] + OBS, because we will eventually call
    # policy and value networks on shape [B, T] +_OBS
    batch_observations_shape = (1, 1) + self.train_env.observation_space.shape
    observations_dtype = self.train_env.observation_space.dtype

    self._rng = trax.get_random_number_generator_and_set_seed(random_seed)
    self._rng, key1 = jax_random.split(self._rng, num=2)

    # Initialize the policy and value network.
    policy_and_value_net_params, self._model_state, policy_and_value_net_apply = (
        ppo.policy_and_value_net(
            rng_key=key1,
            batch_observations_shape=batch_observations_shape,
            observations_dtype=observations_dtype,
            n_actions=n_actions,
            bottom_layers_fn=policy_and_value_model,
            two_towers=policy_and_value_two_towers,
        )
    )
    self._policy_and_value_net_apply = jit(policy_and_value_net_apply)

    # Initialize the optimizer.
    (policy_and_value_opt_state, self._policy_and_value_opt_update,
     self._policy_and_value_get_params) = ppo.optimizer_fn(
         policy_and_value_optimizer, policy_and_value_net_params)

    # Maybe restore the optimization state. If there is nothing to restore, then
    # iteration = 0 and policy_and_value_opt_state is returned as is.
    (restored, self._policy_and_value_opt_state, self._model_state, self._epoch,
     self._total_opt_step) = ppo.maybe_restore_opt_state(
         output_dir, policy_and_value_opt_state, self._model_state)

    if restored:
      logging.info("Restored parameters from iteration [%d]", self._epoch)
      # We should start from the next iteration.
      self._epoch += 1

    # Create summary writers and history.
    self._train_sw = jaxboard.SummaryWriter(
        os.path.join(self._output_dir, "train"))
    self._timing_sw = jaxboard.SummaryWriter(
        os.path.join(self._output_dir, "timing"))
    self._eval_sw = jaxboard.SummaryWriter(
        os.path.join(self._output_dir, "eval"))

    self._n_trajectories_done = 0

    self._last_saved_at = 0

  @property
  def train_env(self):
    return self._train_env

  @train_env.setter
  def train_env(self, new_train_env):
    if self._train_env is not None:
      def assert_same_space(space1, space2):
        assert space1.shape == space2.shape
        assert space1.dtype == space2.dtype
      assert_same_space(
          new_train_env.observation_space, self._train_env.observation_space)
      assert_same_space(
          new_train_env.action_space, self._train_env.action_space)
      # We don't check the reward range, because PPO will work either way.

    self._train_env = new_train_env
    self._should_reset = True

  @property
  def epoch(self):
    return self._epoch

  def train_epoch(self):
    """Train one PPO epoch."""
    epoch_start_time = time.time()

    # Evaluate the policy.
    policy_eval_start_time = time.time()
    if (self._epoch + 1) % self._eval_every_n == 0:
      self._rng, key = jax_random.split(self._rng, num=2)
      self.evaluate()

    policy_eval_time = ppo.get_time(policy_eval_start_time)

    trajectory_collection_start_time = time.time()
    logging.vlog(1, "PPO epoch [% 6d]: collecting trajectories.", self._epoch)
    self._rng, key = jax_random.split(self._rng)
    trajs, n_done, timing_info, self._model_state = ppo.collect_trajectories(
        self.train_env,
        policy_fn=self._get_predictions,
        n_trajectories=self.train_env.batch_size,
        max_timestep=self._max_timestep,
        state=self._model_state,
        rng=key,
        len_history_for_policy=self._len_history_for_policy,
        boundary=self._boundary,
        reset=self._should_reset,
    )
    self._should_reset = False
    trajectory_collection_time = ppo.get_time(trajectory_collection_start_time)

    logging.vlog(1, "Collecting trajectories took %0.2f msec.",
                 trajectory_collection_time)

    avg_reward = float(sum(np.sum(traj[2]) for traj in trajs)) / len(trajs)
    max_reward = max(np.sum(traj[2]) for traj in trajs)
    min_reward = min(np.sum(traj[2]) for traj in trajs)

    self._train_sw.scalar(
        "train/reward_mean_truncated", avg_reward, step=self._epoch)

    logging.vlog(1, "Rewards avg=[%0.2f], max=[%0.2f], min=[%0.2f], all=%s",
                 avg_reward, max_reward, min_reward,
                 [float(np.sum(traj[2])) for traj in trajs])

    logging.vlog(1,
                 "Trajectory Length average=[%0.2f], max=[%0.2f], min=[%0.2f]",
                 float(sum(len(traj[0]) for traj in trajs)) / len(trajs),
                 max(len(traj[0]) for traj in trajs),
                 min(len(traj[0]) for traj in trajs))
    logging.vlog(2, "Trajectory Lengths: %s", [len(traj[0]) for traj in trajs])

    padding_start_time = time.time()
    (_, reward_mask, padded_observations, padded_actions,
     padded_rewards, padded_infos) = ppo.pad_trajectories(
         trajs, boundary=self._boundary)
    padding_time = ppo.get_time(padding_start_time)

    logging.vlog(1, "Padding trajectories took %0.2f msec.",
                 ppo.get_time(padding_start_time))
    logging.vlog(1, "Padded Observations' shape [%s]",
                 str(padded_observations.shape))
    logging.vlog(1, "Padded Actions' shape [%s]", str(padded_actions.shape))
    logging.vlog(1, "Padded Rewards' shape [%s]", str(padded_rewards.shape))

    # Some assertions.
    B, T = padded_actions.shape  # pylint: disable=invalid-name
    assert (B, T) == padded_rewards.shape
    assert (B, T) == reward_mask.shape
    assert (B, T + 1) == padded_observations.shape[:2]
    assert ((B, T + 1) + self.train_env.observation_space.shape ==
            padded_observations.shape)

    log_prob_recompute_start_time = time.time()
    assert ("log_prob_actions" in padded_infos and
            "value_predictions" in padded_infos)
    # These are the actual log-probabs and value predictions seen while picking
    # the actions.
    actual_log_probabs_traj = padded_infos["log_prob_actions"]
    actual_value_predictions_traj = padded_infos["value_predictions"]

    assert (B, T) == actual_log_probabs_traj.shape[:2]
    A = actual_log_probabs_traj.shape[2]  # pylint: disable=invalid-name
    assert (B, T, 1) == actual_value_predictions_traj.shape

    # TODO(afrozm): log-probabs doesn't need to be (B, T+1, A) it can do with
    # (B, T, A), so make that change throughout.

    # NOTE: We don't have the log-probabs and value-predictions for the last
    # observation, so we re-calculate for everything, but use the original ones
    # for all but the last time-step.
    self._rng, key = jax_random.split(self._rng)

    log_probabs_traj, value_predictions_traj, self._model_state, _ = (
        self._get_predictions(padded_observations, self._model_state, rng=key))

    assert (B, T + 1, A) == log_probabs_traj.shape
    assert (B, T + 1, 1) == value_predictions_traj.shape

    # Concatenate the last time-step's log-probabs and value predictions to the
    # actual log-probabs and value predictions and use those going forward.
    log_probabs_traj = np.concatenate(
        (actual_log_probabs_traj, log_probabs_traj[:, -1:, :]), axis=1)
    value_predictions_traj = np.concatenate(
        (actual_value_predictions_traj, value_predictions_traj[:, -1:, :]),
        axis=1)

    log_prob_recompute_time = ppo.get_time(log_prob_recompute_start_time)

    # Compute value and ppo losses.
    self._rng, key1 = jax_random.split(self._rng, num=2)
    logging.vlog(2, "Starting to compute P&V loss.")
    loss_compute_start_time = time.time()
    (cur_combined_loss, component_losses, summaries, self._model_state) = (
        ppo.combined_loss(
            self._policy_and_value_net_params,
            log_probabs_traj,
            value_predictions_traj,
            self._policy_and_value_net_apply,
            padded_observations,
            padded_actions,
            padded_rewards,
            reward_mask,
            gamma=self._gamma,
            lambda_=self._lambda_,
            c1=self._c1,
            c2=self._c2,
            state=self._model_state,
            rng=key1))
    loss_compute_time = ppo.get_time(loss_compute_start_time)
    (cur_ppo_loss, cur_value_loss, cur_entropy_bonus) = component_losses
    logging.vlog(
        1,
        "Calculating P&V loss [%10.2f(%10.2f, %10.2f, %10.2f)] took %0.2f msec.",
        cur_combined_loss, cur_ppo_loss, cur_value_loss, cur_entropy_bonus,
        ppo.get_time(loss_compute_start_time))

    self._rng, key1 = jax_random.split(self._rng, num=2)
    logging.vlog(1, "Policy and Value Optimization")
    optimization_start_time = time.time()
    keys = jax_random.split(key1, num=self._n_optimizer_steps)
    opt_step = 0
    for key in keys:
      k1, k2, k3 = jax_random.split(key, num=3)
      t = time.time()
      # Update the optimizer state.
      self._policy_and_value_opt_state, self._model_state = (
          ppo.policy_and_value_opt_step(
              # We pass the optimizer slots between PPO epochs, so we need to
              # pass the optimization step as well, so for example the
              # bias-correction in Adam is calculated properly. Alternatively we
              # could reset the slots and the step in every PPO epoch, but then
              # the moment estimates in adaptive optimizers would never have
              # enough time to warm up. So it makes sense to reuse the slots,
              # even though we're optimizing a different loss in every new
              # epoch.
              self._total_opt_step,
              self._policy_and_value_opt_state,
              self._policy_and_value_opt_update,
              self._policy_and_value_get_params,
              self._policy_and_value_net_apply,
              log_probabs_traj,
              value_predictions_traj,
              padded_observations,
              padded_actions,
              padded_rewards,
              reward_mask,
              c1=self._c1,
              c2=self._c2,
              gamma=self._gamma,
              lambda_=self._lambda_,
              state=self._model_state,
              rng=k1))
      opt_step += 1
      self._total_opt_step += 1

      # Compute the approx KL for early stopping.
      (log_probab_actions_new, _), self._model_state = (
          self._policy_and_value_net_apply(padded_observations,
                                           self._policy_and_value_net_params,
                                           self._model_state, rng=k2))

      approx_kl = ppo.approximate_kl(log_probab_actions_new, log_probabs_traj,
                                     reward_mask)

      early_stopping = approx_kl > 1.5 * self._target_kl
      if early_stopping:
        logging.vlog(
            1, "Early stopping policy and value optimization after %d steps, "
            "with approx_kl: %0.2f", opt_step, approx_kl)
        # We don't return right-away, we want the below to execute on the last
        # iteration.

      t2 = time.time()
      if (opt_step % self._print_every_optimizer_steps == 0 or
          opt_step == self._n_optimizer_steps or early_stopping):
        # Compute and log the loss.
        (combined_loss, component_losses, _, self._model_state) = (
            ppo.combined_loss(
                self._policy_and_value_net_params,
                log_probabs_traj,
                value_predictions_traj,
                self._policy_and_value_net_apply,
                padded_observations,
                padded_actions,
                padded_rewards,
                reward_mask,
                gamma=self._gamma,
                lambda_=self._lambda_,
                c1=self._c1,
                c2=self._c2,
                state=self._model_state,
                rng=k3))
        logging.vlog(1, "One Policy and Value grad desc took: %0.2f msec",
                     ppo.get_time(t, t2))
        (ppo_loss, value_loss, entropy_bonus) = component_losses
        logging.vlog(
            1, "Combined Loss(value, ppo, entropy_bonus) [%10.2f] ->"
            " [%10.2f(%10.2f,%10.2f,%10.2f)]", cur_combined_loss, combined_loss,
            ppo_loss, value_loss, entropy_bonus)

      if early_stopping:
        break

    optimization_time = ppo.get_time(optimization_start_time)

    logging.vlog(
        1, "Total Combined Loss reduction [%0.2f]%%",
        (100 * (cur_combined_loss - combined_loss) / np.abs(cur_combined_loss)))

    summaries.update({
        "n_optimizer_steps": opt_step,
        "approx_kl": approx_kl,
    })
    for (name, value) in summaries.items():
      self._train_sw.scalar("train/{}".format(name), value, step=self._epoch)

    # Save parameters every time we see the end of at least a fraction of batch
    # number of trajectories that are done (not completed -- completed includes
    # truncated and done).
    # Also don't save too frequently, enforce a minimum gap.
    # Or if this is the last iteration.
    policy_save_start_time = time.time()
    self._n_trajectories_done += n_done
    # TODO(afrozm): Refactor to trax.save_state.
    if ((self._n_trajectories_done >=
         self._done_frac_for_policy_save * self.train_env.batch_size) and
        (self._epoch - self._last_saved_at > self._eval_every_n) and
        (((self._epoch + 1) % self._eval_every_n == 0))):
      self.save()
    policy_save_time = ppo.get_time(policy_save_start_time)

    epoch_time = ppo.get_time(epoch_start_time)

    logging.info(
        "PPO epoch [% 6d], Reward[min, max, avg] [%5.2f,%5.2f,%5.2f], Combined"
        " Loss(ppo, value, entropy) [%2.5f(%2.5f,%2.5f,%2.5f)]", self._epoch,
        min_reward, max_reward, avg_reward, combined_loss, ppo_loss, value_loss,
        entropy_bonus)

    timing_dict = {
        "epoch": epoch_time,
        "policy_eval": policy_eval_time,
        "trajectory_collection": trajectory_collection_time,
        "padding": padding_time,
        "log_prob_recompute": log_prob_recompute_time,
        "loss_compute": loss_compute_time,
        "optimization": optimization_time,
        "policy_save": policy_save_time,
    }

    timing_dict.update(timing_info)

    for k, v in timing_dict.items():
      self._timing_sw.scalar("timing/%s" % k, v, step=self._epoch)

    max_key_len = max(len(k) for k in timing_dict)
    timing_info_list = [
        "%s : % 10.2f" % (k.rjust(max_key_len + 1), v)
        for k, v in sorted(timing_dict.items())
    ]
    logging.info(
        "PPO epoch [% 6d], Timings: \n%s", self._epoch,
        "\n".join(timing_info_list)
    )

    self._epoch += 1

    # Flush summary writers once in a while.
    if (self._epoch + 1) % 1000 == 0:
      self.flush_summaries()

  def evaluate(self):
    """Evaluate the agent."""
    logging.vlog(1, "PPO epoch [% 6d]: evaluating policy.", self._epoch)
    self._rng, key = jax_random.split(self._rng, num=2)
    reward_stats, self._model_state = ppo.evaluate_policy(
        self.eval_env,
        self._get_predictions,
        temperatures=self._eval_temperatures,
        max_timestep=self._max_timestep_eval,
        n_evals=self._n_evals,
        len_history_for_policy=self._len_history_for_policy,
        state=self._model_state,
        rng=key)
    ppo.write_eval_reward_summaries(
        reward_stats, self._eval_sw, epoch=self._epoch)

  def save(self):
    """Save the agent parameters."""
    logging.vlog(1, "PPO epoch [% 6d]: saving model.", self._epoch)
    old_model_files = gfile.glob(
        os.path.join(self._output_dir, "model-??????.pkl"))
    params_file = os.path.join(self._output_dir, "model-%06d.pkl" % self._epoch)
    with gfile.GFile(params_file, "wb") as f:
      pickle.dump(
          (self._policy_and_value_opt_state, self._model_state,
           self._total_opt_step), f)
    # Remove the old model files.
    for path in old_model_files:
      gfile.remove(path)
    # Reset this number.
    self._n_trajectories_done = 0
    self._last_saved_at = self._epoch

  def flush_summaries(self):
    self._train_sw.flush()
    self._timing_sw.flush()
    self._eval_sw.flush()

  @property
  def _policy_and_value_net_params(self):
    return self._policy_and_value_get_params(self._policy_and_value_opt_state)

  # A function to get the policy and value predictions.
  def _get_predictions(self, observations, state, rng=None):
    """Returns log-probs, value predictions and key back."""
    key, key1 = jax_random.split(rng, num=2)

    (log_probs, value_preds), state = self._policy_and_value_net_apply(
        observations, self._policy_and_value_net_params, state, rng=key1)

    return log_probs, value_preds, state, key
