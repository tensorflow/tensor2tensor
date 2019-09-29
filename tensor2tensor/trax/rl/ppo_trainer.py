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

import collections
import functools
import os
import time

from absl import logging
import gym
from jax import jit
from jax import numpy as np
from jax import random as jax_random
import numpy as onp
from tensor2tensor.envs import env_problem_utils
from tensor2tensor.envs import trajectory
from tensor2tensor.trax import jaxboard
from tensor2tensor.trax import models as trax_models
from tensor2tensor.trax import optimizers as trax_opt
from tensor2tensor.trax import trax
from tensor2tensor.trax.rl import base_trainer
from tensor2tensor.trax.rl import ppo
from tensor2tensor.trax.rl import serialization_utils
from tensor2tensor.trax.rl import space_serializer

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

  def __init__(self,
               train_env,
               eval_env,
               output_dir,
               policy_and_value_model=trax_models.FrameStackMLP,
               policy_and_value_optimizer=functools.partial(
                   trax_opt.Adam, learning_rate=1e-3),
               policy_and_value_two_towers=False,
               policy_and_value_vocab_size=None,
               n_optimizer_steps=N_OPTIMIZER_STEPS,
               optimizer_batch_size=64,
               print_every_optimizer_steps=PRINT_EVERY_OPTIMIZER_STEP,
               target_kl=0.01,
               boundary=20,
               max_timestep=100,
               max_timestep_eval=20000,
               random_seed=None,
               gamma=GAMMA,
               lambda_=LAMBDA,
               c1=1.0,
               c2=0.01,
               eval_every_n=1000,
               save_every_n=1000,
               done_frac_for_policy_save=0.5,
               n_evals=1,
               len_history_for_policy=4,
               eval_temperatures=(1.0, 0.5),
               separate_eval=True,
               init_policy_from_world_model_output_dir=None,
               **kwargs):
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
      policy_and_value_vocab_size: Vocabulary size of a policy and value network
        operating on serialized representation. If None, use raw continuous
        representation.
      n_optimizer_steps: Number of optimizer steps.
      optimizer_batch_size: Batch size of an optimizer step.
      print_every_optimizer_steps: How often to log during the policy
        optimization process.
      target_kl: Policy iteration early stopping. Set to infinity to disable
        early stopping.
      boundary: We pad trajectories at integer multiples of this number.
      max_timestep: If set to an integer, maximum number of time-steps in a
        trajectory. Used in the collect procedure.
      max_timestep_eval: If set to an integer, maximum number of time-steps in
        an evaluation trajectory. Used in the collect procedure.
      random_seed: Random seed.
      gamma: Reward discount factor.
      lambda_: N-step TD-error discount factor in GAE.
      c1: Value loss coefficient.
      c2: Entropy loss coefficient.
      eval_every_n: How frequently to eval the policy.
      save_every_n: How frequently to save the policy.
      done_frac_for_policy_save: Fraction of the trajectories that should be
        done to checkpoint the policy.
      n_evals: Number of times to evaluate.
      len_history_for_policy: How much of history to give to the policy.
      eval_temperatures: Sequence of temperatures to try for categorical
        sampling during evaluation.
      separate_eval: Whether to run separate evaluation using a set of
        temperatures. If False, the training reward is reported as evaluation
        reward with temperature 1.0.
      init_policy_from_world_model_output_dir: Model output dir for initializing
        the policy. If None, initialize randomly.
      **kwargs: Additional keyword arguments passed to the base class.
    """
    # Set in base class constructor.
    self._train_env = None
    self._should_reset = None

    super(PPO, self).__init__(train_env, eval_env, output_dir, **kwargs)

    self._n_optimizer_steps = n_optimizer_steps
    self._optimizer_batch_size = optimizer_batch_size
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
    self._save_every_n = save_every_n
    self._done_frac_for_policy_save = done_frac_for_policy_save
    self._n_evals = n_evals
    self._len_history_for_policy = len_history_for_policy
    self._eval_temperatures = eval_temperatures
    self._separate_eval = separate_eval

    action_space = self.train_env.action_space
    assert isinstance(
        action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete))
    if isinstance(action_space, gym.spaces.Discrete):
      n_actions = action_space.n
      n_controls = 1
    else:
      (n_controls,) = action_space.nvec.shape
      assert n_controls > 0
      assert onp.min(action_space.nvec) == onp.max(action_space.nvec), (
          "Every control must have the same number of actions.")
      n_actions = action_space.nvec[0]
    self._n_actions = n_actions
    self._n_controls = n_controls

    self._rng = trax.get_random_number_generator_and_set_seed(random_seed)
    self._rng, key1 = jax_random.split(self._rng, num=2)

    vocab_size = policy_and_value_vocab_size
    self._serialized_sequence_policy = vocab_size is not None
    if self._serialized_sequence_policy:
      self._serialization_kwargs = self._init_serialization(vocab_size)
    else:
      self._serialization_kwargs = {}

    # Initialize the policy and value network.
    policy_and_value_net = ppo.policy_and_value_net(
        n_actions=n_actions,
        n_controls=n_controls,
        vocab_size=vocab_size,
        bottom_layers_fn=policy_and_value_model,
        two_towers=policy_and_value_two_towers,
    )
    self._policy_and_value_net_apply = jit(policy_and_value_net)
    (batch_obs_shape, obs_dtype) = self._batch_obs_shape_and_dtype
    policy_and_value_net_params, self._model_state = (
        policy_and_value_net.initialize_once(batch_obs_shape, obs_dtype, key1))
    if init_policy_from_world_model_output_dir is not None:
      policy_and_value_net_params = ppo.init_policy_from_world_model_checkpoint(
          policy_and_value_net_params, init_policy_from_world_model_output_dir
      )

    # Initialize the optimizer.
    (policy_and_value_opt_state, self._policy_and_value_opt_update,
     self._policy_and_value_get_params) = ppo.optimizer_fn(
         policy_and_value_optimizer, policy_and_value_net_params)

    # Restore the optimizer state.
    self._policy_and_value_opt_state = policy_and_value_opt_state
    self._epoch = 0
    self._total_opt_step = 0
    self.update_optimization_state(
        output_dir, policy_and_value_opt_state=policy_and_value_opt_state)

    # Create summary writers and history.
    self._train_sw = jaxboard.SummaryWriter(
        os.path.join(self._output_dir, "train"))
    self._timing_sw = jaxboard.SummaryWriter(
        os.path.join(self._output_dir, "timing"))
    self._eval_sw = jaxboard.SummaryWriter(
        os.path.join(self._output_dir, "eval"))

    self._n_trajectories_done = 0

    self._last_saved_at = 0
    if self._async_mode:
      logging.info("Saving model on startup to have a model policy file.")
      self.save()

    self._rewards_to_actions = self._init_rewards_to_actions()

  def _init_serialization(self, vocab_size):
    obs_serializer = space_serializer.create(
        self.train_env.observation_space, vocab_size=vocab_size
    )
    act_serializer = space_serializer.create(
        self.train_env.action_space, vocab_size=vocab_size
    )
    repr_length = (
        obs_serializer.representation_length +
        act_serializer.representation_length
    ) * (self._max_timestep + 1)
    return {
        "observation_serializer": obs_serializer,
        "action_serializer": act_serializer,
        "representation_length": repr_length,
    }

  def _init_rewards_to_actions(self):
    # Linear map from the reward sequence to the action sequence, used for
    # scattering advantages over action log-probs and some other things.
    # It has one more timestep at the end, so it's compatible with the value
    # predictions.
    if not self._serialized_sequence_policy:
      rewards_to_actions = np.eye(self._max_timestep + 1)[:, None, :]
      rewards_to_actions = np.broadcast_to(
          rewards_to_actions,
          (self._max_timestep + 1, self._n_controls, self._max_timestep + 1),
      )
      return np.reshape(rewards_to_actions, (self._max_timestep + 1, -1))
    else:
      return serialization_utils.rewards_to_actions_map(
          n_timesteps=(self._max_timestep + 1), **self._serialization_kwargs
      )

  @property
  def _batch_obs_shape_and_dtype(self):
    if not self._serialized_sequence_policy:
      # Batch Observations Shape = [1, 1] + OBS, because we will eventually call
      # policy and value networks on shape [B, T] +_OBS
      shape = (1, 1) + self.train_env.observation_space.shape
      dtype = self.train_env.observation_space.dtype
    else:
      shape = (1, 1)
      dtype = np.int32
    return (shape, dtype)

  # Maybe restore the optimization state. If there is nothing to restore, then
  # epoch = 0 and policy_and_value_opt_state is returned as is.
  def update_optimization_state(self,
                                output_dir,
                                policy_and_value_opt_state=None):
    (self._policy_and_value_opt_state, self._model_state, self._epoch,
     self._total_opt_step) = ppo.maybe_restore_opt_state(
         output_dir, policy_and_value_opt_state, self._model_state)

    if self._epoch > 0:
      logging.info("Restored parameters from epoch [%d]", self._epoch)

  @property
  def train_env(self):
    return self._train_env

  @train_env.setter
  def train_env(self, new_train_env):
    if self._train_env is not None:

      def assert_same_space(space1, space2):
        assert space1.shape == space2.shape
        assert space1.dtype == space2.dtype

      assert_same_space(new_train_env.observation_space,
                        self._train_env.observation_space)
      assert_same_space(new_train_env.action_space,
                        self._train_env.action_space)
      # We don't check the reward range, because PPO will work either way.

    self._train_env = new_train_env
    self._should_reset = True

  @property
  def epoch(self):
    return self._epoch

  def collect_trajectories_async(self,
                                 env,
                                 train=True,
                                 n_trajectories=1,
                                 temperature=1.0):
    """Collects trajectories in an async manner."""

    assert self._async_mode

    # trajectories/train and trajectories/eval are the two subdirectories.
    trajectory_dir = os.path.join(self._output_dir, "trajectories",
                                  "train" if train else "eval")
    epoch = self.epoch

    logging.info(
        "Loading [%s] trajectories from dir [%s] for epoch [%s] and temperature"
        " [%s]", n_trajectories, trajectory_dir, epoch, temperature)

    bt = trajectory.BatchTrajectory.load_from_directory(
        trajectory_dir,
        epoch=epoch,
        temperature=temperature,
        wait_forever=True,
        n_trajectories=n_trajectories)

    if bt is None:
      logging.error(
          "Couldn't load [%s] trajectories from dir [%s] for epoch [%s] and "
          "temperature [%s]", n_trajectories, trajectory_dir, epoch,
          temperature)
      assert bt

    # Doing this is important, since we want to modify `env` so that it looks
    # like `env` was actually played and the trajectories came from it.
    env.trajectories = bt

    trajs = env_problem_utils.get_completed_trajectories_from_env(
        env, n_trajectories)
    n_done = len(trajs)
    timing_info = {}
    return trajs, n_done, timing_info, self._model_state

  def collect_trajectories(self,
                           train=True,
                           temperature=1.0,
                           abort_fn=None,
                           raw_trajectory=False):
    self._rng, key = jax_random.split(self._rng)

    env = self.train_env
    max_timestep = self._max_timestep
    should_reset = self._should_reset
    if not train:  # eval
      env = self.eval_env
      max_timestep = self._max_timestep_eval
      should_reset = True

    n_trajectories = env.batch_size

    # If async, read the required trajectories for the epoch.
    if self._async_mode:
      trajs, n_done, timing_info, self._model_state = self.collect_trajectories_async(
          env,
          train=train,
          n_trajectories=n_trajectories,
          temperature=temperature)
    else:
      trajs, n_done, timing_info, self._model_state = ppo.collect_trajectories(
          env,
          policy_fn=self._policy_fun,
          n_trajectories=n_trajectories,
          max_timestep=max_timestep,
          state=self._model_state,
          rng=key,
          len_history_for_policy=self._len_history_for_policy,
          boundary=self._boundary,
          reset=should_reset,
          temperature=temperature,
          abort_fn=abort_fn,
          raw_trajectory=raw_trajectory,
      )

    if train:
      self._n_trajectories_done += n_done

    return trajs, n_done, timing_info, self._model_state

  def train_epoch(self, evaluate=True):
    """Train one PPO epoch."""
    epoch_start_time = time.time()

    # Evaluate the policy.
    policy_eval_start_time = time.time()
    if evaluate and (self._epoch + 1) % self._eval_every_n == 0:
      self._rng, key = jax_random.split(self._rng, num=2)
      self.evaluate()

    policy_eval_time = ppo.get_time(policy_eval_start_time)

    trajectory_collection_start_time = time.time()
    logging.vlog(1, "PPO epoch [% 6d]: collecting trajectories.", self._epoch)
    self._rng, key = jax_random.split(self._rng)
    trajs, _, timing_info, self._model_state = self.collect_trajectories(
        train=True, temperature=1.0)
    trajs = [(t[0], t[1], t[2], t[4]) for t in trajs]
    self._should_reset = False
    trajectory_collection_time = ppo.get_time(trajectory_collection_start_time)

    logging.vlog(1, "Collecting trajectories took %0.2f msec.",
                 trajectory_collection_time)

    rewards = np.array([np.sum(traj[2]) for traj in trajs])
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)

    self._train_sw.scalar(
        "train/reward_mean_truncated", avg_reward, step=self._epoch)
    if evaluate and not self._separate_eval:
      metrics = {"raw": {1.0: {"mean": avg_reward, "std": std_reward}}}
      ppo.write_eval_reward_summaries(metrics, self._eval_sw, self._epoch)

    logging.vlog(1, "Rewards avg=[%0.2f], max=[%0.2f], min=[%0.2f], all=%s",
                 avg_reward, max_reward, min_reward,
                 [float(np.sum(traj[2])) for traj in trajs])

    logging.vlog(1,
                 "Trajectory Length average=[%0.2f], max=[%0.2f], min=[%0.2f]",
                 float(sum(len(traj[0]) for traj in trajs)) / len(trajs),
                 max(len(traj[0]) for traj in trajs),
                 min(len(traj[0]) for traj in trajs))
    logging.vlog(2, "Trajectory Lengths: %s", [len(traj[0]) for traj in trajs])

    preprocessing_start_time = time.time()
    (padded_observations, padded_actions, padded_rewards, reward_mask,
     padded_infos) = self._preprocess_trajectories(trajs)
    preprocessing_time = ppo.get_time(preprocessing_start_time)

    logging.vlog(1, "Preprocessing trajectories took %0.2f msec.",
                 ppo.get_time(preprocessing_start_time))
    logging.vlog(1, "Padded Observations' shape [%s]",
                 str(padded_observations.shape))
    logging.vlog(1, "Padded Actions' shape [%s]", str(padded_actions.shape))
    logging.vlog(1, "Padded Rewards' shape [%s]", str(padded_rewards.shape))

    # Some assertions.
    B, RT = padded_rewards.shape  # pylint: disable=invalid-name
    B, AT = padded_actions.shape  # pylint: disable=invalid-name
    assert (B, RT) == reward_mask.shape
    assert B == padded_observations.shape[0]

    log_prob_recompute_start_time = time.time()
    # TODO(pkozakowski): The following commented out code collects the network
    # predictions made while stepping the environment and uses them in PPO
    # training, so that we can use non-deterministic networks (e.g. with
    # dropout). This does not work well with serialization, so instead we
    # recompute all network predictions. Let's figure out a solution that will
    # work with both serialized sequences and non-deterministic networks.

    # assert ("log_prob_actions" in padded_infos and
    #         "value_predictions" in padded_infos)
    # These are the actual log-probabs and value predictions seen while picking
    # the actions.
    # actual_log_probabs_traj = padded_infos["log_prob_actions"]
    # actual_value_predictions_traj = padded_infos["value_predictions"]

    # assert (B, T, C) == actual_log_probabs_traj.shape[:3]
    # A = actual_log_probabs_traj.shape[3]  # pylint: disable=invalid-name
    # assert (B, T, 1) == actual_value_predictions_traj.shape

    del padded_infos

    # TODO(afrozm): log-probabs doesn't need to be (B, T+1, C, A) it can do with
    # (B, T, C, A), so make that change throughout.

    # NOTE: We don't have the log-probabs and value-predictions for the last
    # observation, so we re-calculate for everything, but use the original ones
    # for all but the last time-step.
    self._rng, key = jax_random.split(self._rng)

    log_probabs_traj, value_predictions_traj, self._model_state, _ = (
        self._get_predictions(padded_observations, self._model_state, rng=key))

    assert (B, AT) == log_probabs_traj.shape[:2]
    assert (B, AT) == value_predictions_traj.shape

    # TODO(pkozakowski): Commented out for the same reason as before.

    # Concatenate the last time-step's log-probabs and value predictions to the
    # actual log-probabs and value predictions and use those going forward.
    # log_probabs_traj = np.concatenate(
    #     (actual_log_probabs_traj, log_probabs_traj[:, -1:, :]), axis=1)
    # value_predictions_traj = np.concatenate(
    #     (actual_value_predictions_traj, value_predictions_traj[:, -1:, :]),
    #     axis=1)

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
            self._rewards_to_actions,
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
    opt_batch_size = min(self._optimizer_batch_size, B)
    index_batches = ppo.shuffled_index_batches(
        dataset_size=B, batch_size=opt_batch_size
    )
    for (index_batch, key) in zip(index_batches, keys):
      k1, k2, k3 = jax_random.split(key, num=3)
      t = time.time()
      # Update the optimizer state on the sampled minibatch.
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
              log_probabs_traj[index_batch],
              value_predictions_traj[index_batch],
              padded_observations[index_batch],
              padded_actions[index_batch],
              self._rewards_to_actions,
              padded_rewards[index_batch],
              reward_mask[index_batch],
              c1=self._c1,
              c2=self._c2,
              gamma=self._gamma,
              lambda_=self._lambda_,
              state=self._model_state,
              rng=k1))
      opt_step += 1
      self._total_opt_step += 1

      # Compute the approx KL for early stopping. Use the whole dataset - as we
      # only do inference, it should fit in the memory.
      (log_probab_actions_new, _) = (
          self._policy_and_value_net_apply(
              padded_observations,
              params=self._policy_and_value_net_params,
              state=self._model_state,
              rng=k2))

      action_mask = np.dot(
          np.pad(reward_mask, ((0, 0), (0, 1))), self._rewards_to_actions
      )
      approx_kl = ppo.approximate_kl(log_probab_actions_new, log_probabs_traj,
                                     action_mask)

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
                self._rewards_to_actions,
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

    logging.info(
        "PPO epoch [% 6d], Reward[min, max, avg] [%5.2f,%5.2f,%5.2f], Combined"
        " Loss(ppo, value, entropy) [%2.5f(%2.5f,%2.5f,%2.5f)]", self._epoch,
        min_reward, max_reward, avg_reward, combined_loss, ppo_loss, value_loss,
        entropy_bonus)

    # Bump the epoch counter before saving a checkpoint, so that a call to
    # save() after the training loop is a no-op if a checkpoint was saved last
    # epoch - otherwise it would bump the epoch counter on the checkpoint.
    last_epoch = self._epoch
    self._epoch += 1

    # Save parameters every time we see the end of at least a fraction of batch
    # number of trajectories that are done (not completed -- completed includes
    # truncated and done).
    # Also don't save too frequently, enforce a minimum gap.
    policy_save_start_time = time.time()
    # TODO(afrozm): Refactor to trax.save_state.
    if (self._n_trajectories_done >=
        self._done_frac_for_policy_save * self.train_env.batch_size and
        self._epoch % self._save_every_n == 0) or self._async_mode:
      self.save()
    policy_save_time = ppo.get_time(policy_save_start_time)

    epoch_time = ppo.get_time(epoch_start_time)

    timing_dict = {
        "epoch": epoch_time,
        "policy_eval": policy_eval_time,
        "trajectory_collection": trajectory_collection_time,
        "preprocessing": preprocessing_time,
        "log_prob_recompute": log_prob_recompute_time,
        "loss_compute": loss_compute_time,
        "optimization": optimization_time,
        "policy_save": policy_save_time,
    }

    timing_dict.update(timing_info)

    for k, v in timing_dict.items():
      self._timing_sw.scalar("timing/%s" % k, v, step=last_epoch)

    max_key_len = max(len(k) for k in timing_dict)
    timing_info_list = [
        "%s : % 10.2f" % (k.rjust(max_key_len + 1), v)
        for k, v in sorted(timing_dict.items())
    ]
    logging.info("PPO epoch [% 6d], Timings: \n%s", last_epoch,
                 "\n".join(timing_info_list))

    # Flush summary writers once in a while.
    if self._epoch % 1000 == 0:
      self.flush_summaries()

  def evaluate(self):
    """Evaluate the agent."""
    if not self._separate_eval:
      return
    logging.vlog(1, "PPO epoch [% 6d]: evaluating policy.", self._epoch)

    processed_reward_sums = collections.defaultdict(list)
    raw_reward_sums = collections.defaultdict(list)
    for _ in range(self._n_evals):
      for temperature in self._eval_temperatures:
        trajs, _, _, self._model_state = self.collect_trajectories(
            train=False, temperature=temperature)

        processed_reward_sums[temperature].extend(
            sum(traj[2]) for traj in trajs)
        raw_reward_sums[temperature].extend(sum(traj[3]) for traj in trajs)

    # Return the mean and standard deviation for each temperature.
    def compute_stats(reward_dict):
      return {
          temperature: {  # pylint: disable=g-complex-comprehension
              "mean": onp.mean(rewards),
              "std": onp.std(rewards)
          } for (temperature, rewards) in reward_dict.items()
      }

    reward_stats = {
        "processed": compute_stats(processed_reward_sums),
        "raw": compute_stats(raw_reward_sums),
    }

    ppo.write_eval_reward_summaries(
        reward_stats, self._eval_sw, epoch=self._epoch)

  def save(self):
    """Save the agent parameters."""
    logging.vlog(1, "PPO epoch [% 6d]: saving model.", self._epoch)
    ppo.save_opt_state(
        self._output_dir,
        self._policy_and_value_opt_state,
        self._model_state,
        self._epoch,
        self._total_opt_step,
    )
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

  # Prepares the trajectories for policy training.
  def _preprocess_trajectories(self, trajectories):
    (_, reward_mask, observations, actions, rewards, infos) = (
        ppo.pad_trajectories(trajectories, boundary=self._max_timestep)
    )
    assert self.train_env.observation_space.shape == observations.shape[2:]
    if not self._serialized_sequence_policy:
      # Add one timestep at the end, so it's compatible with
      # self._rewards_to_actions.
      pad_width = ((0, 0), (0, 1)) + ((0, 0),) * (actions.ndim - 2)
      actions = np.pad(actions, pad_width)
      actions = np.reshape(actions, (actions.shape[0], -1))
    else:
      (observations, actions) = self._serialize_trajectories(
          observations, actions, reward_mask
      )
    return (observations, actions, rewards, reward_mask, infos)

  def _serialize_trajectories(self, observations, actions, reward_mask):
    (reprs, _) = serialization_utils.serialize_observations_and_actions(
        observations=observations,
        actions=actions,
        mask=reward_mask,
        **self._serialization_kwargs
    )
    # Mask out actions in the representation - otherwise we sample an action
    # based on itself.
    observations = reprs * serialization_utils.observation_mask(
        **self._serialization_kwargs
    )
    actions = reprs
    return (observations, actions)

  # A function to get the policy and value predictions.
  def _get_predictions(self, observations, state, rng=None):
    """Returns log-probs, value predictions and key back."""
    key, key1 = jax_random.split(rng, num=2)

    (log_probs, value_preds) = self._policy_and_value_net_apply(
        observations, params=self._policy_and_value_net_params, state=state,
        rng=key1)

    return log_probs, value_preds, state, key

  def _policy_fun(self, observations, lengths, state, rng):
    (batch_size, n_timesteps) = observations.shape[:2]
    if self._serialized_sequence_policy:
      actions = np.zeros(
          (batch_size, n_timesteps - 1) + self.train_env.action_space.shape,
          dtype=self.train_env.action_space.dtype,
      )
      reward_mask = np.ones((batch_size, n_timesteps - 1), dtype=np.int32)
      (observations, _) = self._serialize_trajectories(
          observations, actions, reward_mask
      )
    (log_probs, value_preds, state, rng) = self._get_predictions(
        observations, state=state, rng=rng
    )
    # We need the log_probs of those actions that correspond to the last actual
    # time-step.
    index = lengths - 1  # Since we want to index using lengths.
    pred_index = self._calc_action_index(index)
    log_probs = log_probs[
        np.arange(batch_size)[:, None, None],
        pred_index[:, :, None],
        np.arange(self._n_actions),
    ]
    value_preds = value_preds[np.arange(batch_size)[:, None], pred_index]
    return (log_probs, value_preds, state, rng)

  def _calc_action_index(self, reward_index):
    # Project the one-hot position in the reward sequence onto the action
    # sequence to figure out which actions correspond to that position.
    one_hot_index = np.eye(self._rewards_to_actions.shape[0])[reward_index]
    action_mask = np.dot(one_hot_index, self._rewards_to_actions)
    # Compute the number of symbols in an action. It's just the number of 1s in
    # the mask.
    action_length = int(np.sum(action_mask[0]))
    # Argmax stops on the first occurrence, so we use it to find the first 1 in
    # the mask.
    action_start_index = np.argmax(action_mask, axis=1)
    return action_start_index[:, None] + np.arange(action_length)[None, :]
