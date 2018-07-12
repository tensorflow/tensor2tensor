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
"""PPO algorithm implementation.

Based on: https://arxiv.org/abs/1707.06347
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.rl.envs.utils import get_policy

import tensorflow as tf


def get_optimiser(config):
  if config.optimizer == "Adam":
    return tf.train.AdamOptimizer(learning_rate=config.learning_rate)
  return config.optimizer(learning_rate=config.learning_rate)


def define_ppo_step(data_points, optimizer, hparams):
  """Define ppo step."""
  observation, action, discounted_reward, norm_advantage, old_pdf = data_points
  new_policy_dist, new_value, _ = get_policy(observation, hparams)
  new_pdf = new_policy_dist.prob(action)

  ratio = new_pdf / old_pdf
  clipped_ratio = tf.clip_by_value(ratio, 1 - hparams.clipping_coef,
                                   1 + hparams.clipping_coef)

  surrogate_objective = tf.minimum(clipped_ratio * norm_advantage,
                                   ratio * norm_advantage)
  policy_loss = -tf.reduce_mean(surrogate_objective)

  value_error = new_value - discounted_reward
  value_loss = hparams.value_loss_coef * tf.reduce_mean(value_error ** 2)

  entropy = new_policy_dist.entropy()
  entropy_loss = -hparams.entropy_loss_coef * tf.reduce_mean(entropy)

  losses = [policy_loss, value_loss, entropy_loss]

  gradients = [list(zip(*optimizer.compute_gradients(loss)))
               for loss in losses]

  gradients_norms = [tf.global_norm(gradient[0]) for gradient in gradients]

  gradients_flat = sum([gradient[0] for gradient in gradients], ())
  gradients_variables_flat = sum([gradient[1] for gradient in gradients], ())

  if hparams.max_gradients_norm:
    gradients_flat, _ = tf.clip_by_global_norm(gradients_flat,
                                               hparams.max_gradients_norm)

  optimize_op = optimizer.apply_gradients(zip(gradients_flat,
                                              gradients_variables_flat))

  with tf.control_dependencies([optimize_op]):
    return [tf.identity(x) for x in losses + gradients_norms]


def define_ppo_epoch(memory, hparams):
  """PPO epoch."""
  observation, reward, done, action, old_pdf, value = memory

  # This is to avoid propagating gradients through simulated environment.
  observation = tf.stop_gradient(observation)
  action = tf.stop_gradient(action)
  reward = tf.stop_gradient(reward)
  if hasattr(hparams, "rewards_preprocessing_fun"):
    reward = hparams.rewards_preprocessing_fun(reward)
  done = tf.stop_gradient(done)
  value = tf.stop_gradient(value)
  old_pdf = tf.stop_gradient(old_pdf)

  advantage = calculate_generalized_advantage_estimator(
      reward, value, done, hparams.gae_gamma, hparams.gae_lambda)

  discounted_reward = tf.stop_gradient(advantage + value)

  advantage_mean, advantage_variance = tf.nn.moments(advantage, axes=[0, 1],
                                                     keep_dims=True)
  advantage_normalized = tf.stop_gradient(
      (advantage - advantage_mean)/(tf.sqrt(advantage_variance) + 1e-8))

  add_lists_elementwise = lambda l1, l2: [x + y for x, y in zip(l1, l2)]

  number_of_batches = (hparams.epoch_length * hparams.optimization_epochs
                       / hparams.optimization_batch_size)

  dataset = tf.data.Dataset.from_tensor_slices(
      (observation, action, discounted_reward, advantage_normalized, old_pdf))
  dataset = dataset.shuffle(buffer_size=hparams.epoch_length,
                            reshuffle_each_iteration=True)
  dataset = dataset.repeat(hparams.optimization_epochs)
  dataset = dataset.batch(hparams.optimization_batch_size)
  iterator = dataset.make_initializable_iterator()
  optimizer = get_optimiser(hparams)

  with tf.control_dependencies([iterator.initializer]):
    ppo_step_rets = tf.scan(
        lambda a, i: add_lists_elementwise(  # pylint: disable=g-long-lambda
            a, define_ppo_step(iterator.get_next(), optimizer, hparams)),
        tf.range(number_of_batches),
        [0., 0., 0., 0., 0., 0.],
        parallel_iterations=1)

  ppo_summaries = [tf.reduce_mean(ret) / number_of_batches
                   for ret in ppo_step_rets]
  summaries_names = ["policy_loss", "value_loss", "entropy_loss",
                     "policy_gradient", "value_gradient", "entropy_gradient"]

  summaries = [tf.summary.scalar(summary_name, summary)
               for summary_name, summary in zip(summaries_names, ppo_summaries)]
  losses_summary = tf.summary.merge(summaries)

  for summary_name, summary in zip(summaries_names, ppo_summaries):
    losses_summary = tf.Print(losses_summary, [summary], summary_name + ": ")

  return losses_summary


def calculate_generalized_advantage_estimator(
    reward, value, done, gae_gamma, gae_lambda):
  """Generalized advantage estimator."""

  # Below is slight weirdness, we set the last reward to 0.
  # This makes the advantage to be 0 in the last timestep
  reward = tf.concat([reward[:-1, :], value[-1:, :]], axis=0)
  next_value = tf.concat([value[1:, :], tf.zeros_like(value[-1:, :])], axis=0)
  next_not_done = 1 - tf.cast(tf.concat([done[1:, :],
                                         tf.zeros_like(done[-1:, :])], axis=0),
                              tf.float32)
  delta = reward + gae_gamma * next_value * next_not_done - value

  return_ = tf.reverse(tf.scan(
      lambda agg, cur: cur[0] + cur[1] * gae_gamma * gae_lambda * agg,
      [tf.reverse(delta, [0]), tf.reverse(next_not_done, [0])],
      tf.zeros_like(delta[0, :]),
      parallel_iterations=1), [0])
  return tf.check_numerics(return_, "return")
