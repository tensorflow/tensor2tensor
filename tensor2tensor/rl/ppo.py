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

import tensorflow as tf


def get_optimizer(config):
  if config.optimizer == "Adam":
    return tf.train.AdamOptimizer(config.learning_rate)
  return config.optimizer(config.learning_rate)


def define_ppo_step(observation, action, reward, done, value, old_pdf,
                    policy_factory, config):
  """Step of PPO."""
  new_policy_dist, new_value, _ = policy_factory(observation)
  new_pdf = new_policy_dist.prob(action)

  ratio = new_pdf / old_pdf
  clipped_ratio = tf.clip_by_value(ratio, 1 - config.clipping_coef,
                                   1 + config.clipping_coef)

  advantage = calculate_generalized_advantage_estimator(
      reward, value, done, config.gae_gamma, config.gae_lambda)

  advantage_mean, advantage_variance = tf.nn.moments(advantage, axes=[0, 1],
                                                     keep_dims=True)
  advantage_normalized = tf.stop_gradient(
      (advantage - advantage_mean)/(tf.sqrt(advantage_variance) + 1e-8))

  surrogate_objective = tf.minimum(clipped_ratio * advantage_normalized,
                                   ratio * advantage_normalized)
  policy_loss = -tf.reduce_mean(surrogate_objective)

  value_error = calculate_generalized_advantage_estimator(
      reward, new_value, done, config.gae_gamma, config.gae_lambda)
  value_loss = config.value_loss_coef * tf.reduce_mean(value_error ** 2)

  entropy = new_policy_dist.entropy()
  entropy_loss = -config.entropy_loss_coef * tf.reduce_mean(entropy)

  optimizer = get_optimizer(config)
  losses = [policy_loss, value_loss, entropy_loss]

  gradients = [list(zip(*optimizer.compute_gradients(loss))) for loss in losses]

  gradients_norms = [tf.global_norm(gradient[0]) for gradient in gradients]

  gradients_flat = sum([gradient[0] for gradient in gradients], ())
  gradients_variables_flat = sum([gradient[1] for gradient in gradients], ())

  optimize_op = optimizer.apply_gradients(zip(gradients_flat,
                                              gradients_variables_flat))

  with tf.control_dependencies([optimize_op]):
    return [tf.identity(x) for x in losses + gradients_norms]


def define_ppo_epoch(memory, policy_factory, config):
  """PPO epoch."""
  observation, reward, done, action, old_pdf, value = memory

  # This is to avoid propagating gradients though simulation of simulation
  observation = tf.stop_gradient(observation)
  action = tf.stop_gradient(action)
  reward = tf.stop_gradient(reward)
  done = tf.stop_gradient(done)
  value = tf.stop_gradient(value)
  old_pdf = tf.stop_gradient(old_pdf)

  ppo_step_rets = tf.scan(
      lambda _1, _2: define_ppo_step(  # pylint: disable=g-long-lambda
          observation, action, reward, done, value,
          old_pdf, policy_factory, config),
      tf.range(config.optimization_epochs),
      [0., 0., 0., 0., 0., 0.],
      parallel_iterations=1)

  ppo_summaries = [tf.reduce_mean(ret) for ret in ppo_step_rets]
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
  # This makes the adventantage to be 0 in the last timestep
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
