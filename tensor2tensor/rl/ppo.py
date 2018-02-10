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


def define_ppo_step(observation, action, reward, done, value, old_pdf,
                    policy_factory, config):
  """A step of PPO."""
  new_policy_dist, new_value, _ = policy_factory(observation)
  new_pdf = new_policy_dist.prob(action)

  ratio = new_pdf/old_pdf
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

  value_error = calculate_discounted_return(
      reward, new_value, done, config.gae_gamma, config.gae_lambda) - value
  value_loss = config.value_loss_coef * tf.reduce_mean(value_error ** 2)

  entropy = new_policy_dist.entropy()
  entropy_loss = -config.entropy_loss_coef * tf.reduce_mean(entropy)

  total_loss = policy_loss + value_loss + entropy_loss

  optimization_op = tf.contrib.layers.optimize_loss(
      loss=total_loss,
      global_step=tf.train.get_or_create_global_step(),
      optimizer=config.optimizer,
      learning_rate=config.learning_rate)

  with tf.control_dependencies([optimization_op]):
    return [tf.identity(x) for x in (policy_loss, value_loss, entropy_loss)]


def define_ppo_epoch(memory, policy_factory, config):
  """An epoch of PPO."""
  observation, reward, done, action, old_pdf, value = memory

  # This is to avoid propagating gradients though simulation of simulation
  observation = tf.stop_gradient(observation)
  action = tf.stop_gradient(action)
  reward = tf.stop_gradient(reward)
  done = tf.stop_gradient(done)
  value = tf.stop_gradient(value)
  old_pdf = tf.stop_gradient(old_pdf)

  policy_loss, value_loss, entropy_loss = tf.scan(
      lambda _1, _2: define_ppo_step(  # pylint: disable=g-long-lambda
          observation, action, reward, done, value,
          old_pdf, policy_factory, config),
      tf.range(config.optimization_epochs),
      [0., 0., 0.],
      parallel_iterations=1)

  summaries = [tf.summary.scalar("policy loss", tf.reduce_mean(policy_loss)),
               tf.summary.scalar("value loss", tf.reduce_mean(value_loss)),
               tf.summary.scalar("entropy loss", tf.reduce_mean(entropy_loss))]

  losses_summary = tf.summary.merge(summaries)

  losses_summary = tf.Print(losses_summary,
                            [tf.reduce_mean(policy_loss)], "policy loss: ")
  losses_summary = tf.Print(losses_summary,
                            [tf.reduce_mean(value_loss)], "value loss: ")
  losses_summary = tf.Print(losses_summary,
                            [tf.reduce_mean(entropy_loss)], "entropy loss: ")

  return losses_summary


def calculate_discounted_return(reward, value, done, discount, unused_lambda):
  """Discounted Monte-Carlo returns."""
  done = tf.cast(done, tf.float32)
  reward2 = done[-1, :] * reward[-1, :] + (1 - done[-1, :]) * value[-1, :]
  reward = tf.concat([reward[:-1,], reward2[None, ...]], axis=0)
  return_ = tf.reverse(tf.scan(
      lambda agg, cur: cur[0] + (1 - cur[1]) * discount * agg,  # fn
      [tf.reverse(reward, [0]),  # elem
       tf.reverse(done, [0])],
      tf.zeros_like(reward[0, :]),  # initializer
      1,
      False), [0])
  return tf.check_numerics(return_, "return")


def calculate_generalized_advantage_estimator(
    reward, value, done, gae_gamma, gae_lambda):
  """Generalized advantage estimator."""
  # Below is slight weirdness, we set the last reward to 0.
  # This makes the adventantage to be 0 in the last timestep.
  reward = tf.concat([reward[:-1, :], value[-1:, :]], axis=0)
  next_value = tf.concat([value[1:, :], tf.zeros_like(value[-1:, :])], axis=0)
  next_not_done = 1 - tf.cast(tf.concat(
      [done[1:, :], tf.zeros_like(done[-1:, :])], axis=0), tf.float32)
  delta = reward + gae_gamma * next_value * next_not_done - value

  return_ = tf.reverse(tf.scan(
      lambda agg, cur: cur[0] + cur[1] * gae_gamma * gae_lambda * agg,
      [tf.reverse(delta, [0]), tf.reverse(next_not_done, [0])],
      tf.zeros_like(delta[0, :]),
      1, False), [0])
  return tf.check_numerics(tf.stop_gradient(return_), "return")
