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

"""PPO algorithm implementation.

Based on: https://arxiv.org/abs/1707.06347
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers
from tensor2tensor.models.research.rl import get_policy
from tensor2tensor.utils import learning_rate
from tensor2tensor.utils import optimize

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


def define_ppo_step(data_points, hparams, action_space, lr, epoch=-1,
                    distributional_size=1, distributional_subscale=0.04):
  """Define ppo step."""
  del distributional_subscale
  (observation, action, discounted_reward, discounted_reward_probs,
   norm_advantage, old_pdf) = data_points

  obs_shape = common_layers.shape_list(observation)
  observation = tf.reshape(
      observation, [obs_shape[0] * obs_shape[1]] + obs_shape[2:]
  )
  (logits, new_value) = get_policy(observation, hparams, action_space,
                                   epoch=epoch,
                                   distributional_size=distributional_size)
  logits = tf.reshape(logits, obs_shape[:2] + [action_space.n])
  new_policy_dist = tfp.distributions.Categorical(logits=logits)

  new_pdf = new_policy_dist.prob(action)

  ratio = new_pdf / old_pdf
  clipped_ratio = tf.clip_by_value(ratio, 1 - hparams.clipping_coef,
                                   1 + hparams.clipping_coef)

  surrogate_objective = tf.minimum(clipped_ratio * norm_advantage,
                                   ratio * norm_advantage)
  policy_loss = -tf.reduce_mean(surrogate_objective)

  if distributional_size > 1:
    new_value = tf.reshape(new_value, obs_shape[:2] + [distributional_size])
    new_value = tf.nn.log_softmax(new_value, axis=-1)
    value_shape = common_layers.shape_list(new_value)
    # The above is the new value distribution. We are also given as discounted
    # reward the value distribution and the corresponding probabilities.
    # The given discounted reward is already rounded to integers but in range
    # increased by 2x for greater fidelity. Increase range of new_values here.
    new_value_shifted = tf.concat([new_value[1:], new_value[-1:]], axis=0)
    new_value_mean = (new_value + new_value_shifted) / 2
    new_value = tf.concat([tf.expand_dims(new_value, axis=-1),
                           tf.expand_dims(new_value_mean, axis=-1)], -1)
    new_value = tf.reshape(new_value, value_shape[:-1] + [2 * value_shape[-1]])
    # Cast discounted reward to integers and gather the new log-probs for them.
    discounted_reward = tf.cast(discounted_reward, tf.int32)
    value_loss = tf.batch_gather(new_value, discounted_reward)
    # Weight the gathered (new) log-probs by the old probabilities.
    discounted_reward_probs = tf.expand_dims(discounted_reward_probs, axis=1)
    value_loss = - tf.reduce_sum(value_loss * discounted_reward_probs, axis=-1)
    # Take the mean over batch and time as final loss, multiply by coefficient.
    value_loss = hparams.value_loss_coef * tf.reduce_mean(value_loss)
  else:
    new_value = tf.reshape(new_value, obs_shape[:2])
    value_error = new_value - discounted_reward
    value_loss = hparams.value_loss_coef * tf.reduce_mean(value_error ** 2)

  entropy = new_policy_dist.entropy()
  entropy_loss = -hparams.entropy_loss_coef * tf.reduce_mean(entropy)

  losses = [policy_loss, value_loss, entropy_loss]
  loss = sum(losses)
  variables = tf.global_variables(hparams.policy_network + "/.*")
  train_op = optimize.optimize(loss, lr, hparams, variables=variables)

  with tf.control_dependencies([train_op]):
    return [tf.identity(x) for x in losses]


def _distributional_to_value(value_d, size, subscale, threshold):
  """Get a scalar value out of a value distribution in distributional RL."""
  half = size // 2
  value_range = (tf.to_float(tf.range(-half, half)) + 0.5) * subscale
  probs = tf.nn.softmax(value_d)

  if threshold == 0.0:
    return tf.reduce_sum(probs * value_range, axis=-1)

  # accumulated_probs[..., i] is the sum of probabilities in buckets upto i
  # so it is the probability that value <= i'th bucket value
  accumulated_probs = tf.cumsum(probs, axis=-1)
  # New probs are 0 on all lower buckets, until the threshold
  probs = tf.where(accumulated_probs < threshold, tf.zeros_like(probs), probs)
  probs /= tf.reduce_sum(probs, axis=-1, keepdims=True)  # Re-normalize.
  return tf.reduce_sum(probs * value_range, axis=-1)


def define_ppo_epoch(memory, hparams, action_space, batch_size,
                     distributional_size=1, distributional_subscale=0.04,
                     distributional_threshold=0.0, epoch=-1):
  """PPO epoch."""
  observation, reward, done, action, old_pdf, value_sm = memory

  # This is to avoid propagating gradients through simulated environment.
  observation = tf.stop_gradient(observation)
  action = tf.stop_gradient(action)
  reward = tf.stop_gradient(reward)
  if hasattr(hparams, "rewards_preprocessing_fun"):
    reward = hparams.rewards_preprocessing_fun(reward)
  done = tf.stop_gradient(done)
  value_sm = tf.stop_gradient(value_sm)
  old_pdf = tf.stop_gradient(old_pdf)

  value = value_sm
  if distributional_size > 1:
    value = _distributional_to_value(
        value_sm, distributional_size, distributional_subscale,
        distributional_threshold)

  advantage = calculate_generalized_advantage_estimator(
      reward, value, done, hparams.gae_gamma, hparams.gae_lambda)

  if distributional_size > 1:
    # Create discounted reward values range.
    half = distributional_size // 2
    value_range = tf.to_float(tf.range(-half, half)) + 0.5  # Mid-bucket value.
    value_range *= distributional_subscale
    # Acquire new discounted rewards by using the above range as end-values.
    end_values = tf.expand_dims(value_range, 0)
    discounted_reward = discounted_rewards(
        reward, done, hparams.gae_gamma, end_values)
    # Re-normalize the discounted rewards to integers, in [0, dist_size] range.
    discounted_reward /= distributional_subscale
    discounted_reward += half
    discounted_reward = tf.maximum(discounted_reward, 0.0)
    discounted_reward = tf.minimum(discounted_reward, distributional_size)
    # Multiply the rewards by 2 for greater fidelity and round to integers.
    discounted_reward = tf.stop_gradient(tf.round(2 * discounted_reward))
    # The probabilities corresponding to the end values from old predictions.
    discounted_reward_prob = tf.stop_gradient(value_sm[-1])
    discounted_reward_prob = tf.nn.softmax(discounted_reward_prob, axis=-1)
  else:
    discounted_reward = tf.stop_gradient(advantage + value[:-1])
    discounted_reward_prob = discounted_reward  # Unused in this case.

  advantage_mean, advantage_variance = tf.nn.moments(advantage, axes=[0, 1],
                                                     keep_dims=True)
  advantage_normalized = tf.stop_gradient(
      (advantage - advantage_mean)/(tf.sqrt(advantage_variance) + 1e-8))

  add_lists_elementwise = lambda l1, l2: [x + y for x, y in zip(l1, l2)]

  number_of_batches = ((hparams.epoch_length-1) * hparams.optimization_epochs
                       // hparams.optimization_batch_size)
  epoch_length = hparams.epoch_length
  if hparams.effective_num_agents is not None:
    number_of_batches *= batch_size
    number_of_batches //= hparams.effective_num_agents
    epoch_length //= hparams.effective_num_agents

  assert number_of_batches > 0, "Set the paremeters so that number_of_batches>0"
  lr = learning_rate.learning_rate_schedule(hparams)

  shuffled_indices = [tf.random.shuffle(tf.range(epoch_length - 1))
                      for _ in range(hparams.optimization_epochs)]
  shuffled_indices = tf.concat(shuffled_indices, axis=0)
  shuffled_indices = shuffled_indices[:number_of_batches *
                                      hparams.optimization_batch_size]
  indices_of_batches = tf.reshape(shuffled_indices,
                                  shape=(-1, hparams.optimization_batch_size))
  input_tensors = [observation, action, discounted_reward,
                   discounted_reward_prob, advantage_normalized, old_pdf]

  ppo_step_rets = tf.scan(
      lambda a, i: add_lists_elementwise(  # pylint: disable=g-long-lambda
          a, define_ppo_step(
              [tf.gather(t, indices_of_batches[i, :]) for t in input_tensors],
              hparams, action_space, lr,
              epoch=epoch,
              distributional_size=distributional_size,
              distributional_subscale=distributional_subscale
          )),
      tf.range(number_of_batches),
      [0., 0., 0.],
      parallel_iterations=1)

  ppo_summaries = [tf.reduce_mean(ret) / number_of_batches
                   for ret in ppo_step_rets]
  ppo_summaries.append(lr)
  summaries_names = [
      "policy_loss", "value_loss", "entropy_loss", "learning_rate"
  ]

  summaries = [tf.summary.scalar(summary_name, summary)
               for summary_name, summary in zip(summaries_names, ppo_summaries)]
  losses_summary = tf.summary.merge(summaries)

  for summary_name, summary in zip(summaries_names, ppo_summaries):
    losses_summary = tf.Print(losses_summary, [summary], summary_name + ": ")

  return losses_summary


def calculate_generalized_advantage_estimator(
    reward, value, done, gae_gamma, gae_lambda):
  # pylint: disable=g-doc-args
  """Generalized advantage estimator.

  Returns:
    GAE estimator. It will be one element shorter than the input; this is
    because to compute GAE for [0, ..., N-1] one needs V for [1, ..., N].
  """
  # pylint: enable=g-doc-args

  next_value = value[1:, :]
  next_not_done = 1 - tf.cast(done[1:, :], tf.float32)
  delta = (reward[:-1, :] + gae_gamma * next_value * next_not_done
           - value[:-1, :])

  return_ = tf.reverse(tf.scan(
      lambda agg, cur: cur[0] + cur[1] * gae_gamma * gae_lambda * agg,
      [tf.reverse(delta, [0]), tf.reverse(next_not_done, [0])],
      tf.zeros_like(delta[0, :]),
      parallel_iterations=1), [0])
  return tf.check_numerics(return_, "return")


def discounted_rewards(reward, done, gae_gamma, end_values):
  """Discounted rewards."""
  not_done = tf.expand_dims(1 - tf.cast(done, tf.float32), axis=2)
  end_values = end_values * not_done[-1, :, :]
  return_ = tf.scan(
      lambda agg, cur: cur + gae_gamma * agg,
      tf.expand_dims(reward, axis=2) * not_done,
      initializer=end_values,
      reverse=True,
      back_prop=False,
      parallel_iterations=2)
  return tf.check_numerics(return_, "return")
