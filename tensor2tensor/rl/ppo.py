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

import tensorflow as tf
import tensorflow_probability as tfp


def define_ppo_step(data_points, hparams, action_space, lr):
  """Define ppo step."""
  observation, action, discounted_reward, norm_advantage, old_pdf = data_points

  obs_shape = common_layers.shape_list(observation)
  observation = tf.reshape(
      observation, [obs_shape[0] * obs_shape[1]] + obs_shape[2:]
  )
  (logits, new_value) = get_policy(observation, hparams, action_space)
  logits = tf.reshape(logits, obs_shape[:2] + [action_space.n])
  new_value = tf.reshape(new_value, obs_shape[:2])
  new_policy_dist = tfp.distributions.Categorical(logits=logits)

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
  loss = sum(losses)
  variables = tf.global_variables(hparams.policy_network + "/.*")
  train_op = optimize.optimize(loss, lr, hparams, variables=variables)

  with tf.control_dependencies([train_op]):
    return [tf.identity(x) for x in losses]


def define_ppo_epoch(memory, hparams, action_space, batch_size):
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

  discounted_reward = tf.stop_gradient(advantage + value[:-1])

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
                   advantage_normalized, old_pdf]

  ppo_step_rets = tf.scan(
      lambda a, i: add_lists_elementwise(  # pylint: disable=g-long-lambda
          a, define_ppo_step([tf.gather(t, indices_of_batches[i, :])
                              for t in input_tensors],
                             hparams, action_space, lr
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
