# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

"""Collect trajectories from interactions of agent with environment."""

import tensorflow as tf


def define_collect(policy_factory, batch_env, config):
  """Collect trajectories."""
  memory_shape = [config.epoch_length] + [batch_env.observ.shape.as_list()[0]]
  memories_shapes_and_types = [
      # observation
      (memory_shape + [batch_env.observ.shape.as_list()[1]], tf.float32),
      (memory_shape, tf.float32),      # reward
      (memory_shape, tf.bool),         # done
      (memory_shape + batch_env.action_shape, tf.float32),  # action
      (memory_shape, tf.float32),      # pdf
      (memory_shape, tf.float32),      # value function
  ]
  memory = [tf.Variable(tf.zeros(shape, dtype), trainable=False)
            for (shape, dtype) in memories_shapes_and_types]
  cumulative_rewards = tf.Variable(
      tf.zeros(config.num_agents, tf.float32), trainable=False)

  should_reset_var = tf.Variable(True, trainable=False)
  reset_op = tf.cond(should_reset_var,
                     lambda: batch_env.reset(tf.range(config.num_agents)),
                     lambda: 0.0)
  with tf.control_dependencies([reset_op]):
    reset_once_op = tf.assign(should_reset_var, False)

  with tf.control_dependencies([reset_once_op]):

    def step(index, scores_sum, scores_num):
      """Single step."""
      # Note - the only way to ensure making a copy of tensor is to run simple
      # operation. We are waiting for tf.copy:
      # https://github.com/tensorflow/tensorflow/issues/11186
      obs_copy = batch_env.observ + 0
      actor_critic = policy_factory(tf.expand_dims(obs_copy, 0))
      policy = actor_critic.policy
      action = policy.sample()
      postprocessed_action = actor_critic.action_postprocessing(action)
      simulate_output = batch_env.simulate(postprocessed_action[0, ...])
      pdf = policy.prob(action)[0]
      with tf.control_dependencies(simulate_output):
        reward, done = simulate_output
        done = tf.reshape(done, (config.num_agents,))
        to_save = [obs_copy, reward, done, action[0, ...], pdf,
                   actor_critic.value[0]]
        save_ops = [tf.scatter_update(memory_slot, index, value)
                    for memory_slot, value in zip(memory, to_save)]
        cumulate_rewards_op = cumulative_rewards.assign_add(reward)
        agent_indices_to_reset = tf.where(done)[:, 0]
      with tf.control_dependencies([cumulate_rewards_op]):
        scores_sum_delta = tf.reduce_sum(
            tf.gather(cumulative_rewards, agent_indices_to_reset))
        scores_num_delta = tf.count_nonzero(done, dtype=tf.int32)
      with tf.control_dependencies(save_ops + [scores_sum_delta,
                                               scores_num_delta]):
        reset_env_op = batch_env.reset(agent_indices_to_reset)
        reset_cumulative_rewards_op = tf.scatter_update(
            cumulative_rewards, agent_indices_to_reset,
            tf.zeros(tf.shape(agent_indices_to_reset)))
      with tf.control_dependencies([reset_env_op,
                                    reset_cumulative_rewards_op]):
        return [index + 1, scores_sum + scores_sum_delta,
                scores_num + scores_num_delta]

    init = [tf.constant(0), tf.constant(0.0), tf.constant(0)]
    index, scores_sum, scores_num = tf.while_loop(
        lambda c, _1, _2: c < config.epoch_length,
        step,
        init,
        parallel_iterations=1,
        back_prop=False)
  mean_score = tf.cond(tf.greater(scores_num, 0),
                       lambda: scores_sum / tf.cast(scores_num, tf.float32),
                       lambda: 0.)
  printing = tf.Print(0, [mean_score, scores_sum, scores_num], "mean_score: ")
  with tf.control_dependencies([printing]):
    return tf.identity(index), memory
