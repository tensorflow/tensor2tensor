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
"""Collect trajectories from interactions of agent with environment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import copy

from tensor2tensor.rl.envs.batch_env_factory import batch_env_factory
from tensor2tensor.rl.envs.tf_atari_wrappers import WrapperBase
from tensor2tensor.rl.envs.utils import get_policy

import tensorflow as tf


def _rollout_metadata(batch_env):
  """Metadata for rollouts."""
  batch_env_shape = batch_env.observ.get_shape().as_list()
  batch_size = [batch_env_shape[0]]
  shapes_types_names = [
      (batch_size + batch_env_shape[1:], tf.float32, "observation"),
      (batch_size, tf.float32, "reward"),
      (batch_size, tf.bool, "done"),
      (batch_size + batch_env.action_shape, batch_env.action_dtype, "action"),
      (batch_size, tf.float32, "pdf"),
      (batch_size, tf.float32, "value_function"),
  ]
  return shapes_types_names


class _MemoryWrapper(WrapperBase):
  """Memory wrapper."""

  def __init__(self, batch_env):
    super(_MemoryWrapper, self).__init__(batch_env)
    infinity = 10000000
    meta_data = list(zip(*_rollout_metadata(batch_env)))
    shapes = meta_data[0][:4]
    dtypes = meta_data[1][:4]
    self.speculum = tf.FIFOQueue(infinity, shapes=shapes, dtypes=dtypes)
    observs_shape = batch_env.observ.shape
    observ_dtype = tf.float32
    self._observ = tf.Variable(tf.zeros(observs_shape, observ_dtype),
                               trainable=False)

  def simulate(self, action):

    # There is subtlety here. We need to collect data
    # obs, action = policy(obs), done, reward = env(abs, action)
    # Thus we need to enqueue data before assigning new observation

    reward, done = self._batch_env.simulate(action)

    with tf.control_dependencies([reward, done]):
      enqueue_op = self.speculum.enqueue(
          [self._observ.read_value(), reward, done, action])

    with tf.control_dependencies([enqueue_op]):
      assign = self._observ.assign(self._batch_env.observ)

    with tf.control_dependencies([assign]):
      return tf.identity(reward), tf.identity(done)


def define_collect(hparams, scope, eval_phase,
                   collect_level=-1,
                   policy_to_actions_lambda=None):
  """Collect trajectories."""
  to_initialize = []
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    batch_env = batch_env_factory(hparams)
    to_initialize.append(batch_env)
    environment_wrappers = hparams.environment_spec.wrappers
    wrappers = copy.copy(environment_wrappers) if environment_wrappers else []
    # Put memory wrapper at the level you want to gather observations at.
    # Negative indices need to be shifted for insert to work correctly.
    collect_level = collect_level if \
      collect_level >= 0 else len(wrappers) + collect_level + 1
    wrappers.insert(collect_level, [_MemoryWrapper, {}])
    rollout_metadata = None
    speculum = None
    for w in wrappers:
      batch_env = w[0](batch_env, **w[1])
      to_initialize.append(batch_env)
      if w[0] == _MemoryWrapper:
        rollout_metadata = _rollout_metadata(batch_env)
        speculum = batch_env.speculum

    def initialization_lambda(sess):
      for batch_env in to_initialize:
        batch_env.initialize(sess)

    memory = [tf.get_variable("collect_memory_{}".format(name),
                              shape=[hparams.epoch_length]+shape,
                              dtype=dtype,
                              initializer=tf.zeros_initializer(),
                              trainable=False)
              for (shape, dtype, name) in rollout_metadata]

    cumulative_rewards = tf.get_variable("cumulative_rewards", len(batch_env),
                                         trainable=False)

    eval_phase = tf.convert_to_tensor(eval_phase)
    should_reset_var = tf.Variable(True, trainable=False)
    zeros_tensor = tf.zeros(len(batch_env))

  if "force_beginning_resets" in hparams:
    force_beginning_resets = hparams.force_beginning_resets
  else:
    force_beginning_resets = False

  def group():
    return tf.group(batch_env.reset(tf.range(len(batch_env))),
                    tf.assign(cumulative_rewards, zeros_tensor))
  reset_op = tf.cond(
      tf.logical_or(should_reset_var, tf.convert_to_tensor(
          force_beginning_resets)),
      group, tf.no_op)

  with tf.control_dependencies([reset_op]):
    reset_once_op = tf.assign(should_reset_var, False)

  with tf.control_dependencies([reset_once_op]):

    def step(index, scores_sum, scores_num):
      """Single step."""
      index %= hparams.epoch_length  # Only needed in eval runs.
      # Note - the only way to ensure making a copy of tensor is to run simple
      # operation. We are waiting for tf.copy:
      # https://github.com/tensorflow/tensorflow/issues/11186
      obs_copy = batch_env.observ + 0

      def env_step(arg1, arg2):  # pylint: disable=unused-argument
        """Step of the environment."""
        actor_critic = get_policy(tf.expand_dims(obs_copy, 0), hparams)
        policy = actor_critic.policy
        if policy_to_actions_lambda:
          action = policy_to_actions_lambda(policy)
        else:
          action = tf.cond(eval_phase,
                           policy.mode,
                           policy.sample)

        postprocessed_action = actor_critic.action_postprocessing(action)
        simulate_output = batch_env.simulate(postprocessed_action[0, ...])

        pdf = policy.prob(action)[0]
        value_function = actor_critic.value[0]
        pdf = tf.reshape(pdf, shape=(hparams.num_agents,))
        value_function = tf.reshape(value_function, shape=(hparams.num_agents,))

        with tf.control_dependencies(simulate_output):
          return tf.identity(pdf), tf.identity(value_function)

      pdf, value_function = tf.while_loop(
          lambda _1, _2: tf.equal(speculum.size(), 0),
          env_step,
          [tf.constant(0.0, shape=(hparams.num_agents,)),
           tf.constant(0.0, shape=(hparams.num_agents,))],
          parallel_iterations=1,
          back_prop=False,)

      with tf.control_dependencies([pdf, value_function]):
        obs, reward, done, action = speculum.dequeue()

        done = tf.reshape(done, (len(batch_env),))
        to_save = [obs, reward, done, action,
                   pdf, value_function]
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
            tf.gather(zeros_tensor, agent_indices_to_reset))
      with tf.control_dependencies([reset_env_op,
                                    reset_cumulative_rewards_op]):
        return [index + 1, scores_sum + scores_sum_delta,
                scores_num + scores_num_delta]

    def stop_condition(i, _, resets):
      return tf.cond(eval_phase,
                     lambda: resets < hparams.num_eval_agents,
                     lambda: i < hparams.epoch_length)

    init = [tf.constant(0), tf.constant(0.0), tf.constant(0)]
    index, scores_sum, scores_num = tf.while_loop(
        stop_condition,
        step,
        init,
        parallel_iterations=1,
        back_prop=False)
  mean_score = tf.cond(tf.greater(scores_num, 0),
                       lambda: scores_sum / tf.cast(scores_num, tf.float32),
                       lambda: 0.)
  printing = tf.Print(0, [mean_score, scores_sum, scores_num], "mean_score: ")
  with tf.control_dependencies([index, printing]):
    memory = [tf.identity(mem) for mem in memory]
    mean_score_summary = tf.cond(
        tf.greater(scores_num, 0),
        lambda: tf.summary.scalar("mean_score_this_iter", mean_score),
        str)
    summaries = tf.summary.merge(
        [mean_score_summary,
         tf.summary.scalar("episodes_finished_this_iter", scores_num)])
    return memory, summaries, initialization_lambda
