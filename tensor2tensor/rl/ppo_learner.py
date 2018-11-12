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

"""PPO learner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

from tensor2tensor.models.research.rl import get_policy
from tensor2tensor.rl import ppo
from tensor2tensor.rl.envs.tf_atari_wrappers import StackWrapper
from tensor2tensor.rl.envs.tf_atari_wrappers import WrapperBase
from tensor2tensor.rl.policy_learner import PolicyLearner
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


class PPOLearner(PolicyLearner):
  """PPO for policy learning."""

  def __init__(self, *args, **kwargs):
    super(PPOLearner, self).__init__(*args, **kwargs)
    self._num_completed_iterations = 0

  def train(self,
            env_fn,
            hparams,
            simulated,
            save_continuously,
            epoch,
            num_env_steps=None,
            env_step_multiplier=1,
            eval_env_fn=None,
            report_fn=None):
    if not save_continuously:
      # We do not save model, as that resets frames that we need at restarts.
      # But we need to save at the last step, so we set it very high.
      hparams.save_models_every_epochs = 1000000

    if simulated:
      simulated_str = "sim"
    else:
      simulated_str = "real"
    name_scope = "ppo_{}{}".format(simulated_str, epoch + 1)
    event_dir = os.path.join(self.base_event_dir, "ppo_summaries",
                             str(epoch) + simulated_str)

    with tf.Graph().as_default():
      with tf.name_scope(name_scope):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
          env = env_fn(in_graph=True)
          (train_summary_op, eval_summary_op, initializers) = (
              _define_train(
                  env,
                  hparams,
                  eval_env_fn,
                  frame_stack_size=self.frame_stack_size,
                  force_beginning_resets=simulated))

        if num_env_steps is None:
          iteration_increment = hparams.epochs_num
        else:
          iteration_increment = int(
              math.ceil(
                  num_env_steps / (env.batch_size * hparams.epoch_length)))
        iteration_increment *= env_step_multiplier

        self._num_completed_iterations += iteration_increment
        _run_train(
            hparams,
            event_dir,
            self.agent_model_dir,
            self._num_completed_iterations,
            train_summary_op,
            eval_summary_op,
            initializers,
            report_fn=report_fn)

  def evaluate(self, env_fn, hparams, stochastic):
    if stochastic:
      policy_to_actions_lambda = lambda policy: policy.sample()
    else:
      policy_to_actions_lambda = lambda policy: policy.mode()

    with tf.Graph().as_default():
      with tf.name_scope("rl_eval"):
        eval_env = env_fn(in_graph=True)
        (collect_memory, _, collect_init) = _define_collect(
            eval_env,
            hparams,
            "ppo_eval",
            eval_phase=True,
            frame_stack_size=self.frame_stack_size,
            force_beginning_resets=False,
            policy_to_actions_lambda=policy_to_actions_lambda)
        model_saver = tf.train.Saver(
            tf.global_variables(".*network_parameters.*"))

        with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          collect_init(sess)
          trainer_lib.restore_checkpoint(self.agent_model_dir, model_saver,
                                         sess)
          sess.run(collect_memory)


def _define_train(train_env, ppo_hparams, eval_env_fn=None, **collect_kwargs):
  """Define the training setup."""
  memory, collect_summary, train_initialization = (
      _define_collect(
          train_env,
          ppo_hparams,
          "ppo_train",
          eval_phase=False,
          policy_to_actions_lambda=(lambda policy: policy.sample()),
          **collect_kwargs))
  ppo_summary = ppo.define_ppo_epoch(
      memory, ppo_hparams, train_env.action_space, train_env.batch_size)
  train_summary = tf.summary.merge([collect_summary, ppo_summary])

  if ppo_hparams.eval_every_epochs:
    assert eval_env_fn is not None
    eval_env = eval_env_fn(in_graph=True)
    (_, eval_collect_summary, eval_initialization) = (
        _define_collect(
            eval_env,
            ppo_hparams,
            "ppo_eval",
            eval_phase=True,
            policy_to_actions_lambda=(lambda policy: policy.mode()),
            **collect_kwargs))
    return (train_summary, eval_collect_summary, (train_initialization,
                                                  eval_initialization))
  else:
    return (train_summary, None, (train_initialization,))


def _run_train(ppo_hparams,
               event_dir,
               model_dir,
               num_target_iterations,
               train_summary_op,
               eval_summary_op,
               initializers,
               report_fn=None):
  """Train."""
  summary_writer = tf.summary.FileWriter(
      event_dir, graph=tf.get_default_graph(), flush_secs=60)

  model_saver = tf.train.Saver(tf.global_variables(".*network_parameters.*"))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for initializer in initializers:
      initializer(sess)
    num_completed_iterations = trainer_lib.restore_checkpoint(
        model_dir, model_saver, sess)

    # Fail-friendly, complete only unfinished epoch
    if num_target_iterations <= num_completed_iterations:
      tf.logging.info(
          "Skipping PPO training. Requested %d iterations while %d train "
          "iterations already reached", num_target_iterations,
          num_completed_iterations)
      return

    for epoch_index in range(num_completed_iterations, num_target_iterations):
      summary = sess.run(train_summary_op)
      if summary_writer:
        summary_writer.add_summary(summary, epoch_index)

      if (ppo_hparams.eval_every_epochs and
          epoch_index % ppo_hparams.eval_every_epochs == 0):
        eval_summary = sess.run(eval_summary_op)
        if summary_writer:
          summary_writer.add_summary(eval_summary, epoch_index)
        if report_fn:
          summary_proto = tf.Summary()
          summary_proto.ParseFromString(eval_summary)
          for elem in summary_proto.value:
            if "mean_score" in elem.tag:
              report_fn(elem.simple_value, epoch_index)
              break

      if (model_saver and ppo_hparams.save_models_every_epochs and
          (epoch_index % ppo_hparams.save_models_every_epochs == 0 or
           (epoch_index + 1) == num_target_iterations)):
        ckpt_path = os.path.join(
            model_dir,
            "model.ckpt-{}".format(epoch_index + 1))
        model_saver.save(sess, ckpt_path)


def _rollout_metadata(batch_env):
  """Metadata for rollouts."""
  batch_env_shape = batch_env.observ.get_shape().as_list()
  batch_size = [batch_env_shape[0]]
  shapes_types_names = [
      # TODO(piotrmilos): possibly retrieve the observation type for batch_env
      (batch_size + batch_env_shape[1:], batch_env.observ_dtype, "observation"),
      (batch_size, tf.float32, "reward"),
      (batch_size, tf.bool, "done"),
      (batch_size + list(batch_env.action_shape), batch_env.action_dtype,
       "action"),
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
    # In memory wrapper we do not collect pdfs neither value_function
    # thus we only need the first 4 entries of meta_data
    shapes = meta_data[0][:4]
    dtypes = meta_data[1][:4]
    self.speculum = tf.FIFOQueue(infinity, shapes=shapes, dtypes=dtypes)
    observs_shape = batch_env.observ.shape
    # TODO(piotrmilos): possibly retrieve the observation type for batch_env
    self._observ = tf.Variable(
        tf.zeros(observs_shape, self.observ_dtype), trainable=False)

  def __str__(self):
    return "MemoryWrapper(%s)" % str(self._batch_env)

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


def _define_collect(batch_env, ppo_hparams, scope, frame_stack_size, eval_phase,
                    policy_to_actions_lambda, force_beginning_resets):
  """Collect trajectories.

  Args:
    batch_env: Batch environment.
    ppo_hparams: PPO hparams, defined in tensor2tensor.models.research.rl.
    scope: var scope.
    frame_stack_size: TODO(koz4k): Write docstring.
    eval_phase: TODO(koz4k): Write docstring.
    policy_to_actions_lambda: TODO(koz4k): Write docstring.
    force_beginning_resets: TODO(koz4k): Write docstring.

  Returns:
    Returns memory (observations, rewards, dones, actions,
    pdfs, values_functions)
    containing a rollout of environment from nested wrapped structure.
  """
  epoch_length = ppo_hparams.epoch_length

  to_initialize = []
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    num_agents = batch_env.batch_size

    to_initialize.append(batch_env)
    wrappers = [(StackWrapper, {
        "history": frame_stack_size
    }), (_MemoryWrapper, {})]
    rollout_metadata = None
    speculum = None
    for w in wrappers:
      tf.logging.info("Applying wrapper %s(%s) to env %s." % (str(
          w[0]), str(w[1]), str(batch_env)))
      batch_env = w[0](batch_env, **w[1])
      to_initialize.append(batch_env)

    rollout_metadata = _rollout_metadata(batch_env)
    speculum = batch_env.speculum

    def initialization_lambda(sess):
      for batch_env in to_initialize:
        batch_env.initialize(sess)

    memory = [
        tf.get_variable(
            "collect_memory_%d_%s" % (epoch_length, name),
            shape=[epoch_length] + shape,
            dtype=dtype,
            initializer=tf.zeros_initializer(),
            trainable=False) for (shape, dtype, name) in rollout_metadata
    ]

    cumulative_rewards = tf.get_variable(
        "cumulative_rewards", len(batch_env), trainable=False)

    eval_phase_t = tf.convert_to_tensor(eval_phase)
    should_reset_var = tf.Variable(True, trainable=False)
    zeros_tensor = tf.zeros(len(batch_env))

  force_beginning_resets = tf.convert_to_tensor(force_beginning_resets)

  def reset_ops_group():
    return tf.group(
        batch_env.reset(tf.range(len(batch_env))),
        tf.assign(cumulative_rewards, zeros_tensor))

  reset_op = tf.cond(
      tf.logical_or(should_reset_var.read_value(), force_beginning_resets),
      reset_ops_group, tf.no_op)

  with tf.control_dependencies([reset_op]):
    reset_once_op = tf.assign(should_reset_var, False)

  with tf.control_dependencies([reset_once_op]):

    def step(index, scores_sum, scores_num):
      """Single step."""
      index %= epoch_length  # Only needed in eval runs.
      # Note - the only way to ensure making a copy of tensor is to run simple
      # operation. We are waiting for tf.copy:
      # https://github.com/tensorflow/tensorflow/issues/11186
      obs_copy = batch_env.observ + 0

      def env_step(arg1, arg2, arg3):  # pylint: disable=unused-argument
        """Step of the environment."""
        actor_critic = get_policy(
            tf.expand_dims(obs_copy, 0), ppo_hparams, batch_env.action_space)
        policy = actor_critic.policy
        action = policy_to_actions_lambda(policy)

        postprocessed_action = actor_critic.action_postprocessing(action)
        reward, done = batch_env.simulate(postprocessed_action[0, ...])

        pdf = policy.prob(action)[0]
        value_function = actor_critic.value[0]
        pdf = tf.reshape(pdf, shape=(num_agents,))
        value_function = tf.reshape(value_function, shape=(num_agents,))
        done = tf.reshape(done, shape=(num_agents,))

        with tf.control_dependencies([reward, done]):
          return tf.identity(pdf), tf.identity(value_function), \
                 tf.identity(done)

      # TODO(piotrmilos): while_body is executed at most once,
      # thus should be replaced with tf.cond
      pdf, value_function, top_level_done = tf.while_loop(
          lambda _1, _2, _3: tf.equal(speculum.size(), 0),
          env_step,
          [
              tf.constant(0.0, shape=(num_agents,)),
              tf.constant(0.0, shape=(num_agents,)),
              tf.constant(False, shape=(num_agents,))
          ],
          parallel_iterations=1,
          back_prop=False,
      )

      with tf.control_dependencies([pdf, value_function]):
        obs, reward, done, action = speculum.dequeue()

        to_save = [obs, reward, done, action, pdf, value_function]
        save_ops = [
            tf.scatter_update(memory_slot, index, value)
            for memory_slot, value in zip(memory, to_save)
        ]
        cumulate_rewards_op = cumulative_rewards.assign_add(reward)

        agent_indices_to_reset = tf.where(top_level_done)[:, 0]
      with tf.control_dependencies([cumulate_rewards_op]):
        # TODO(piotrmilos): possibly we need cumulative_rewards.read_value()
        scores_sum_delta = tf.reduce_sum(
            tf.gather(cumulative_rewards.read_value(), agent_indices_to_reset))
        scores_num_delta = tf.count_nonzero(done, dtype=tf.int32)
      with tf.control_dependencies(save_ops +
                                   [scores_sum_delta, scores_num_delta]):
        reset_env_op = batch_env.reset(agent_indices_to_reset)
        reset_cumulative_rewards_op = tf.scatter_update(
            cumulative_rewards, agent_indices_to_reset,
            tf.gather(zeros_tensor, agent_indices_to_reset))
      with tf.control_dependencies([reset_env_op, reset_cumulative_rewards_op]):
        return [
            index + 1, scores_sum + scores_sum_delta,
            scores_num + scores_num_delta
        ]

    def stop_condition(i, _, resets):
      return tf.cond(eval_phase_t, lambda: resets < num_agents,
                     lambda: i < epoch_length)

    init = [tf.constant(0), tf.constant(0.0), tf.constant(0)]
    index, scores_sum, scores_num = tf.while_loop(
        stop_condition, step, init, parallel_iterations=1, back_prop=False)

  # We handle force_beginning_resets differently. We assume that all envs are
  # reseted at the end of episod (though it happens at the beginning of the
  # next one
  scores_num = tf.cond(force_beginning_resets,
                       lambda: scores_num + len(batch_env), lambda: scores_num)

  with tf.control_dependencies([scores_sum]):
    scores_sum = tf.cond(
        force_beginning_resets,
        lambda: scores_sum + tf.reduce_sum(cumulative_rewards.read_value()),
        lambda: scores_sum)

  mean_score = tf.cond(
      tf.greater(scores_num, 0),
      lambda: scores_sum / tf.cast(scores_num, tf.float32), lambda: 0.)
  printing = tf.Print(0, [mean_score, scores_sum, scores_num], "mean_score: ")
  with tf.control_dependencies([index, printing]):
    memory = [mem.read_value() for mem in memory]
    # When generating real data together with PPO training we must use single
    # agent. For PPO to work we reshape the history, as if it was generated
    # by real_ppo_effective_num_agents.
    if ppo_hparams.effective_num_agents is not None and not eval_phase:
      new_memory = []
      effective_num_agents = ppo_hparams.effective_num_agents
      assert epoch_length % ppo_hparams.effective_num_agents == 0, (
          "The rollout of ppo_hparams.epoch_length will be distributed amongst"
          "effective_num_agents of agents")
      new_epoch_length = int(epoch_length / effective_num_agents)
      for mem, info in zip(memory, rollout_metadata):
        shape, _, name = info
        new_shape = [effective_num_agents, new_epoch_length] + shape[1:]
        perm = list(range(len(shape) + 1))
        perm[0] = 1
        perm[1] = 0
        mem = tf.transpose(mem, perm=perm)
        mem = tf.reshape(mem, shape=new_shape)
        mem = tf.transpose(
            mem,
            perm=perm,
            name="collect_memory_%d_%s" % (new_epoch_length, name))
        new_memory.append(mem)
      memory = new_memory

    mean_score_summary = tf.cond(
        tf.greater(scores_num, 0),
        lambda: tf.summary.scalar("mean_score_this_iter", mean_score), str)
    summaries = tf.summary.merge([
        mean_score_summary,
        tf.summary.scalar("episodes_finished_this_iter", scores_num)
    ])
    return memory, summaries, initialization_lambda
