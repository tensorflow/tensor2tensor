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

"""Batch of environments inside the TensorFlow graph."""

# The code was based on Danijar Hafner's code from tf.agents:
# https://github.com/tensorflow/agents/blob/master/agents/tools/in_graph_batch_env.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers
from tensor2tensor.rl.envs import in_graph_batch_env
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


class HistoryBuffer(object):
  """History Buffer."""

  def __init__(self, initial_frame_chooser, observ_shape, observ_dtype,
               num_initial_frames, length):
    self.length = length
    self._observ_dtype = observ_dtype
    initial_shape = (length, num_initial_frames) + observ_shape
    self._initial_frames = tf.py_func(
        initial_frame_chooser, [tf.constant(length)], observ_dtype
    )
    self._initial_frames.set_shape(initial_shape)
    self._history_buff = tf.Variable(tf.zeros(initial_shape, observ_dtype),
                                     trainable=False)

  def get_all_elements(self):
    return self._history_buff.read_value()

  def move_by_one_element(self, element):
    last_removed = self.get_all_elements()[:, 1:, ...]
    element = tf.expand_dims(element, dim=1)
    moved = tf.concat([last_removed, element], axis=1)
    with tf.control_dependencies([moved]):
      with tf.control_dependencies([self._history_buff.assign(moved)]):
        return self._history_buff.read_value()

  def reset(self, indices):
    initial_frames = tf.gather(self._initial_frames, indices)
    scatter_op = tf.scatter_update(self._history_buff, indices, initial_frames)
    with tf.control_dependencies([scatter_op]):
      return self._history_buff.read_value()


def compute_uncertainty_reward(logits, predictions):
  """Uncertainty reward based on logits."""
  # TODO(rsepassi): Add support for L1/L2 loss models. Current code only
  # works for softmax models.
  vocab_size = logits.shape[-1]
  assert vocab_size > 1
  log_probs = common_layers.log_prob_from_logits(logits)
  max_log_probs = common_layers.index_last_dim_with_indices(log_probs,
                                                            predictions)
  # Threshold
  neg_log_prob = tf.nn.relu(-max_log_probs - 0.02)
  # Sum across all but the batch dimension
  reduce_dims = list(range(len(neg_log_prob.shape)))[1:]
  summed = tf.reduce_sum(neg_log_prob, axis=reduce_dims)
  return summed / 10


class SimulatedBatchEnv(in_graph_batch_env.InGraphBatchEnv):
  """Batch of environments inside the TensorFlow graph.

  The batch of environments will be stepped and reset inside of the graph using
  a tf.py_func(). The current batch of observations, actions, rewards, and done
  flags are held in according variables.
  """

  def __init__(self, environment_spec, length):
    """Batch of environments inside the TensorFlow graph."""
    super(SimulatedBatchEnv, self).__init__(
        environment_spec.observation_space, environment_spec.action_space
    )

    self.length = length
    self._min_reward = environment_spec.reward_range[0]
    self._num_frames = environment_spec.video_num_input_frames
    self._intrinsic_reward_scale = environment_spec.intrinsic_reward_scale

    # TODO(koz4k): Pass by argument.
    model_hparams = trainer_lib.create_hparams(
        FLAGS.hparams_set, problem_name=FLAGS.problem)
    model_hparams.force_full_predict = True
    self._model = registry.model(FLAGS.model)(
        model_hparams, tf.estimator.ModeKeys.PREDICT)

    self.history_buffer = HistoryBuffer(
        environment_spec.initial_frame_chooser, self.observ_shape,
        self.observ_dtype, self._num_frames, self.length
    )

    self._observ = tf.Variable(
        tf.zeros((len(self),) + self.observ_shape, self.observ_dtype),
        trainable=False)

  def initialize(self, sess):
    # Currently not needed. Keeping it just in case.
    pass

  def __str__(self):
    return "SimulatedEnv"

  def __len__(self):
    """Number of combined environments."""
    return self.length

  def simulate(self, action):
    with tf.name_scope("environment/simulate"):
      actions = tf.concat([tf.expand_dims(action, axis=1)] * self._num_frames,
                          axis=1)
      history = self.history_buffer.get_all_elements()
      with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        # We only need 1 target frame here, set it.
        hparams_target_frames = self._model.hparams.video_num_target_frames
        self._model.hparams.video_num_target_frames = 1
        model_output = self._model.infer(
            {"inputs": history, "input_action": actions})
        self._model.hparams.video_num_target_frames = hparams_target_frames

      observ = tf.cast(tf.squeeze(model_output["targets"], axis=1),
                       self.observ_dtype)

      reward = tf.to_float(model_output["target_reward"])
      reward = tf.reshape(reward, shape=(self.length,)) + self._min_reward

      if self._intrinsic_reward_scale:
        # Use the model's uncertainty about its prediction as an intrinsic
        # reward. The uncertainty is measured by the log probability of the
        # predicted pixel value.
        if "targets_logits" not in model_output:
          raise ValueError("The use of intrinsic rewards requires access to "
                           "the logits. Ensure that model.infer returns "
                           "'targets_logits'")
        uncertainty_reward = compute_uncertainty_reward(
            model_output["targets_logits"], model_output["targets"])
        uncertainty_reward = tf.minimum(
            1., self._intrinsic_reward_scale * uncertainty_reward)
        uncertainty_reward = tf.Print(uncertainty_reward, [uncertainty_reward],
                                      message="uncertainty_reward", first_n=1,
                                      summarize=8)
        reward += uncertainty_reward

      done = tf.constant(False, tf.bool, shape=(self.length,))

      with tf.control_dependencies([observ]):
        with tf.control_dependencies(
            [self._observ.assign(observ),
             self.history_buffer.move_by_one_element(observ)]):
          return tf.identity(reward), tf.identity(done)

  def _reset_non_empty(self, indices):
    """Reset the batch of environments.

    Args:
      indices: The batch indices of the environments to reset; defaults to all.

    Returns:
      Batch tensor of the new observations.
    """
    with tf.control_dependencies([self.history_buffer.reset(indices)]):
      with tf.control_dependencies([self._observ.assign(
          self.history_buffer.get_all_elements()[:, -1, ...])]):
        return tf.gather(self._observ.read_value(), indices)

  @property
  def observ(self):
    """Access the variable holding the current observation."""
    return self._observ.read_value()

  @property
  def history_observations(self):
    return self.history_buffer.get_all_elements()
