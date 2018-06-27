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
from tensor2tensor.rl.envs.utils import get_action_space
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib

import tensorflow as tf

from tensorflow.contrib.training import HParams


flags = tf.flags
FLAGS = flags.FLAGS


class HistoryBuffer(object):
  """History Buffer."""

  def __init__(self, input_dataset, length):
    self.input_data_iterator = (
        input_dataset.batch(length).make_initializable_iterator())
    self.length = length
    initial_frames = self.get_initial_observations()
    initial_shape = [length] + common_layers.shape_list(initial_frames)[1:]
    self._history_buff = tf.Variable(tf.zeros(initial_shape, tf.float32),
                                     trainable=False)

  def initialize(self, sess):
    sess.run(self.input_data_iterator.initializer)

  def get_initial_observations(self):
    return tf.cast(self.input_data_iterator.get_next(), tf.float32)

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
    initial_frames = tf.gather(self.get_initial_observations(), indices)
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

  def __init__(self, environment_spec, length, other_hparams):
    """Batch of environments inside the TensorFlow graph."""
    del other_hparams
    self.length = length
    initial_frames_problem = environment_spec.initial_frames_problem
    self._min_reward = initial_frames_problem.min_reward
    self._num_frames = environment_spec.video_num_input_frames
    self._intrinsic_reward_scale = environment_spec.intrinsic_reward_scale

    model_hparams = trainer_lib.create_hparams(
        FLAGS.hparams_set, problem_name=FLAGS.problem)
    model_hparams.force_full_predict = True
    self._model = registry.model(FLAGS.model)(
        model_hparams, tf.estimator.ModeKeys.PREDICT)

    _, self.action_shape, self.action_dtype = get_action_space(environment_spec)

    hparams = HParams(video_num_input_frames=
                      environment_spec.video_num_input_frames,
                      video_num_target_frames=
                      environment_spec.video_num_target_frames,
                      environment_spec=environment_spec)

    if environment_spec.simulation_random_starts:
      dataset = initial_frames_problem.dataset(tf.estimator.ModeKeys.TRAIN,
                                               FLAGS.data_dir,
                                               shuffle_files=True,
                                               hparams=hparams)
      dataset = dataset.shuffle(buffer_size=100)
    else:
      dataset = initial_frames_problem.dataset(tf.estimator.ModeKeys.TRAIN,
                                               FLAGS.data_dir,
                                               shuffle_files=False,
                                               hparams=hparams).take(1)

    dataset = dataset.map(lambda x: x["inputs"]).repeat()
    self.history_buffer = HistoryBuffer(dataset, self.length)

    shape = (self.length, initial_frames_problem.frame_height,
             initial_frames_problem.frame_width,
             initial_frames_problem.num_channels)
    self._observ = tf.Variable(tf.zeros(shape, tf.float32), trainable=False)

  def initialize(self, sess):
    self.history_buffer.initialize(sess)

  def __len__(self):
    """Number of combined environments."""
    return self.length

  def simulate(self, action):
    with tf.name_scope("environment/simulate"):
      actions = tf.concat([tf.expand_dims(action, axis=1)] * self._num_frames,
                          axis=1)
      history = self.history_buffer.get_all_elements()
      with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        model_output = self._model.infer(
            {"inputs": history, "input_action": actions})

      observ = tf.to_float(tf.squeeze(model_output["targets"], axis=1))

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
        return tf.identity(self._observ.read_value())

  @property
  def observ(self):
    """Access the variable holding the current observation."""
    return tf.identity(self._observ)
