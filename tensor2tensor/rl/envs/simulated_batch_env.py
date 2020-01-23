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

"""Batch of environments inside the TensorFlow graph."""

# The code was based on Danijar Hafner's code from tf.agents:
# https://github.com/tensorflow/agents/blob/master/agents/tools/in_graph_batch_env.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os

import numpy as np

from tensor2tensor.data_generators.gym_env import DummyWorldModelProblem
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_video
from tensor2tensor.rl.envs import in_graph_batch_env
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib

import tensorflow.compat.v1 as tf


# Lazy load PIL.Image
def PIL_Image():  # pylint: disable=invalid-name
  from PIL import Image  # pylint: disable=g-import-not-at-top
  return Image


# Lazy load PIL.Image
def PIL_ImageDraw():  # pylint: disable=invalid-name
  from PIL import ImageDraw  # pylint: disable=g-import-not-at-top
  return ImageDraw


class HistoryBuffer(object):
  """History Buffer."""

  def __init__(self, initial_frame_chooser, observ_shape, observ_dtype,
               num_initial_frames, batch_size):
    self.batch_size = batch_size
    self._observ_dtype = observ_dtype
    initial_shape = (batch_size, num_initial_frames) + observ_shape
    self._initial_frames = tf.py_func(
        initial_frame_chooser, [tf.constant(batch_size)], observ_dtype
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

  def __init__(
      self, reward_range, observation_space, action_space, frame_stack_size,
      frame_height, frame_width, initial_frame_chooser, batch_size, model_name,
      model_hparams, model_dir, intrinsic_reward_scale=0.0, sim_video_dir=None
  ):
    """Batch of environments inside the TensorFlow graph."""
    super(SimulatedBatchEnv, self).__init__(observation_space, action_space)

    self._ffmpeg_works = common_video.ffmpeg_works()
    self.batch_size = batch_size
    self._min_reward = reward_range[0]
    self._num_frames = frame_stack_size
    self._intrinsic_reward_scale = intrinsic_reward_scale
    self._episode_counter = tf.get_variable(
        "episode_counter", initializer=tf.zeros((), dtype=tf.int32),
        trainable=False, dtype=tf.int32)
    if sim_video_dir:
      self._video_every_epochs = 100
      self._video_dir = sim_video_dir
      self._video_writer = None
      self._video_counter = 0
      tf.gfile.MakeDirs(self._video_dir)
      self._video_condition = tf.equal(
          self._episode_counter.read_value() % self._video_every_epochs, 0)
    else:
      self._video_condition = tf.constant(False, dtype=tf.bool, shape=())

    model_hparams = copy.copy(model_hparams)
    problem = DummyWorldModelProblem(action_space, reward_range,
                                     frame_height, frame_width)
    trainer_lib.add_problem_hparams(model_hparams, problem)
    model_hparams.force_full_predict = True
    self._model = registry.model(model_name)(
        model_hparams, tf.estimator.ModeKeys.PREDICT
    )

    self.history_buffer = HistoryBuffer(
        initial_frame_chooser, self.observ_shape, self.observ_dtype,
        self._num_frames, self.batch_size
    )

    self._observ = tf.Variable(
        tf.zeros((batch_size,) + self.observ_shape, self.observ_dtype),
        trainable=False
    )

    self._reset_model = tf.get_variable(
        "reset_model", [], trainable=False, initializer=tf.zeros_initializer())

    self._model_dir = model_dir

  def initialize(self, sess):
    model_loader = tf.train.Saver(
        var_list=tf.global_variables(scope="next_frame*")  # pylint:disable=unexpected-keyword-arg
    )
    if tf.gfile.IsDirectory(self._model_dir):
      trainer_lib.restore_checkpoint(
          self._model_dir, saver=model_loader, sess=sess, must_restore=True
      )
    else:
      model_loader.restore(sess=sess, save_path=self._model_dir)

  def __str__(self):
    return "SimulatedEnv"

  def __len__(self):
    """Number of combined environments."""
    return self.batch_size

  def simulate(self, action):
    with tf.name_scope("environment/simulate"):
      actions = tf.concat([tf.expand_dims(action, axis=1)] * self._num_frames,
                          axis=1)
      history = self.history_buffer.get_all_elements()
      with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        # We only need 1 target frame here, set it.
        hparams_target_frames = self._model.hparams.video_num_target_frames
        self._model.hparams.video_num_target_frames = 1
        model_output = self._model.infer({
            "inputs": history,
            "input_action": actions,
            "reset_internal_states": self._reset_model.read_value()
        })
        self._model.hparams.video_num_target_frames = hparams_target_frames

      observ = tf.cast(tf.squeeze(model_output["targets"], axis=1),
                       self.observ_dtype)

      reward = tf.to_float(model_output["target_reward"])
      reward = tf.reshape(reward, shape=(self.batch_size,)) + self._min_reward

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

      done = tf.constant(False, tf.bool, shape=(self.batch_size,))

      with tf.control_dependencies([observ]):
        dump_frame_op = tf.cond(self._video_condition,
                                lambda: tf.py_func(self._video_dump_frame,  # pylint: disable=g-long-lambda
                                                   [observ, reward], []),
                                tf.no_op)
        with tf.control_dependencies(
            [self._observ.assign(observ),
             self.history_buffer.move_by_one_element(observ), dump_frame_op]):
          clear_reset_model_op = tf.assign(self._reset_model, tf.constant(0.0))
          with tf.control_dependencies([clear_reset_model_op]):
            return tf.identity(reward), tf.identity(done)

  def _reset_non_empty(self, indices):
    """Reset the batch of environments.

    Args:
      indices: The batch indices of the environments to reset; defaults to all.

    Returns:
      Batch tensor of the new observations.
    """
    reset_video_op = tf.cond(
        self._video_condition,
        lambda: tf.py_func(self._video_reset_writer, [], []),
        tf.no_op)
    with tf.control_dependencies([reset_video_op]):
      inc_op = tf.assign_add(self._episode_counter, 1)
      with tf.control_dependencies([self.history_buffer.reset(indices),
                                    inc_op]):
        initial_frame_dump_op = tf.cond(
            self._video_condition,
            lambda: tf.py_func(self._video_dump_frames,  # pylint: disable=g-long-lambda
                               [self.history_buffer.get_all_elements()], []),
            tf.no_op)
        observ_assign_op = self._observ.assign(
            self.history_buffer.get_all_elements()[:, -1, ...])
        with tf.control_dependencies([observ_assign_op, initial_frame_dump_op]):
          reset_model_op = tf.assign(self._reset_model, tf.constant(1.0))
          with tf.control_dependencies([reset_model_op]):
            return tf.gather(self._observ.read_value(), indices)

  @property
  def observ(self):
    """Access the variable holding the current observation."""
    return self._observ.read_value()

  @property
  def history_observations(self):
    return self.history_buffer.get_all_elements()

  def _video_dump_frame(self, obs, rews):
    if not self._ffmpeg_works:
      return
    if self._video_writer is None:
      self._video_counter += 1
      self._video_writer = common_video.WholeVideoWriter(
          fps=10,
          output_path=os.path.join(self._video_dir,
                                   "{}.avi".format(self._video_counter)),
          file_format="avi")
    img = PIL_Image().new("RGB", (obs.shape[-2], 11),)
    draw = PIL_ImageDraw().Draw(img)
    draw.text((0, 0), "r:{:3}".format(int(rews[-1])), fill=(255, 0, 0))
    self._video_writer.write(np.concatenate([np.asarray(img), obs[-1]], axis=0))

  def _video_dump_frames(self, obs):
    if not self._ffmpeg_works:
      return
    zeros = np.zeros(obs.shape[0])
    for i in range(obs.shape[1]):
      self._video_dump_frame(obs[:, i, :], zeros)

  def _video_reset_writer(self):
    if self._video_writer:
      self._video_writer.finish_to_disk()
    self._video_writer = None

  def close(self):
    self._video_reset_writer()
