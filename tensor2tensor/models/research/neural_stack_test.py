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

"""Tests NeuralStackCell, NeuralQueueCell and NeuralStackModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mock
import numpy as np

from tensor2tensor.layers import modalities
from tensor2tensor.models.research import neural_stack

import tensorflow as tf


def build_fake_controller(cell):
  """Create a scalar variable to track the timestep.

  Args:
    cell: The NeuralStackCell to add the variable to.
  """
  cell.current_step = cell.add_variable(
      "current_step", [],
      initializer=tf.constant_initializer(-1),
      dtype=tf.int32,
      trainable=False)


def call_fake_controller(push_values, pop_values, read_values, output_values):
  """Mock a RNN controller from a set of expected outputs.

  Args:
    push_values: Expected controller push values.
    pop_values: Expected controller pop values.
    read_values: Expected controller read values.
    output_values: Expected controller output values.

  Returns:
    A callable which behaves like the call method of an NeuralStackCell.
  """
  def call(cell, inputs, state, batch_size):
    del inputs
    del batch_size
    next_step = tf.assign_add(cell.current_step, tf.constant(1))
    return (
        tf.slice(tf.constant(push_values), [next_step, 0], [1, -1]),
        tf.slice(tf.constant(pop_values), [next_step, 0], [1, -1]),
        tf.slice(tf.constant(read_values), [next_step, 0, 0], [1, -1, -1]),
        tf.slice(tf.constant(output_values), [next_step, 0, 0], [1, -1, -1]),
        state
    )
  return call


class NeuralStackCellTest(tf.test.TestCase):

  def test_controller_shapes(self):
    """Check that all the NeuralStackCell tensor shapes are correct.
    """

    batch_size = 5
    embedding_size = 3
    memory_size = 6
    num_units = 8

    stack = neural_stack.NeuralStackCell(num_units, memory_size, embedding_size)

    stack.build(None)

    self.assertEqual([1, embedding_size], stack.output_size)
    self.assertEqual([1, memory_size, memory_size], stack.read_mask.shape)
    self.assertEqual([3, 3, 1, 1], stack.write_shift_convolution.shape)

    stack_input = tf.zeros([batch_size, 1, embedding_size], dtype=tf.float32)

    zero_state = stack.zero_state(batch_size, tf.float32)

    (controller_state,
     previous_values,
     memory_values,
     read_strengths,
     write_strengths) = zero_state

    self.assertEqual([batch_size, num_units], controller_state.shape)
    self.assertEqual([batch_size, 1, embedding_size], previous_values.shape)
    self.assertEqual([batch_size, memory_size, embedding_size],
                     memory_values.shape)
    self.assertEqual([batch_size, 1, memory_size, 1], read_strengths.shape)
    self.assertEqual([batch_size, 1, memory_size, 1], write_strengths.shape)

    rnn_input = tf.concat([
        tf.reshape(
            previous_values,
            shape=[batch_size, embedding_size]),
        tf.reshape(
            stack_input,
            shape=[batch_size, embedding_size])
    ], axis=1)
    self.assertEqual([batch_size, 2 * embedding_size], rnn_input.shape)

    (push_strengths,
     pop_strengths,
     new_values,
     outputs,
     controller_next_state) = stack.call_controller(rnn_input,
                                                    controller_state,
                                                    batch_size)

    self.assertEqual([batch_size, 1, 1, 1], push_strengths.shape)
    self.assertEqual([batch_size, 1, 1, 1], pop_strengths.shape)
    self.assertEqual([batch_size, 1, embedding_size], new_values.shape)
    self.assertEqual([batch_size, 1, embedding_size], outputs.shape)
    self.assertEqual([batch_size, num_units], controller_next_state.shape)

    (outputs, (controller_next_state,
               read_values,
               next_memory_values,
               next_read_strengths,
               next_write_strengths)) = stack.call(stack_input, zero_state)

    self.assertEqual([batch_size, 1, embedding_size], outputs.shape)
    self.assertEqual([batch_size, num_units], controller_next_state.shape)
    self.assertEqual([batch_size, 1, embedding_size], read_values.shape)
    self.assertEqual([batch_size, memory_size, embedding_size],
                     next_memory_values.shape)
    self.assertEqual([batch_size, 1, memory_size, 1], next_read_strengths.shape)
    self.assertEqual([batch_size, 1, memory_size, 1],
                     next_write_strengths.shape)

    # Make sure that stack output shapes match stack input shapes
    self.assertEqual(controller_next_state.shape, controller_state.shape)
    self.assertEqual(read_values.shape, previous_values.shape)
    self.assertEqual(next_memory_values.shape, memory_values.shape)
    self.assertEqual(next_read_strengths.shape, read_strengths.shape)
    self.assertEqual(next_write_strengths.shape, write_strengths.shape)

  @mock.patch.object(neural_stack.NeuralStackCell, "build_controller",
                     build_fake_controller)
  @mock.patch.object(neural_stack.NeuralStackCell, "call_controller",
                     call_fake_controller(
                         push_values=[[1.0], [1.0], [0.0]],
                         pop_values=[[0.0], [0.0], [1.0]],
                         read_values=[[[1.0, 0.0, 0.0]],
                                      [[0.0, 1.0, 0.0]],
                                      [[0.0, 0.0, 1.0]]],
                         output_values=[[[0.0, 0.0, 0.0]],
                                        [[0.0, 0.0, 0.0]],
                                        [[0.0, 0.0, 0.0]]]))
  def test_push_pop(self):
    """Test pushing a popping from a NeuralStackCell.
    """
    input_values = np.array([[[[1.0, 0.0, 0.0]],
                              [[0.0, 1.0, 0.0]],
                              [[0.0, 0.0, 1.0]]]])

    expected_values = np.array([[[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, 1.0],
                                 [0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0]]])
    expected_read_strengths = np.array([
        [[[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]])
    expected_write_strengths = np.array([
        [[[0.0], [0.0], [0.], [1.0], [0.0], [0.0]]]])
    expected_top = np.array([[[1.0, 0.0, 0.0]]])

    stack = neural_stack.NeuralStackCell(8, 6, 3)
    stack_input = tf.constant(input_values, dtype=tf.float32)
    (outputs, state) = tf.nn.dynamic_rnn(cell=stack,
                                         inputs=stack_input,
                                         time_major=False,
                                         dtype=tf.float32)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      _, state_vals = sess.run([outputs, state])
      (_, stack_top, values, read_strengths, write_strengths) = state_vals

      self.assertAllClose(expected_top, stack_top)
      self.assertAllClose(expected_values, values)
      self.assertAllClose(expected_read_strengths, read_strengths)
      self.assertAllClose(expected_write_strengths, write_strengths)


class NeuralQueueCellTest(tf.test.TestCase):

  @mock.patch.object(neural_stack.NeuralQueueCell, "build_controller",
                     build_fake_controller)
  @mock.patch.object(neural_stack.NeuralQueueCell, "call_controller",
                     call_fake_controller(
                         push_values=[[1.0], [1.0], [0.0]],
                         pop_values=[[0.0], [0.0], [1.0]],
                         read_values=[[[1.0, 0.0, 0.0]],
                                      [[0.0, 1.0, 0.0]],
                                      [[0.0, 0.0, 1.0]]],
                         output_values=[[[0.0, 0.0, 0.0]],
                                        [[0.0, 0.0, 0.0]],
                                        [[0.0, 0.0, 0.0]]]))
  def test_enqueue_dequeue(self):
    """Test enqueueing a dequeueing from a NeuralQueueCell.
    """
    input_values = np.array([[[[1.0, 0.0, 0.0]],
                              [[0.0, 1.0, 0.0]],
                              [[0.0, 0.0, 1.0]]]])
    expected_values = np.array([[[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, 1.0],
                                 [0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0]]])
    expected_read_strengths = np.array([
        [[[0.0], [1.0], [0.0], [0.0], [0.0], [0.0]]]])
    expected_write_strengths = np.array([
        [[[0.0], [0.0], [0.0], [1.0], [0.0], [0.0]]]])
    expected_front = np.array([[[0.0, 1.0, 0.0]]])

    queue = neural_stack.NeuralQueueCell(8, 6, 3)
    rnn_input = tf.constant(input_values, dtype=tf.float32)
    (outputs, state) = tf.nn.dynamic_rnn(cell=queue,
                                         inputs=rnn_input,
                                         time_major=False,
                                         dtype=tf.float32)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      _, state_vals = sess.run([outputs, state])
      (_, queue_front, values, read_strengths, write_strengths) = state_vals

      self.assertAllClose(expected_front, queue_front)
      self.assertAllClose(expected_values, values)
      self.assertAllClose(expected_read_strengths, read_strengths)
      self.assertAllClose(expected_write_strengths, write_strengths)


class NeuralStackModelTest(tf.test.TestCase):

  def test_model_shapes(self):
    """Test a few of the important output shapes for NeuralStackModel.
    """
    batch_size = 100
    seq_length = 80
    embedding_size = 64
    vocab_size = 128

    hparams = neural_stack.neural_stack()
    problem_hparams = tf.contrib.training.HParams()

    problem_hparams.add_hparam("modality", {
        "inputs": modalities.ModalityType.SYMBOL,
        "targets": modalities.ModalityType.SYMBOL,
    })
    problem_hparams.add_hparam("vocab_size", {
        "inputs": vocab_size,
        "targets": vocab_size,
    })
    model = neural_stack.NeuralStackModel(hparams,
                                          problem_hparams=problem_hparams)

    features = {
        "inputs": tf.ones([batch_size, seq_length, 1, 1],
                          dtype=tf.int32),
        "targets": tf.ones([batch_size, seq_length, 1, 1], dtype=tf.int32)
    }

    transformed_features = model.bottom(features)

    self.assertEqual([batch_size, seq_length, 1, embedding_size],
                     transformed_features["inputs"].shape)

    logits = model.body(transformed_features)

    self.assertEqual([batch_size, seq_length, 1, embedding_size], logits.shape)


if __name__ == "__main__":
  tf.test.main()
