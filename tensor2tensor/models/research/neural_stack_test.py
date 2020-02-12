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

"""Tests NeuralStackCell, NeuralQueueCell and NeuralStackModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mock
import numpy as np

from tensor2tensor.layers import modalities
from tensor2tensor.models.research import neural_stack
from tensor2tensor.utils import contrib

import tensorflow.compat.v1 as tf


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


def call_fake_controller(push_values, pop_values, write_values, output_values):
  """Mock a RNN controller from a set of expected outputs.

  Args:
    push_values: Expected controller push values.
    pop_values: Expected controller pop values.
    write_values: Expected controller write values.
    output_values: Expected controller output values.

  Returns:
    A callable which behaves like the call method of an NeuralStackCell.
  """
  def call(cell, inputs, prev_read_values, controller_state, batch_size):
    del inputs
    del prev_read_values
    del batch_size
    next_step = tf.constant(0)
    if hasattr(cell, "current_step"):
      next_step = tf.assign_add(cell.current_step, tf.constant(1))
    return neural_stack.NeuralStackControllerInterface(
        push_strengths=tf.slice(tf.constant(push_values),
                                [next_step, 0, 0, 0],
                                [1, -1, -1, -1]),
        pop_strengths=tf.slice(tf.constant(pop_values),
                               [next_step, 0, 0, 0],
                               [1, -1, -1, -1]),
        write_values=tf.slice(tf.constant(write_values),
                              [next_step, 0, 0],
                              [1, -1, -1]),
        outputs=tf.slice(tf.constant(output_values),
                         [next_step, 0, 0],
                         [1, -1, -1]),
        state=controller_state
    )
  return call


def assert_controller_shapes(test, controller_outputs, controller_shapes):
  for name, output, shape in zip(controller_outputs._fields, controller_outputs,
                                 controller_shapes):
    test.assertEqual(shape, output.shape, "%s shapes don't match" % name)


def assert_cell_shapes(test, output_state, zero_state):
  for name, output, zero in zip(output_state._fields, output_state,
                                zero_state):
    test.assertEqual(zero.shape, output.shape, "%s shapes don't match" % name)


class NeuralStackCellTest(tf.test.TestCase):

  def test_cell_shapes(self):
    """Check that all the NeuralStackCell tensor shapes are correct.
    """
    batch_size = 5
    embedding_size = 3
    memory_size = 6
    num_units = 8

    stack = neural_stack.NeuralStackCell(num_units, memory_size, embedding_size)
    stack.build(None)

    self.assertEqual([1, 1, memory_size, memory_size],
                     stack.get_read_mask(0).shape)

    stack_input = tf.zeros([batch_size, 1, embedding_size], dtype=tf.float32)
    zero_state = stack.zero_state(batch_size, tf.float32)
    (outputs, (stack_next_state)) = stack.call(stack_input, zero_state)

    # Make sure that stack output shapes match stack input shapes
    self.assertEqual(outputs.shape, stack_input.shape)

    assert_cell_shapes(self, stack_next_state, zero_state)

  @mock.patch.object(neural_stack.NeuralStackCell, "build_controller",
                     build_fake_controller)
  @mock.patch.object(neural_stack.NeuralStackCell, "call_controller",
                     call_fake_controller(
                         push_values=[[[[1.0]]], [[[1.0]]], [[[0.0]]]],
                         pop_values=[[[[0.0]]], [[[0.0]]], [[[1.0]]]],
                         write_values=[[[1.0, 0.0, 0.0]],
                                       [[0.0, 1.0, 0.0]],
                                       [[0.0, 0.0, 1.0]]],
                         output_values=[[[0.0, 0.0, 0.0]],
                                        [[0.0, 0.0, 0.0]],
                                        [[0.0, 0.0, 0.0]]]))
  def test_push_pop(self):
    """Test pushing a popping from a NeuralStackCell.

    The sequence of operations is:
      push([1.0, 0.0, 0.0])
      push([0.0, 1.0, 0.0])
      pop()
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

    batch_size = 1
    embedding_size = 3
    memory_size = 6
    num_units = 8

    stack = neural_stack.NeuralStackCell(num_units, memory_size, embedding_size)
    stack_input = tf.constant(input_values, dtype=tf.float32)

    stack_zero_state = tf.zeros([batch_size, num_units])
    controller_outputs = stack.call_controller(None, None, stack_zero_state,
                                               batch_size)
    assert_controller_shapes(self, controller_outputs,
                             stack.get_controller_shape(batch_size))

    (outputs, state) = tf.nn.dynamic_rnn(cell=stack,
                                         inputs=stack_input,
                                         time_major=False,
                                         dtype=tf.float32)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      _, state_vals = sess.run([outputs, state])
      (_, stack_top, values, read_strengths, write_strengths) = state_vals

      self.assertAllClose(expected_values, values)
      self.assertAllClose(expected_write_strengths, write_strengths)
      self.assertAllClose(expected_read_strengths, read_strengths)
      self.assertAllClose(expected_top, stack_top)


class NeuralQueueCellTest(tf.test.TestCase):

  @mock.patch.object(neural_stack.NeuralQueueCell, "build_controller",
                     build_fake_controller)
  @mock.patch.object(neural_stack.NeuralQueueCell, "call_controller",
                     call_fake_controller(
                         push_values=[[[[1.0]]], [[[1.0]]], [[[0.0]]]],
                         pop_values=[[[[0.0]]], [[[0.0]]], [[[1.0]]]],
                         write_values=[[[1.0, 0.0, 0.0]],
                                       [[0.0, 1.0, 0.0]],
                                       [[0.0, 0.0, 1.0]]],
                         output_values=[[[0.0, 0.0, 0.0]],
                                        [[0.0, 0.0, 0.0]],
                                        [[0.0, 0.0, 0.0]]]))
  def test_enqueue_dequeue(self):
    """Test enqueueing a dequeueing from a NeuralQueueCell.

    The sequence of operations is:
      enqueue([1.0, 0.0, 0.0])
      enqueue([0.0, 1.0, 0.0])
      dequeue()
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

    batch_size = 1
    num_units = 8
    embedding_size = 3
    memory_size = 6

    queue = neural_stack.NeuralQueueCell(num_units, memory_size, embedding_size)
    rnn_input = tf.constant(input_values, dtype=tf.float32)

    queue_zero_state = tf.zeros([batch_size, num_units])
    controller_outputs = queue.call_controller(None, None, queue_zero_state,
                                               batch_size)
    assert_controller_shapes(self, controller_outputs,
                             queue.get_controller_shape(batch_size))

    (outputs, state) = tf.nn.dynamic_rnn(cell=queue,
                                         inputs=rnn_input,
                                         time_major=False,
                                         dtype=tf.float32)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      _, state_vals = sess.run([outputs, state])
      (_, queue_front, values, read_strengths, write_strengths) = state_vals

      self.assertAllClose(expected_values, values)
      self.assertAllClose(expected_write_strengths, write_strengths)
      self.assertAllClose(expected_read_strengths, read_strengths)
      self.assertAllClose(expected_front, queue_front)


class NeuralDequeCellTest(tf.test.TestCase):

  def test_cell_shapes(self):
    """Check that all the NeuralStackCell tensor shapes are correct.
    """
    batch_size = 5
    embedding_size = 4
    memory_size = 12
    num_units = 8

    deque = neural_stack.NeuralDequeCell(num_units, memory_size, embedding_size)
    deque.build(None)

    self.assertEqual([1, 1, memory_size, memory_size],
                     deque.get_read_mask(0).shape)
    self.assertEqual([1, 1, memory_size, memory_size],
                     deque.get_read_mask(1).shape)

    deque_input = tf.zeros([batch_size, 1, embedding_size], dtype=tf.float32)
    zero_state = deque.zero_state(batch_size, tf.float32)
    (outputs, (deque_next_state)) = deque.call(deque_input, zero_state)

    # Make sure that deque output shapes match deque input shapes
    self.assertEqual(outputs.shape, deque_input.shape)

    assert_cell_shapes(self, deque_next_state, zero_state)

  @mock.patch.object(neural_stack.NeuralDequeCell, "build_controller",
                     build_fake_controller)
  @mock.patch.object(neural_stack.NeuralDequeCell, "call_controller",
                     call_fake_controller(
                         push_values=[[[[1.0]], [[0.0]]],
                                      [[[1.0]], [[0.0]]],
                                      [[[1.0]], [[0.0]]],
                                      [[[0.0]], [[1.0]]],
                                      [[[0.0]], [[0.0]]],
                                      [[[0.0]], [[0.0]]]],
                         pop_values=[[[[0.0]], [[0.0]]],
                                     [[[0.0]], [[0.0]]],
                                     [[[0.0]], [[0.0]]],
                                     [[[0.0]], [[0.0]]],
                                     [[[0.0]], [[1.0]]],
                                     [[[0.0]], [[1.0]]]],
                         write_values=[[[1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0]],
                                       [[0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0]],
                                       [[0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0]],
                                       [[0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0]],
                                       [[0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0]],
                                       [[0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0]]],
                         output_values=[[[0.0, 0.0, 0.0, 0.0]],
                                        [[0.0, 0.0, 0.0, 0.0]],
                                        [[0.0, 0.0, 0.0, 0.0]],
                                        [[0.0, 0.0, 0.0, 0.0]],
                                        [[0.0, 0.0, 0.0, 0.0]],
                                        [[0.0, 0.0, 0.0, 0.0]]]))
  def test_enqueue_dequeue(self):
    """Test enqueueing a dequeueing from a NeuralDequeCell.

    The sequence of operations is:
      enqueue_bottom([1.0, 0.0, 0.0, 0.0])
      enqueue_bottom([0.0, 1.0, 0.0, 0.0])
      enqueue_bottom([0.0, 0.0, 1.0, 0.0])
      enqueue_top([0.0, 0.0, 0.0, 1.0])
      dequeue_top()
      dequeue_top()
    """
    input_values = np.array([[[[1.0, 0.0, 0.0, 0.0]],
                              [[0.0, 1.0, 0.0, 0.0]],
                              [[0.0, 0.0, 1.0, 0.0]],
                              [[0.0, 0.0, 0.0, 1.0]],
                              [[0.0, 0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0, 0.0]]]])

    expected_values = np.array([[[0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 1.0, 0.0, 0.0],
                                 [1.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0],
                                 [0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0]]])

    expected_read_strengths = np.array([[[[0.0], [0.0], [0.0], [1.0], [1.0],
                                          [0.0], [0.0], [0.0], [0.0], [0.0],
                                          [0.0], [0.0]]]])

    expected_write_strengths = np.array([[[[0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [1.0]],
                                          [[1.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0]]]])

    expected_read_values = np.array([[[0.0, 0.0, 1.0, 0.0],
                                      [0.0, 1.0, 0.0, 0.0]]])

    batch_size = input_values.shape[0]
    memory_size = input_values.shape[1] * 2
    embedding_size = input_values.shape[3]
    num_units = 8

    deque = neural_stack.NeuralDequeCell(num_units, memory_size, embedding_size)
    rnn_input = tf.constant(input_values, dtype=tf.float32)

    deque_zero_state = tf.zeros([batch_size, num_units])
    controller_outputs = deque.call_controller(None, None,
                                               deque_zero_state,
                                               batch_size)
    assert_controller_shapes(self, controller_outputs,
                             deque.get_controller_shape(batch_size))

    (outputs, state) = tf.nn.dynamic_rnn(cell=deque,
                                         inputs=rnn_input,
                                         time_major=False,
                                         dtype=tf.float32)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      _, state_vals = sess.run([outputs, state])
      (_, read_values,
       memory_values,
       read_strengths,
       write_strengths) = state_vals

      print(read_values)
      self.assertAllClose(expected_values, memory_values)
      self.assertAllClose(expected_write_strengths, write_strengths)
      self.assertAllClose(expected_read_strengths, read_strengths)
      self.assertAllClose(expected_read_values, read_values)


class NeuralStackModelTest(tf.test.TestCase):

  def test_model_shapes(self):
    """Test a few of the important output shapes for NeuralStackModel.
    """
    batch_size = 100
    seq_length = 80
    embedding_size = 64
    vocab_size = 128

    hparams = neural_stack.neural_stack()
    problem_hparams = contrib.training().HParams()

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
