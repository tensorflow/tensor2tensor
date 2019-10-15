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

"""Stacks and Queues implemented as encoder-decoder models.

Based off of the following research:

Learning to Transduce with Unbounded Memory
Edward Grefenstette, Karl Moritz Hermann, Mustafa Suleyman, Phil Blunsom
https://arxiv.org/abs/1506.02516, 2015

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

# This is the interface between the RNN controller and the neural stack.
NeuralStackControllerInterface = collections.namedtuple(
    "NeuralStackControllerInterface",
    "push_strengths, pop_strengths, write_values, outputs, state")

# This is recurrent state of the neural stack RNN cell.
NeuralStackState = collections.namedtuple(
    "NeuralStackState",
    "controller_state, read_values, memory_values, read_strengths, " +
    "write_strengths")


class NeuralStackCell(tf.nn.rnn_cell.RNNCell):
  """An RNN cell base class that can implement a stack or queue.
  """

  def __init__(self, num_units, memory_size, embedding_size,
               num_read_heads=1, num_write_heads=1, reuse=None):
    """Create a new NeuralStackCell.

    Args:
      num_units: The number of hidden units in the RNN cell.
      memory_size: The maximum memory size allocated for the stack.
      embedding_size:  The embedding width of the individual stack values.
      num_read_heads: This should always be 1 for a regular stack.
      num_write_heads: This should always be 1 for a regular stack.
      reuse: Whether to reuse the weights.
    """
    super(NeuralStackCell, self).__init__(dtype=tf.float32, _reuse=reuse)
    self._num_units = num_units
    self._embedding_size = embedding_size
    self._memory_size = memory_size
    self._num_read_heads = num_read_heads
    self._num_write_heads = num_write_heads

  @property
  def state_size(self):
    """The NeuralStackCell maintains a tuple of state values.

    Returns:
      (controller_state.shape,
       read_values.shape,
       memory_values.shape,
       read_strengths.shape,
       write_strengths.shape)
    """
    return (tf.TensorShape([self._num_units]),
            tf.TensorShape([self._num_read_heads, self._embedding_size]),
            tf.TensorShape([self._memory_size, self._embedding_size]),
            tf.TensorShape([1, self._memory_size, 1]),
            tf.TensorShape([self._num_write_heads, self._memory_size, 1]))

  @property
  def output_size(self):
    return tf.TensorShape([1, self._embedding_size])

  def initialize_write_strengths(self, batch_size):
    """Initialize write strengths to write to the first memory address.

    This is exposed as its own function so that it can be overridden to provide
    alternate write adressing schemes.

    Args:
      batch_size: The size of the current batch.

    Returns:
      A tf.float32 tensor of shape [num_write_heads, memory_size, 1] where the
      first element in the second dimension is set to 1.0.
    """
    return tf.expand_dims(
        tf.one_hot([[0] * self._num_write_heads] * batch_size,
                   depth=self._memory_size, dtype=tf.float32), axis=3)

  def zero_state(self, batch_size, dtype):
    """Initialize the tuple of state values to zeros except write strengths.

    Args:
      batch_size: The size of the current batch.
      dtype: The default datatype to initialize to.

    Returns:
      A new NeuralStackState tuple.
    """
    parent_state = NeuralStackState(*super(NeuralStackCell, self).zero_state(
        batch_size, dtype))
    return NeuralStackState(
        controller_state=parent_state.controller_state,
        read_values=parent_state.read_values,
        memory_values=parent_state.memory_values,
        read_strengths=parent_state.read_strengths,
        write_strengths=self.initialize_write_strengths(batch_size))

  def get_read_mask(self, read_head_index):
    """Creates a mask which allows us to attenuate subsequent read strengths.

    This is exposed as its own function so that it can be overridden to provide
    alternate read adressing schemes.

    Args:
      read_head_index: Identifies which read head we're getting the mask for.

    Returns:
      A tf.float32 tensor of shape [1, 1, memory_size, memory_size]
    """
    if read_head_index == 0:
      return tf.expand_dims(
          common_layers.mask_pos_lt(self._memory_size, self._memory_size),
          axis=0)
    else:
      raise ValueError("Read head index must be 0 for stack.")

  def get_write_head_offset(self, write_head_index):
    """Lookup the offset to shift the write head at each step.

    By default, we move each write head forward by 1.

    This is exposed as its own function so that it can be overridden to provide
    alternate write adressing schemes.

    Args:
      write_head_index: Identifies which write head we're getting the index for.

    Returns:
      An integer offset to move the write head at each step.
    """
    if write_head_index == 0:
      return 1
    else:
      raise ValueError("Write head index must be 0 for stack.")

  def add_scalar_projection(self, name, size):
    """A helper function for mapping scalar controller outputs.

    Args:
      name: A prefix for the variable names.
      size: The desired number of scalar outputs.

    Returns:
      A tuple of (weights, bias) where weights has shape [num_units, size] and
      bias has shape [size].
    """
    weights = self.add_variable(
        name + "_projection_weights",
        shape=[self._num_units, size],
        dtype=self.dtype)
    bias = self.add_variable(
        name + "_projection_bias",
        shape=[size],
        initializer=tf.zeros_initializer(dtype=self.dtype))
    return weights, bias

  def add_vector_projection(self, name, size):
    """A helper function for mapping embedding controller outputs.

    Args:
      name: A prefix for the variable names.
      size: The desired number of embedding outputs.

    Returns:
      A tuple of (weights, bias) where weights has shape
      [num_units, size * embedding_size] and bias has shape
      [size * embedding_size].
    """
    weights = self.add_variable(
        name + "_projection_weights",
        shape=[self._num_units, size * self._embedding_size],
        dtype=self.dtype)
    bias = self.add_variable(
        name + "_projection_bias",
        shape=[size * self._embedding_size],
        initializer=tf.zeros_initializer(dtype=self.dtype))
    return weights, bias

  def build_controller(self):
    """Create the RNN and output projections for controlling the stack.
    """
    with tf.name_scope("controller"):
      self.rnn = tf.contrib.rnn.BasicRNNCell(self._num_units)
      self._input_proj = self.add_variable(
          "input_projection_weights",
          shape=[self._embedding_size * (self._num_read_heads + 1),
                 self._num_units],
          dtype=self.dtype)
      self._input_bias = self.add_variable(
          "input_projection_bias",
          shape=[self._num_units],
          initializer=tf.zeros_initializer(dtype=self.dtype))
      self._push_proj, self._push_bias = self.add_scalar_projection(
          "push", self._num_write_heads)
      self._pop_proj, self._pop_bias = self.add_scalar_projection(
          "pop", self._num_write_heads)
      self._value_proj, self._value_bias = self.add_vector_projection(
          "value", self._num_write_heads)
      self._output_proj, self._output_bias = self.add_vector_projection(
          "output", 1)

  def build(self, _):
    """Build the controller.
    """
    self.build_controller()
    self.built = True

  def get_controller_shape(self, batch_size):
    """Define the output shapes of the neural stack controller.

    Making this a separate functions so that it can be used in unit tests.

    Args:
      batch_size: The size of the current batch of data.

    Returns:
      A tuple of shapes for each output returned from the controller.
    """
    return (
        # push_strengths,
        [batch_size, self._num_write_heads, 1, 1],
        # pop_strengths
        [batch_size, self._num_write_heads, 1, 1],
        # write_values
        [batch_size, self._num_write_heads, self._embedding_size],
        # outputs
        [batch_size, 1, self._embedding_size],
        # state
        [batch_size, self._num_units])

  def call_controller(self, input_value, read_values, prev_state, batch_size):
    """Make a call to the neural stack controller.

    See Section 3.1 of Grefenstette et al., 2015.

    Args:
      input_value: The input to the neural stack cell should be a tf.float32
        tensor with shape [batch_size, 1, embedding_size]
      read_values: The values of the read heads at the previous timestep.
      prev_state: The hidden state from the previous time step.
      batch_size: The size of the current batch of input values.

    Returns:
      A tuple of outputs and the new NeuralStackControllerInterface.
    """
    with tf.name_scope("controller"):
      # Concatenate the current input value with the read values from the
      # previous timestep before feeding them into the controller.
      controller_inputs = tf.concat([
          tf.contrib.layers.flatten(input_value),
          tf.contrib.layers.flatten(read_values),
      ], axis=1)

      rnn_input = tf.tanh(tf.nn.bias_add(tf.matmul(
          controller_inputs, self._input_proj), self._input_bias))

      (rnn_output, state) = self.rnn(rnn_input, prev_state)

      push_strengths = tf.sigmoid(tf.nn.bias_add(tf.matmul(
          rnn_output, self._push_proj), self._push_bias))

      pop_strengths = tf.sigmoid(tf.nn.bias_add(tf.matmul(
          rnn_output, self._pop_proj), self._pop_bias))

      write_values = tf.tanh(tf.nn.bias_add(tf.matmul(
          rnn_output, self._value_proj), self._value_bias))

      outputs = tf.tanh(tf.nn.bias_add(tf.matmul(
          rnn_output, self._output_proj), self._output_bias))

      # Reshape all the outputs according to the shapes specified by
      # get_controller_shape()
      projected_outputs = [push_strengths,
                           pop_strengths,
                           write_values,
                           outputs,
                           state]
      next_state = [
          tf.reshape(output, shape=output_shape) for output, output_shape
          in zip(projected_outputs, self.get_controller_shape(batch_size))]
      return NeuralStackControllerInterface(*next_state)

  def call(self, inputs, prev_state):
    """Evaluates one timestep of the current neural stack cell.

    See section 3.4 of Grefenstette et al., 2015.

    Args:
      inputs: The inputs to the neural stack cell should be a tf.float32 tensor
        with shape [batch_size, embedding_size]
      prev_state: The NeuralStackState from the previous timestep.

    Returns:
      A tuple of the output of the stack as well as the new NeuralStackState.
    """
    batch_size = tf.shape(inputs)[0]

    # Call the controller and get controller interface values.
    with tf.control_dependencies([prev_state.read_strengths]):
      controller_output = self.call_controller(
          inputs, prev_state.read_values, prev_state.controller_state,
          batch_size)

    # Always write input values to memory regardless of push strength.
    # See Equation-1 in Grefenstette et al., 2015.
    new_memory_values = prev_state.memory_values + tf.reduce_sum(
        tf.expand_dims(controller_output.write_values, axis=2) *
        prev_state.write_strengths,
        axis=1)

    # Attenuate the read strengths of existing memory values depending on the
    # current pop strength.
    # See Equation-2 in Grefenstette et al., 2015.
    new_read_strengths = prev_state.read_strengths
    for h in range(self._num_read_heads - 1, -1, -1):
      new_read_strengths = tf.nn.relu(new_read_strengths - tf.nn.relu(
          tf.slice(controller_output.pop_strengths,
                   [0, h, 0, 0],
                   [-1, 1, -1, -1]) -
          tf.expand_dims(
              tf.reduce_sum(new_read_strengths * self.get_read_mask(h), axis=2),
              axis=3)))

    # Combine all write heads and their associated push values into a single set
    # of read weights.
    new_read_strengths += tf.reduce_sum(
        controller_output.push_strengths * prev_state.write_strengths,
        axis=1, keep_dims=True)

    # Calculate the "top" value of the stack by looking at read strengths.
    # See Equation-3 in Grefenstette et al., 2015.
    new_read_values = tf.reduce_sum(
        tf.minimum(
            new_read_strengths,
            tf.nn.relu(1 - tf.expand_dims(
                tf.reduce_sum(
                    new_read_strengths * tf.concat([
                        self.get_read_mask(h)
                        for h in range(self._num_read_heads)
                    ], axis=1),
                    axis=2),
                axis=3))
        ) * tf.expand_dims(new_memory_values, axis=1),
        axis=2)

    # Temporarily split write strengths apart so they can be shifted in
    # different directions.
    write_strengths_by_head = tf.split(prev_state.write_strengths,
                                       self._num_write_heads,
                                       axis=1)
    # Shift the write strengths for each write head in the direction indicated
    # by get_write_head_offset().
    new_write_strengths = tf.concat([
        tf.roll(write_strength, shift=self.get_write_head_offset(h), axis=2)
        for h, write_strength in enumerate(write_strengths_by_head)
    ], axis=1)

    return (controller_output.outputs, NeuralStackState(
        controller_state=controller_output.state,
        read_values=new_read_values,
        memory_values=new_memory_values,
        read_strengths=new_read_strengths,
        write_strengths=new_write_strengths))


class NeuralQueueCell(NeuralStackCell):
  """An subclass of the NeuralStackCell which reads from the opposite direction.

  See section 3.2 of Grefenstette et al., 2015.
  """

  def get_read_mask(self, read_head_index):
    """Uses mask_pos_lt() instead of mask_pos_gt() to reverse read values.

    Args:
      read_head_index: Identifies which read head we're getting the mask for.

    Returns:
      A tf.float32 tensor of shape [1, 1, memory_size, memory_size].
    """
    if read_head_index == 0:
      return tf.expand_dims(
          common_layers.mask_pos_gt(self._memory_size, self._memory_size),
          axis=0)
    else:
      raise ValueError("Read head index must be 0 for queue.")


class NeuralDequeCell(NeuralStackCell):
  """An subclass of the NeuralStackCell which reads/writes in both directions.

  See section 3.3 of Grefenstette et al., 2015.
  """

  def __init__(self, num_units, memory_size, embedding_size, reuse=None):
    # Override constructor to set 2 read/write heads.
    super(NeuralDequeCell, self).__init__(num_units,
                                          memory_size,
                                          embedding_size,
                                          num_read_heads=2,
                                          num_write_heads=2,
                                          reuse=reuse)

  def get_read_mask(self, read_head_index):
    if read_head_index == 0:
      # Use the same read mask as the queue for the bottom of the deque.
      return tf.expand_dims(
          common_layers.mask_pos_gt(self._memory_size, self._memory_size),
          axis=0)
    elif read_head_index == 1:
      # Use the same read mask as the stack for the top of the deque.
      return tf.expand_dims(
          common_layers.mask_pos_lt(self._memory_size, self._memory_size),
          axis=0)
    else:
      raise ValueError("Read head index must be either 0 or 1 for deque.")

  def get_write_head_offset(self, write_head_index):
    if write_head_index == 0:
      # Move the bottom write position back at each timestep.
      return -1
    elif write_head_index == 1:
      # Move the top write position forward at each timestep.
      return 1
    else:
      raise ValueError("Write head index must be 0 or 1 for deque.")

  def initialize_write_strengths(self, batch_size):
    """Initialize write strengths which write in both directions.

    Unlike in Grefenstette et al., It's writing out from the center of the
    memory so that it doesn't need to shift the entire memory forward at each
    step.

    Args:
      batch_size: The size of the current batch.

    Returns:
      A tf.float32 tensor of shape [num_write_heads, memory_size, 1].
    """
    memory_center = self._memory_size // 2
    return tf.expand_dims(
        tf.concat([
            # The write strength for the deque bottom.
            # Should be shifted back at each timestep.
            tf.one_hot([[memory_center - 1]] * batch_size,
                       depth=self._memory_size, dtype=tf.float32),
            # The write strength for the deque top.
            # Should be shifted forward at each timestep.
            tf.one_hot([[memory_center]] * batch_size,
                       depth=self._memory_size, dtype=tf.float32)
        ], axis=1), axis=3)


@registry.register_model
class NeuralStackModel(t2t_model.T2TModel):
  """An encoder-decoder T2TModel that uses NeuralStackCells.
  """

  def cell(self, hidden_size):
    """Build an RNN cell.

    This is exposed as its own function so that it can be overridden to provide
    different types of RNN cells.

    Args:
      hidden_size: The hidden size of the cell.

    Returns:
      A new RNNCell with the given hidden size.
    """
    return NeuralStackCell(hidden_size,
                           self._hparams.memory_size,
                           self._hparams.embedding_size)

  def _rnn(self, inputs, name, initial_state=None, sequence_length=None):
    """A helper method to build tf.nn.dynamic_rnn.

    Args:
      inputs: The inputs to the RNN. A tensor of shape
              [batch_size, max_seq_length, embedding_size]
      name: A namespace for the RNN.
      initial_state: An optional initial state for the RNN.
      sequence_length: An optional sequence length for the RNN.

    Returns:
      A tf.nn.dynamic_rnn operator.
    """
    layers = [self.cell(layer_size)
              for layer_size in self._hparams.controller_layer_sizes]
    with tf.variable_scope(name):
      return tf.nn.dynamic_rnn(
          tf.contrib.rnn.MultiRNNCell(layers),
          inputs,
          initial_state=initial_state,
          sequence_length=sequence_length,
          dtype=tf.float32,
          time_major=False)

  def body(self, features):
    """Build the main body of the model.

    Args:
      features: A dict of "inputs" and "targets" which have already been passed
        through an embedding layer. Inputs should have shape
        [batch_size, max_seq_length, 1, embedding_size]. Targets should have
        shape [batch_size, max_seq_length, 1, 1]

    Returns:
      The logits which get passed to the top of the model for inference.
      A tensor of shape [batch_size, seq_length, 1, embedding_size]
    """
    inputs = features.get("inputs")
    targets = features["targets"]

    if inputs is not None:
      inputs = common_layers.flatten4d3d(inputs)
      _, final_encoder_state = self._rnn(tf.reverse(inputs, axis=[1]),
                                         "encoder")
    else:
      final_encoder_state = None

    shifted_targets = common_layers.shift_right(targets)
    decoder_outputs, _ = self._rnn(
        common_layers.flatten4d3d(shifted_targets),
        "decoder",
        initial_state=final_encoder_state)
    return decoder_outputs


@registry.register_model
class NeuralQueueModel(NeuralStackModel):
  """Subcalss of NeuralStackModel which implements a queue.
  """

  def cell(self, hidden_size):
    """Build a NeuralQueueCell instead of a NeuralStackCell.

    Args:
      hidden_size: The hidden size of the cell.

    Returns:
      A new NeuralQueueCell with the given hidden size.
    """
    return NeuralQueueCell(hidden_size,
                           self._hparams.memory_size,
                           self._hparams.embedding_size)


@registry.register_model
class NeuralDequeModel(NeuralStackModel):
  """Subclass of NeuralStackModel which implements a double-ended queue.
  """

  def cell(self, hidden_size):
    """Build a NeuralDequeCell instead of a NeuralStackCell.

    Args:
      hidden_size: The hidden size of the cell.

    Returns:
      A new NeuralDequeCell with the given hidden size.
    """
    return NeuralDequeCell(hidden_size,
                           self._hparams.memory_size,
                           self._hparams.embedding_size)


@registry.register_hparams
def lstm_transduction():
  """HParams for LSTM base on transduction tasks."""
  hparams = common_hparams.basic_params1()
  hparams.daisy_chain_variables = False
  hparams.batch_size = 10
  hparams.clip_grad_norm = 1.0
  hparams.hidden_size = 128
  hparams.num_hidden_layers = 4
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.optimizer = "RMSProp"
  hparams.learning_rate = 0.01
  hparams.weight_decay = 0.0

  hparams.add_hparam("memory_size", 128)
  hparams.add_hparam("embedding_size", 32)
  return hparams


@registry.register_hparams
def neural_stack():
  """HParams for neural stacks and queues."""
  hparams = common_hparams.basic_params1()
  hparams.daisy_chain_variables = False
  hparams.batch_size = 10
  hparams.clip_grad_norm = 1.0
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.optimizer = "RMSProp"
  hparams.learning_rate = 0.0001
  hparams.weight_decay = 0.0

  hparams.add_hparam("controller_layer_sizes", [256, 512])
  hparams.add_hparam("memory_size", 128)
  hparams.add_hparam("embedding_size", 64)
  hparams.hidden_size = hparams.embedding_size
  return hparams


@registry.register_hparams
def neural_deque():
  """HParams for neural deques."""
  hparams = common_hparams.basic_params1()
  hparams.daisy_chain_variables = False
  hparams.batch_size = 10
  hparams.clip_grad_norm = 1.0
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.optimizer = "RMSProp"
  hparams.learning_rate = 0.0001
  hparams.weight_decay = 0.0

  hparams.add_hparam("controller_layer_sizes", [256, 512])
  hparams.add_hparam("memory_size", 256)
  hparams.add_hparam("embedding_size", 64)
  hparams.hidden_size = hparams.embedding_size
  return hparams
