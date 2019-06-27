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

"""Implementations of common recurrent neural network cells (RNNs)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.trax.layers import combinators as cb
from tensor2tensor.trax.layers import convolution
from tensor2tensor.trax.layers import core


def GRUCell(n_units):
  """Builds a traditional GRU cell with dense internal transformations.

  Gated Recurrent Unit paper: https://arxiv.org/abs/1412.3555


  Args:
    n_units: Number of hidden units.

  Returns:
    A Stax model representing a traditional GRU RNN cell.
  """
  return GeneralGRUCell(
      candidate_transform=lambda: core.Dense(n_units),
      memory_transform_fn=None,
      gate_nonlinearity=core.Sigmoid,
      candidate_nonlinearity=core.Tanh)


def ConvGRUCell(n_units, kernel_size=(3, 3)):
  """Builds a convolutional GRU.

  Paper: https://arxiv.org/abs/1511.06432.

  Args:
    n_units: Number of hidden units
    kernel_size: Kernel size for convolution

  Returns:
    A Stax model representing a GRU cell with convolution transforms.
  """

  def BuildConv():
    return convolution.Conv(
        filters=n_units, kernel_size=kernel_size, padding='SAME')

  return GeneralGRUCell(
      candidate_transform=BuildConv,
      memory_transform_fn=None,
      gate_nonlinearity=core.Sigmoid,
      candidate_nonlinearity=core.Tanh)


def GeneralGRUCell(candidate_transform,
                   memory_transform_fn=None,
                   gate_nonlinearity=core.Sigmoid,
                   candidate_nonlinearity=core.Tanh,
                   dropout_rate_c=0.1,
                   sigmoid_bias=0.5):
  r"""Parametrized Gated Recurrent Unit (GRU) cell construction.

  GRU update equations:
  $$ Update gate: u_t = \sigmoid(U' * s_{t-1} + B') $$
  $$ Reset gate: r_t = \sigmoid(U'' * s_{t-1} + B'') $$
  $$ Candidate memory: c_t = \tanh(U * (r_t \odot s_{t-1}) + B) $$
  $$ New State: s_t = u_t \odot s_{t-1} + (1 - u_t) \odot c_t $$

  See combinators.Gate for details on the gating function.


  Args:
    candidate_transform: Transform to apply inside the Candidate branch. Applied
      before nonlinearities.
    memory_transform_fn: Optional transformation on the memory before gating.
    gate_nonlinearity: Function to use as gate activation. Allows trying
      alternatives to Sigmoid, such as HardSigmoid.
    candidate_nonlinearity: Nonlinearity to apply after candidate branch. Allows
      trying alternatives to traditional Tanh, such as HardTanh
    dropout_rate_c: Amount of dropout on the transform (c) gate. Dropout works
      best in a GRU when applied exclusively to this branch.
    sigmoid_bias: Constant to add before sigmoid gates. Generally want to start
      off with a positive bias.

  Returns:
    A model representing a GRU cell with specified transforms.
  """
  gate_block = [  # u_t
      candidate_transform(),
      core.AddConstant(constant=sigmoid_bias),
      gate_nonlinearity(),
  ]
  reset_block = [  # r_t
      candidate_transform(),
      core.AddConstant(constant=sigmoid_bias),  # Want bias to start positive.
      gate_nonlinearity(),
  ]
  candidate_block = [
      cb.Dup(),
      reset_block,
      cb.Multiply(),  # Gate S{t-1} with sigmoid(candidate_transform(S{t-1}))
      candidate_transform(),  # Final projection + tanh to get Ct
      candidate_nonlinearity(),  # Candidate gate

      # Only apply dropout on the C gate. Paper reports 0.1 as a good default.
      core.Dropout(rate=dropout_rate_c)
  ]
  memory_transform = memory_transform_fn() if memory_transform_fn else []
  return cb.Model(
      cb.Dup(), cb.Dup(),
      cb.Parallel(memory_transform, gate_block, candidate_block),
      cb.Gate(),
  )
