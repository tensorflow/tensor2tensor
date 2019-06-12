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

"""Simple net for playing Atari games using PPO."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.trax import layers as tl


def AtariCnn(hidden_sizes=(32, 32), output_size=128):
  """An Atari CNN."""
  # Input's shape = (B, T, H, W, C)
  return tl.Model(
      tl.ToFloat(),
      tl.Div(divisor=255.0),
      # Have 4 copies of the input, each one shifted to the right by one.
      tl.Branch(
          [],
          [tl.ShiftRight()],
          [tl.ShiftRight(), tl.ShiftRight()],
          [tl.ShiftRight(), tl.ShiftRight(), tl.ShiftRight()]
      ),
      # Concatenated on the last axis.
      tl.Concatenate(axis=-1),  # (B, T, H, W, 4C)
      tl.Conv(hidden_sizes[0], (5, 5), (2, 2), 'SAME'),
      tl.Relu(),
      tl.Conv(hidden_sizes[1], (5, 5), (2, 2), 'SAME'),
      tl.Relu(),
      tl.Flatten(n_axes_to_keep=2),  # B, T and rest.
      tl.Dense(output_size),
      tl.Relu(),
      # Eventually this is shaped (B, T, output_size)
  )
