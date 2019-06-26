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


def AtariCnn(hidden_sizes=(32, 32), output_size=128, mode='train'):
  """An Atari CNN."""
  del mode

  # TODO(jonni): Include link to paper?
  # Input shape: (B, T, H, W, C)
  # Output shape: (B, T, output_size)
  return tl.Model(
      tl.ToFloat(),
      tl.Div(divisor=255.0),

      # Set up 4 successive game frames, concatenated on the last axis.
      tl.Dup(), tl.Dup(), tl.Dup(),
      tl.Parallel(None, _shift_right(1), _shift_right(2), _shift_right(3)),
      tl.Concatenate(n_items=4, axis=-1),  # (B, T, H, W, 4C)

      tl.Conv(hidden_sizes[0], (5, 5), (2, 2), 'SAME'),
      tl.Relu(),
      tl.Conv(hidden_sizes[1], (5, 5), (2, 2), 'SAME'),
      tl.Relu(),
      tl.Flatten(n_axes_to_keep=2),  # B, T and rest.
      tl.Dense(output_size),
      tl.Relu(),
  )


def _shift_right(n):  # pylint: disable=invalid-name
  return [tl.ShiftRight()] * n
