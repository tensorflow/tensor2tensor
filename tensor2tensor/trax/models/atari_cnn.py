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


def FrameStack(n_frames):
  """Stacks a fixed number of frames along the dimension 1."""
  # Input shape: (B, T, ..., C).
  # Output shape: (B, T, ..., C * n_frames).
  assert n_frames >= 1
  if n_frames == 1:
    return ()
  return (
      # Make n_frames copies of the input sequence.
      [tl.Dup()] * (n_frames - 1),
      # Shift copies to the right by [0, .., n_frames - 1] frames.
      tl.Parallel(*map(_shift_right, range(n_frames))),
      # Concatenate along the channel dimension.
      tl.Concatenate(n_items=n_frames, axis=-1),
  )


def AtariCnn(n_frames=4, hidden_sizes=(32, 32), output_size=128, mode='train'):
  """An Atari CNN."""
  del mode

  # TODO(jonni): Include link to paper?
  # Input shape: (B, T, H, W, C)
  # Output shape: (B, T, output_size)
  return tl.Model(
      tl.ToFloat(),
      tl.Div(divisor=255.0),

      # Set up n_frames successive game frames, concatenated on the last axis.
      FrameStack(n_frames=n_frames),  # (B, T, H, W, 4C)

      tl.Conv(hidden_sizes[0], (5, 5), (2, 2), 'SAME'),
      tl.Relu(),
      tl.Conv(hidden_sizes[1], (5, 5), (2, 2), 'SAME'),
      tl.Relu(),
      tl.Flatten(n_axes_to_keep=2),  # B, T and rest.
      tl.Dense(output_size),
      tl.Relu(),
  )


def FrameStackMLP(n_frames=4, hidden_sizes=(64,), output_size=64,
                  mode='train'):
  """MLP operating on a fixed number of last frames."""
  del mode

  return tl.Model(
      FrameStack(n_frames=n_frames),
      [[tl.Dense(d_hidden), tl.Relu()] for d_hidden in hidden_sizes],
      tl.Dense(output_size),
  )


def _shift_right(n):  # pylint: disable=invalid-name
  return [tl.ShiftRight()] * n
