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

"""MLP."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.trax import layers as tl


def MLP(n_hidden_layers=2,
        d_hidden=512,
        activation_fn=tl.Relu,
        n_output_classes=10,
        mode="train"):
  """A multi-layer feedforward (perceptron) network."""
  del mode

  return tl.Model(
      tl.Flatten(),
      [[tl.Dense(d_hidden), activation_fn()] for _ in range(n_hidden_layers)],
      tl.Dense(n_output_classes),
      tl.LogSoftmax(),
  )
