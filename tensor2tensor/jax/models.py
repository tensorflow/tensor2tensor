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

"""J2J models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin

from jax.experimental import stax


@gin.configurable()
def mlp(num_hidden_layers=2,
        hidden_size=512,
        activation_fn=stax.Relu,
        num_output_classes=10):
  layers = [stax.Flatten]
  layers += [stax.Dense(hidden_size), activation_fn] * num_hidden_layers
  layers += [stax.Dense(num_output_classes), stax.LogSoftmax]
  return stax.serial(*layers)
