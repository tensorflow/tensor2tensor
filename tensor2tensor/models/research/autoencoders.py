# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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

"""Autoencoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import common_layers
from tensor2tensor.models import basic
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_model
class BasicDiscreteAutoencoder(basic.BasicAutoencoder):

  def bottleneck(self, x, res_size):
    hparams = self._hparams
    x = tf.tanh(tf.layers.dense(x, hparams.bottleneck_size, name="bottleneck"))
    d = x + tf.stop_gradient(2 * tf.to_float(tf.less(0.0, x)) - 1.0 - x)
    y = tf.nn.dropout(x, keep_prob=1.0 - hparams.dropout)
    x = common_layers.mix(d, y, hparams.discretize_warmup_steps,
                          hparams.mode == tf.estimator.ModeKeys.TRAIN)
    x = tf.layers.dense(x, res_size, name="unbottleneck")
    return x


@registry.register_hparams
def basic_discrete_autoencoder():
  """Basic autoencoder model."""
  hparams = basic.basic_autoencoder()
  hparams.hidden_size = 128
  hparams.bottleneck_size = 512
  hparams.bottleneck_warmup_steps = 3000
  hparams.add_hparam("discretize_warmup_steps", 5000)
  return hparams
