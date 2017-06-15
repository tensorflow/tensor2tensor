# Copyright 2017 Google Inc.
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

"""Xception."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.models import common_hparams
from tensor2tensor.models import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


def residual_block(x, hparams, train):
  """A stack of convolution blocks with residual connection."""
  k = (hparams.kernel_height, hparams.kernel_width)
  dilations_and_kernels = [((1, 1), k) for _ in xrange(3)]
  y = common_layers.subseparable_conv_block(
      x,
      hparams.hidden_size,
      dilations_and_kernels,
      padding="SAME",
      separability=0,
      name="residual_block")
  x = common_layers.layer_norm(x + y, hparams.hidden_size, name="lnorm")
  return tf.nn.dropout(x, 1.0 - hparams.dropout * tf.to_float(train))


def xception_internal(inputs, hparams, train):
  """Xception body."""
  with tf.variable_scope("xception"):
    cur = inputs
    for i in xrange(hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % i):
        cur = residual_block(cur, hparams, train)
    return cur


@registry.register_model
class Xception(t2t_model.T2TModel):

  def model_fn_body(self, features, train):
    return xception_internal(features["inputs"], self._hparams, train)


@registry.register_hparams
def xception_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.batch_size = 4096
  hparams.hidden_size = 768
  hparams.dropout = 0.2
  hparams.symbol_dropout = 0.2
  hparams.label_smoothing = 0.1
  hparams.clip_grad_norm = 2.0
  hparams.num_hidden_layers = 8
  hparams.kernel_height = 3
  hparams.kernel_width = 3
  hparams.learning_rate_decay_scheme = "exp50k"
  hparams.learning_rate = 0.05
  hparams.learning_rate_warmup_steps = 3000
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 3.0
  hparams.num_sampled_classes = 0
  hparams.sampling_method = "argmax"
  hparams.optimizer_adam_epsilon = 1e-6
  hparams.optimizer_adam_beta1 = 0.85
  hparams.optimizer_adam_beta2 = 0.997
  hparams.add_hparam("imagenet_use_2d", True)
  return hparams
