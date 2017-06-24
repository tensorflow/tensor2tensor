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

"""BlueNet: and out of the blue network to experiment with shake-shake."""

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


def residual_module(x, hparams, train, n, sep):
  """A stack of convolution blocks with residual connection."""
  k = (hparams.kernel_height, hparams.kernel_width)
  dilations_and_kernels = [((1, 1), k) for _ in xrange(n)]
  with tf.variable_scope("residual_module%d_sep%d" % (n, sep)):
    y = common_layers.subseparable_conv_block(
        x,
        hparams.hidden_size,
        dilations_and_kernels,
        padding="SAME",
        separability=sep,
        name="block")
    x = common_layers.layer_norm(x + y, hparams.hidden_size, name="lnorm")
  return tf.nn.dropout(x, 1.0 - hparams.dropout * tf.to_float(train))


def residual_module1(x, hparams, train):
  return residual_module(x, hparams, train, 1, 1)


def residual_module1_sep(x, hparams, train):
  return residual_module(x, hparams, train, 1, 0)


def residual_module2(x, hparams, train):
  return residual_module(x, hparams, train, 2, 1)


def residual_module2_sep(x, hparams, train):
  return residual_module(x, hparams, train, 2, 0)


def residual_module3(x, hparams, train):
  return residual_module(x, hparams, train, 3, 1)


def residual_module3_sep(x, hparams, train):
  return residual_module(x, hparams, train, 3, 0)


def norm_module(x, hparams, train):
  del train  # Unused.
  return common_layers.layer_norm(x, hparams.hidden_size, name="norm_module")


def identity_module(x, hparams, train):
  del hparams, train  # Unused.
  return x


def run_modules(blocks, cur, hparams, train, dp):
  """Run blocks in parallel using dp as data_parallelism."""
  assert len(blocks) % dp.n == 0
  res = []
  for i in xrange(len(blocks) // dp.n):
    res.extend(dp(blocks[i * dp.n:(i + 1) * dp.n], cur, hparams, train))
  return res


@registry.register_model
class BlueNet(t2t_model.T2TModel):

  def model_fn_body_sharded(self, sharded_features, train):
    dp = self._data_parallelism
    dp._reuse = False  # pylint:disable=protected-access
    hparams = self._hparams
    blocks = [identity_module, norm_module,
              residual_module1, residual_module1_sep,
              residual_module2, residual_module2_sep,
              residual_module3, residual_module3_sep]
    inputs = sharded_features["inputs"]

    cur = tf.concat(inputs, axis=0)
    cur_shape = cur.get_shape()
    for i in xrange(hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % i):
        processed = run_modules(blocks, cur, hparams, train, dp)
        cur = common_layers.shakeshake(processed)
        cur.set_shape(cur_shape)

    return list(tf.split(cur, len(inputs), axis=0)), 0.0


@registry.register_hparams
def bluenet_base():
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


@registry.register_hparams
def bluenet_tiny():
  hparams = bluenet_base()
  hparams.batch_size = 1024
  hparams.hidden_size = 128
  hparams.num_hidden_layers = 4
  hparams.learning_rate_decay_scheme = "none"
  return hparams
