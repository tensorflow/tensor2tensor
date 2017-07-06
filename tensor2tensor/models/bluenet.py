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


def conv_module(kw, kh, sep, div):
  def convfn(x, hparams):
    return common_layers.subseparable_conv(
        x, hparams.hidden_size // div, (kw, kh),
        padding="SAME", separability=sep,
        name="conv_%d%d_sep%d_div%d" % (kw, kh, sep, div))
  return convfn


def layernorm_module(x, hparams):
  return common_layers.layer_norm(x, hparams.hidden_size, name="layer_norm")


def noamnorm_module(x, hparams):
  del hparams  # Unused.
  return common_layers.noam_norm(x)


def identity_module(x, hparams):
  del hparams  # Unused.
  return x


def first_binary_module(x, y, hparams):
  del y, hparams  # Unused.
  return x


def second_binary_module(x, y, hparams):
  del x, hparams  # Unused.
  return y


def sum_binary_module(x, y, hparams):
  del hparams  # Unused.
  return x + y


def shakeshake_binary_module(x, y, hparams):
  del hparams  # Unused.
  return common_layers.shakeshake2(x, y)


def run_binary_modules(modules, cur1, cur2, hparams):
  """Run binary modules."""
  selection_var = tf.get_variable("selection", [len(modules)],
                                  initializer=tf.zeros_initializer())
  inv_t = 100.0 * common_layers.inverse_exp_decay(
      hparams.anneal_until, min_value=0.01)
  selected_weights = tf.nn.softmax(selection_var * inv_t)
  all_res = [modules[n](cur1, cur2, hparams) for n in xrange(len(modules))]
  all_res = tf.concat([tf.expand_dims(r, axis=0) for r in all_res], axis=0)
  res = all_res * tf.reshape(selected_weights, [-1, 1, 1, 1, 1])
  return tf.reduce_sum(res, axis=0)


def run_unary_modules_basic(modules, cur, hparams):
  """Run unary modules."""
  selection_var = tf.get_variable("selection", [len(modules)],
                                  initializer=tf.zeros_initializer())
  inv_t = 100.0 * common_layers.inverse_exp_decay(
      hparams.anneal_until, min_value=0.01)
  selected_weights = tf.nn.softmax(selection_var * inv_t)
  all_res = [modules[n](cur, hparams) for n in xrange(len(modules))]
  all_res = tf.concat([tf.expand_dims(r, axis=0) for r in all_res], axis=0)
  res = all_res * tf.reshape(selected_weights, [-1, 1, 1, 1, 1])
  return tf.reduce_sum(res, axis=0)


def run_unary_modules_sample(modules, cur, hparams, k):
  """Run modules, sampling k."""
  selection_var = tf.get_variable("selection", [len(modules)],
                                  initializer=tf.zeros_initializer())
  selection = tf.multinomial(tf.expand_dims(selection_var, axis=0), k)
  selection = tf.squeeze(selection, axis=0)   # [k] selected classes.
  to_run = tf.one_hot(selection, len(modules))  # [k x nmodules] one-hot.
  to_run = tf.reduce_sum(to_run, axis=0)  # [nmodules], 0=not run, 1=run.
  all_res = [tf.cond(tf.less(to_run[n], 0.1),
                     lambda: tf.zeros_like(cur),
                     lambda i=n: modules[i](cur, hparams))
             for n in xrange(len(modules))]
  inv_t = 100.0 * common_layers.inverse_exp_decay(
      hparams.anneal_until, min_value=0.01)
  selected_weights = tf.nn.softmax(selection_var * inv_t - 1e9 * (1.0 - to_run))
  all_res = tf.concat([tf.expand_dims(r, axis=0) for r in all_res], axis=0)
  res = all_res * tf.reshape(selected_weights, [-1, 1, 1, 1, 1])
  return tf.reduce_sum(res, axis=0)


def run_unary_modules(modules, cur, hparams):
  if len(modules) < 8:
    return run_unary_modules_basic(modules, cur, hparams)
  return run_unary_modules_sample(modules, cur, hparams, 4)


def batch_deviation(x):
  """Average deviation of the batch."""
  x_mean = tf.reduce_mean(x, axis=[0], keep_dims=True)
  x_variance = tf.reduce_mean(
      tf.square(x - x_mean), axis=[0], keep_dims=True)
  return tf.reduce_mean(tf.sqrt(x_variance))


@registry.register_model
class BlueNet(t2t_model.T2TModel):

  def model_fn_body(self, features):
    hparams = self._hparams
    conv_modules = [conv_module(kw, kw, sep, div)
                    for kw in [3, 5, 7]
                    for sep in [0, 1]
                    for div in [1]] + [identity_module]
    activation_modules = [identity_module,
                          lambda x, _: tf.nn.relu(x),
                          lambda x, _: tf.nn.elu(x),
                          lambda x, _: tf.tanh(x)]
    norm_modules = [identity_module, layernorm_module, noamnorm_module]
    binary_modules = [first_binary_module, second_binary_module,
                      sum_binary_module, shakeshake_binary_module]
    inputs = features["inputs"]

    def run_unary(x, name):
      """A single step of unary modules."""
      x_shape = x.get_shape()
      with tf.variable_scope(name):
        with tf.variable_scope("norm"):
          x = run_unary_modules(norm_modules, x, hparams)
          x.set_shape(x_shape)
        with tf.variable_scope("activation"):
          x = run_unary_modules(activation_modules, x, hparams)
          x.set_shape(x_shape)
        with tf.variable_scope("conv"):
          x = run_unary_modules(conv_modules, x, hparams)
          x.set_shape(x_shape)
      return tf.nn.dropout(x, 1.0 - hparams.dropout), batch_deviation(x)

    cur1, cur2, extra_loss = inputs, inputs, 0.0
    cur_shape = inputs.get_shape()
    for i in xrange(hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % i):
        cur1, loss1 = run_unary(cur1, "unary1")
        cur2, loss2 = run_unary(cur2, "unary2")
        extra_loss += (loss1 + loss2) / float(hparams.num_hidden_layers)
        with tf.variable_scope("binary1"):
          next1 = run_binary_modules(binary_modules, cur1, cur2, hparams)
          next1.set_shape(cur_shape)
        with tf.variable_scope("binary2"):
          next2 = run_binary_modules(binary_modules, cur1, cur2, hparams)
          next2.set_shape(cur_shape)
        cur1, cur2 = next1, next2

    anneal = common_layers.inverse_exp_decay(hparams.anneal_until)
    extra_loss *= hparams.batch_deviation_loss_factor * anneal
    return cur1, extra_loss


@registry.register_hparams
def bluenet_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.batch_size = 4096
  hparams.hidden_size = 256
  hparams.dropout = 0.2
  hparams.symbol_dropout = 0.2
  hparams.label_smoothing = 0.1
  hparams.clip_grad_norm = 2.0
  hparams.num_hidden_layers = 8
  hparams.kernel_height = 3
  hparams.kernel_width = 3
  hparams.learning_rate_decay_scheme = "exp10k"
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
  hparams.add_hparam("anneal_until", 40000)
  hparams.add_hparam("batch_deviation_loss_factor", 0.001)
  return hparams


@registry.register_hparams
def bluenet_tiny():
  hparams = bluenet_base()
  hparams.batch_size = 1024
  hparams.hidden_size = 128
  hparams.num_hidden_layers = 4
  hparams.learning_rate_decay_scheme = "none"
  return hparams
