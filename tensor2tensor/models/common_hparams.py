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

"""Hyperparameters and ranges common to multiple models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import six
from six.moves import zip  # pylint: disable=redefined-builtin
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_hparams("basic_1")
def basic_params1():
  """A set of basic hyperparameters."""
  return tf.contrib.training.HParams(
      batch_size=4096,  # in tokens per batch per gpu
      # This flag controls the number of length buckets in the data reader.
      # Too many buckets slows down data reading - this needs fixing.
      # Too few buckets mean lots of wasted padding.
      # If this value is 1, we have buckets with maximum lengths:
      # [8, 12, 16, 24, 32, 48 ... (max_length or batch_size)]
      # If this value is 2, we have buckets with maximum lengths:
      # [8, 10, 12, 14, 16, 20, 24 ... (max_length or batch_size)]
      batching_mantissa_bits=1,
      num_hidden_layers=4,
      kernel_height=3,
      kernel_width=1,
      hidden_size=64,
      compress_steps=0,
      dropout=0.2,
      clip_grad_norm=2.0,
      initializer="orthogonal",
      initializer_gain=1.5,
      label_smoothing=0.1,
      optimizer="Adam",
      optimizer_adam_epsilon=1e-6,
      optimizer_adam_beta1=0.85,
      optimizer_adam_beta2=0.997,
      optimizer_momentum_momentum=0.9,
      weight_decay=0.1,
      weight_noise=0.0,
      learning_rate_decay_scheme="none",
      learning_rate_warmup_steps=100,
      learning_rate=0.1,
      sampling_method="argmax",  # "argmax" or "random"
      problem_choice="adaptive",  # "uniform", "adaptive", "distributed"
      multiply_embedding_mode="sqrt_depth",
      symbol_modality_num_shards=16,
      # setting the max length in a minibatch. 0 means default behavior,
      # max_length = hparams.batch_size * length_multiplier
      max_length=0,
      # in SymbolModality, share the output embeddings and the softmax
      # variables.
      # You can also share the input embeddings with the output embeddings
      # by using a problem_hparams that uses the same modality object for
      # the input_modality and target_modality.
      shared_embedding_and_softmax_weights=int(False),
      # For each feature for which you want to override the default input
      # modality, add an entry to this semicolon-separated string. Entries are
      # formatted "feature_name:modality_type:modality_name", e.g.
      # "inputs:image:small_image_modality;other_inputs:audio:identity".
      input_modalities="",
      # To override the default target modality, specify
      # "modality_type:modality_name", e.g. "image:small_image_modality".
      target_modality="")


class RangedHParams(object):
  """Defines parameter ranges for tuning."""

  # From ParameterConfig proto
  LINEAR_SCALE = 1
  LOG_SCALE = 2
  REVERSE_LOG_SCALE = 3

  def __init__(self):
    self._categorical_params = {}
    self._discrete_params = {}
    self._float_params = {}
    self._int_params = {}

  def _check_reset_and_type_change(self, name, orig_ctr):
    """Check if name is in orig_ctr or in one of the other type containers."""
    # Resetting a hyperparameter
    if name in orig_ctr:
      tf.logging.warning("Overwriting hparam %s", name)

    ctr_names = [(self._categorical_params,
                  "categorical"), (self._discrete_params, "discrete"),
                 (self._float_params, "float"), (self._int_params, "int")]
    ctrs, names = list(zip(*ctr_names))
    orig_name = names[ctrs.index(orig_ctr)]

    for ctr, ctr_name in ctr_names:
      if ctr is orig_ctr:
        continue

      # Using a different type for the same hyperparameter name
      if name in ctr:
        raise ValueError("Setting hyperparameter %s as type %s, but a "
                         "hyperparemeter of the same name was originally "
                         "registered as type %s" % (name, ctr_name, orig_name))

  def set_categorical(self, name, categories, length=None):
    self._check_reset_and_type_change(name, self._categorical_params)
    self._categorical_params[name] = (name, categories, length)

  def set_discrete(self, name, feasible_points, scale=None, length=None):
    self._check_reset_and_type_change(name, self._discrete_params)
    self._discrete_params[name] = (name, feasible_points, scale, length)

  def set_float(self, name, min_val, max_val, scale=None, length=None):
    self._check_reset_and_type_change(name, self._float_params)
    self._float_params[name] = (name, min_val, max_val, scale, length)

  def set_int(self, name, min_val, max_val, scale=None, length=None):
    self._check_reset_and_type_change(name, self._int_params)
    self._int_params[name] = (name, min_val, max_val, scale, length)


def fill_ranged_hparams_from_hparams(hparams, ranged_hparams):
  """Fill ranged_hparams with singleton values from hparams.

  HParams are placed in RangedHParams with the following functions, according to
  type:
    * int: set_discrete
    * float: set_float
    * str: set_categorical

  Args:
    hparams: tf.contrib.training.HParams; contains the hyperparameters to copy
      over to ranged_hparams.
    ranged_hparams: RangedHParams; will have hparams values copied to it.

  Raises:
    ValueError: if hparams contains a hyperparameter not of type
      {int, float, str, bool}.
  """
  for name, (hp_type, is_multivalent) in six.iteritems(hparams._hparam_types):  # pylint: disable=protected-access

    if is_multivalent:
      raise ValueError("Multivalent hparams not supported in RangedHParams. "
                       "Hyperparameter %s is multivalent." % name)
    val = getattr(hparams, name)
    if hp_type == int:
      ranged_hparams.set_discrete(name, [val])
    elif hp_type == float:
      ranged_hparams.set_float(name, val, val)
    elif hp_type == str:
      ranged_hparams.set_categorical(name, [val])
    else:
      raise ValueError("Unsupported type %s for param %s" % (hp_type, name))


@registry.register_ranged_hparams("basic1")
def basic_range1(ranged_hparams):
  """A basic range of hyperparameters."""
  rhp = ranged_hparams

  hparams = basic_params1()
  fill_ranged_hparams_from_hparams(hparams, rhp)

  rhp.set_discrete("batch_size", [1024, 2048, 4096])
  rhp.set_discrete("num_hidden_layers", [1, 2, 3, 4, 5, 6])
  rhp.set_discrete("hidden_size", [32, 64, 128, 256, 512], scale=rhp.LOG_SCALE)
  rhp.set_discrete("kernel_height", [1, 3, 5, 7])
  rhp.set_discrete("kernel_width", [1, 3, 5, 7])
  rhp.set_discrete("compress_steps", [0, 1, 2])
  rhp.set_float("dropout", 0.0, 0.5)
  rhp.set_float("weight_decay", 1e-4, 10.0, scale=rhp.LOG_SCALE)
  rhp.set_float("label_smoothing", 0.0, 0.2)
  rhp.set_float("clip_grad_norm", 0.01, 50.0, scale=rhp.LOG_SCALE)
  rhp.set_float("learning_rate", 0.005, 2.0, scale=rhp.LOG_SCALE)
  rhp.set_categorical("initializer",
                      ["uniform", "orthogonal", "uniform_unit_scaling"])
  rhp.set_float("initializer_gain", 0.5, 3.5)
  rhp.set_categorical("learning_rate_decay_scheme",
                      ["none", "sqrt", "noam", "exp10k"])
  rhp.set_float("optimizer_adam_epsilon", 1e-7, 1e-2, scale=rhp.LOG_SCALE)
  rhp.set_float("optimizer_adam_beta1", 0.8, 0.9)
  rhp.set_float("optimizer_adam_beta2", 0.995, 0.999)
  rhp.set_categorical("optimizer",
                      ["Adam", "Adagrad", "Momentum", "RMSProp", "SGD"])
