# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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
      # Fixed batch size turns off bucketing during training mode
      # and uses batch_size as minibatch size (use small batch_size<=32)
      use_fixed_batch_size=False,
      num_hidden_layers=4,
      kernel_height=3,
      kernel_width=1,
      hidden_size=64,
      compress_steps=0,
      # All hyperparameters ending in "dropout" are automatically set to 0.0
      # when not in training mode.
      dropout=0.2,
      clip_grad_norm=2.0,
      grad_noise_scale=0.0,
      summarize_grads=False,
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
      learning_rate_cosine_cycle_steps=250000,
      learning_rate=0.1,
      sampling_method="argmax",  # "argmax" or "random"
      sampling_temp=1.0,  # temperature for sampling
      problem_choice="adaptive",  # "uniform", "adaptive", "distributed"
      # expand the logits a piece at a time - saves memory.
      factored_logits=False,
      multiply_embedding_mode="sqrt_depth",
      # Parameters related to mixtures of experts.
      moe_hidden_sizes="2048",  # hidden layer sizes (comma-separated)
      moe_num_experts=64,  # number of experts per layer
      moe_k=2,  # how many experts to use for each batch element
      moe_loss_coef=1e-2,
      # Sequences of operations to perform on layer input and layer output.
      # Used by common_layers.layer_preprocess, common_layers.layer_postprocess
      # Each character repsesnts an operation:
      # none: no preprocessing
      #    d: apply dropout
      #    n: apply normalization (see norm_type and norm_epsilon)
      #    a: add layer input (residual connection - only during postprocess)
      # The special string "none" is used instead of the empty string
      # to indicate no pre/postprocesisng, since the empty string causes
      # trouble for hyperparameter tuning.
      # TODO(noam): The current settings ("", "dan") are the published version
      # of the transformer.  ("n", "da") seems better for harder-to-learn
      # models, so it should probably be the default.
      layer_preprocess_sequence="none",
      layer_postprocess_sequence="dan",
      # dropout rate to use during layer_preprocess and layer_postprocess
      layer_prepostprocess_dropout=0.1,
      # What type of normalization to use
      norm_type="layer",  # "batch", layer", "noam", "none".
      # epsilon parameter to normalization function
      norm_epsilon=1e-6,
      symbol_modality_num_shards=16,
      # During training, we drop sequences whose inputs and targets are shorter
      # than min_length
      min_length=0,
      # During training, we drop sequences whose inputs or targets are longer
      # than max_length.
      # If max_length==0, we use hparams.batch_size instead.
      max_length=0,
      # Maximum length in the smallest length bucket.  Setting this
      # flag too high will result in wasteful padding of short
      # sequences.  Due to some (hopefully) temporary hacks in the
      # data reading and batching code, setting this flag too low
      # results in a very long batch-shuffling queue.
      # TODO(noam): change this once the Datasets API changes.
      min_length_bucket=8,
      # This flag controls the number of length buckets in the data
      # reader.  The buckets have maximum lengths from
      # min_bucket_length to (max_length or batch_size), increasing
      # (approximately) by factors of length_bucket_step.
      length_bucket_step=1.1,
      # If set to True, drop sequences longer than max_length during eval.
      # This affects the validity of the evaluation metrics.
      eval_drop_long_sequences=False,
      # TODO(lukaszkaiser): these parameters should probably be set elsewhere.
      # in SymbolModality, share the output embeddings and the softmax
      # variables.
      # You can also share the input embeddings with the output embeddings
      # by using a problem_hparams that uses the same modality object for
      # the input_modality and target_modality.
      shared_embedding_and_softmax_weights=False,
      # In SymbolModality, skip the top layer, assume we're providing logits.
      symbol_modality_skip_top=False,
      # For each feature for which you want to override the default input
      # modality, add an entry to this semicolon-separated string. Entries are
      # formatted "feature_name:modality_type:modality_name", e.g.
      # "inputs:symbol:default;other_inputs:audio:identity".
      input_modalities="default",  # We don't use empty string in params.
      # To override the default target modality, specify
      # "modality_type:modality_name", e.g. "symbol:ctc".
      target_modality="default",
      # The maximum length of "input" sequence.
      # Sequences longer than this value will be truncated. 0 or negative values
      # mean there is no maximum or truncation.
      # You can change this behavior by overridding preprocess_example() method
      # in your problem class.
      max_input_seq_length=0,
      # The maximum length of "target" sequence.
      # Sequences longer than this value will be truncated. 0 or negative values
      # mean there is no maximum or truncation.
      # You can change this behavior by overridding preprocess_example() method
      # in your problem class.
      max_target_seq_length=0,
      # This flag allows us to optionally treat a seq-to-seq problem
      # as a language model.  Legal values are:
      #
      # "none" - Do not prepend the inputs to the targets.
      # "prepend_inputs_masked_attention"
      #     replace "targets" in preprocessing with
      #     tf.concat([inputs, [0], targets], axis=1)
      #     i.e. we prepend the inputs to the targets with a single
      #     padding token in between.  Use masked self-attention on the
      #     entire resulting sequence.  During training, we compute losses on
      #     the combined sequence.  During eval, we compute the metrics
      #     on only the targets portion.
      # "prepend_inputs_full_attention"
      #     similar to the previous option except that each
      #     position in the inputs portion can see the
      #     entire inputs portion.  This removes the challenge of
      #     autoregressively predicting the inputs portion.
      prepend_mode="none",
      # Scheduled sampling is interesting for auto-regressive models.
      # It runs an additional step using the generated output as autoregressive
      # targets, which can improve the models inference results later. The
      # parameter scheduled_sampling_prob determines with what probability
      # will such additional step be run. It's turned off (0.0) by default.
      # This probability will exponentially warm up for the number of
      # steps determined by scheduled_sampling_warmup_steps.
      # The tensor used for the second step will consist of outputs from
      # the first step mixed with gold truth, with the proportion of gold
      # determined by scheduled_sampling_gold_mixin_prob.
      scheduled_sampling_prob=0.0,
      scheduled_sampling_warmup_steps=50000,
      scheduled_sampling_gold_mixin_prob=0.5,
      # This is the actual batch size, *not* tokens per batch (i.e. for
      # language models this is the number of sentences in the batch)
      tpu_batch_size_per_shard=24,
  )


class RangedHParams(object):
  """Defines parameter ranges for tuning."""

  # From ParameterConfig proto
  LINEAR_SCALE = 1
  LOG_SCALE = 2
  REVERSE_LOG_SCALE = 3

  def __init__(self):
    self._categorical_params = {}
    self._discrete_params = {}
    self._discrete_float_params = {}
    self._float_params = {}
    self._int_params = {}

  def _check_reset_and_type_change(self, name, orig_ctr):
    """Check if name is in orig_ctr or in one of the other type containers."""
    # Resetting a hyperparameter
    if name in orig_ctr:
      tf.logging.warning("Overwriting hparam %s", name)

    ctr_names = [(self._categorical_params,
                  "categorical"), (self._discrete_params, "discrete"),
                 (self._float_params, "float"), (self._int_params, "int"),
                 (self._discrete_float_params, "discrete_float")]
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
    if name in self._discrete_float_params:
      del self._discrete_float_params[name]
    self._check_reset_and_type_change(name, self._float_params)
    self._float_params[name] = (name, min_val, max_val, scale, length)

  def set_discrete_float(self, name, val):
    self._check_reset_and_type_change(name, self._discrete_float_params)
    self._discrete_float_params[name] = (name, [val])

  def set_int(self, name, min_val, max_val, scale=None, length=None):
    self._check_reset_and_type_change(name, self._int_params)
    self._int_params[name] = (name, min_val, max_val, scale, length)

  def fix_select_params(self, hp):
    ctrs = [
        self._categorical_params, self._discrete_params,
        self._discrete_float_params, self._float_params, self._int_params
    ]
    for key, val in hp.values().iteritems():
      for ctr in ctrs:
        if key in ctr:
          del ctr[key]
      self.set_discrete(key, [val])


def fill_ranged_hparams_from_hparams(hparams, ranged_hparams):
  """Fill ranged_hparams with singleton values from hparams.

  HParams are placed in RangedHParams with the following functions, according to
  type:
    * int: set_discrete
    * bool: set_discrete
    * float: set_discrete_float
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
    elif hp_type == bool:
      ranged_hparams.set_discrete(name, [int(val)])
    elif hp_type == float:
      ranged_hparams.set_discrete_float(name, val)
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
  rhp.set_categorical(
      "optimizer",
      ["Adam", "Adagrad", "Momentum", "RMSProp", "SGD", "YellowFin"])
