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

"""T2TModel Base Class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import functools
import math
import os
import time
import six

from tensor2tensor.data_generators import multi_problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators.problem import problem_hparams_to_features
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
from tensor2tensor.layers.common_attention import mixed_precision_is_enabled
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import decoding
from tensor2tensor.utils import expert_utils as eu
from tensor2tensor.utils import hparams_lib
from tensor2tensor.utils import learning_rate
from tensor2tensor.utils import metrics
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import optimize
from tensor2tensor.utils import quantization
from tensor2tensor.utils import registry

import tensorflow as tf

from tensorflow.python.layers import base
from tensorflow.python.ops import inplace_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import tf_inspect as inspect

_no_problem_err_str = (
    "The default implementation of %s requires that the "
    "model be used with a Problem. If using a Problem, augment the "
    "hparams object with trainer_lib.add_problem_hparams. If not, "
    "override %s.")
_no_problem_err = (
    lambda method_name: _no_problem_err_str % (method_name, method_name))


def _flatten_dict(original_dict):
  """Flatten dict of dicts into a single dict with appropriate prefixes.

  Handles only 2 levels of nesting in the original dict.

  Args:
    original_dict: Dict which may contain one or more dicts.
  Returns:
    flat_dict: Dict without any nesting. Any dicts in the original dict have
      their keys as prefixes in the new dict.
  Raises:
    ValueError if the original dict has more than two levels of nesting.
  """
  flat_dict = {}
  for key, value in original_dict.items():
    if isinstance(value, dict):
      for name, tensor in value.items():
        if isinstance(tensor, dict):
          raise ValueError("flatten_dict only handles 2 levels of nesting.")
        flat_key = "__" + key + "_" + name
        flat_dict[flat_key] = tensor
    else:
      flat_dict[key] = value

  return flat_dict


def _unflatten_dict(flat_dict, prefixes):
  """Returns a dict of dicts if any prefixes match keys in the flat dict.

    The function handles the case where the prefix may not be a dict.

  Args:
    flat_dict: A dict without any nesting.
    prefixes: A list of strings which may have been dicts in the
      original structure.

  """
  original_dict = {}
  for key, value in flat_dict.items():
    prefix_found = False
    for prefix in prefixes:
      full_prefix = "__" + prefix + "_"
      if key.startswith(full_prefix):
        # Add a dict to the original dict with key=prefix
        if prefix not in original_dict:
          original_dict[prefix] = {}
        original_dict[prefix][key[len(full_prefix):]] = value
        prefix_found = True
        break
    if not prefix_found:
      # No key matched a prefix in the for loop.
      original_dict[key] = value

  return original_dict


class T2TModel(base.Layer):
  """Abstract base class for models.

  `T2TModel` has three typical usages:

  1. Estimator: The method `make_estimator_model_fn` builds a `model_fn` for
     the tf.Estimator workflow of training, evaluation, and prediction.
     It performs the method `call`, which performs the core computation,
     followed by `estimator_spec_train`, `estimator_spec_eval`, or
     `estimator_spec_predict` depending on the tf.Estimator mode.
  2. Layer: The method `call` enables `T2TModel` to be used a callable by
     itself. It calls the following methods:

     * `bottom`, which transforms features according to `problem_hparams`' input
       and target `Modality`s;
     * `body`, which takes features and performs the core model computation to
        return output and any auxiliary loss terms;
     * `top`, which takes features and the body output, and transforms them
       according to `problem_hparams`' input and target `Modality`s to return
       the final logits;
     * `loss`, which takes the logits, forms any missing training loss, and sums
       all loss terms.
  3. Inference: The method `infer` enables `T2TModel` to make sequence
     predictions by itself.

  Subclasses generally only need to override `body`.
  """
  REGISTERED_NAME = None  # Updated on registration.

  def __init__(self,
               hparams,
               mode=tf.estimator.ModeKeys.TRAIN,
               problem_hparams=None,
               data_parallelism=None,
               decode_hparams=None,
               **kwargs):
    """Creates a T2TModel.

    Args:
      hparams: HParams, model hyperparameters.
      mode: tf.estimator.ModeKeys, the execution mode.
      problem_hparams: HParams, hyperparameters for the
        Problem. If provided here or in hparams.problem_hparams, the model will
        automatically determine bottom, top, and loss methods. If not provided,
        calling the model will only invoke body.
      data_parallelism: a expert_utils.Parallelism object,
        specifies devices for data parallelism.
      decode_hparams: a hyperparameter object with decoding parameters.
        See decoding.decode_hparams.
      **kwargs: arguments to pass to base.Layer constructor.
    """
    # Determine name first: use registered name if possible, class name else.
    default_name = registry.default_name(type(self))
    name = self.REGISTERED_NAME or default_name
    super(T2TModel, self).__init__(
        trainable=mode == tf.estimator.ModeKeys.TRAIN, name=name, **kwargs)

    if not problem_hparams and hasattr(hparams, "problem_hparams"):
      problem_hparams = hparams.problem_hparams
    self._problem_hparams = problem_hparams

    # Setup hparams
    hparams = hparams_lib.copy_hparams(hparams)
    if self._problem_hparams and hparams.shared_embedding_and_softmax_weights:
      # If vocabularies differ, unset shared_embedding_and_softmax_weights.
      input_vocab_size = self._problem_hparams.vocab_size.get("inputs")
      target_vocab_size = self._problem_hparams.vocab_size.get("targets")
      if input_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
        input_vocab_size += (-input_vocab_size) % hparams.vocab_divisor
      if target_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
        target_vocab_size += (-target_vocab_size) % hparams.vocab_divisor
      if (input_vocab_size is not None and target_vocab_size is not None and
          input_vocab_size != target_vocab_size):
        log_info("Unsetting shared_embedding_and_softmax_weights.")
        hparams.shared_embedding_and_softmax_weights = 0

      if hparams.hidden_size:
        hidden_size = hparams.hidden_size
      else:
        hidden_size = 1024
      mlperf_log.transformer_print(
          key=mlperf_log.MODEL_HP_EMBEDDING_SHARED_WEIGHTS,
          value={
              "vocab_size": target_vocab_size,
              "hidden_size": hidden_size
          },
          hparams=hparams)

    if self._problem_hparams:
      for feature_name, modality in six.iteritems(
          self._problem_hparams.modality):
        # If prepend mode, set weights_fn to appropriately handle it.
        if (modality in (modalities.ModalityType.CTC_SYMBOL,
                         modalities.ModalityType.IDENTITY_SYMBOL,
                         modalities.ModalityType.SYMBOL,
                         modalities.ModalityType.SYMBOL_ONE_HOT)):
          if (hparams.prepend_mode == "prepend_inputs_full_attention" or
              (hparams.prepend_mode == "prepend_inputs_masked_attention" and
               mode != tf.estimator.ModeKeys.TRAIN)):
            weights_fn = common_layers.weights_prepend_inputs_to_targets
            hparams.weights_fn[feature_name] = weights_fn

    self._original_hparams = hparams
    self.set_mode(mode)

    self._decode_hparams = hparams_lib.copy_hparams(
        decode_hparams or decoding.decode_hparams())
    self._data_parallelism = data_parallelism or eu.Parallelism([""])
    self._num_datashards = self._data_parallelism.n
    self._ps_devices = self._data_parallelism.ps_devices
    self._eager_var_store = create_eager_var_store()
    if not common_layers.is_xla_compiled():
      self.summarize_hparams()
    self._variable_scopes = {}

  def _add_variable_scope(self, key, vs):
    if key not in self._variable_scopes:
      self._variable_scopes[key] = vs

  def summarize_hparams(self):
    def create_hparams_summary(hparams, name):
      hparams_strs = [tf.convert_to_tensor([k, str(v)])
                      for k, v in hparams.values().items()]
      tf.summary.text(name, tf.cast(tf.stack(hparams_strs), tf.string))

    create_hparams_summary(self._hparams, "%s_hparams" % self.name)
    if self._problem_hparams:
      create_hparams_summary(self._problem_hparams,
                             "%s_problem_hparams" % self.name)

  # Replace the two methods below in order to add custom SessionRunHooks to
  # the training procedure.
  @staticmethod
  def train_hooks(hook_context):
    return []

  @staticmethod
  def eval_hooks(hook_context):
    return []

  @property
  def hparams(self):
    return self._hparams

  @property
  def problem_hparams(self):
    return self._problem_hparams

  @property
  def is_training(self):
    return self._hparams.mode == tf.estimator.ModeKeys.TRAIN

  @property
  def is_predicting(self):
    return self._hparams.mode == tf.estimator.ModeKeys.PREDICT

  @property
  def has_input(self):
    if self._problem_hparams:
      return "inputs" in self._problem_hparams.modality
    else:
      return True

  @property
  def _custom_getter(self):
    if self.hparams.weight_dtype == "bfloat16":
      if self.hparams.optimizer != "Adafactor":
        raise NotImplementedError(
            "weight_dtype=bfloat16 only implemented with Adafactor optimizer")
      activation_dtype = tf.float32
      if self.hparams.activation_dtype == "bfloat16":
        activation_dtype = tf.bfloat16
      return quantization.EighthPowerEncoding().custom_getter(
          activation_dtype=activation_dtype)
    elif self.hparams.activation_dtype == "bfloat16":
      return quantization.bfloat16_activations_var_getter
    elif mixed_precision_is_enabled(hparams=self.hparams):
      return quantization.float16_activations_var_getter
    else:
      return None

  @property
  def _target_modality_is_real(self):
    """Whether the target modality is real-valued."""
    vocab_size = self._problem_hparams.vocab_size["targets"]
    if vocab_size is not None and hasattr(self._hparams, "vocab_divisor"):
      vocab_size += (-vocab_size) % self._hparams.vocab_divisor
    modality = self._problem_hparams.modality["targets"]
    modality_name = self._hparams.name.get(
        "targets",
        modalities.get_name(modality))(self._hparams, vocab_size)
    return modality_name.startswith("Real")

  def call(self, inputs, **kwargs):
    del kwargs
    features = inputs
    set_custom_getter_compose(self._custom_getter)
    tf.get_variable_scope().set_initializer(
        optimize.get_variable_initializer(self.hparams))
    with self._eager_var_store.as_default():
      self._fill_problem_hparams_features(features)
      summarize_features(features, num_shards=self._num_datashards)
      sharded_features = self._shard_features(features)
      sharded_logits, losses = self.model_fn_sharded(sharded_features)
      if isinstance(sharded_logits, dict):
        concat_logits = {}
        for k, v in six.iteritems(sharded_logits):
          concat_logits[k] = tf.concat(v, 0)
        return concat_logits, losses
      else:
        return tf.concat(sharded_logits, 0), losses

  @staticmethod
  def has_symmetric_shards(model_name):
    # model_fn is sharded symmetrically unless the model overrides body_sharded
    # method to manually control the sharding.
    model_cls = registry.model(model_name)
    return not model_cls.use_body_sharded()

  @staticmethod
  def use_body_sharded():
    return False

  def body_sharded(self, sharded_features):
    raise NotImplementedError("Models that wish to manually control sharding, "
                              "e.g. MoE models, should override body_sharded "
                              "and set use_body_sharded to True.")

  def model_fn_sharded(self, sharded_features):
    dp = self._data_parallelism
    datashard_to_features = self._to_features_per_datashard(sharded_features)
    if self.use_body_sharded():
      # MoE models override body_sharded
      transformed_features = dp(self.bottom, datashard_to_features)
      body_out = self.body_sharded(
          self._to_single_features_dict(transformed_features))
      body_out, losses = self._normalize_body_output(body_out)
      if "training" in losses:
        log_info("Skipping T2TModel top and loss because training loss "
                 "returned from body")
        sharded_logits = body_out
      else:
        if isinstance(body_out, dict):
          sharded_logits = collections.OrderedDict()
          sharded_losses = collections.OrderedDict()
          for k, v in sorted(six.iteritems(body_out)):
            sharded_logits[k] = dp(self.top, v, datashard_to_features)
            sharded_losses[k] = dp(self.loss, sharded_logits[k],
                                   datashard_to_features)
          training_loss_dict = average_sharded_losses([({
              "training": l
          } for l in loss) for loss in sharded_losses.values()])
          losses.update(training_loss_dict)
        else:
          sharded_logits = dp(self.top, body_out, datashard_to_features)
          sharded_losses = dp(self.loss, sharded_logits, datashard_to_features)
          if isinstance(sharded_losses, tuple):
            nums, dens = sharded_losses
            sharded_losses = zip(nums, dens)
          training_loss_dict = average_sharded_losses([{
              "training": loss
          } for loss in sharded_losses])
          losses.update(training_loss_dict)
    else:
      sharded_logits, sharded_losses = dp(self.model_fn, datashard_to_features)
      if isinstance(sharded_logits[0], dict):
        temp_dict = {k: [] for k, _ in six.iteritems(sharded_logits[0])}
        for k, _ in six.iteritems(sharded_logits[0]):
          for l in sharded_logits:
            temp_dict[k].append(l[k])
        sharded_logits = temp_dict
      losses = average_sharded_losses(sharded_losses)

    # TODO(rsepassi): Reenable scheduled sampling
    # Disabled because of model_fn_sharded refactor
    #
    # do_scheduled_sampling = (  # Only do it if training and set for it.
    #     self.hparams.scheduled_sampling_prob > 0.0 and
    #     self.hparams.mode == tf.estimator.ModeKeys.TRAIN)
    # if do_scheduled_sampling:
    #   sharded_logits, losses = scheduled_sampling(
    #       self.hparams, self._problem_hparams, dp,
    #       sharded_logits, losses, sharded_features,
    #       transformed_features, self)

    return sharded_logits, losses

  def model_fn(self, features):
    with tf.variable_scope(tf.get_variable_scope(), use_resource=True) as vs:
      self._add_variable_scope("model_fn", vs)
      transformed_features = self.bottom(features)

      if self.hparams.activation_dtype == "bfloat16":
        for k, v in sorted(six.iteritems(transformed_features)):
          if v.dtype == tf.float32:
            transformed_features[k] = tf.cast(v, tf.bfloat16)

      with tf.variable_scope("body") as body_vs:
        self._add_variable_scope("body", body_vs)
        log_info("Building model body")
        body_out = self.body(transformed_features)
      output, losses = self._normalize_body_output(body_out)

      if "training" in losses:
        log_info("Skipping T2TModel top and loss because training loss "
                 "returned from body")
        logits = output
      else:
        logits = self.top(output, features)
        losses["training"] = 0.0
        if (self._hparams.mode != tf.estimator.ModeKeys.PREDICT and
            self._hparams.mode != "attack"):
          losses["training"] = self.loss(logits, features)

      return logits, losses

  def bottom(self, features):
    """Transforms features to feed into body.

    Args:
      features: dict of str to Tensor. Typically it is the preprocessed data
        batch after Problem's preprocess_example().

    Returns:
      transformed_features: dict of same key-value pairs as features. The value
        Tensors are newly transformed.
    """
    if not self._problem_hparams:
      log_warn("Without a Problem, T2TModel.bottom is a passthrough.")
      return features

    transformed_features = collections.OrderedDict()
    all_previous_modalities = []
    target_modality = _create_target_modality(self._problem_hparams.modality)

    # Transform features via its corresponding modality.
    for feature_name, modality in sorted(
        six.iteritems(self._problem_hparams.modality)):
      if feature_name not in features:
        tf.logging.warning("Missing feature %s - ignoring." % feature_name)
        continue
      vocab_size = self._problem_hparams.vocab_size[feature_name]
      if vocab_size is not None and hasattr(self._hparams, "vocab_divisor"):
        vocab_size += (-vocab_size) % self._hparams.vocab_divisor
      modality_name = self._hparams.name.get(
          feature_name,
          modalities.get_name(modality))(self._hparams, vocab_size)
      # Use if-else clauses to preserve behavior of previous changes: namely,
      # the variable scope name for the targets feature if there is only one
      # target modality; and to reuse variable scopes for only input modalities.
      if feature_name in target_modality:
        if len(target_modality) > 1:
          variable_scope_name = "%s/%s" % (modality_name, feature_name)
        else:
          variable_scope_name = modality_name
        bottom = self._hparams.bottom.get(
            feature_name,
            modalities.get_targets_bottom(modality))
        # TODO(aidangomez): share variables?
        with tf.variable_scope(variable_scope_name) as vs:
          self._add_variable_scope(variable_scope_name, vs)
          log_info("Transforming feature '%s' with %s.targets_bottom",
                   feature_name,
                   modality_name)
          transformed_features[feature_name] = bottom(features[feature_name],
                                                      self._hparams,
                                                      vocab_size)
      else:
        bottom = self._hparams.bottom.get(feature_name,
                                          modalities.get_bottom(modality))
        do_reuse = modality_name in all_previous_modalities
        with tf.variable_scope(modality_name, reuse=do_reuse) as vs:
          self._add_variable_scope(modality_name, vs)
          log_info("Transforming feature '%s' with %s.bottom",
                   feature_name,
                   modality_name)
          transformed_features[feature_name] = bottom(features[feature_name],
                                                      self._hparams,
                                                      vocab_size)
        all_previous_modalities.append(modality_name)

    for key in features:
      if key not in transformed_features:
        # For features without a modality, we pass them along as is
        transformed_features[key] = features[key]
      else:
        # Other features get passed along with the "raw" suffix
        transformed_features[key + "_raw"] = features[key]

    return transformed_features

  def body(self, features):
    """Computes the targets' pre-logit activations given transformed inputs.

    Most `T2TModel` subclasses will override this method.

    Args:
      features: dict of str to Tensor, where each Tensor has shape [batch_size,
        ..., hidden_size]. It typically contains keys `inputs` and `targets`.

    Returns:
      output: Tensor of pre-logit activations with shape [batch_size, ...,
              hidden_size].
      losses: Either single loss as a scalar, a list, a Tensor (to be averaged),
              or a dictionary of losses. If losses is a dictionary with the key
              "training", losses["training"] is considered the final training
              loss and output is considered logits; self.top and self.loss will
              be skipped.
    """
    raise NotImplementedError("Abstract Method")

  def _top_single(self, body_output, feature_name, features):
    if not self._problem_hparams:
      log_warn("Without a Problem, T2TModel.top is a passthrough.")
      return body_output

    modality = self._problem_hparams.modality[feature_name]
    vocab_size = self._problem_hparams.vocab_size[feature_name]
    if vocab_size is not None and hasattr(self._hparams, "vocab_divisor"):
      vocab_size += (-vocab_size) % self._hparams.vocab_divisor
    name = self._hparams.name.get(
        feature_name,
        modalities.get_name(modality))(self._hparams, vocab_size)
    with tf.variable_scope(name) as tm_vs:
      self._add_variable_scope(tm_vs.name, tm_vs)
      log_info("Transforming body output with %s.top", name)
      top = self._hparams.top.get(feature_name, modalities.get_top(modality))
      top_is_pointwise = getattr(top, "pointwise", False)
      last_only = (top_is_pointwise and
                   self.hparams.mode == tf.estimator.ModeKeys.PREDICT and
                   not self.hparams.force_full_predict)
      if not last_only:
        logits = top(body_output, features.get("targets"),
                     self._hparams, vocab_size)
      else:
        # Take body outputs for the last position only, and targets too.
        if "decode_loop_step" not in features:
          last_position_body_output = tf.expand_dims(
              body_output[:, -1, :, :], axis=[1])
          last_position_targets = tf.expand_dims(
              features["targets"][:, -1, :, :], axis=[1])
        else:
          body_output_shape = body_output.shape.as_list()
          last_position_body_output = tf.slice(
              body_output, [0, features["decode_loop_step"][0], 0, 0], [
                  body_output_shape[0], 1, body_output_shape[2],
                  body_output_shape[3]
              ])
          target_shape = features["targets"].shape.as_list()
          last_position_targets = tf.slice(
              features["targets"], [0, features["decode_loop_step"][0], 0, 0],
              [target_shape[0], 1, target_shape[2], target_shape[3]])
        logits = top(last_position_body_output, last_position_targets,
                     self._hparams, vocab_size)
    return logits

  def top(self, body_output, features):
    """Computes logits given body output and features.

    Args:
      body_output: dict of str to Tensor, comprising one key-value pair for each
        target. Each value denotes the target's pre-logit activations.
        Alternatively, it may be a single Tensor denoting the pre-logits for
        that target.
      features: dict of str to Tensor. Typically it is the preprocessed data
        batch after Problem's preprocess_example().

    Returns:
      logits: dict of str to Tensor, denoting each logits for each target; or
        a single Tensor denoting the logits for that target.
        When targets are generated at training time:
          logits == {
            "self_generated_targets": <generated targets tensor>
            "logits": <original logits Tensor or dict>
          }
    """
    if isinstance(body_output, dict):
      logits = {}
      for k, v in six.iteritems(body_output):
        # TODO(aidangomez): share variables here?
        with tf.variable_scope(k) as top_vs:
          self._add_variable_scope("top_%s" % k, top_vs)
          logits[k] = self._top_single(v, k, features)
      return logits
    else:
      return self._top_single(body_output, "targets", features)

  def _loss_single(self, logits, feature_name, feature, weights=None):
    # The current bfloat16 version still uses float32 for most parts of backward
    # propagation to keep model quality, so cast back before computing the loss
    # value.
    if not self._problem_hparams:
      log_warn(_no_problem_err("loss"))
      return (tf.constant(0., dtype=tf.float32),
              tf.constant(1., dtype=tf.float32))

    # Calculate loss contribution.
    modality = self._problem_hparams.modality[feature_name]
    vocab_size = self._problem_hparams.vocab_size[feature_name]
    if vocab_size is not None and hasattr(self._hparams, "vocab_divisor"):
      vocab_size += (-vocab_size) % self._hparams.vocab_divisor
    loss = self._hparams.loss.get(feature_name, modalities.get_loss(modality))
    targets_weights_fn = self._hparams.weights_fn.get(
        "targets", modalities.get_weights_fn(modality))
    if weights is None:
      loss_num, loss_den = loss(logits, feature, self._hparams, vocab_size,
                                weights_fn=targets_weights_fn)
    else:

      def weights_fn(labels):
        """Per-token weights for loss."""
        # Use target_weights_fn() given by modality as well as explicitly given
        # weights.
        modality_weights = targets_weights_fn(labels)

        # Broadcast 'weights' along minor dimensions (TF's default is major).
        explicit_weights = weights
        if len(explicit_weights.shape) < len(modality_weights.shape):
          explicit_weights = common_layers.expand_squeeze_to_nd(
              weights, modality_weights.shape.ndims)

        return explicit_weights * modality_weights

      # Ensure that target.modality_loss() supports "weights_fn" keyword
      # argument. If it doesn't and "weights" is specified, raise an exception.
      argument_names = inspect.getargspec(loss).args
      if "weights_fn" not in argument_names:
        raise ValueError(
            "Explicit 'weights' given but default loss for modality doesn't "
            "support 'weights_fn' keyword argument: %s.loss(%s)." %
            (modality, ", ".join(argument_names)))

      loss_num, loss_den = loss(
          logits, feature, self._hparams, vocab_size, weights_fn=weights_fn)

    loss_num *= self._problem_hparams.loss_multiplier

    if hasattr(self.hparams, "problem") and hasattr(
        self.hparams.problem, "task_list"):
      if weights is not None:
        raise NotImplementedError("weights not yet implemented in "
                                  "multitask setting.")
      loss_num, loss_den, summaries = multi_problem.aggregate_task_losses(
          self.hparams,
          self._problem_hparams,
          logits,
          feature_name,
          feature
      )

      for key, val in summaries:
        tf.summary.scalar(key, val)

    return loss_num, loss_den

  def loss(self, logits, features):
    if isinstance(logits, dict):
      losses = {}
      for k, v in six.iteritems(logits):
        losses[k] = self._loss_single(
            v,
            k,
            features[k],
            weights=features.get(k + "_mask"))

        n, d = losses[k]
        if common_layers.should_generate_summaries():
          tf.summary.scalar(k + "_loss", n / d)
          tf.summary.scalar(k + "_loss_num", n)
          tf.summary.scalar(k + "_loss_den", d)
          if getattr(self.hparams, "visualize_logits_histogram", False):
            hist = tf.summary.histogram
            hist(k + "_predict", tf.argmax(tf.squeeze(v), axis=-1))
            hist(k + "_targets", features[k])

      return tf.add_n([n / d for n, d in losses.values()])
    else:
      return self._loss_single(
          logits,
          "targets",
          features["targets"],
          weights=features.get("targets_mask"))

  def optimize(self, loss, num_async_replicas=1, use_tpu=False):
    """Return a training op minimizing loss."""
    lr = learning_rate.learning_rate_schedule(self.hparams)
    if num_async_replicas > 1:
      log_info("Dividing learning rate by num_async_replicas: %d",
               num_async_replicas)
    lr /= math.sqrt(float(num_async_replicas))
    train_op = optimize.optimize(loss, lr, self.hparams, use_tpu=use_tpu)
    return train_op

  def set_mode(self, mode):
    """Set hparams with the given mode."""
    log_info("Setting T2TModel mode to '%s'", mode)
    hparams = hparams_lib.copy_hparams(self._original_hparams)
    hparams.add_hparam("mode", mode)
    # When not in training mode, set all forms of dropout to zero.
    if mode != tf.estimator.ModeKeys.TRAIN:
      for key in hparams.values():
        if key.endswith("dropout") or key == "label_smoothing":
          log_info("Setting hparams.%s to 0.0", key)
          setattr(hparams, key, 0.0)
    self._hparams = hparams

  def prepare_features_for_infer(self, features):
    """Called before inference to allow adding infer-specific features."""
    pass

  def eval_autoregressive(self, features=None, decode_length=50):
    """Autoregressive eval.

    Quadratic time in decode_length.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.

    Returns:
      logits: `Tensor`
      losses: a dictionary: {loss-name (string): floating point `Scalar`}.
          Contains a single key "training".
    """
    results = self._slow_greedy_infer(features, decode_length=decode_length)
    return results["logits"], results["losses"]

  def _fill_problem_hparams_features(self, features):
    if features is not None:
      for k, v in sorted(
          six.iteritems(problem_hparams_to_features(self._problem_hparams))):
        if k not in features:
          features[k] = tf.constant(v, name=k)

  def infer(self,
            features=None,
            decode_length=50,
            beam_size=1,
            top_beams=1,
            alpha=0.0,
            use_tpu=False):
    """A inference method.

    Quadratic time in decode_length.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
      use_tpu: bool, whether to build the inference graph for TPU.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }
      if slow greedy decoding is used then the dict will also contain {
          "logits": `Tensor` of shape [batch_size, time, 1, 1, vocab_size].
          "losses": a dictionary: {loss-name (string): floating point `Scalar`
      }
    """
    set_custom_getter_compose(self._custom_getter)
    with self._eager_var_store.as_default():
      # TODO(rsepassi): Make decoding work with real-valued model outputs
      # (i.e. if the target modality is RealModality).
      self.prepare_features_for_infer(features)
      if not self.has_input and beam_size > 1:
        log_warn("Beam searching for a model with no inputs.")
      if not self.has_input and self.hparams.sampling_method != "random":
        log_warn("Non-random sampling for a model with no inputs.")
      self._fill_problem_hparams_features(features)

      if self._problem_hparams:
        target_modality = self._problem_hparams.modality["targets"]
        if target_modality == modalities.ModalityType.CLASS_LABEL:
          beam_size = 1  # No use to run beam-search for a single class.
      if beam_size == 1:
        log_info("Greedy Decoding")
        results = self._greedy_infer(features, decode_length, use_tpu)
      else:
        log_info("Beam Decoding with beam size %d" % beam_size)
        results = self._beam_decode(features, decode_length, beam_size,
                                    top_beams, alpha, use_tpu)

      return results

  def _beam_decode(self,
                   features,
                   decode_length,
                   beam_size,
                   top_beams,
                   alpha,
                   use_tpu=False):
    """Beam search decoding.

    Models should ideally implement a more efficient version of this function.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
      use_tpu: A bool, whether to do beam decode on TPU.

    Returns:
       samples: an integer `Tensor`. Top samples from the beam search
    """
    return self._beam_decode_slow(features, decode_length, beam_size, top_beams,
                                  alpha, use_tpu)

  def _beam_decode_slow(self, features, decode_length, beam_size, top_beams,
                        alpha, use_tpu=False):
    """Slow version of Beam search decoding.

    Quadratic time in decode_length.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
      use_tpu: A bool, whether to do slow beam decode on TPU.

    Returns:
      samples: an integer `Tensor`. Top samples from the beam search.

    Raises:
      NotImplementedError: If use_tpu is set to true.
    """
    batch_size = common_layers.shape_list(features["inputs"])[0]

    def symbols_to_logits_fn(ids, i=None):
      """Go from ids to logits."""
      ids = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      ids = tf.pad(ids[:, 1:], [[0, 0], [0, 1], [0, 0], [0, 0]])
      if "partial_targets" in features:
        pt = features["partial_targets"]
        pt_length = common_layers.shape_list(pt)[1]
        pt = tf.tile(pt, [1, beam_size])
        pt = tf.reshape(pt, [batch_size * beam_size, pt_length, 1, 1])
        ids = tf.concat([pt, ids], axis=1)

      features["targets"] = ids
      if i is not None:
        features["decode_loop_step"] = i
      self._coverage = None
      logits, _ = self(features)  # pylint: disable=not-callable
      # now self._coverage is a coverage tensor for the first datashard.
      # it has shape [batch_size] and contains floats between 0 and
      # source_length.
      if self._problem_hparams:
        modality = self._problem_hparams.modality["targets"]
        top = self._hparams.top.get("targets", modalities.get_top(modality))
        if getattr(top, "pointwise", False):
          return tf.squeeze(logits, axis=[1, 2, 3])
      # -1 due to the pad above.
      current_output_position = common_layers.shape_list(ids)[1] - 1
      logits = logits[:, current_output_position, :, :]
      return tf.squeeze(logits, axis=[1, 2])

    def _clone_examples_for_beam(old_feature, n):
      """Clone each example n times."""
      old_shape = common_layers.shape_list(old_feature)
      assert len(old_shape) >= 1

      # Expand the inputs in to the beam size.
      feature = tf.expand_dims(old_feature, 1)
      feature = tf.tile(feature, [1, n] + [1] * (len(old_shape) - 1))
      new_shape = common_layers.shape_list(feature)
      feature = tf.reshape(feature,
                           [new_shape[0] * new_shape[1]] + new_shape[2:])
      return feature

    initial_ids = tf.zeros([batch_size], dtype=tf.int32)

    # Clone select features multiple times to account for beam size.
    old_features = {}
    for feature_name in ["inputs", "knowledge"]:
      if feature_name not in features:
        continue
      old_features[feature_name] = features[feature_name]
      features[feature_name] = _clone_examples_for_beam(
          features[feature_name], beam_size)

    vocab_size = self._problem_hparams.vocab_size["targets"]
    if vocab_size is not None and hasattr(self._hparams, "vocab_divisor"):
      vocab_size += (-vocab_size) % self._hparams.vocab_divisor

    # Setting decode length to input length + decode_length
    if "partial_targets" not in features:
      inputs = features["inputs"]
      decode_length = (common_layers.shape_list(inputs)[1] +
                       features.get("decode_length", decode_length))
    ids, scores, _ = beam_search.beam_search(
        symbols_to_logits_fn,
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        alpha,
        stop_early=(top_beams == 1),
        use_tpu=use_tpu)

    # Set features back to the unexpanded form to not to confuse the
    # Estimator!
    features.update(old_features)

    # Return `top_beams` decodings (also remove initial id from the beam search)
    # TODO(lukaszkaiser): make it work multi-problem.
    if top_beams == 1:
      samples = ids[:, 0, 1:]
    else:
      samples = ids[:, :top_beams, 1:]

    return {"outputs": samples, "scores": scores}

  def _greedy_infer(self, features, decode_length, use_tpu=False):
    """A greedy inference method.

    Models should ideally implement a more efficient version of this function.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      use_tpu: A bool, whether to build the inference graph for TPU.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": None
          "logits": `Tensor` of shape [batch_size, time, 1, 1, vocab_size].
          "losses": a dictionary: {loss-name (string): floating point `Scalar`}
      }
    """
    if use_tpu:
      return self._slow_greedy_infer_tpu(features, decode_length)
    return self._slow_greedy_infer(features, decode_length)

  def _slow_greedy_infer_tpu(self, features, decode_length):
    """A slow greedy inference method on TPU.

    Quadratic time in decode_length.

    Args:
      features: An map of string to `Tensor`.
      decode_length: An integer, how many additional timesteps to decode.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": None
          "logits": `Tensor` of shape [batch_size, time, 1, 1, vocab_size].
          "losses": a dictionary: {loss-name (string): floating point `Scalar`}
      }
    """
    if not features:
      features = {}
    inputs_old = None
    if "inputs" in features and len(features["inputs"].shape) < 4:
      inputs_old = features["inputs"]
      features["inputs"] = tf.expand_dims(features["inputs"], 2)
    if not self.has_input:
      # Prepare partial targets.
      # In either features["inputs"] or features["targets"].
      # We force the outputs to begin with these sequences.
      partial_targets = features.get("inputs")
      if partial_targets is None:
        partial_targets = features["targets"]
      features["partial_targets"] = tf.to_int64(partial_targets)
    # Save the targets in a var and reassign it after the tf.while loop to avoid
    # having targets being in a 'while' frame. This ensures targets when used
    # in metric functions stays in the same frame as other vars.
    targets_old = features.get("targets", None)

    target_modality = self._problem_hparams.modality["targets"]

    def infer_step(i, recent_output, recent_logits, unused_loss):
      """Inference step."""
      if not tf.executing_eagerly():
        recent_output.set_shape([None, None, None, 1])
      padded = tf.pad(recent_output, [[0, 0], [0, 1], [0, 0], [0, 0]])
      features["targets"] = padded
      # This is inefficient in that it generates samples at all timesteps,
      # not just the last one, except if target_modality is pointwise.
      features["decode_loop_step"] = i
      samples, logits, losses = self.sample(features)
      # Concatenate the already-generated recent_output with last timestep
      # of the newly-generated samples.z
      top = self._hparams.top.get("targets",
                                  modalities.get_top(target_modality))
      if getattr(top, "pointwise", False):
        cur_sample = samples[:, -1, :, :]
      else:
        cur_sample = samples[:, i, :, :]
      samples = tf.transpose(recent_output, perm=[1, 0, 2, 3])
      samples = inplace_ops.alias_inplace_update(samples, i,
                                                 tf.to_int64(cur_sample))
      samples = tf.transpose(samples, perm=[1, 0, 2, 3])
      if not tf.executing_eagerly():
        samples.set_shape([None, None, None, 1])

      # Assuming we have one shard for logits.
      recent_logits = tf.transpose(recent_logits, perm=[1, 0, 2, 3, 4])
      recent_logits = inplace_ops.alias_inplace_update(
          recent_logits, i, tf.squeeze(logits[:, -1:], axis=1))
      logits = tf.transpose(recent_logits, perm=[1, 0, 2, 3, 4])
      loss = sum([l for l in losses.values() if l is not None])
      return i + 1, samples, logits, loss

    # Create an initial output tensor. This will be passed
    # to the infer_step, which adds one timestep at every iteration.
    if "partial_targets" in features:
      initial_output = tf.to_int64(features["partial_targets"])
      while len(initial_output.get_shape().as_list()) < 4:
        initial_output = tf.expand_dims(initial_output, 2)
      batch_size = common_layers.shape_list(initial_output)[0]
    else:
      batch_size = common_layers.shape_list(features["inputs"])[0]
      initial_output = tf.zeros((batch_size, 0, 1, 1), dtype=tf.int64)
    # Hack: foldl complains when the output shape is less specified than the
    # input shape, so we confuse it about the input shape.
    initial_output = tf.slice(initial_output, [0, 0, 0, 0],
                              common_layers.shape_list(initial_output))
    target_modality = self._problem_hparams.modality["targets"]
    if target_modality == modalities.ModalityType.CLASS_LABEL:
      decode_length = 1
    else:
      if "partial_targets" in features:
        prefix_length = common_layers.shape_list(features["partial_targets"])[1]
      else:
        prefix_length = common_layers.shape_list(features["inputs"])[1]
      decode_length = prefix_length + decode_length

    # Initial values of result, logits and loss.
    result = tf.concat(
        [initial_output,
         tf.zeros([batch_size, decode_length, 1, 1], tf.int64)],
        axis=1)
    # tensor padded to [batch_size, decode_length, 1, 1, vocab_size]
    vocab_size = self._problem_hparams.vocab_size["targets"]
    if vocab_size is not None and hasattr(self._hparams, "vocab_divisor"):
      vocab_size += (-vocab_size) % self._hparams.vocab_divisor
    logits = tf.zeros((batch_size, decode_length, 1, 1, vocab_size))
    if not tf.executing_eagerly():
      logits.set_shape([None, None, None, None, None])
    loss = 0.0

    def while_exit_cond(i, result, logits, loss):  # pylint: disable=unused-argument
      """Exit the loop either if reach decode_length or EOS."""
      not_overflow = i < decode_length

      if self._problem_hparams.stop_at_eos:

        def fn_not_eos():
          # Check if the last predicted element is a EOS
          return tf.reduce_any(
              tf.not_equal(
                  tf.squeeze(result[:, -1, :, :]), text_encoder.EOS_ID))

        not_eos = tf.cond(
            # We only check for early stopping if there is at least 1 element (
            # otherwise not_eos will crash).
            tf.not_equal(i, 0),
            fn_not_eos,
            lambda: True,
        )

        return tf.cond(
            tf.equal(batch_size, 1),
            # If batch_size == 1, we check EOS for early stopping.
            lambda: tf.logical_and(not_overflow, not_eos),
            # Else, just wait for max length
            lambda: not_overflow)
      return not_overflow

    _, result, logits, loss = tf.while_loop(
        while_exit_cond,
        infer_step, [tf.constant(0), result, logits, loss],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([batch_size, decode_length, 1, 1]),
            tf.TensorShape([batch_size, decode_length, 1, 1, vocab_size]),
            tf.TensorShape([]),
        ],
        back_prop=False,
        parallel_iterations=1)
    if inputs_old is not None:  # Restore to not confuse Estimator.
      features["inputs"] = inputs_old
    # Reassign targets back to the previous value.
    if targets_old is not None:
      features["targets"] = targets_old
    losses = {"training": loss}
    if "partial_targets" in features:
      partial_target_length = common_layers.shape_list(
          features["partial_targets"])[1]
      result = tf.slice(result, [0, partial_target_length, 0, 0],
                        [-1, -1, -1, -1])
    return {
        "outputs": result,
        "scores": None,
        "logits": logits,
        "losses": losses,
    }

  def _slow_greedy_infer(self, features, decode_length):
    """A slow greedy inference method.

    Quadratic time in decode_length.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": None
          "logits": `Tensor` of shape [batch_size, time, 1, 1, vocab_size].
          "losses": a dictionary: {loss-name (string): floating point `Scalar`}
      }
    """
    if not features:
      features = {}
    inputs_old = None
    if "inputs" in features and len(features["inputs"].shape) < 4:
      inputs_old = features["inputs"]
      features["inputs"] = tf.expand_dims(features["inputs"], 2)
    if not self.has_input:
      # Prepare partial targets.
      # In either features["inputs"] or features["targets"].
      # We force the outputs to begin with these sequences.
      partial_targets = features.get("inputs")
      if partial_targets is None:
        partial_targets = features["targets"]
      features["partial_targets"] = tf.to_int64(partial_targets)
    # Save the targets in a var and reassign it after the tf.while loop to avoid
    # having targets being in a 'while' frame. This ensures targets when used
    # in metric functions stays in the same frame as other vars.
    targets_old = features.get("targets", None)

    target_modality = self._problem_hparams.modality["targets"]

    def infer_step(recent_output, recent_logits, unused_loss):
      """Inference step."""
      if not tf.executing_eagerly():
        if self._target_modality_is_real:
          dim = self._problem_hparams.vocab_size["targets"]
          if dim is not None and hasattr(self._hparams, "vocab_divisor"):
            dim += (-dim) % self._hparams.vocab_divisor
          recent_output.set_shape([None, None, None, dim])
        else:
          recent_output.set_shape([None, None, None, 1])
      padded = tf.pad(recent_output, [[0, 0], [0, 1], [0, 0], [0, 0]])
      features["targets"] = padded
      # This is inefficient in that it generates samples at all timesteps,
      # not just the last one, except if target_modality is pointwise.
      samples, logits, losses = self.sample(features)
      # Concatenate the already-generated recent_output with last timestep
      # of the newly-generated samples.
      top = self._hparams.top.get("targets",
                                  modalities.get_top(target_modality))
      if getattr(top, "pointwise", False):
        cur_sample = samples[:, -1, :, :]
      else:
        cur_sample = samples[:,
                             common_layers.shape_list(recent_output)[1], :, :]
      if self._target_modality_is_real:
        cur_sample = tf.expand_dims(cur_sample, axis=1)
        samples = tf.concat([recent_output, cur_sample], axis=1)
      else:
        cur_sample = tf.to_int64(tf.expand_dims(cur_sample, axis=1))
        samples = tf.concat([recent_output, cur_sample], axis=1)
        if not tf.executing_eagerly():
          samples.set_shape([None, None, None, 1])

      # Assuming we have one shard for logits.
      logits = tf.concat([recent_logits, logits[:, -1:]], 1)
      loss = sum([l for l in losses.values() if l is not None])
      return samples, logits, loss

    # Create an initial output tensor. This will be passed
    # to the infer_step, which adds one timestep at every iteration.
    if "partial_targets" in features:
      initial_output = tf.to_int64(features["partial_targets"])
      while len(initial_output.get_shape().as_list()) < 4:
        initial_output = tf.expand_dims(initial_output, 2)
      batch_size = common_layers.shape_list(initial_output)[0]
    else:
      batch_size = common_layers.shape_list(features["inputs"])[0]
      if self._target_modality_is_real:
        dim = self._problem_hparams.vocab_size["targets"]
        if dim is not None and hasattr(self._hparams, "vocab_divisor"):
          dim += (-dim) % self._hparams.vocab_divisor
        initial_output = tf.zeros((batch_size, 0, 1, dim), dtype=tf.float32)
      else:
        initial_output = tf.zeros((batch_size, 0, 1, 1), dtype=tf.int64)
    # Hack: foldl complains when the output shape is less specified than the
    # input shape, so we confuse it about the input shape.
    initial_output = tf.slice(initial_output, [0, 0, 0, 0],
                              common_layers.shape_list(initial_output))
    target_modality = self._problem_hparams.modality["targets"]
    if target_modality == modalities.ModalityType.CLASS_LABEL:
      decode_length = 1
    else:
      if "partial_targets" in features:
        prefix_length = common_layers.shape_list(features["partial_targets"])[1]
      else:
        prefix_length = common_layers.shape_list(features["inputs"])[1]
      decode_length = prefix_length + decode_length

    # Initial values of result, logits and loss.
    result = initial_output
    vocab_size = self._problem_hparams.vocab_size["targets"]
    if vocab_size is not None and hasattr(self._hparams, "vocab_divisor"):
      vocab_size += (-vocab_size) % self._hparams.vocab_divisor
    if self._target_modality_is_real:
      logits = tf.zeros((batch_size, 0, 1, vocab_size))
      logits_shape_inv = [None, None, None, None]
    else:
      # tensor of shape [batch_size, time, 1, 1, vocab_size]
      logits = tf.zeros((batch_size, 0, 1, 1, vocab_size))
      logits_shape_inv = [None, None, None, None, None]
    if not tf.executing_eagerly():
      logits.set_shape(logits_shape_inv)

    loss = 0.0

    def while_exit_cond(result, logits, loss):  # pylint: disable=unused-argument
      """Exit the loop either if reach decode_length or EOS."""
      length = common_layers.shape_list(result)[1]

      not_overflow = length < decode_length

      if self._problem_hparams.stop_at_eos:

        def fn_not_eos():
          return tf.not_equal(  # Check if the last predicted element is a EOS
              tf.squeeze(result[:, -1, :, :]), text_encoder.EOS_ID)

        not_eos = tf.cond(
            # We only check for early stopping if there is at least 1 element (
            # otherwise not_eos will crash).
            tf.not_equal(length, 0),
            fn_not_eos,
            lambda: True,
        )

        return tf.cond(
            tf.equal(batch_size, 1),
            # If batch_size == 1, we check EOS for early stopping.
            lambda: tf.logical_and(not_overflow, not_eos),
            # Else, just wait for max length
            lambda: not_overflow)
      return not_overflow

    result, logits, loss = tf.while_loop(
        while_exit_cond,
        infer_step, [result, logits, loss],
        shape_invariants=[
            tf.TensorShape([None, None, None, None]),
            tf.TensorShape(logits_shape_inv),
            tf.TensorShape([]),
        ],
        back_prop=False,
        parallel_iterations=1)
    if inputs_old is not None:  # Restore to not confuse Estimator.
      features["inputs"] = inputs_old
    # Reassign targets back to the previous value.
    if targets_old is not None:
      features["targets"] = targets_old
    losses = {"training": loss}
    if "partial_targets" in features:
      partial_target_length = common_layers.shape_list(
          features["partial_targets"])[1]
      result = tf.slice(result, [0, partial_target_length, 0, 0],
                        [-1, -1, -1, -1])
    return {
        "outputs": result,
        "scores": None,
        "logits": logits,
        "losses": losses,
    }

  def sample(self, features):
    """Run the model and extract samples.

    Args:
      features: an map of string to `Tensor`.

    Returns:
       samples: an integer `Tensor`.
       logits: a list of `Tensor`s, one per datashard.
       losses: a dictionary: {loss-name (string): floating point `Scalar`}.
    """
    logits, losses = self(features)  # pylint: disable=not-callable
    if self._target_modality_is_real:
      return logits, logits, losses  # Raw numbers returned from real modality.
    if self.hparams.sampling_method == "argmax":
      samples = tf.argmax(logits, axis=-1)
    else:
      assert self.hparams.sampling_method == "random"

      def multinomial_squeeze(logits, temperature=1.0):
        logits_shape = common_layers.shape_list(logits)
        reshaped_logits = (
            tf.reshape(logits, [-1, logits_shape[-1]]) / temperature)
        choices = tf.multinomial(reshaped_logits, 1)
        choices = tf.reshape(choices, logits_shape[:-1])
        return choices

      samples = multinomial_squeeze(logits, self.hparams.sampling_temp)

    return samples, logits, losses

  def _shard_features(self, features):  # pylint: disable=missing-docstring
    sharded_features = {}
    for k, v in sorted(six.iteritems(features)):
      v = tf.convert_to_tensor(v)
      v_shape = common_layers.shape_list(v)
      if not v_shape:
        v = tf.expand_dims(v, axis=-1)
        v_shape = [1]
      if v_shape == [1]:
        v = tf.tile(v, tf.to_int32([self._num_datashards]))
      sharded_features[k] = self._data_parallelism(
          tf.identity, tf.split(v, self._num_datashards, 0))
    return sharded_features

  def _to_features_per_datashard(self, features):
    datashard_features = []
    assert len(features[list(features.keys())[0]]) == self._num_datashards
    for d in range(self._num_datashards):
      f = {k: v[d] for k, v in six.iteritems(features)}
      datashard_features.append(f)
    return datashard_features

  def _to_single_features_dict(self, datashard_features):
    assert len(datashard_features) == self._num_datashards
    features = collections.defaultdict(list)
    for feats in datashard_features:
      for k, v in six.iteritems(feats):
        features[k].append(v)
    return features

  @staticmethod
  def get_train_hooks(model_name, hook_context):
    model_cls = registry.model(model_name)
    return model_cls.train_hooks(hook_context)

  @staticmethod
  def get_eval_hooks(model_name, hook_context):
    model_cls = registry.model(model_name)
    return model_cls.eval_hooks(hook_context)

  @staticmethod
  def make_estimator_model_fn(model_name,
                              hparams,
                              decode_hparams=None,
                              use_tpu=False):
    model_cls = registry.model(model_name)

    def wrapping_model_fn(features, labels, mode, params=None, config=None):
      return model_cls.estimator_model_fn(
          hparams,
          features,
          labels,
          mode,
          config=config,
          params=params,
          decode_hparams=decode_hparams,
          use_tpu=use_tpu)

    return wrapping_model_fn

  @classmethod
  def estimator_model_fn(cls,
                         hparams,
                         features,
                         labels,
                         mode,
                         config=None,
                         params=None,
                         decode_hparams=None,
                         use_tpu=False):
    """Model fn for Estimator.

    Args:
      hparams: HParams, model hyperparameters
      features: dict<str name, Tensor feature>
      labels: Tensor
      mode: tf.estimator.ModeKeys
      config: RunConfig, possibly with data_parallelism attribute
      params: dict, may include batch_size, use_tpu
      decode_hparams: HParams, used when mode == PREDICT.
      use_tpu: A bool, whether to build the inference graph for TPU.

    Returns:
      TPUEstimatorSpec if use tpu else EstimatorSpec
    """
    if mode == tf.estimator.ModeKeys.TRAIN:
      create_dummy_vars()
    hparams = hparams_lib.copy_hparams(hparams)

    # Instantiate model
    data_parallelism = None
    if not use_tpu and config:
      data_parallelism = config.data_parallelism
    reuse = tf.get_variable_scope().reuse
    model = cls(
        hparams,
        mode,
        data_parallelism=data_parallelism,
        decode_hparams=decode_hparams,
        _reuse=reuse)

    # PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
      if use_tpu:
        inputs = features.get("inputs")
        if inputs is None:
          inputs = features["targets"]
        shape = inputs.get_shape().as_list()
        if shape[0] is None:
          shape[0] = decode_hparams.batch_size or hparams.batch_size
        if shape[1] is None:
          shape[1] = hparams.max_input_seq_length or hparams.max_length
        inputs.set_shape(shape)
      return model.estimator_spec_predict(features, use_tpu=use_tpu)

    # TRAIN and EVAL modes
    if hparams.eval_run_autoregressive and mode == tf.estimator.ModeKeys.EVAL:
      logits, losses_dict = model.eval_autoregressive(features)
    else:
      logits, losses_dict = model(features)  # pylint: disable=not-callable

    # Support model-generated labels by overriding features["targets"] with
    # logits["self_generated_targets"].
    if isinstance(logits, dict) and "self_generated_targets" in logits:
      # Overwrite 'features["targets"]' and 'labels'
      # by logits["self_generated_targets"].
      tf.logging.info("Replacing targets with model-provided targets.")
      features["targets"] = labels = logits.pop("self_generated_targets")
      assert list(logits.keys()) == ["logits"], (
          # See "Returns" in the "top" method docstring for the expected
          # "logits" format when targets are generated at training time.
          "Expect only key 'logits' when there is 'self_generated_targets'. "
          "Found {}".format(logits.keys())
      )
      # Recover the original logits tensor from the logits dict.
      logits = logits["logits"]  # Can be a tf.Tensor or a dict.

    # Set known shapes
    if common_layers.is_xla_compiled():
      if isinstance(logits, dict):
        for k, v in sorted(six.iteritems(logits)):
          if "scalar/" in k:
            continue

          shape = v.get_shape().as_list()
          if shape[0] is None:
            shape[0] = params["batch_size"]
          if shape[1] is None:
            shape[1] = hparams.max_length
          v.set_shape(shape)
      else:
        shape = logits.get_shape().as_list()
        if shape[0] is None:
          shape[0] = params["batch_size"]
        if shape[1] is None:
          shape[1] = hparams.max_length
        logits.set_shape(shape)

    assert "training" in losses_dict

    # Attack mode
    if mode == "attack":
      return logits

    # Summarize losses
    model._summarize_losses(losses_dict)  # pylint: disable=protected-access

    # Accumulate losses
    loss = sum(losses_dict[key] for key in sorted(losses_dict.keys()))

    # EVAL mode
    if mode == tf.estimator.ModeKeys.EVAL:
      return model.estimator_spec_eval(features, logits, labels, loss,
                                       losses_dict)

    # TRAIN mode
    assert mode == tf.estimator.ModeKeys.TRAIN
    num_async_replicas = 1
    if config and not use_tpu:
      num_async_replicas = config.t2t_device_info["num_async_replicas"]
    return model.estimator_spec_train(
        loss, num_async_replicas=num_async_replicas, use_tpu=use_tpu)

  def initialize_from_ckpt(self, ckpt_dir):
    return initialize_from_ckpt(ckpt_dir=ckpt_dir, hparams=self._hparams)

  def create_train_host_call(self):
    return create_host_call(self.hparams.model_dir)

  def create_eval_host_call(self):
    return self.create_train_host_call()

  def estimator_spec_train(self, loss, num_async_replicas=1, use_tpu=False):
    """Constructs `tf.estimator.EstimatorSpec` for TRAIN (training) mode."""
    train_op = self.optimize(loss, num_async_replicas=num_async_replicas,
                             use_tpu=use_tpu)

    if use_tpu:
      if self._hparams.warm_start_from:
        def scaffold_fn():
          self.initialize_from_ckpt(self._hparams.warm_start_from)
          return tf.train.Scaffold()
      else:
        scaffold_fn = None

      # Note: important to call this before remove_summaries()
      if self.hparams.tpu_enable_host_call:
        host_call = self.create_train_host_call()
      else:
        host_call = None

      remove_summaries()

      return tf.contrib.tpu.TPUEstimatorSpec(
          tf.estimator.ModeKeys.TRAIN,
          loss=loss,
          train_op=train_op,
          host_call=host_call,
          scaffold_fn=scaffold_fn)
    else:
      if self._hparams.warm_start_from:
        self.initialize_from_ckpt(self._hparams.warm_start_from)

      # When loading weights from a pre-trained model, you want to be able to
      # load separate weights into the encoder and decoder.
      if self._hparams.warm_start_from_second:
        self.initialize_from_ckpt(self._hparams.warm_start_from_second)

      return tf.estimator.EstimatorSpec(
          tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)

  def estimator_spec_eval(self, features, logits, labels, loss, losses_dict):
    """Constructs `tf.estimator.EstimatorSpec` for EVAL (evaluation) mode."""
    del losses_dict
    hparams = self.hparams

    if not hasattr(hparams, "problem"):
      raise NotImplementedError(_no_problem_err("estimator_spec_eval"))

    problem = hparams.problem

    if common_layers.is_xla_compiled():
      # Note: important to call this before remove_summaries()
      if self.hparams.tpu_enable_host_call:
        host_call = self.create_eval_host_call()
      else:
        host_call = None

      remove_summaries()

      eval_metrics_fn = create_tpu_eval_metrics_fn(problem, hparams)

      batch_size = [feature.shape.as_list()[0] for _, feature
                    in features.items() if feature.shape.ndims][0]

      # Add batch dimension to all features since tpu requires the batch
      # dimension on all tensors.
      for name, feature in features.items():
        if not feature.shape.as_list():
          # All features must have a batch dimension
          feature = tf.tile(tf.expand_dims(feature, 0), [batch_size])
        features[name] = feature

      eval_metrics_fn_args = dict(
          logits=logits,  # possibly a dict
          labels=labels,
          features=features,  # dict
      )

      eval_metrics_fn_flat_args = _flatten_dict(eval_metrics_fn_args)
      return tf.contrib.tpu.TPUEstimatorSpec(
          tf.estimator.ModeKeys.EVAL,
          eval_metrics=(eval_metrics_fn, eval_metrics_fn_flat_args),
          host_call=host_call,
          loss=loss)
    else:
      task_list = [problem]
      if hasattr(problem, "task_list"):
        task_list = problem.task_list

      eval_metrics_fns = metrics.create_evaluation_metrics(task_list, hparams)
      eval_metrics = {}
      for metric_name, metric_fn in six.iteritems(eval_metrics_fns):
        if isinstance(logits, dict):
          # the key is located in the center of metric_name: "metrics-%s/%s/%s"
          k = metric_name.split("/")[1]
          if k in logits:
            eval_metrics[metric_name] = metric_fn(logits[k], features,
                                                  features[k])
          else:
            # We do not make it an error because we sometimes run models that
            # predict only parts of the targets defined by the Problem class.
            # For example, an autoencoder or pure-video model can run on a gym
            # problem even if another model is also predicting other things,
            # like actions or rewards.
            tf.logging.warning("No key %s in logits for evaluation." % k)
        else:
          eval_metrics[metric_name] = metric_fn(logits, features,
                                                features["targets"])
      if isinstance(logits, dict):
        predictions = logits
      else:
        predictions = {"predictions": logits}

      evaluation_hooks = []
      # Create a SummarySaverHook
      eval_dir = os.path.join(
          self.hparams.model_dir,
          self.hparams.get("eval_dir_name", "eval"))
      eval_summary_hook = tf.train.SummarySaverHook(
          save_steps=1,
          output_dir=eval_dir,
          summary_op=tf.summary.merge_all())
      evaluation_hooks.append(eval_summary_hook)

      evaluation_hooks += problem.eval_hooks(features, logits, hparams)

      return tf.estimator.EstimatorSpec(
          tf.estimator.ModeKeys.EVAL,
          predictions=predictions,
          eval_metric_ops=eval_metrics,
          evaluation_hooks=evaluation_hooks,
          loss=loss)

  def estimator_spec_predict(self, features, use_tpu=False):
    """Constructs `tf.estimator.EstimatorSpec` for PREDICT (inference) mode."""
    decode_hparams = self._decode_hparams
    top_beams = decode_hparams.beam_size if decode_hparams.return_beams else 1
    infer_out = self.infer(
        features,
        beam_size=decode_hparams.beam_size,
        top_beams=top_beams,
        alpha=decode_hparams.alpha,
        decode_length=decode_hparams.extra_length,
        use_tpu=use_tpu)
    if isinstance(infer_out, dict):
      outputs = infer_out["outputs"]
      scores = infer_out["scores"]
    else:
      outputs = infer_out
      scores = None

    inputs = features.get("inputs")
    if inputs is None:
      inputs = features["targets"]

    predictions = {
        "outputs": outputs,
        "scores": scores,
        "inputs": inputs,
        "targets": features.get("infer_targets"),
    }

    # Pass through remaining features
    for name, feature in features.items():
      if name not in list(predictions.keys()) + ["infer_targets"]:
        if name == "decode_loop_step":
          continue
        if not feature.shape.as_list():
          # All features must have a batch dimension
          batch_size = common_layers.shape_list(outputs)[0]
          feature = tf.tile(tf.expand_dims(feature, 0), [batch_size])
        predictions[name] = feature

    _del_dict_non_tensors(predictions)

    export_out = {"outputs": predictions["outputs"]}
    if "scores" in predictions:
      export_out["scores"] = predictions["scores"]

    # Necessary to rejoin examples in the correct order with the Cloud ML Engine
    # batch prediction API.
    if "batch_prediction_key" in predictions:
      export_out["batch_prediction_key"] = predictions["batch_prediction_key"]

    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            tf.estimator.export.PredictOutput(export_out)
    }
    if use_tpu:
      # Note: important to call this before remove_summaries()
      if self.hparams.tpu_enable_host_call:
        host_call = self.create_eval_host_call()
      else:
        host_call = None

      remove_summaries()

      return tf.contrib.tpu.TPUEstimatorSpec(
          tf.estimator.ModeKeys.PREDICT,
          predictions=predictions,
          host_call=host_call,
          export_outputs=export_outputs)
    else:
      return tf.estimator.EstimatorSpec(
          tf.estimator.ModeKeys.PREDICT,
          predictions=predictions,
          export_outputs=export_outputs)

  def _normalize_body_output(self, body_out):
    if isinstance(body_out, tuple):
      output, losses = body_out
      if isinstance(losses, (list, tuple)):
        losses = {"extra": tf.add_n([tf.reduce_mean(l) for l in losses])}
      elif isinstance(losses, dict):
        pass
      else:
        losses = {"extra": tf.reduce_mean(losses)}
    else:
      output = body_out
      losses = {"extra": 0.0}

    return output, losses

  def _summarize_losses(self, losses_dict):
    """Adds `tf.summary`s to all terms in the losses dictionary."""
    if common_layers.should_generate_summaries():
      with tf.name_scope("losses"):
        for loss_name, loss_val in sorted(losses_dict.items()):
          tf.summary.scalar(loss_name, loss_val)


def _with_timing(fn, msg, silent=False):

  def fn_with_timing(*args, **kwargs):
    start_time = time.time()
    res = fn(*args, **kwargs)
    if not silent:
      log_info("Doing %s took %.3f sec." % (msg, time.time() - start_time))
    return res

  return fn_with_timing


def create_dummy_vars():
  """Dummy vars for restore to work when not using TPU codepath."""
  var_names = set([v.name for v in tf.global_variables()])
  if "losses_avg/problem_0/total_loss:0" in var_names:
    return
  with tf.variable_scope("losses_avg"):
    with tf.variable_scope("problem_0"):
      for var_name in ["total", "extra", "training"]:
        tf.get_variable(
            "%s_loss" % var_name, initializer=100.0, trainable=False)
  with tf.variable_scope("train_stats"):
    tf.get_variable("problem_0_steps", initializer=0, trainable=False)


# These metrics are implemented with py_funcs and therefore do no work with TPU
TPU_METRIC_BLACKLIST = set([
    metrics.Metrics.APPROX_BLEU,
    metrics.Metrics.ROUGE_2_F,
    metrics.Metrics.ROUGE_L_F,
    metrics.Metrics.IMAGE_SUMMARY,
])


def create_tpu_eval_metrics_fn(problem, model_hparams):
  """Create the metrics_fn that TPUEstimatorSpec expects."""

  metric_fns = []
  eval_metrics = problem.eval_metric_fns(model_hparams)

  tm = _create_target_modality(problem.get_hparams(model_hparams).modality)
  if isinstance(tm, dict):
    for k, v in six.iteritems(tm):
      weights_fn = modalities.get_weights_fn(v)

      def make_metric_fn(metric_fn):
        def wrapped_metric_fn(logits, labels, features, weights_fn=weights_fn):
          kwargs = {}
          args, _, keywords, _ = inspect.getargspec(metric_fn)
          if ("features" in args) or keywords:
            kwargs["features"] = features
          num, den = metric_fn(logits, labels, weights_fn=weights_fn, **kwargs)
          return tf.metrics.mean(num, den)

        return wrapped_metric_fn

      for metric, metric_fn in six.iteritems(eval_metrics):
        if metric in TPU_METRIC_BLACKLIST:
          log_warn("Skipping eval metric %s in TPU_METRIC_BLACKLIST", metric)
          continue
        name = "%s/metrics-%s/%s" % (k, problem.name, metric)
        metric_fns.append((name, make_metric_fn(metric_fn)))
  else:
    weights_fn = modalities.get_weights_fn(tm)

    def make_metric_fn(metric_fn):
      def wrapped_metric_fn(logits, labels, features):
        kwargs = {}
        args, _, keywords, _ = inspect.getargspec(metric_fn)
        if ("features" in args) or keywords:
          kwargs["features"] = features
        num, den = metric_fn(logits, labels, weights_fn=weights_fn, **kwargs)
        return tf.metrics.mean(num, den)

      return wrapped_metric_fn

    for metric, metric_fn in six.iteritems(eval_metrics):
      if metric in TPU_METRIC_BLACKLIST:
        log_warn("Skipping eval metric %s in TPU_METRIC_BLACKLIST", metric)
        continue
      name = "metrics-%s/%s" % (problem.name, metric)
      metric_fns.append((name, make_metric_fn(metric_fn)))

  def all_metrics_fn(**kwargs):
    """Construct metrics dictionary."""

    original_kwargs = _unflatten_dict(kwargs, prefixes=["logits", "features"])
    del kwargs

    logits = original_kwargs["logits"]
    labels = original_kwargs["labels"]
    features = original_kwargs["features"]
    del original_kwargs

    metrics_dict = {}

    for name, fn in metric_fns:
      if isinstance(logits, dict) and isinstance(labels, dict):
        for k, v in six.iteritems(logits):
          metrics_dict["%s/%s" % (k, name)] = fn(v, labels[k], features)
      elif isinstance(logits, dict):
        tf.logging.warning("Logits is a dict, but labels is not; only "
                           "evaluating logits['targets'] against labels.")
        metrics_dict["%s/%s" % ("targets", name)] = fn(logits["targets"],
                                                       labels, features)
      else:
        metrics_dict[name] = fn(logits, labels, features)

    return metrics_dict

  return all_metrics_fn


def remove_summaries():
  """Remove summaries from the default graph."""
  g = tf.get_default_graph()
  key = tf.GraphKeys.SUMMARIES
  log_debug("Remove summaries %s" % str(g.get_collection(key)))
  del g.get_collection_ref(key)[:]
  assert not g.get_collection(key)


def create_host_call(model_dir):
  """Construct a host_call writing scalar summaries.

  Args:
    model_dir: String containing path to train

  Returns:
    (fn, args) Pair to be called by TPUEstimator as the host_call.
  """
  graph = tf.get_default_graph()
  summaries = graph.get_collection(tf.GraphKeys.SUMMARIES)
  gs_t = tf.reshape(tf.to_int32(tf.train.get_global_step()), [1])
  summary_kwargs = collections.OrderedDict()
  for t in summaries:
    # TODO(aidangomez): enable ImageSummary support when we have a faster method
    # see @shibow's comment in cl/202344570
    if t.op.type not in ["ScalarSummary"]:
      tf.logging.warn("Ignoring unsupported tf.Summary type %s" % t.op.type)
      continue

    name = t.op.name
    tensor = t.op.inputs[1]
    if t.op.type == "ScalarSummary":
      assert tensor.shape.is_compatible_with([])
      if tensor.dtype == tf.int64:
        tensor = tf.to_int32(tensor)
      summary_kwargs["ScalarSummary" + name] = tf.reshape(tensor, [1])
    elif t.op.type == "ImageSummary":
      # TODO(aidangomez): as we move to support more types, update
      # common_layers.tpu_safe_image_summary
      if tensor.dtype != tf.float32:
        tf.logging.warn(
            "Currently T2T on TPU only supports ImageSummary of "
            "tf.float32-type Tensors. Skipping Tensor "
            "%s with dtype %s..." % (tensor.name, tensor.dtype))
        continue
      # tensor = tf.to_float(tensor)
      summary_kwargs["ImageSummary" + name] = tensor
  # When no supported summaries are found, don't create host_call. Otherwise,
  # TPU outfeed queue would enqueue global_step while host_call doesn't dequeue
  # it, eventually causing hang.
  if not summary_kwargs:
    return None
  summary_kwargs["global_step"] = gs_t
  log_info("summary_kwargs %s" % str(summary_kwargs))

  def host_call_fn(**kwargs):
    """Training host call. Creates summaries for training metrics.

    Args:
      **kwargs: Dict of {str: Tensor} , with `Tensor` of shape `[batch]`. Must
        contain key "global_step" with value of current global_step Tensor.

    Returns:
      List of summary ops to run on the CPU host.
    """
    gs = tf.to_int64(kwargs.pop("global_step")[0])
    with tf.contrib.summary.create_file_writer(model_dir).as_default():
      with tf.contrib.summary.always_record_summaries():
        # We need to use tf.contrib.summary in order to feed the `step`.
        for name, value in sorted(six.iteritems(kwargs)):
          if name.startswith("ScalarSummary"):
            name = name[len("ScalarSummary"):]
            tf.contrib.summary.scalar(
                name, tf.reduce_mean(tf.to_float(value)), step=gs)
          elif name.startswith("ImageSummary"):
            name = name[len("ImageSummary"):]
            tf.contrib.summary.image(name, value, step=gs)

        return tf.contrib.summary.all_summary_ops()

  return (host_call_fn, summary_kwargs)


def _del_dict_non_tensors(d):
  for k in list(d.keys()):
    if not isinstance(d[k], tf.Tensor):
      del d[k]


class DummyVariableStore(object):

  @contextlib.contextmanager
  def as_default(self):
    yield


def create_eager_var_store():
  if tf.executing_eagerly():
    return variable_scope.EagerVariableStore()
  else:
    return DummyVariableStore()


def scheduled_sampling(hparams, problem_hparams, dp, sharded_logits, losses,
                       sharded_features, transformed_features, model):
  """Scheduled sampling."""
  modality = problem_hparams.modality["targets"]
  vocab_size = problem_hparams.vocab_size["targets"]
  if vocab_size is not None and hasattr(hparams, "vocab_divisor"):
    vocab_size += (-vocab_size) % hparams.vocab_divisor

  def sample(x):
    """Multinomial sampling from a n-dimensional tensor."""
    samples = tf.multinomial(tf.reshape(x, [-1, vocab_size]), 1)
    reshaped_samples = tf.reshape(samples, common_layers.shape_list(x)[:-1])
    return tf.to_int32(reshaped_samples)

  def mix_gold_sampled(gold_targets, sampled_targets):
    return tf.where(
        tf.less(
            tf.random_uniform(common_layers.shape_list(sampled_targets)),
            hparams.scheduled_sampling_gold_mixin_prob), gold_targets,
        sampled_targets)

  def sampled_results():
    """Generate scheduled sampling results."""
    sampled_targets = dp(sample, sharded_logits)
    new_targets = dp(mix_gold_sampled, sharded_features["targets"],
                     sampled_targets)
    new_features = transformed_features
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      modality_name = hparams.name.get(
          "targets",
          modalities.get_name(modality))(hparams, vocab_size)
      with tf.variable_scope(modality_name):
        bottom = hparams.bottom.get(
            "targets", modalities.get_targets_bottom(modality))
        new_features["targets"] = dp(bottom, new_targets, hparams, vocab_size)
      with tf.variable_scope("body"):
        body_outputs, losses = model.model_fn_sharded(new_features)
        if not isinstance(losses, dict):  # If it's a single extra loss.
          losses = {"extra": losses}
      with tf.variable_scope(modality_name):
        top = hparams.top.get("targets", modalities.get_top(modality))
        new_sharded_logits = dp(top,
                                body_outputs,
                                sharded_features["targets"],
                                hparams,
                                vocab_size)
        if "training" not in losses:
          loss = hparams.loss.get("targets", modalities.get_loss(modality))
          weights_fn = hparams.weights_fn.get(
              "targets", modalities.get_weights_fn(modality))
          sharded_loss_num, sharded_loss_den = dp(loss,
                                                  sharded_logits,
                                                  sharded_features["targets"],
                                                  hparams,
                                                  vocab_size,
                                                  weights_fn=weights_fn)
          training_loss = (tf.add_n(sharded_loss_num) /
                           tf.maximum(1.0, tf.add_n(sharded_loss_den)))
          training_loss *= problem_hparams.loss_multiplier
          losses["training"] = training_loss
    return new_sharded_logits, losses

  # Run the above conditionally.
  prob = hparams.scheduled_sampling_prob
  prob *= common_layers.inverse_exp_decay(
      hparams.scheduled_sampling_warmup_steps, min_value=0.001)
  sharded_logits, losses = tf.cond(
      tf.less(tf.random_uniform([]), prob), sampled_results,
      lambda: (sharded_logits, losses))
  return sharded_logits, losses


def average_sharded_losses(sharded_losses):
  """Average losses across datashards.

  Args:
    sharded_losses: list<dict<str loss_name, Tensor loss>>. The loss
      can be a single Tensor or a 2-tuple (numerator and denominator).

  Returns:
    losses: dict<str loss_name, Tensor avg_loss>
  """
  losses = {}
  for loss_name in sorted(sharded_losses[0]):
    all_shards = [shard_losses[loss_name] for shard_losses in sharded_losses]
    if isinstance(all_shards[0], tuple):
      sharded_num, sharded_den = zip(*all_shards)
      mean_loss = (
          tf.add_n(sharded_num) / tf.maximum(
              tf.cast(1.0, sharded_den[0].dtype), tf.add_n(sharded_den)))
    else:
      mean_loss = tf.reduce_mean(all_shards)

    losses[loss_name] = mean_loss
  return losses


def summarize_features(features, num_shards=1):
  """Generate summaries for features."""
  if not common_layers.should_generate_summaries():
    return

  with tf.name_scope("input_stats"):
    for (k, v) in sorted(six.iteritems(features)):
      if (isinstance(v, tf.Tensor) and (v.get_shape().ndims > 1) and
          (v.dtype != tf.string)):
        tf.summary.scalar("%s_batch" % k, tf.shape(v)[0] // num_shards)
        tf.summary.scalar("%s_length" % k, tf.shape(v)[1])
        nonpadding = tf.to_float(tf.not_equal(v, 0))
        nonpadding_tokens = tf.reduce_sum(nonpadding)
        tf.summary.scalar("%s_nonpadding_tokens" % k, nonpadding_tokens)
        tf.summary.scalar("%s_nonpadding_fraction" % k,
                          tf.reduce_mean(nonpadding))


_already_logged = set()


def _eager_log(level, *args):
  if tf.executing_eagerly() and args in _already_logged:
    return
  _already_logged.add(args)
  getattr(tf.logging, level)(*args)


def log_debug(*args):
  _eager_log("debug", *args)


def log_info(*args):
  _eager_log("info", *args)


def log_warn(*args):
  _eager_log("warn", *args)


def _compose_custom_getters(getter_a, getter_b):
  """Compose two custom getters.

  Example use:
  tf.get_variable_scope().set_custom_getter(
    compose_custom_getters(tf.get_variable_scope().custom_getter, new_getter))

  This composes getters in the same way as creating a new variable scope with
  the new_getter, but it does not actually create a new variable scope.

  Args:
    getter_a: a custom getter - generally from the existing variable scope.
    getter_b: a custom getter

  Returns:
    a custom getter
  """
  if not getter_a:
    return getter_b
  if not getter_b:
    return getter_a

  def getter_fn(getter, *args, **kwargs):
    return getter_b(functools.partial(getter_a, getter), *args, **kwargs)

  return getter_fn


def set_custom_getter_compose(custom_getter):
  """Set a custom getter in the current variable scope.

  Do not overwrite the existing custom getter - rather compose with it.

  Args:
    custom_getter: a custom getter.
  """
  tf.get_variable_scope().set_custom_getter(
      _compose_custom_getters(tf.get_variable_scope().custom_getter,
                              custom_getter))


def _create_target_modality(modality_dict):
  # TODO(trandustin): We require this in order to apply methods utilized
  # differently for modalities which are "targets"
  # (e.g., modality.target_bottom). In the future, remove need for this
  # behavior.
  return {k: v for k, v in six.iteritems(modality_dict) if "target" in k
          and k != "targets_segmentation" and k != "targets_position"}


def initialize_from_ckpt(ckpt_dir, hparams):
  """Initialize variables from given directory."""
  model_dir = hparams.get("model_dir", None)
  already_has_ckpt = (
      model_dir and tf.train.latest_checkpoint(model_dir) is not None)
  if already_has_ckpt:
    return

  tf.logging.info("Checkpoint dir: %s", ckpt_dir)
  reader = tf.contrib.framework.load_checkpoint(ckpt_dir)
  variable_map = {}
  for var in tf.contrib.framework.get_trainable_variables():
    var_name = var.name.split(":")[0]
    if reader.has_tensor(var_name):
      tf.logging.info("Loading variable from checkpoint: %s", var_name)
      variable_map[var_name] = var
    else:
      tf.logging.info("Cannot find variable in checkpoint, skipping: %s",
                      var_name)
  tf.train.init_from_checkpoint(ckpt_dir, variable_map)
