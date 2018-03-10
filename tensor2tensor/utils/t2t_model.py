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

"""T2TModel Base Class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import copy
import math
import time

# Dependency imports

import six

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators.problem import problem_hparams_to_features
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import decoding
from tensor2tensor.utils import expert_utils as eu
from tensor2tensor.utils import learning_rate
from tensor2tensor.utils import metrics
from tensor2tensor.utils import optimize
from tensor2tensor.utils import registry

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.layers import base
from tensorflow.python.ops import variable_scope

_no_problem_err_str = (
    "The default implementation of %s requires that the "
    "model be used with a Problem. If using a Problem, augment the "
    "hparams object with trainer_lib.add_problem_hparams. If not, "
    "override %s.")
_no_problem_err = (
    lambda method_name: _no_problem_err_str % (method_name, method_name))


class T2TModel(base.Layer):
  """Abstract base class for models.

  Subclassess generally only need to override `body`.
  """
  REGISTERED_NAME = None  # Updated on registration.

  def __init__(self,
               hparams,
               mode=tf.estimator.ModeKeys.TRAIN,
               problem_hparams=None,
               data_parallelism=None,
               decode_hparams=None):
    """Create a T2TModel.

    Args:
      hparams: tf.contrib.training.HParams, model hyperparameters.
      mode: tf.estimator.ModeKeys, the execution mode.
      problem_hparams: tf.contrib.training.HParams, hyperparameters for the
        Problem. If provided here or in hparams.problems, the model will
        automatically determine bottom, top, and loss methods. If not provided,
        calling the model will only invoke body.
      data_parallelism: a expert_utils.Parallelism object,
        specifies devices for data parallelism.
      decode_hparams: a hyperparameter object with decoding parameters.
        See decoding.decode_hparams.

    Returns:
      a T2TModel
    """
    # Determine name first: use registered name if possible, class name else.
    default_name = registry.default_name(type(self))
    name = self.REGISTERED_NAME or default_name
    super(T2TModel, self).__init__(
        trainable=mode == tf.estimator.ModeKeys.TRAIN, name=name)

    if not problem_hparams and hasattr(hparams, "problems"):
      problem_hparams = hparams.problems[0]
    print(problem_hparams)
    self._problem_hparams = problem_hparams

    # Setup hparams
    # If vocabularies differ, unset shared_embedding_and_softmax_weights.
    hparams = copy.copy(hparams)
    if self._problem_hparams and hparams.shared_embedding_and_softmax_weights:
      same_vocab_sizes = True
      if "inputs" in self._problem_hparams.input_modality:
        if (self._problem_hparams.input_modality["inputs"] !=
            self._problem_hparams.target_modality):
          same_vocab_sizes = False
      if not same_vocab_sizes:
        log_info("Unsetting shared_embedding_and_softmax_weights.")
        hparams.shared_embedding_and_softmax_weights = 0
    self._original_hparams = hparams
    self.set_mode(mode)

    self._decode_hparams = copy.copy(decode_hparams or
                                     decoding.decode_hparams())
    self._data_parallelism = data_parallelism or eu.Parallelism([""])
    self._num_datashards = self._data_parallelism.n
    self._ps_devices = self._data_parallelism.ps_devices
    self._eager_var_store = create_eager_var_store()
    if self._problem_hparams:
      self._create_modalities(self._problem_hparams, self._hparams)

  @property
  def hparams(self):
    return self._hparams

  @property
  def has_input(self):
    if self._problem_hparams:
      return "inputs" in self._problem_hparams.input_modality
    else:
      return True

  def call(self, features):
    tf.get_variable_scope().set_initializer(
        optimize.get_variable_initializer(self.hparams))
    with self._eager_var_store.as_default():
      self._fill_problem_hparams_features(features)
      sharded_features = self._shard_features(features)
      sharded_logits, losses = self.model_fn_sharded(sharded_features)
      if isinstance(sharded_logits, dict):
        concat_logits = {}
        for k, v in six.iteritems(sharded_logits):
          concat_logits[k] = tf.concat(v, 0)
        return concat_logits, losses
      else:
        return tf.concat(sharded_logits, 0), losses

  @property
  def use_body_sharded(self):
    return False

  def body_sharded(self, sharded_features):
    raise NotImplementedError("Models that wish to manually control sharding, "
                              "e.g. MoE models, should override body_sharded "
                              "and set use_body_sharded to True.")

  def model_fn_sharded(self, sharded_features):
    dp = self._data_parallelism
    summarize_features(sharded_features, num_shards=dp.n)
    datashard_to_features = self._to_features_per_datashard(sharded_features)
    if self.use_body_sharded:
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
          sharded_logits = {}
          sharded_losses = {}
          for k, v in six.iteritems(body_out):
            sharded_logits[k] = dp(self.top, v, datashard_to_features)
            sharded_losses[k] = dp(self.loss, sharded_logits[k],
                                   datashard_to_features)
          training_loss_dict = average_sharded_losses([{
              "training": l
          } for l in loss for loss in sharded_losses.values()])
          losses.update(training_loss_dict)
        else:
          sharded_logits = dp(self.top, body_out, datashard_to_features)
          sharded_losses = dp(self.loss, sharded_logits, datashard_to_features)
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
    transformed_features = self.bottom(features)

    with tf.variable_scope("body"):
      log_info("Building model body")
      body_out = self.body(transformed_features)
    output, losses = self._normalize_body_output(body_out)

    if "training" in losses:
      log_info("Skipping T2TModel top and loss because training loss "
               "returned from body")
      logits = output
    else:
      logits = self.top(output, features)
      losses["training"] = self.loss(logits, features)
    return logits, losses

  def bottom(self, features):
    """Transform features to feed into body."""
    if not self._problem_hparams:
      log_warn("Without a Problem, T2TModel.bottom is a passthrough.")
      return features

    transformed_features = {}
    all_previous_modalities = []

    # Transform the input features
    for key, input_modality in six.iteritems(
        self._problem_hparams.input_modality):
      if key not in features:
        tf.logging.warning("Missing feature %s - ignoring." % key)
        continue
      do_reuse = input_modality.name in all_previous_modalities
      with tf.variable_scope(input_modality.name, reuse=do_reuse):
        log_info("Transforming feature '%s' with %s.bottom", key,
                 input_modality.name)
        transformed_features[key] = input_modality.bottom(features[key])
      all_previous_modalities.append(input_modality.name)

    # Transform the targets (for autoregressive models)
    print(self._problem_hparams)
    target_modality = self._problem_hparams.target_modality
    if isinstance(target_modality, dict):
      for k, v in six.iteritems(target_modality):
        with tf.variable_scope(
            "%s/%s" %
            (v.name,
             k)):  # TODO(aidangomez): share variables across modalities?
          log_info("Transforming 'targets' with %s.targets_bottom", v.name)
          transformed_features[k] = v.targets_bottom(features[k])
    else:
      with tf.variable_scope(target_modality.name):
        log_info("Transforming 'targets' with %s.targets_bottom",
                 target_modality.name)
        transformed_features["targets"] = target_modality.targets_bottom(
            features["targets"])

    for key in features:
      if key not in transformed_features:
        # For features without a modality, we pass them along as is
        transformed_features[key] = features[key]
      else:
        # Other features get passed along with the "raw" suffix
        transformed_features[key + "_raw"] = features[key]

    return transformed_features

  def body(self, features):
    """Most models will override this function.

    Compute label logits for one shard as a function of the transformed
    features.

    Args:
      features: A dictionary of key to Tensor.  Each Tensor has shape
         [batch_size, ?, ?, hidden_size].

    Returns:
      output: tensor of logits with shape [batch_size, O, P, body_output_size.
      losses: either single loss as a scalar, a list, a tensor (to be averaged)
              or a dictionary of losses.
    """
    raise NotImplementedError("Abstract Method")

  def _top_single(self, body_output, target_modality, features):
    if not target_modality:
      log_warn("Without a Problem, T2TModel.top is a passthrough.")
      return body_output

    with tf.variable_scope(target_modality.name):
      log_info("Transforming body output with %s.top", target_modality.name)
      last_only = (
          target_modality.top_is_pointwise and
          self.hparams.mode == tf.estimator.ModeKeys.PREDICT and
          not self.hparams.force_full_predict)
      if not last_only:
        logits = target_modality.top(body_output, features["targets"])
      else:
        # Take body outputs for the last position only, and targets too.
        last_position_body_output = tf.expand_dims(
            body_output[:, -1, :, :], axis=[1])
        last_position_targets = tf.expand_dims(
            features["targets"][:, -1:, :, :], axis=[1])
        logits = target_modality.top(last_position_body_output,
                                     last_position_targets)
    return logits

  def top(self, body_output, features):
    if isinstance(body_output, dict):
      if self._problem_hparams:
        target_modality = self._problem_hparams.target_modality
      else:
        target_modality = {k: None for k in body_output.keys()}
      assert set(body_output.keys()) == set(target_modality.keys()), (
          "The keys of model_body's returned logits dict must match the keys "
          "of problem_hparams.target_modality's dict.")
      logits = {}
      for k, v in six.iteritems(body_output):
        with tf.variable_scope(k):  # TODO(aidangomez): share variables here?
          logits[k] = self._top_single(v, target_modality[k], features)
      return logits
    else:
      if self._problem_hparams:
        target_modality = self._problem_hparams.target_modality
      else:
        target_modality = None
      assert not isinstance(target_modality, dict), (
          "model_body must return a dictionary of logits when "
          "problem_hparams.target_modality is a dict.")
      return self._top_single(body_output, target_modality, features)

  def _loss_single(self, logits, target_modality, features):
    if not target_modality:
      log_warn(_no_problem_err("loss"))
      return (tf.constant(0., dtype=tf.float32),
              tf.constant(1., dtype=tf.float32))

    loss_num, loss_den = target_modality.loss(logits, features["targets"])
    loss_num *= self._problem_hparams.loss_multiplier
    return loss_num, loss_den

  def loss(self, logits, features):
    if isinstance(logits, dict):
      if self._problem_hparams:
        target_modality = self._problem_hparams.target_modality
      else:
        target_modality = {k: None for k in logits.keys()}
      assert set(logits.keys()) == set(target_modality.keys()), (
          "The keys of model_body's returned logits dict must match the keys "
          "of problem_hparams.target_modality's dict.")
      losses = {}
      for k, v in six.iteritems(logits):
        losses[k] = self._loss_single(v, target_modality[k], features)
      return tf.add_n([n / d for n, d in losses.values()])
    else:
      if self._problem_hparams:
        target_modality = self._problem_hparams.target_modality
      else:
        target_modality = None
      assert not isinstance(target_modality, dict), (
          "model_body must return a dictionary of logits when "
          "problem_hparams.target_modality is a dict.")
      return self._loss_single(logits, target_modality, features)

  def optimize(self, loss, num_async_replicas=1):
    """Return a training op minimizing loss."""
    log_info("Base learning rate: %f", self.hparams.learning_rate)
    lr = learning_rate.learning_rate_schedule(self.hparams)
    if num_async_replicas > 1:
      log_info("Dividing learning rate by num_async_replicas: %d",
               num_async_replicas)
    lr /= math.sqrt(float(num_async_replicas))
    train_op = optimize.optimize(
        loss, lr, self.hparams, use_tpu=common_layers.is_on_tpu())
    return train_op

  def set_mode(self, mode):
    """Set hparams with the given mode."""
    log_info("Setting T2TModel mode to '%s'", mode)
    hparams = copy.copy(self._original_hparams)
    hparams.add_hparam("mode", mode)
    # When not in training mode, set all forms of dropout to zero.
    if mode != tf.estimator.ModeKeys.TRAIN:
      for key in hparams.values():
        if key.endswith("dropout"):
          log_info("Setting hparams.%s to 0.0", key)
          setattr(hparams, key, 0.0)
    self._hparams = hparams

  def _create_modalities(self, problem_hparams, hparams):
    """Construct modalities in problem_hparams."""

    input_modality_overrides = {}
    for override_str in hparams.input_modalities.split(";"):
      if override_str != "default":
        parts = override_str.split(":")
        feature_name = parts[0]
        modality_name = ":".join(parts[1:])
        input_modality_overrides[feature_name] = modality_name

    target_modality_name = None
    if hparams.target_modality and hparams.target_modality != "default":
      target_modality_name = hparams.target_modality

    input_modality = {}
    for f, modality_spec in six.iteritems(problem_hparams.input_modality):
      if f in input_modality_overrides:
        _warn_changed_modality_type(input_modality_overrides[f],
                                    modality_spec[0], f)
        modality_spec = (input_modality_overrides[f], modality_spec[1])
      input_modality[f] = registry.create_modality(modality_spec, hparams)
    problem_hparams.input_modality = input_modality

    if isinstance(problem_hparams.target_modality, dict):
      target_modality = {}
      for f, modality_spec in six.iteritems(problem_hparams.target_modality):
        if target_modality_name:
          _warn_changed_modality_type(target_modality_name, modality_spec[0],
                                      "target_modality/%s" % f)
          modality_spec = (target_modality_name, modality_spec[1])
        target_modality[f] = registry.create_modality(modality_spec, hparams)
    else:
      target_modality_spec = problem_hparams.target_modality
      if target_modality_name:
        _warn_changed_modality_type(target_modality_name,
                                    target_modality_spec[0], "target")
        target_modality_spec = (target_modality_name, target_modality_spec[1])
      target_modality = registry.create_modality(target_modality_spec, hparams)
    problem_hparams.target_modality = target_modality

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
      for k, v in six.iteritems(
          problem_hparams_to_features(self._problem_hparams)):
        if k not in features:
          features[k] = tf.constant(v, name=k)

  def infer(self,
            features=None,
            decode_length=50,
            beam_size=1,
            top_beams=1,
            alpha=0.0):
    """A inference method.

    Quadratic time in decode_length.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.

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
        target_modality = self._problem_hparams.target_modality
        if target_modality.is_class_modality:
          beam_size = 1  # No use to run beam-search for a single class.
      if beam_size == 1:
        log_info("Greedy Decoding")
        results = self._greedy_infer(features, decode_length)
      else:
        log_info("Beam Decoding with beam size %d" % beam_size)
        results = self._beam_decode(features, decode_length, beam_size,
                                    top_beams, alpha)

      return results

  def _beam_decode(self, features, decode_length, beam_size, top_beams, alpha):
    """Beam search decoding.

    Models should ideally implement a more efficient version of this function.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.

    Returns:
       samples: an integer `Tensor`. Top samples from the beam search
    """
    return self._beam_decode_slow(features, decode_length, beam_size, top_beams,
                                  alpha)

  def _beam_decode_slow(self, features, decode_length, beam_size, top_beams,
                        alpha):
    """Slow version of Beam search decoding.

    Quadratic time in decode_length.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.

    Returns:
       samples: an integer `Tensor`. Top samples from the beam search
    """
    batch_size = common_layers.shape_list(features["inputs"])[0]

    def symbols_to_logits_fn(ids):
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
      self._coverage = None
      logits, _ = self(features)  # pylint: disable=not-callable
      # now self._coverage is a coverage tensor for the first datashard.
      # it has shape [batch_size] and contains floats between 0 and
      # source_length.
      if self._problem_hparams:
        modality = self._problem_hparams.target_modality
        if modality.top_is_pointwise:
          return tf.squeeze(logits, axis=[1, 2, 3])
      # -1 due to the pad above.
      current_output_position = common_layers.shape_list(ids)[1] - 1
      logits = logits[:, current_output_position, :, :]
      return tf.squeeze(logits, axis=[1, 2])

    initial_ids = tf.zeros([batch_size], dtype=tf.int32)

    if self.has_input:
      inputs_old = features["inputs"]
      features["inputs"] = tf.expand_dims(features["inputs"], 1)
      if len(features["inputs"].shape) < 5:
        features["inputs"] = tf.expand_dims(features["inputs"], 4)
      # Expand the inputs in to the beam size.
      features["inputs"] = tf.tile(features["inputs"], [1, beam_size, 1, 1, 1])
      s = common_layers.shape_list(features["inputs"])
      features["inputs"] = tf.reshape(features["inputs"],
                                      [s[0] * s[1], s[2], s[3], s[4]])

    target_modality = self._problem_hparams.target_modality
    vocab_size = target_modality.top_dimensionality
    # Setting decode length to input length + decode_length
    decode_length = tf.constant(decode_length)
    if "partial_targets" not in features:
      decode_length += common_layers.shape_list(features["inputs"])[1]
    ids, scores = beam_search.beam_search(
        symbols_to_logits_fn,
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        alpha,
        stop_early=(top_beams == 1))

    # Set inputs back to the unexpanded inputs to not to confuse the Estimator!
    if self.has_input:
      features["inputs"] = inputs_old

    # Return `top_beams` decodings (also remove initial id from the beam search)
    # TODO(lukaszkaiser): make it work multi-problem.
    if top_beams == 1:
      samples = ids[:, 0, 1:]
    else:
      samples = ids[:, :top_beams, 1]

    return {"outputs": samples, "scores": scores}

  def _greedy_infer(self, features, decode_length):
    """A greedy inference method.

    Models should ideally implement a more efficient version of this function.

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
    return self._slow_greedy_infer(features, decode_length)

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
      features["partial_targets"] = tf.to_int64(features["inputs"])
    # Save the targets in a var and reassign it after the tf.while loop to avoid
    # having targets being in a 'while' frame. This ensures targets when used
    # in metric functions stays in the same frame as other vars.
    targets_old = features.get("targets", None)

    target_modality = self._problem_hparams.target_modality

    def infer_step(recent_output, recent_logits, unused_loss):
      """Inference step."""
      if not context.in_eager_mode():
        recent_output.set_shape([None, None, None, 1])
      padded = tf.pad(recent_output, [[0, 0], [0, 1], [0, 0], [0, 0]])
      features["targets"] = padded
      # This is inefficient in that it generates samples at all timesteps,
      # not just the last one, except if target_modality is pointwise.
      samples, logits, losses = self.sample(features)
      # Concatenate the already-generated recent_output with last timestep
      # of the newly-generated samples.
      if target_modality.top_is_pointwise:
        cur_sample = samples[:, -1, :, :]
      else:
        cur_sample = samples[:,
                             common_layers.shape_list(recent_output)[1], :, :]
      cur_sample = tf.to_int64(tf.expand_dims(cur_sample, axis=1))
      samples = tf.concat([recent_output, cur_sample], axis=1)
      if not context.in_eager_mode():
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
      initial_output = tf.zeros((batch_size, 0, 1, 1), dtype=tf.int64)
    # Hack: foldl complains when the output shape is less specified than the
    # input shape, so we confuse it about the input shape.
    initial_output = tf.slice(initial_output, [0, 0, 0, 0],
                              common_layers.shape_list(initial_output))
    target_modality = self._problem_hparams.target_modality
    if target_modality.is_class_modality:
      decode_length = 1
    else:
      decode_length = common_layers.shape_list(
          features["inputs"])[1] + decode_length
    # Initial values of result, logits and loss.
    result = initial_output
    # tensor of shape [batch_size, time, 1, 1, vocab_size]
    logits = tf.zeros((batch_size, 0, 1, 1, target_modality.top_dimensionality))
    if not context.in_eager_mode():
      logits.set_shape([None, None, None, None, None])
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
            # We only check for early stoping if there is at least 1 element (
            # otherwise not_eos will crash)
            tf.not_equal(length, 0),
            fn_not_eos,
            lambda: True,
        )

        return tf.cond(
            tf.equal(batch_size, 1),
            # If batch_size == 1, we check EOS for early stoping
            lambda: tf.logical_and(not_overflow, not_eos),
            # Else, just wait for max length
            lambda: not_overflow)
      return not_overflow

    result, logits, loss = tf.while_loop(
        while_exit_cond,
        infer_step, [result, logits, loss],
        shape_invariants=[
            tf.TensorShape([None, None, None, None]),
            tf.TensorShape([None, None, None, None, None]),
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
    sharded_features = dict()
    for k, v in six.iteritems(features):
      v = tf.convert_to_tensor(v)
      v_shape = common_layers.shape_list(v)
      if not v_shape:
        v = tf.expand_dims(v, axis=-1)
        v_shape = [1]
      if v_shape == [1]:
        v = tf.tile(v, [self._num_datashards])
      sharded_features[k] = self._data_parallelism(tf.identity,
                                                   tf.split(
                                                       v, self._num_datashards,
                                                       0))
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
      params: dict, may include batch_size
      decode_hparams: HParams, used when mode == PREDICT.
      use_tpu: bool, whether using TPU

    Returns:
      TPUEstimatorSpec if use tpu else EstimatorSpec
    """
    _create_dummy_vars()
    hparams = copy.deepcopy(hparams)

    # Instantiate model
    data_parallelism = None
    if not use_tpu and config:
      data_parallelism = config.data_parallelism
    model = cls(
        hparams,
        mode,
        data_parallelism=data_parallelism,
        decode_hparams=decode_hparams)

    # PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
      assert not use_tpu
      return model.estimator_spec_predict(features)

    # TRAIN and EVAL modes
    if hparams.eval_run_autoregressive and mode == tf.estimator.ModeKeys.EVAL:
      logits, losses_dict = model.eval_autoregressive(features)
    else:
      logits, losses_dict = model(features)  # pylint: disable=not-callable

    # Set known shapes
    if use_tpu:
      if isinstance(logits, dict):
        for k, v in six.iteritems(logits):
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

    # Summarize losses
    with tf.name_scope("losses"):
      for loss_name, loss_val in losses_dict.items():
        tf.summary.scalar(loss_name, loss_val)

    # Accumulate losses
    loss = sum(losses_dict.values())

    # EVAL mode
    if mode == tf.estimator.ModeKeys.EVAL:
      return model.estimator_spec_eval(features, logits, labels, loss,
                                       losses_dict)

    # TRAIN mode
    assert mode == tf.estimator.ModeKeys.TRAIN
    num_async_replicas = (1 if (use_tpu or not config) else
                          config.t2t_device_info["num_async_replicas"])
    return model.estimator_spec_train(
        loss, num_async_replicas=num_async_replicas)

  def estimator_spec_train(self, loss, num_async_replicas=1):
    """Construct EstimatorSpec for TRAIN mode."""
    train_op = self.optimize(loss, num_async_replicas=num_async_replicas)

    if common_layers.is_on_tpu():
      _remove_summaries()  # summaries not currently working on TPU
      return tf.contrib.tpu.TPUEstimatorSpec(
          tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)
    else:
      return tf.estimator.EstimatorSpec(
          tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)

  def estimator_spec_eval(self, features, logits, labels, loss, losses_dict):
    """Construct EstimatorSpec for EVAL mode."""
    hparams = self.hparams

    if not hasattr(hparams, "problem_instances"):
      raise NotImplementedError(_no_problem_err("estimator_spec_eval"))

    problem = hparams.problem_instances[0]
    if common_layers.is_on_tpu():
      _remove_summaries()
      if isinstance(logits, dict):
        eval_metrics_fn = _create_tpu_eval_metrics_fn(problem, hparams)
        # For TPU, logits dict will be passed as keyword arguments to
        # eval_metrics_fn. Here we add the labels to those arguments.
        logits.update({"labels": labels})
        return tf.contrib.tpu.TPUEstimatorSpec(
            tf.estimator.ModeKeys.EVAL,
            eval_metrics=(eval_metrics_fn, logits),
            loss=loss)
      else:
        eval_metrics_fn = _create_tpu_eval_metrics_fn(problem, hparams)
        return tf.contrib.tpu.TPUEstimatorSpec(
            tf.estimator.ModeKeys.EVAL,
            eval_metrics=(eval_metrics_fn, [logits, labels]),
            loss=loss)
    else:
      eval_metrics_fns = metrics.create_evaluation_metrics([problem], hparams)
      eval_metrics = {}
      for metric_name, metric_fn in six.iteritems(eval_metrics_fns):
        if isinstance(logits, dict):
          # the key is located in the center of metric_name: "metrics-%s/%s/%s"
          k = metric_name.split("/")[1]
          eval_metrics[metric_name] = metric_fn(logits[k], features)
          return tf.estimator.EstimatorSpec(
              tf.estimator.ModeKeys.EVAL,
              predictions=logits,
              eval_metric_ops=eval_metrics,
              loss=loss)
        else:
          eval_metrics[metric_name] = metric_fn(logits, features)
          return tf.estimator.EstimatorSpec(
              tf.estimator.ModeKeys.EVAL,
              predictions={"predictions": logits},
              eval_metric_ops=eval_metrics,
              loss=loss)

  def estimator_spec_predict(self, features):
    """Construct EstimatorSpec for PREDICT mode."""
    decode_hparams = self._decode_hparams
    infer_out = self.infer(
        features,
        beam_size=decode_hparams.beam_size,
        top_beams=(decode_hparams.beam_size
                   if decode_hparams.return_beams else 1),
        alpha=decode_hparams.alpha,
        decode_length=decode_hparams.extra_length)
    if isinstance(infer_out, dict):
      outputs = infer_out["outputs"]
      scores = infer_out["scores"]
    else:
      outputs = infer_out
      scores = None

    batched_problem_choice = (
        features["problem_choice"] * tf.ones(
            (common_layers.shape_list(features["inputs"])[0],), dtype=tf.int32))
    predictions = {
        "outputs": outputs,
        "scores": scores,
        "inputs": features.get("inputs"),
        "targets": features.get("infer_targets"),
        "problem_choice": batched_problem_choice,
    }
    _del_dict_nones(predictions)

    export_out = {"outputs": predictions["outputs"]}
    if "scores" in predictions:
      export_out["scores"] = predictions["scores"]

    _remove_summaries()

    return tf.estimator.EstimatorSpec(
        tf.estimator.ModeKeys.PREDICT,
        predictions=predictions,
        export_outputs={
            "output": tf.estimator.export.PredictOutput(export_out)
        })

  def _normalize_body_output(self, body_out):
    if isinstance(body_out, tuple):
      output, losses = body_out
      if not isinstance(losses, dict):
        losses = {"extra": tf.reduce_mean(losses)}
    else:
      output = body_out
      losses = {"extra": 0.0}

    return output, losses


def _warn_changed_modality_type(new_name, old_name, feature_name):
  new_type, new_name = registry.parse_modality_name(new_name)
  old_type, old_name = registry.parse_modality_name(old_name)
  if new_type != old_type:
    log_warn("%s has a designated modality type %s (%s) but has been "
             "overridden with a modality of type %s (%s).", feature_name,
             old_type, old_name, new_type, new_name)


def _with_timing(fn, msg, silent=False):

  def fn_with_timing(*args, **kwargs):
    start_time = time.time()
    res = fn(*args, **kwargs)
    if not silent:
      log_info("Doing %s took %.3f sec." % (msg, time.time() - start_time))
    return res

  return fn_with_timing


def _create_dummy_vars():
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
])


def _create_tpu_eval_metrics_fn(problem, hparams):
  """Create the metrics_fn that TPUEstimatorSpec expects."""

  metric_fns = []
  eval_metrics = problem.eval_metrics()

  tm = problem.get_hparams().target_modality
  if isinstance(tm, dict):
    for k, v in six.iteritems(tm):
      if isinstance(v, tuple):
        v = registry.create_modality(v, hparams)
      weights_fn = v.targets_weights_fn

      def make_metric_fn(metric_fn):

        def wrapped_metric_fn(logits, labels, weights_fn=weights_fn):
          num, den = metric_fn(logits, labels, weights_fn=weights_fn)
          return tf.metrics.mean(num, den)

        return wrapped_metric_fn

      for metric in eval_metrics:
        if metric in TPU_METRIC_BLACKLIST:
          log_warn("Skipping eval metric %s in TPU_METRIC_BLACKLIST", metric)
          continue
        name = "%s/metrics-%s/%s" % (k, problem.name, metric)
        metric_fns.append((name, make_metric_fn(metrics.METRICS_FNS[metric])))
  else:
    if isinstance(tm, tuple):
      tm = registry.create_modality(tm, hparams)
    weights_fn = tm.targets_weights_fn

    def make_metric_fn(metric_fn):

      def wrapped_metric_fn(logits, labels):
        num, den = metric_fn(logits, labels, weights_fn=weights_fn)
        return tf.metrics.mean(num, den)

      return wrapped_metric_fn

    for metric in eval_metrics:
      if metric in TPU_METRIC_BLACKLIST:
        log_warn("Skipping eval metric %s in TPU_METRIC_BLACKLIST", metric)
        continue
      name = "metrics-%s/%s" % (problem.name, metric)
      metric_fns.append((name, make_metric_fn(metrics.METRICS_FNS[metric])))

  def all_metrics_fn(logits=None, labels=None, **kwargs):
    """Construct metrics dictionary."""
    metrics_dict = {}

    if logits is None:
      logits = kwargs

    for name, fn in metric_fns:
      if isinstance(logits, dict):
        for k, v in six.iteritems(logits):
          if isinstance(labels, dict):
            metrics_dict["%s/%s" % (name, k)] = fn(v, labels[k])
          else:
            metrics_dict["%s/%s" % (name, k)] = fn(v, labels)
      else:
        metrics_dict[name] = fn(logits, labels)

    return metrics_dict

  return all_metrics_fn


def _remove_summaries():
  g = tf.get_default_graph()
  key = tf.GraphKeys.SUMMARIES
  del g.get_collection_ref(key)[:]
  assert not g.get_collection(key)


def _del_dict_nones(d):
  for k in list(d.keys()):
    if d[k] is None:
      del d[k]


class DummyVariableStore(object):

  @contextlib.contextmanager
  def as_default(self):
    yield


def create_eager_var_store():
  if context.in_eager_mode():
    return variable_scope.EagerVariableStore()
  else:
    return DummyVariableStore()


def scheduled_sampling(hparams, problem_hparams, dp, sharded_logits, losses,
                       sharded_features, transformed_features, model):
  """Scheduled sampling."""
  target_modality = problem_hparams.target_modality

  def sample(x):
    """Multinomial sampling from a n-dimensional tensor."""
    vocab_size = target_modality.top_dimensionality
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
      with tf.variable_scope(target_modality.name):
        new_features["targets"] = target_modality.targets_bottom_sharded(
            new_targets, dp)
      with tf.variable_scope("body"):
        body_outputs, losses = model.model_fn_sharded(new_features)
        if not isinstance(losses, dict):  # If it's a single extra loss.
          losses = {"extra": losses}
      with tf.variable_scope(target_modality.name):
        new_sharded_logits = target_modality.top_sharded(
            body_outputs, sharded_features["targets"], dp)
        if "training" not in losses:
          training_loss = target_modality.loss_sharded(
              sharded_logits, sharded_features["targets"], dp)
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
  for loss_name in sharded_losses[0]:
    all_shards = [shard_losses[loss_name] for shard_losses in sharded_losses]
    if isinstance(all_shards[0], tuple):
      sharded_num, sharded_den = zip(*all_shards)
      mean_loss = (
          tf.add_n(sharded_num) / tf.maximum(1.0, tf.add_n(sharded_den)))
    else:
      mean_loss = tf.reduce_mean(all_shards)

    losses[loss_name] = mean_loss
  return losses


def summarize_features(features, num_shards=1):
  with tf.name_scope("input_stats"):
    for (k, v) in six.iteritems(features):
      if isinstance(v, tf.Tensor) and v.get_shape().ndims > 1:
        tf.summary.scalar("%s_batch" % k, tf.shape(v)[0] // num_shards)
        tf.summary.scalar("%s_length" % k, tf.shape(v)[1])
        nonpadding = tf.to_float(tf.not_equal(v, 0))
        nonpadding_tokens = tf.reduce_sum(nonpadding)
        tf.summary.scalar("%s_nonpadding_tokens" % k, nonpadding_tokens)
        tf.summary.scalar("%s_nonpadding_fraction" % k,
                          tf.reduce_mean(nonpadding))


_already_logged = set()


def _eager_log(level, *args):
  if context.in_eager_mode() and args in _already_logged:
    return
  _already_logged.add(args)
  getattr(tf.logging, level)(*args)


def log_info(*args):
  _eager_log("info", *args)


def log_warn(*args):
  _eager_log("warn", *args)
