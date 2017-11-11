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

"""T2TModel Base Class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import time

# Dependency imports

import six
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import expert_utils as eu
from tensor2tensor.utils import registry

import tensorflow as tf


def _with_timing(fn, msg):

  def fn_with_timing(*args, **kwargs):
    start_time = time.time()
    res = fn(*args, **kwargs)
    tf.logging.info("Doing %s took %.3f sec." % (msg, time.time() - start_time))
    return res

  return fn_with_timing


def is_class_modality(mod):
  # TODO(lukaszkaiser): should be based on type, like CLASS_LABEL, not string.
  prefix = "class_label_modality_"
  if len(mod.name) < len(prefix):
    return False
  return mod.name[:len(prefix)] == prefix


class T2TModel(object):
  """Abstract base class for models.

  Subclassess generally only need to override `build_model`.
  """

  def __init__(self,
               hparams,
               mode,
               problem_hparams,
               problem_idx=0,
               data_parallelism=None,
               ps_devices=None,
               decode_hparams=None):
    """Create a T2TModel.

    Args:
      hparams: a hyperparameters object.
      mode: The execution mode, as defined in tf.estimator.ModeKeys.
      problem_hparams: a hyperparameters object.
      problem_idx: an integer.
      data_parallelism: a expert_utils.parallelism
        (specifies devices for data parallelism).
      ps_devices: a list of devices to be used for experts
      decode_hparams: a hyperparameter object with decoding parameters.

    Returns:
      a T2TModel
    """
    if data_parallelism is None:
      data_parallelism = eu.Parallelism([""])
    if ps_devices is None:
      ps_devices = [""]
    hparams = copy.copy(hparams)
    hparams.add_hparam("mode", mode)
    # When not in training mode, set all forms of dropout to zero.
    if mode != tf.estimator.ModeKeys.TRAIN:
      for key in hparams.values():
        if key[-len("dropout"):] == "dropout":
          setattr(hparams, key, 0.0)
    # If vocabularies differ, unset shared_embedding_and_softmax_weights.
    if hparams.shared_embedding_and_softmax_weights:
      same_vocab_sizes = True
      for problem in hparams.problems:
        if "inputs" in problem.input_modality:
          if problem.input_modality["inputs"] != problem.target_modality:
            same_vocab_sizes = False
      if not same_vocab_sizes:
        tf.logging.info("Unsetting shared_embedding_and_softmax_weights.")
        hparams.shared_embedding_and_softmax_weights = 0
    self._hparams = hparams
    self._decode_hparams = copy.copy(decode_hparams)
    self._data_parallelism = data_parallelism
    self._num_datashards = data_parallelism.n
    self._ps_devices = ps_devices
    self._problem_hparams = problem_hparams
    self._problem_idx = problem_idx
    self._create_modalities(problem_hparams, hparams)

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

    target_modality_spec = problem_hparams.target_modality
    if target_modality_name:
      _warn_changed_modality_type(target_modality_name, target_modality_spec[0],
                                  "target")
      target_modality_spec = (target_modality_name, target_modality_spec[1])
    target_modality = registry.create_modality(target_modality_spec, hparams)
    problem_hparams.target_modality = target_modality

  @property
  def has_input(self):
    return self._problem_hparams.input_modality

  def prepare_features_for_infer(self, features):
    """Called before inference to allow adding infer-specific features."""
    pass

  def eval_autoregressive(self,
                          features=None,
                          decode_length=50):
    """Autoregressive eval.

    Quadratic time in decode_length.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.

    Returns:
      sharded_logits: a list of `Tensor`s. Assumes one datashard.
      losses: a dictionary: {loss-name (string): floating point `Scalar`}.
          Contains a single key "training".
    """
    _, logits, losses = self._slow_greedy_infer(
        features,
        decode_length=decode_length)
    return [logits], losses

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
       samples: an integer `Tensor`.
    """
    # TODO(rsepassi): Make decoding work with real-valued model outputs
    # (i.e. if the target modality is RealModality).
    self.prepare_features_for_infer(features)
    if not self.has_input and beam_size > 1:
      tf.logging.warn("Beam searching for a model with no inputs.")
    if not self.has_input and self._hparams.sampling_method != "random":
      tf.logging.warn("Non-random sampling for a model with no inputs.")
    if is_class_modality(
        self._hparams.problems[self._problem_idx].target_modality):
      beam_size = 1  # No use to run beam-search for a single class.
    if beam_size == 1:
      tf.logging.info("Greedy Decoding")
      samples, _, _ = self._greedy_infer(features, decode_length)
    else:
      tf.logging.info("Beam Decoding with beam size %d" % beam_size)
      samples = self._beam_decode(features, decode_length, beam_size, top_beams,
                                  alpha)
    return samples

  def _beam_decode(self, features, decode_length, beam_size, top_beams,
                   alpha):
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
    batch_size = tf.shape(features["inputs"])[0]
    batch_size = tf.Print(batch_size, [batch_size], "beam_decode batch_size=")

    def symbols_to_logits_fn(ids):
      """Go from ids to logits."""
      ids = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      ids = tf.pad(ids[:, 1:], [[0, 0], [0, 1], [0, 0], [0, 0]])
      if "partial_targets" in features:
        pt = features["partial_targets"]
        pt_length = tf.shape(pt)[1]
        pt = tf.tile(pt, [1, beam_size])
        pt = tf.reshape(pt, [batch_size * beam_size, pt_length, 1, 1])
        ids = tf.concat([pt, ids], axis=1)

      features["targets"] = ids
      self._coverage = None
      sharded_logits, _ = self.model_fn(features, False)
      # now self._coverage is a coverage tensor for the first datashard.
      # it has shape [batch_size] and contains floats between 0 and
      # source_length.
      logits = sharded_logits[0]  # Assuming we have one shard.
      modality = self._hparams.problems[self._problem_idx].target_modality
      if modality.top_is_pointwise:
        return tf.squeeze(logits, axis=[1, 2, 3])
      current_output_position = tf.shape(ids)[1] - 1  # -1 due to the pad above.
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
      s = tf.shape(features["inputs"])
      features["inputs"] = tf.reshape(features["inputs"],
                                      [s[0] * s[1], s[2], s[3], s[4]])

    target_modality = self._hparams.problems[self._problem_idx].target_modality
    vocab_size = target_modality.top_dimensionality
    # Setting decode length to input length + decode_length
    decode_length = tf.constant(decode_length)
    if "partial_targets" not in features:
      decode_length += tf.shape(features["inputs"])[1]
    ids, scores = beam_search.beam_search(symbols_to_logits_fn, initial_ids,
                                          beam_size, decode_length, vocab_size,
                                          alpha, stop_early=(top_beams == 1))

    # Set inputs back to the unexpanded inputs to not to confuse the Estimator!
    if self.has_input:
      features["inputs"] = inputs_old

    # Return `top_beams` decodings (also remove initial id from the beam search)
    return_scores = True  # TODO(lukaszkaiser): make it work multi-problem.
    if top_beams == 1:
      if return_scores:
        return {"outputs": ids[:, 0, 1:], "scores": scores}
      return ids[:, 0, 1:]
    else:
      if return_scores:
        return {"outputs": ids[:, :top_beams, 1:], "scores": scores}
      return ids[:, :top_beams, 1:]

  def  _greedy_infer(self, features, decode_length):
    """A greedy inference method.

    Models should ideally implement a more efficient version of this function.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.

    Returns:
       samples: an integer `Tensor`.
       logits: `Tensor` of shape [batch_size, time, 1, 1, vocab_size].
       losses: a dictionary: {loss-name (string): floating point `Scalar`}
    """
    return self._slow_greedy_infer(features, decode_length)

  def _slow_greedy_infer(self, features, decode_length):
    """A slow greedy inference method.

    Quadratic time in decode_length.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.

    Returns:
       samples: an integer `Tensor`.
       logits: `Tensor` of shape [batch_size, time, 1, 1, vocab_size].
       losses: a dictionary: {loss-name (string): floating point `Scalar`}
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

    target_modality = self._hparams.problems[self._problem_idx].target_modality
    def infer_step(recent_output, recent_logits, unused_loss):
      """Inference step."""
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
        cur_sample = samples[:, tf.shape(recent_output)[1], :, :]
      cur_sample = tf.to_int64(tf.expand_dims(cur_sample, axis=1))
      samples = tf.concat([recent_output, cur_sample], axis=1)
      samples.set_shape([None, None, None, 1])

      # Assuming we have one shard for logits.
      logits = tf.concat([recent_logits, logits[0][:, -1:]], 1)
      loss = sum([l for l in losses.values() if l is not None])
      return samples, logits, loss

    # Create an initial output tensor. This will be passed
    # to the infer_step, which adds one timestep at every iteration.
    if "partial_targets" in features:
      initial_output = tf.to_int64(features["partial_targets"])
      while len(initial_output.get_shape().as_list()) < 4:
        initial_output = tf.expand_dims(initial_output, 2)
      batch_size = tf.shape(initial_output)[0]
    else:
      batch_size = tf.shape(features["inputs"])[0]
      initial_output = tf.zeros((batch_size, 0, 1, 1), dtype=tf.int64)
    # Hack: foldl complains when the output shape is less specified than the
    # input shape, so we confuse it about the input shape.
    initial_output = tf.slice(initial_output, [0, 0, 0, 0],
                              tf.shape(initial_output))
    target_modality = self._hparams.problems[self._problem_idx].target_modality
    if is_class_modality(target_modality):
      decode_length = 1
    else:
      decode_length = tf.shape(features["inputs"])[1] + decode_length
    # Initial values of result, logits and loss.
    result = initial_output
    # tensor of shape [batch_size, time, 1, 1, vocab_size]
    logits = tf.zeros((batch_size, 0, 1, 1, target_modality.top_dimensionality))
    logits.set_shape([None, None, None, None, None])
    loss = 0.0

    def while_exit_cond(result, logits, loss):  # pylint: disable=unused-argument
      """Exit the loop either if reach decode_length or EOS."""
      length = tf.shape(result)[1]

      not_overflow = length < decode_length

      if self._problem_hparams.stop_at_eos:
        def fn_not_eos():
          return tf.not_equal(  # Check if the last predicted element is a EOS
              tf.squeeze(result[:, -1, :, :]),
              text_encoder.EOS_ID
          )

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
            lambda: not_overflow
        )
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
      partial_target_length = tf.shape(features["partial_targets"])[1]
      result = tf.slice(
          result, [0, partial_target_length, 0, 0], [-1, -1, -1, -1])
    return result, logits, losses

  def sample(self, features):
    """Run the model and extract samples.

    Args:
      features: an map of string to `Tensor`.

    Returns:
       samples: an integer `Tensor`.
       logits: a list of `Tensor`s, one per datashard.
       losses: a dictionary: {loss-name (string): floating point `Scalar`}.
    """
    sharded_logits, losses = self.model_fn(features, False)
    if self._hparams.sampling_method == "argmax":
      sharded_samples = self._data_parallelism(tf.argmax, sharded_logits, 4)
    else:
      assert self._hparams.sampling_method == "random"

      def _multinomial_squeeze(logits, temperature=1.0):
        reshaped_logits = (
            tf.reshape(logits, [-1, tf.shape(logits)[-1]])/temperature)
        choices = tf.multinomial(reshaped_logits, 1)
        choices = tf.reshape(choices,
                             tf.shape(logits)[:logits.get_shape().ndims - 1])
        return choices

      sharded_samples = self._data_parallelism(_multinomial_squeeze,
                                               sharded_logits,
                                               self._hparams.sampling_temp)
    return tf.concat(sharded_samples, 0), sharded_logits, losses

  def _shard_features(self, features):  # pylint: disable=missing-docstring
    sharded_features = dict()
    for k, v in six.iteritems(features):
      v = tf.convert_to_tensor(v)
      if not v.shape.as_list():
        v = tf.expand_dims(v, axis=-1)
        v = tf.tile(v, [self._num_datashards])
      sharded_features[k] = self._data_parallelism(tf.identity,
                                                   tf.split(
                                                       v, self._num_datashards,
                                                       0))
    return sharded_features

  def model_fn(self, features, skip=False):
    """Computes the entire model and produces sharded logits and losses.

    Args:
      features: A dictionary of feature name to tensor.
      skip: a boolean, if we're just dummy-calling and actually skip this model
        (but we need to create variables to not confuse distributed training).

    Returns:
      sharded_logits: a list of `Tensor`s, one per datashard.
      losses: a dictionary: {loss-name (string): floating point `Scalar`}.
    """
    start_time = time.time()
    dp = self._data_parallelism

    sharded_features = self._shard_features(features)

    # Construct the model bottom for inputs.
    transformed_features = {}
    all_previous_modalities = []

    for key, input_modality in six.iteritems(
        self._problem_hparams.input_modality):
      previous_modalities = [
          self._hparams.problems[i].input_modality[key].name
          for i in xrange(self._problem_idx)
      ]
      all_previous_modalities.extend(previous_modalities)
      do_reuse = input_modality.name in all_previous_modalities
      transformed_features[key + "_raw"] = sharded_features[key]
      with tf.variable_scope(input_modality.name, reuse=do_reuse):
        transformed_features[key] = input_modality.bottom_sharded(
            sharded_features[key], dp)
      all_previous_modalities.append(input_modality.name)

    # Target space id just gets copied to every shard.
    if "target_space_id" in features:
      transformed_features["target_space_id"] = [
          features["target_space_id"]] * self._num_datashards

    # For features without a modality ending in "_raw", we pass them raw.
    for key, feature in sharded_features.items():
      if key not in transformed_features and key.endswith("_raw"):
        transformed_features[key] = feature

    # Targets are transformed by the autoregressive part of the modality
    previous_tgt_modalities = [
        self._hparams.problems[i].target_modality.name
        for i in xrange(self._problem_idx)
    ]
    all_previous_modalities.extend(previous_tgt_modalities)

    target_modality = self._problem_hparams.target_modality
    target_reuse = target_modality.name in previous_tgt_modalities
    with tf.variable_scope(target_modality.name, reuse=target_reuse):
      transformed_features["targets"] = target_modality.targets_bottom_sharded(
          sharded_features["targets"], dp)

    # Allows later access to pre-embedding raw targets.
    transformed_features["targets_raw"] = sharded_features["targets"]

    # Construct the model body.
    with tf.variable_scope("body", reuse=self._problem_idx > 0):
      if skip:
        body_outputs = transformed_features["targets"]
        losses = {"extra": 0.0}
      else:
        body_outputs, losses = self.model_fn_body_sharded(
            transformed_features)
        if not isinstance(losses, dict):  # If it's a single extra loss.
          losses = {"extra": losses}

    with tf.variable_scope(target_modality.name, reuse=target_reuse):
      last_only = (target_modality.top_is_pointwise and
                   self._hparams.mode == tf.estimator.ModeKeys.PREDICT)
      if not last_only:
        sharded_logits = target_modality.top_sharded(
            body_outputs, sharded_features["targets"], dp)
        training_loss = target_modality.loss_sharded(
            sharded_logits, sharded_features["targets"], dp)

        training_loss *= self._problem_hparams.loss_multiplier
      else:
        # Take body outputs for the last position only, and targets too.
        last_position_body_outputs = [
            tf.expand_dims(body_shard[:, -1, :, :], axis=[1])
            for body_shard in body_outputs
        ]
        last_position_targets = [
            tf.expand_dims(target_shard[:, -1:, :, :], axis=[1])
            for target_shard in sharded_features["targets"]
        ]
        sharded_logits = target_modality.top_sharded(last_position_body_outputs,
                                                     last_position_targets,
                                                     self._data_parallelism)
        training_loss = None
    losses["training"] = training_loss

    # Scheduled sampling.
    do_scheduled_sampling = (  # Only do it if training and set for it.
        self._hparams.scheduled_sampling_prob > 0.0 and
        self._hparams.mode == tf.estimator.ModeKeys.TRAIN and
        not skip)
    if do_scheduled_sampling:

      def sample(x):
        """Multinomial sampling from a n-dimensional tensor."""
        vocab_size = target_modality.top_dimensionality
        samples = tf.multinomial(tf.reshape(x, [-1, vocab_size]), 1)
        reshaped_samples = tf.reshape(samples, tf.shape(x)[:-1])
        return tf.to_int32(reshaped_samples)

      def mix_gold_sampled(gold_targets, sampled_targets):
        return tf.where(
            tf.less(tf.random_uniform(tf.shape(sampled_targets)),
                    self._hparams.scheduled_sampling_gold_mixin_prob),
            gold_targets, sampled_targets)

      def sampled_results():
        """Generate scheduled sampling results."""
        sampled_targets = dp(sample, sharded_logits)
        new_targets = dp(mix_gold_sampled,
                         sharded_features["targets"], sampled_targets)
        new_features = transformed_features
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
          with tf.variable_scope(target_modality.name):
            new_features["targets"] = target_modality.targets_bottom_sharded(
                new_targets, dp)
          with tf.variable_scope("body"):
            body_outputs, losses = self.model_fn_body_sharded(new_features)
            if not isinstance(losses, dict):  # If it's a single extra loss.
              losses = {"extra": losses}
          with tf.variable_scope(target_modality.name):
            new_sharded_logits = target_modality.top_sharded(
                body_outputs, sharded_features["targets"], dp)
            training_loss = target_modality.loss_sharded(
                sharded_logits, sharded_features["targets"], dp)
            training_loss *= self._problem_hparams.loss_multiplier
          losses["training"] = training_loss
        return new_sharded_logits, losses
      # Run the above conditionally.
      prob = self._hparams.scheduled_sampling_prob
      prob *= common_layers.inverse_exp_decay(
          self._hparams.scheduled_sampling_warmup_steps, min_value=0.001)
      sharded_logits, losses = tf.cond(
          tf.less(tf.random_uniform([]), prob),
          sampled_results,
          lambda: (sharded_logits, losses))

    tf.logging.info("This model_fn took %.3f sec." % (time.time() - start_time))
    return sharded_logits, losses

  def model_fn_body_sharded(self, sharded_features):
    """Mixture-of-experts models will override this function.

    Compute model body on all datashards.

    Args:
      sharded_features: map from string to list of Tensors each with shape
         [batch, ?, ?, body_input_size]

    Returns:
      sharded_body_output:
          a list of Tensors, each with shape [batch, O, P, body_output_size]
      extra_loss: a Scalar.
    """
    with tf.name_scope("model"):
      datashard_to_features = [{
          k: v[d]
          for k, v in six.iteritems(sharded_features)
      } for d in xrange(self._num_datashards)]
      output = self._data_parallelism(
          _with_timing(self.model_fn_body, "model_fn_body"),
          datashard_to_features)
      if isinstance(output, tuple):
        losses_sharded = output[1]
        if isinstance(losses_sharded[0], dict):
          loss = {}
          for k in losses_sharded[0].keys():
            k_loss_sharded = [losses[k] for losses in losses_sharded]
            loss[k] = tf.reduce_mean(k_loss_sharded)
        else:
          loss = {"extra": tf.reduce_mean(losses_sharded)}
        output = output[0]
      else:
        loss = {"extra": 0.0}
      return output, loss

  def model_fn_body(self, features):
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

  @property
  def hparams(self):
    return self._hparams


def _warn_changed_modality_type(new_name, old_name, feature_name):
  new_type, new_name = registry.parse_modality_name(new_name)
  old_type, old_name = registry.parse_modality_name(old_name)
  if new_type != old_type:
    tf.logging.warning("%s has a designated modality type %s (%s) but has been "
                       "overridden with a modality of type %s (%s).",
                       feature_name, old_type, old_name, new_type, new_name)
