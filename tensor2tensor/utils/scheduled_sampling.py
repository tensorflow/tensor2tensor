# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""Scheduled Sampling.

This module implemented scheduled sampling as described in (Bengio et al, 2015).
The entry points are two functions,

`sequential_scheduled_sampling_for_t2tmodel()`:
  scheduled sampling adapted to instances of T2TModel.

`sequential_scheduled_sampling()`:
  raw implementation of scheduled sampling. May be used independent of T2T.

**WARNING** This code is VERY slow. Its runtime is at least O(n^2) for
sequences of length n. For models with self-attention, its runtime is O(n^3).

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from tensor2tensor.layers import common_layers
import tensorflow.compat.v1 as tf

from tensorflow.python.ops import inplace_ops  # pylint: disable=g-direct-tensorflow-import


def sequential_scheduled_sampling_for_t2tmodel(t2tmodel, features):
  """Schedule Sampling for T2TModels.

  Args:
    t2tmodel: T2TModel instance.
    features: {str: Tensor}. Input features.

  Returns:
    ss_logits: [batch_size, seq_len, 1, 1, vocab_size].
    losses_dict: {str: scalar Tensor}. Losses to minimize.
  """
  targets = features["targets"]
  targets_size = common_layers.shape_list(targets)
  batch_size = targets_size[0]
  seq_len = targets_size[1]
  targets = tf.reshape(targets, [batch_size, seq_len])

  adapter = ScheduledSamplingAdapter(t2tmodel, features)
  ss_tokens, ss_logits, losses_dict = sequential_scheduled_sampling(
      infer_fn=adapter.infer_fn,
      mix_fn=adapter.mix_fn,
      loss_fn=adapter.loss_fn,
      targets=targets)

  _ = ss_tokens  # unused.
  targets_vocab_size = t2tmodel.problem_hparams.vocab_size["targets"]
  ss_logits = tf.reshape(ss_logits,
                         [batch_size, seq_len, 1, 1, targets_vocab_size])

  return ss_logits, losses_dict


def sequential_scheduled_sampling(infer_fn, mix_fn, loss_fn, targets):
  """Scheduled Sampling.

  Args:
    infer_fn: Function. Computes logits for all timesteps.
    mix_fn: Function. Mixes gold and sample tokens.
    loss_fn: Function. Computes loss between gold tokens and logits.
    targets: Tensor of shape [batch_size, seq_len]. Gold tokens.

  Returns:
    ss_tokens: Tensor of shape [batch_size, seq_len]. Scheduled sampling tokens.
    ss_logits: Tensor of shape [batch_size, seq_len, vocab_size]. Logits for
      next token when conditioning on ss_tokens.
    losses_dict: {str: scalar Tensor}. Losses to optimize.
  """
  targets_shape = common_layers.shape_list(targets)
  batch_size = targets_shape[0]
  seq_len = targets_shape[1]

  if not targets.shape.is_fully_defined():
    # TODO(duckworthd): When running on GPU, I get the following error. Solve
    # it to enable use on other devices.
    #
    #   Cannot use 'Identity_186' as input to
    #   'transformer/parallel_0_7/transformer/transformer/symbol_modality_16282_512/shared/convert_gradient_to_tensor_HBc3xYw22Mw'
    #   because 'Identity_186' is in a while loop.

    raise ValueError(
        "The following code only works on TPU. As targets.shape isn't fully "
        "defined, I am assuming you are using a different device.")

  def cond_fn(i, ss_tokens):
    """True if i < seq_len."""
    _ = ss_tokens
    return i < seq_len

  def body_fn(i, ss_tokens):
    """Constructs conditioning tokens for scheduled sampling."""
    # next_token_logits depends on timesteps 0...i-1.
    #
    # [batch_size, seq_len] -> [batch_size, seq_len, vocab_size]
    ss_tokens_logits = infer_fn(ss_tokens)

    # Same as 'next_token_logits = ss_tokens_logits[:, i, :]'.
    vocab_size = common_layers.shape_list(ss_tokens_logits)[2]
    next_token_logits = tf.slice(
        ss_tokens_logits, begin=[0, i, 0], size=[batch_size, 1, vocab_size])
    next_token_logits = tf.squeeze(next_token_logits, axis=[1])

    # [batch_size, vocab_size] -> [batch_size]
    sampled_next_tokens = _sample_next_tokens(next_token_logits)

    # Same as 'gold_next_tokens = targets[:, i]'.
    gold_next_tokens = tf.slice(targets, begin=[0, i], size=[batch_size, 1])
    gold_next_tokens = tf.squeeze(gold_next_tokens, axis=[1])

    next_tokens = mix_fn(gold_next_tokens, sampled_next_tokens)
    ss_tokens = _update_timestep(ss_tokens, timestep=i, values=next_tokens)

    return i+1, tf.stop_gradient(ss_tokens)

  # tf.while_loop() over all timesteps. Generate scheduled sampling tokens.
  i = 0
  ss_tokens = tf.zeros([batch_size, seq_len], dtype=tf.int32)
  i, ss_tokens = tf.while_loop(cond_fn, body_fn, [i, ss_tokens])

  ss_logits = infer_fn(ss_tokens)
  return ss_tokens, ss_logits, loss_fn(targets, ss_logits)


def _mix_tokens(p_sample, gold_targets, sampled_targets):
  """Interleave sampled and gold tokens randomly.

  Args:
    p_sample: float in [0, 1]. Probability a token will come from
      'sampled_targets'. 0 means all-gold, 1 means all-sampled.
    gold_targets: Tensor. Gold token IDs.
    sampled_targets: Tensor. Sampled token IDs. Same shape as 'gold_targets'.

  Returns:
    Tensor of same shape as 'gold_targets' containing a mix of tokens from
    'gold_targets' and 'sampled_targets'.
  """
  targets_shape = common_layers.shape_list(sampled_targets)
  return tf.where(
      tf.less(tf.random_uniform(targets_shape), p_sample),
      sampled_targets, gold_targets)


def _sample_next_tokens(logits):
  """Sample tokens for next timestep."""
  batch_size = common_layers.shape_list(logits)[0]
  next_tokens = tf.random.categorical(logits, 1)
  next_tokens = tf.cast(next_tokens, tf.int32)
  next_tokens = tf.reshape(next_tokens, [batch_size])
  return next_tokens


def _update_timestep(x, timestep, values):
  """Set x[:, timestep] = values.

  This operation is **NOT** differentiable.

  Args:
    x: Tensor of shape [batch_size, seq_len, ...]
    timestep: int or scalar Tensor. Index to update in x.
    values: Tensor of shape [batch_size, ...]. New values for x[:, i].

  Returns:
    Copy of 'x' after setting x[:, timestep] = values.
  """
  perm = range(x.shape.ndims)
  perm[0], perm[1] = perm[1], perm[0]
  x = tf.transpose(x, perm)
  x = inplace_ops.alias_inplace_update(x, timestep, values)
  x = tf.transpose(x, perm)
  return x


def inverse_decay_mix_prob(warmup_schedule_name, p_max, num_warmup_steps):
  """Interpolate from 0.001 to 'p_max' over 'num_warmup_steps'."""
  warmup_schedule_fn = {
      "exp": common_layers.inverse_exp_decay,
      "linear": common_layers.inverse_lin_decay,
      "sigmoid": common_layers.inverse_sigmoid_decay,
  }[warmup_schedule_name]
  return p_max * warmup_schedule_fn(num_warmup_steps, min_value=0.001)


class ScheduledSamplingAdapter(object):
  """Adapts T2TModel for sequential_scheduled_sampling()."""

  def __init__(self, t2tmodel, features):
    self._t2tmodel = t2tmodel
    self._features = features

    hparams = self._t2tmodel.hparams
    assert hparams.mode == tf.estimator.ModeKeys.TRAIN, hparams.mode

  def infer_fn(self, partial_targets):
    """Computes logits for all timesteps.

    Args:
      partial_targets: [batch_size, seq_len]. Targets to condition on.

    Returns:
      next_token_logits: [batch_size, seq_len, vocab_size]
    """
    batch_size, seq_len = common_layers.shape_list(partial_targets)
    partial_targets = tf.reshape(partial_targets, [batch_size, seq_len, 1, 1])
    features = copy.copy(self._features)
    features["targets"] = partial_targets

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      transformed_features = self._t2tmodel.bottom(features)

      with tf.variable_scope("body"):
        body_outputs, losses = self._t2tmodel._normalize_body_output(  # pylint: disable=protected-access
            self._t2tmodel.body(transformed_features))
        assert losses == {"extra": 0.0}, (
            "Auxiliary losses are not propagated in this code. %s"
            % (losses,))

      logits = self._t2tmodel.top(body_outputs, features)

    vocab_size = self._t2tmodel.problem_hparams.vocab_size["targets"]
    logits = tf.reshape(logits, [batch_size, seq_len, vocab_size])
    return logits

  def mix_fn(self, gold_tokens, sampled_tokens):
    """Mixes gold and sampled tokens randomly."""
    hparams = self._t2tmodel.hparams
    p_sample = inverse_decay_mix_prob(
        hparams.scheduled_sampling_warmup_schedule,
        hparams.scheduled_sampling_gold_mixin_prob,
        hparams.scheduled_sampling_warmup_steps)
    return _mix_tokens(
        p_sample=p_sample,
        gold_targets=gold_tokens,
        sampled_targets=sampled_tokens)

  def loss_fn(self, targets, logits):
    """Constructs loss dict.

    Args:
      targets: [batch_size, seq_len]
      logits: [batch_size, seq_len, vocab_size]

    Returns:
      {str: Tensor of shape []}. Losses.
    """
    batch_size, seq_len, vocab_size = common_layers.shape_list(logits)
    targets = tf.reshape(targets, [batch_size, seq_len, 1, 1])
    logits = tf.reshape(logits, [batch_size, seq_len, 1, 1, vocab_size])
    features = copy.copy(self._features)
    features["targets"] = targets

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      losses = {
          "training": self._t2tmodel.loss(logits, features),
      }

    return losses
