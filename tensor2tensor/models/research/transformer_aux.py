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

"""Transformer with auxiliary losses from https://arxiv.org/abs/1803.00144."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


def shift_and_pad(tensor, shift, axis=0):
  """Shifts and pads with zero along an axis.

  Example:
    shift_and_pad([1, 2, 3, 4], 2)  --> [0, 0, 1, 2]
    shift_and_pad([1, 2, 3, 4], -2) --> [3, 4, 0, 0]

  Args:
    tensor: Tensor; to be shifted and padded.
    shift: int; number of positions to shift by.
    axis: int; along which axis to shift and pad.

  Returns:
    A Tensor with the same shape as the input tensor.
  """
  shape = tensor.shape
  rank = len(shape)
  assert 0 <= abs(axis) < rank

  length = int(shape[axis])
  assert 0 <= abs(shift) < length

  paddings = [(0, 0)] * rank
  begin = [0] * rank
  size = [-1] * rank

  if shift > 0:
    paddings[axis] = (shift, 0)
    size[axis] = length - shift
  elif shift < 0:
    paddings[axis] = (0, -shift)
    begin[axis] = -shift

  ret = tf.pad(tf.slice(tensor, begin, size), paddings)

  return ret


@registry.register_model
class TransformerAux(transformer.Transformer):
  """Attention net. See file docstring."""

  def _extract_shift_values(self):
    """Parses the shift string.

    The hparams should contain the key shift_values, which maps to a
    comma-separated string of integers. These integers specify the number of
    timesteps to predict/reconstruct to compute auxiliary losses.

    For instance, "-4,2,6" means to reconstruct the target 4 steps before and
    predict the targets 2 steps and 6 steps ahead.

    Returns:
      List of int != 0 shift values to compute the auxiliary losses.
    """
    shift_values_str = self._hparams.get("shift_values", "")
    shift_values = [int(x) for x in shift_values_str.split(",")]

    tf.logging.info(
        "Computing auxiliary losses for the following shifts: %s",
        shift_values)

    return shift_values

  def auxiliary_loss(self, body_output, features, shift):
    """Auxiliary predict loss.

    Args:
      body_output: Tensor with shape [batch_size, decoder_length, hidden_dim].
      features: Map of features to the model. Must contain the following:
          "targets": Target decoder outputs.
              [batch_size, decoder_length, 1, hidden_dim]
      shift: int != 0, amount to shift/pad the target sequence.
        If shift > 0, it represents the number of previous timesteps to
        reconstruct; if shift < 0, it represents the number of future timesteps
        to predict.

    Returns:
      A 2-tuple of the numerator and denominator of the cross-entropy loss.

    Raises:
      ValueError: if features does not contain a targets_raw tensor.
    """
    assert isinstance(shift, int) and shift != 0
    name = "reconst_%d" % shift if shift > 0 else "predict_%d" % abs(shift)

    if features and "targets_raw" in features:
      targets = features["targets_raw"]
      targets = common_layers.flatten4d3d(targets)
    else:
      raise ValueError(
          "Feature map must contain a targets_raw tensor.")

    with tf.variable_scope(name):
      logits = self.top(body_output, features)
      labels = shift_and_pad(targets, shift, axis=1)
      return common_layers.padded_cross_entropy(
          logits,
          labels,
          self._hparams.label_smoothing)

  def body(self, features):
    """Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs.
              [batch_size, input_length, 1, hidden_dim].
          "targets": Target decoder outputs.
              [batch_size, target_length, 1, hidden_dim]
          "target_space_id": A scalar int from data_generators.problem.SpaceID.

    Returns:
      A 2-tuple containing:
          Logit tensor. [batch_size, decoder_length, vocab_size]
          Map of keys to loss tensors. Should contain the following:
              "training": Training loss (shift == 0).
              "auxiliary": Auxiliary loss (shift != 0).
    """
    output = super(TransformerAux, self).body(features)
    output, losses = self._normalize_body_output(output)

    aux = 0.0
    for shift in self._extract_shift_values():
      loss_num, loss_den = self.auxiliary_loss(output, features, shift)
      aux += loss_num / loss_den
    losses["auxiliary"] = aux

    return output, losses


@registry.register_hparams
def transformer_aux_base():
  """Set of hyperparameters."""
  hparams = transformer.transformer_base()
  hparams.shared_embedding_and_softmax_weights = False
  hparams.add_hparam("shift_values", "1,2,3,4")
  return hparams


@registry.register_hparams
def transformer_aux_tiny():
  """Set of hyperparameters."""
  hparams = transformer.transformer_tiny()
  hparams.shared_embedding_and_softmax_weights = False
  hparams.add_hparam("shift_values", "1,2")
  return hparams
