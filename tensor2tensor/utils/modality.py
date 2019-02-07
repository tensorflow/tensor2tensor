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

"""Modality base class - defines the bottom and top of the model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import misc_utils

import tensorflow as tf


class Modality(object):
  """Abstract Modality class for data transformations.

  An abstract class representing modalities for transforming data to a space
  interpretable by T2T models. It has 4 functions:
  * bottom: called on inputs entering the model.
  * targets_bottom: called on targets entering the model (e.g., the decoder).
  * top: called on model outputs to generate predictions (e.g., logits).
  * loss: called on predictions (outputs of top) and targets.

  For example, think about a modality for images:
  * `bottom` represents the part of the model applied to an incoming image,
    e.g., an entry flow of a convolutional network.
  * `top` represents the top part of a model that is generating images, e.g., a
    PixelCNN network.
  * `targets_bottom` represents the auto-regressive part of the network.  It is
    applied to the already-generated part of an image, which is given to the
    decoder to generate the next part. In some cases, e.g., for text, it is the
    same as the `bottom` function, and that is the default we use. But, e.g.,
    for images, a different function might be needed to regress properly.
  * `loss` would compare the generated image to the target image and score it.

  All the functions have simple and sharded versions. A sub-class only needs to
  implement the simple version, the default sharding will be used then.
  """

  def __init__(self, model_hparams, vocab_size=None):
    self._model_hparams = model_hparams
    if vocab_size is not None and hasattr(model_hparams, "vocab_divisor"):
      vocab_size += (0 - vocab_size) % model_hparams.vocab_divisor
    self._vocab_size = vocab_size

  @property
  def name(self):
    return misc_utils.camelcase_to_snakecase(type(self).__name__)

  @property
  def top_dimensionality(self):
    """Integer, the last dimension of the predictions (vocab size)."""
    return self._vocab_size

  @property
  def _body_input_depth(self):
    return self._model_hparams.hidden_size

  @property
  def top_is_pointwise(self):
    """Whether the top mapping of the modality is pointwise.

    An example of a pointwise top mapping is a linear layer followed by
    a softmax. Given a tensor [batch, length, height, depth] it operates
    only on the last axis, on every point in [batch, length, height] fully
    independently. In contrast, a classifier that first averages over length
    and height is not pointwise, as it depends on the whole field. It is useful
    to know if a top is pointwise to speed up decoding in certain models.

    Returns:
      A Boolean, True if the modality is pointwise, False otherwise (default).
    """
    return False

  @property
  def targets_weights_fn(self):
    """The weights function to use for loss and eval metrics.

    A weights function takes labels and returns a Tensor that assigns weights
    (usually either 1. or 0.) to each one.

    Common weights functions are:
      * weights_all: 1. for all labels
      * weights_nonzero: 1. for all non-zero labels (e.g. to deal with padding)

    Returns:
      Callable: (targets) -> weights Tensor
    """
    return common_layers.weights_all

  def bottom(self, x):
    """Transform one shard of input.

    Args:
      x: An int32 Tensor with shape [batch, p0, p1, input_channels]
    Returns:
      A float32 Tensor with shape [batch, p0, p1, body_input_depth]
    """
    raise NotImplementedError("Abstract Method")

  def targets_bottom(self, x):
    """Transform one shard of targets.

    Args:
      x: An int32 Tensor with shape [batch, p0, p1, target_channels]
    Returns:
      A float32 Tensor with shape [batch, p0, p1, body_input_depth]
    """
    with tf.variable_scope("targets_bottom"):
      return self.bottom(x)

  def top(self, body_output, targets):
    """Generate predictions/logits for one shard of output.

    Most classes will override this function.

    Args:
      body_output: A Tensor with shape [batch, p0, p1, body_output_depth]
      targets: A Tensor with shape [batch, p0, p1, targets_channels,
        top_dimensionality]
    Returns:
      A Tensor of class logits.
    """
    raise NotImplementedError("Abstract Method")

  def loss(self, top_out, targets, weights_fn=None):
    """Compute loss numerator and denominator for one shard of output."""
    logits = top_out
    if weights_fn is None:
      weights_fn = self.targets_weights_fn
    logits = common_attention.maybe_upcast(logits, hparams=self._model_hparams)
    return common_layers.padded_cross_entropy(
        logits,
        targets,
        self._model_hparams.label_smoothing,
        weights_fn=weights_fn)

  @property
  def is_class_modality(self):
    return self.name.startswith("class_label")
