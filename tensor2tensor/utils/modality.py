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
  """

  def __init__(self, model_hparams, vocab_size=None):
    # __init__ args are unused in any methods. They're maintained for
    # backwards compatibility for now. In the future, Modality classes will be
    # removed altogether.
    del model_hparams, vocab_size

  @classmethod
  def name(cls, model_hparams, vocab_size=None):
    del model_hparams, vocab_size  # unused arg
    return misc_utils.camelcase_to_snakecase(type(cls).__name__)

  @staticmethod
  def targets_weights_fn(model_hparams):
    """The weights function to use for loss and eval metrics.

    A weights function takes labels and returns a Tensor that assigns weights
    (usually either 1. or 0.) to each one.

    Common weights functions are:
      * weights_all: 1. for all labels
      * weights_nonzero: 1. for all non-zero labels (e.g. to deal with padding)

    Args:
      model_hparams: tf.HParams, model hyperparmeters.

    Returns:
      Callable: (targets) -> weights Tensor
    """
    del model_hparams  # unused arg
    return common_layers.weights_all

  @staticmethod
  def bottom(x, model_hparams, vocab_size=None):
    """Transform one shard of input.

    Args:
      x: An int32 Tensor with shape [batch, p0, p1, input_channels]
      model_hparams: tf.HParams, model hyperparmeters.
      vocab_size: int, vocabulary size.

    Returns:
      A float32 Tensor with shape [batch, p0, p1, body_input_depth]
    """
    raise NotImplementedError("Abstract Method")

  @classmethod
  def targets_bottom(cls, x, model_hparams, vocab_size=None):
    """Transform one shard of targets.

    Args:
      x: An int32 Tensor with shape [batch, p0, p1, target_channels]
      model_hparams: tf.HParams, model hyperparmeters.
      vocab_size: int, vocabulary size.

    Returns:
      A float32 Tensor with shape [batch, p0, p1, body_input_depth]
    """
    with tf.variable_scope("targets_bottom"):
      return cls.bottom(x, model_hparams, vocab_size)

  @staticmethod
  def top(body_output, targets, model_hparams, vocab_size=None):
    """Generate predictions/logits for one shard of output.

    Most classes will override this function.

    Args:
      body_output: A Tensor with shape [batch, p0, p1, body_output_depth]
      targets: A Tensor with shape [batch, p0, p1, targets_channels,
        top_dimensionality]
      model_hparams: tf.HParams, model hyperparmeters.
      vocab_size: int, vocabulary size.

    Returns:
      A Tensor of class logits.
    """
    raise NotImplementedError("Abstract Method")

  @classmethod
  def loss(cls,
           top_out,
           targets,
           model_hparams,
           vocab_size=None,
           weights_fn=None):
    """Compute loss numerator and denominator for one shard of output."""
    del vocab_size  # unused arg
    if weights_fn is None:
      weights_fn = cls.targets_weights_fn(model_hparams)
    logits = top_out
    logits = common_attention.maybe_upcast(logits, hparams=model_hparams)
    return common_layers.padded_cross_entropy(
        logits,
        targets,
        model_hparams.label_smoothing,
        weights_fn=weights_fn)
