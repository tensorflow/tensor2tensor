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

"""Modality base class - defines the bottom and top of the model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

# Dependency imports

from tensor2tensor.layers import common_layers

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
    self._vocab_size = vocab_size

  @property
  def name(self):
    camelcase_name = type(self).__name__  # DeCamelCase for TF readability.
    return re.sub("([A-Z]+)", r"_\1", camelcase_name).lower()[1:]

  @property
  def top_dimensionality(self):
    """Integer, the last dimension of the predictions (vocab size)."""
    raise NotImplementedError("Abstract Method")

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

  def bottom(self, x):
    """Transform one shard of input.

    Args:
      x: An int32 Tensor with shape [batch, p0, p1, input_channels]
    Returns:
      A float32 Tensor with shape [batch, p0, p1, body_input_depth]
    """
    raise NotImplementedError("Abstract Method")

  def bottom_sharded(self, xs, data_parallelism):
    """Transform the inputs.

    Args:
      xs: A list of num_datashards Tensors (one per shard)
        each with shape [batch, p0, p1, depth]
      data_parallelism: a expert_utils.Parallelism object
    Returns:
      shaded_body_input: A list of num_datashards Tensors, each with shape
        [batch, p0, p1, body_input_depth].
    """
    return data_parallelism(self.bottom, xs)

  def targets_bottom(self, x):
    """Transform one shard of targets.

    Args:
      x: An int32 Tensor with shape [batch, p0, p1, target_channels]
    Returns:
      A float32 Tensor with shape [batch, p0, p1, body_input_depth]
    """
    with tf.variable_scope("targets_bottom"):
      return self.bottom(x)

  def targets_bottom_sharded(self, xs, data_parallelism):
    """Transform the targets.

    Args:
      xs: A list of num_datashards Tensors (one per shard)
        each with shape [batch, p0, p1, target_channels]
      data_parallelism: a expert_utils.Parallelism object
    Returns:
      shaded_body_input: A list of num_datashards Tensors, each with shape
        [batch, p0, p1, body_input_depth].
    """
    return data_parallelism(self.targets_bottom, xs)

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

  def top_sharded(self, sharded_body_output, sharded_targets, data_parallelism):
    """Generate predictions/logits for all shards.

    Classes with cross-shard interaction will override this function.

    Args:
      sharded_body_output: A list of Tensors.
      sharded_targets: A list of Tensors.
      data_parallelism: a expert_utils.Parallelism object.
    Returns:
      sharded_logits: A list of Tensors.
    """
    return data_parallelism(self.top, sharded_body_output, sharded_targets)

  def loss(self, top_out, targets, weights_fn=common_layers.weights_nonzero):
    """Compute loss numerator and denominator for one shard of output."""
    logits = top_out
    return common_layers.padded_cross_entropy(
        logits,
        targets,
        self._model_hparams.label_smoothing,
        weights_fn=weights_fn)

  def loss_sharded(self, sharded_top_out, sharded_targets, data_parallelism):
    """Compute loss for all shards."""
    sharded_loss_num, sharded_loss_den = data_parallelism(
        self.loss, sharded_top_out, sharded_targets)
    loss = tf.add_n(sharded_loss_num) / tf.maximum(1.0,
                                                   tf.add_n(sharded_loss_den))
    return loss
