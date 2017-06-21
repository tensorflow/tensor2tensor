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

"""Modality base class - defines the bottom and top of the model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

# Dependency imports

from tensor2tensor.models import common_layers

import tensorflow as tf


class Modality(object):
  """Abstract Modality class for data transformations.

  An abstract class representing modalities for transforming data to a space
  interpretable by sequence models. It has 3 functions:
  * bottom:  called on inputs entering the model.
  * targets_bottom: called on targets entering the model (e.g., the decoder).
  * top:   called on targets to generate predictions.

  For example, think about a modality for images. The inputs_bottom function
  represents the part of the model applied to an incoming image, e.g., an entry
  flow of a convolutional network. The targets_top function represents the top
  part of a model that is generating images, e.g., a PixelCNN network. The final
  function targets_bottom represents the auto-regressive part of the network.
  It is applied to the already-generated part of an image, which is given to
  the decoder to generate the next part. In some cases, e.g., for text, it is
  the same as the inputs_bottom function, as that is the default we use. But,
  e.g., for images, a different function might be needed to regress properly.

  All 3 functions have simple and sharded versions. A sub-class only needs
  to implement the simple version, the default sharding will be used then.
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
    """Transform one shard of output.

    Most classes will override this function.

    Args:
      body_output: A Tensor with shape [batch, p0, p1, body_output_depth]
      targets: A Tensor with shape [batch, p0, p1, targets_channels,
        top_dimensionality]
    Returns:
      A Tensor of class logits.
    """
    raise NotImplementedError("Abstract Method")

  def top_sharded(self,
                  sharded_body_output,
                  sharded_targets,
                  data_parallelism,
                  weights_fn=common_layers.weights_nonzero):
    """Transform all shards of targets.

    Classes with cross-shard interaction will override this function.

    Args:
      sharded_body_output: A list of Tensors.
      sharded_targets: A list of Tensors.
      data_parallelism: a expert_utils.Parallelism object.
      weights_fn: function from targets to target weights.
    Returns:
      shaded_logits: A list of Tensors.
      training_loss: a Scalar.
    """
    sharded_logits = data_parallelism(self.top, sharded_body_output,
                                      sharded_targets)
    loss_num, loss_den = data_parallelism(
        common_layers.padded_cross_entropy,
        sharded_logits,
        sharded_targets,
        self._model_hparams.label_smoothing,
        weights_fn=weights_fn)
    loss = tf.add_n(loss_num) / tf.maximum(1.0, tf.add_n(loss_den))
    return sharded_logits, loss
