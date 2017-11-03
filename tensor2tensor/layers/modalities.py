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

"""Modalities define the bottom and top of the model (not the body)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_layers
from tensor2tensor.utils import expert_utils as eu
from tensor2tensor.utils import modality
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_symbol_modality("default")
class SymbolModality(modality.Modality):
  """Modality for sets of discrete symbols.

  Input:
    Embedding.

  Output:
    Linear transformation + softmax.
  """

  @property
  def name(self):
    return "symbol_modality_%d_%d" % (self._vocab_size, self._body_input_depth)

  @property
  def top_dimensionality(self):
    return self._vocab_size

  def _get_weights(self):
    """Create or get concatenated embedding or softmax variable.

    Returns:
       a list of self._num_shards Tensors.
    """
    num_shards = self._model_hparams.symbol_modality_num_shards
    shards = []
    for i in xrange(num_shards):
      shard_size = (self._vocab_size // num_shards) + (
          1 if i < self._vocab_size % num_shards else 0)
      var_name = "weights_%d" % i
      shards.append(
          tf.get_variable(
              var_name, [shard_size, self._body_input_depth],
              initializer=tf.random_normal_initializer(
                  0.0, self._body_input_depth**-0.5)))
    if num_shards == 1:
      ret = shards[0]
    else:
      ret = tf.concat(shards, 0)
    ret = eu.convert_gradient_to_tensor(ret)
    return ret

  def bottom_simple(self, x, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
      # Squeeze out the channels dimension.
      x = tf.squeeze(x, axis=3)
      var = self._get_weights()
      ret = tf.gather(var, x)
      if self._model_hparams.multiply_embedding_mode == "sqrt_depth":
        ret *= self._body_input_depth**0.5
      ret *= tf.expand_dims(tf.to_float(tf.not_equal(x, 0)), -1)
      return ret

  def bottom(self, x):
    self._bottom_was_called = True
    if self._model_hparams.shared_embedding_and_softmax_weights:
      return self.bottom_simple(x, "shared", reuse=None)
    else:
      return self.bottom_simple(x, "input_emb", reuse=None)

  def targets_bottom(self, x):
    if self._model_hparams.shared_embedding_and_softmax_weights:
      try:
        return self.bottom_simple(x, "shared", reuse=True)
      except ValueError:
        # perhaps there were no inputs, and this is a new variable.
        return self.bottom_simple(x, "shared", reuse=None)
    else:
      return self.bottom_simple(x, "target_emb", reuse=None)

  def top(self, body_output, _):
    """Generate logits.

    Args:
      body_output: A Tensor with shape [batch, p0, p1, body_input_depth]
    Returns:
      logits: A Tensor with shape  [batch, p0, p1, ?, vocab_size].
    """
    if self._model_hparams.shared_embedding_and_softmax_weights:
      scope_name = "shared"
      reuse = True
    else:
      scope_name = "softmax"
      reuse = False
    if self._model_hparams.symbol_modality_skip_top:
      return tf.expand_dims(body_output, 3)
    with tf.variable_scope(scope_name, reuse=reuse):
      var = self._get_weights()
      if (self._model_hparams.factored_logits and
          self._model_hparams.mode == tf.estimator.ModeKeys.TRAIN):
        # insert channels dimension
        body_output = tf.expand_dims(body_output, 3)
        logits = common_layers.FactoredTensor(body_output, var)
      else:
        shape = tf.shape(body_output)[:-1]
        body_output = tf.reshape(body_output, [-1, self._body_input_depth])
        logits = tf.matmul(body_output, var, transpose_b=True)
        logits = tf.reshape(
            logits, tf.concat([shape, [1, self._vocab_size]], 0))
      return logits


@registry.register_image_modality
class SmallImageModality(modality.Modality):
  """Performs strided conv compressions for small image data."""

  def __init__(self, model_hparams, vocab_size):
    super(SmallImageModality, self).__init__(model_hparams, vocab_size)
    self._channels = 3

  @property
  def top_dimensionality(self):
    return 256

  def bottom(self, inputs):
    with tf.variable_scope(self.name):
      inputs = common_layers.standardize_images(inputs)
      tf.summary.image("inputs", inputs, max_outputs=2)
      return common_layers.conv_block(
          inputs,
          self._body_input_depth, [((1, 1), (3, 3))],
          first_relu=False,
          padding="SAME",
          force2d=True,
          name="small_image_conv")

  def targets_bottom(self, inputs):
    with tf.variable_scope(self.name):
      # Reshape inputs to 2-d tensor and embed the RGB pixel values.
      shape = tf.shape(inputs)
      inputs = common_layers.flatten4d3d(inputs)
      ret = common_layers.embedding(
          tf.to_int32(inputs),
          self.top_dimensionality,
          self._body_input_depth,
          name="input_rgb_embedding")
      if self._model_hparams.multiply_embedding_mode == "sqrt_depth":
        ret *= self._body_input_depth**0.5
      ret = tf.reshape(ret, [shape[0], shape[1], shape[2],
                             self._body_input_depth * 3])
      return tf.layers.dense(ret, self._body_input_depth)

  def top(self, body_output, _):
    with tf.variable_scope("rgb_softmax"):
      shape = tf.shape(body_output)
      dim = body_output.get_shape().as_list()[-1] // 3
      out = tf.reshape(body_output, [shape[0], shape[1], shape[2],
                                     self._channels, dim])
      res = tf.layers.dense(out, self.top_dimensionality)
      if not tf.get_variable_scope().reuse:
        res_argmax = tf.cast(tf.argmax(res, axis=-1), tf.uint8)
        tf.summary.image("result", res_argmax, max_outputs=1)
      return res

  def loss(self, top_out, targets, weights_fn=common_layers.weights_all):
    # Call the default implementation, but weight 1.0 on 0s by default.
    # (Since we're processing images and so have no padding and some pixel 0s.)
    return super(SmallImageModality, self).loss(
        top_out, targets, weights_fn=weights_fn)


@registry.register_image_modality("default")
class ImageModality(modality.Modality):
  """Performs embedding and strided conv compressions for large image data."""

  @property
  def top_dimensionality(self):
    return 256

  def bottom(self, inputs):
    """Transform input from data space to model space.

    Perform the Xception "Entry flow", which consists of two convolutional
    filter upscalings followed by three residually connected separable
    convolution blocks.

    Args:
      inputs: A Tensor with shape [batch, ...]
    Returns:
      body_input: A Tensor with shape [batch, ?, ?, body_input_depth].
    """
    with tf.variable_scope(self.name):

      def xnet_resblock(x, filters, res_relu, name):
        with tf.variable_scope(name):
          y = common_layers.separable_conv_block(
              x,
              filters, [((1, 1), (3, 3)), ((1, 1), (3, 3))],
              first_relu=True,
              padding="SAME",
              force2d=True,
              name="sep_conv_block")
          y = common_layers.pool(y, (3, 3), "MAX", "SAME", strides=(2, 2))
          return y + common_layers.conv_block(
              x,
              filters, [((1, 1), (1, 1))],
              padding="SAME",
              strides=(2, 2),
              first_relu=res_relu,
              force2d=True,
              name="res_conv0")

      inputs = common_layers.standardize_images(inputs)
      # TODO(lukaszkaiser): summaries here don't work in multi-problem case yet.
      # tf.summary.image("inputs", inputs, max_outputs=2)
      x = common_layers.conv_block(
          inputs,
          32, [((1, 1), (3, 3))],
          first_relu=False,
          padding="SAME",
          strides=(2, 2),
          force2d=True,
          name="conv0")
      x = common_layers.conv_block(
          x, 64, [((1, 1), (3, 3))], padding="SAME", force2d=True, name="conv1")
      x = xnet_resblock(x, min(128, self._body_input_depth), True, "block0")
      x = xnet_resblock(x, min(256, self._body_input_depth), False, "block1")
      return xnet_resblock(x, self._body_input_depth, False, "block2")

  def top(self, body_output, _):
    # TODO(lukaszkaiser): work on a better way to generate large images.
    with tf.variable_scope(self.name):
      decompressed_inputs = common_layers.deconv_stride2_multistep(
          body_output,
          self._model_hparams.compress_steps,
          body_output.get_shape()[-1],
          name="deconv")
      return common_layers.conv(
          decompressed_inputs, self._vocab_size, (1, 1), padding="SAME")


@registry.register_audio_modality("default")
class AudioModality(modality.Modality):
  """Performs strided conv compressions for audio data."""

  def bottom(self, inputs):
    """Transform input from data space to model space.

    Args:
      inputs: A Tensor with shape [batch, ...]
    Returns:
      body_input: A Tensor with shape [batch, ?, ?, body_input_depth].
    """
    with tf.variable_scope(self.name):
      # TODO(aidangomez): Will need to sort out a better audio pipeline
      def xnet_resblock(x, filters, res_relu, name):
        with tf.variable_scope(name):
          # Typically audio samples are >100k samples in length and have a width
          # of 2 or 4. Mono audio has a single channel while stereo has 2.
          y = common_layers.separable_conv_block(
              x,
              filters, [((1, 1), (3, 3)), ((1, 1), (3, 3))],
              first_relu=True,
              padding="SAME",
              force2d=True,
              name="sep_conv_block")
          y = common_layers.pool(y, (3, 3), "MAX", "SAME", strides=(2, 2))
          return y + common_layers.conv_block(
              x,
              filters, [((1, 1), (1, 1))],
              padding="SAME",
              strides=(2, 2),
              first_relu=res_relu,
              force2d=True,
              name="res_conv0")

      x = tf.to_float(inputs) / 255.
      x.set_shape([None, None, None, 1])
      for i in xrange(self._model_hparams.audio_compression):
        x = xnet_resblock(x, 2**(i + 1), True, "compress_block_%d" % i)
      return xnet_resblock(x, self._body_input_depth, False,
                           "compress_block_final")


@registry.register_audio_modality
class AudioSpectralModality(modality.Modality):
  """Performs strided conv compressions for audio spectral data."""

  def bottom(self, inputs):
    """Transform input from data space to model space.

    Args:
      inputs: A Tensor with shape [batch, ...]
    Returns:
      body_input: A Tensor with shape [batch, ?, ?, body_input_depth].
    """
    with tf.variable_scope(self.name):
      # TODO(aidangomez): Will need to sort out a better audio pipeline
      def xnet_resblock(x, filters, res_relu, name):
        with tf.variable_scope(name):
          # We only stride along the length dimension to preserve the spectral
          # bins (which are tiny in dimensionality relative to length)
          y = common_layers.separable_conv_block(
              x,
              filters, [((1, 1), (3, 3)), ((1, 1), (3, 3))],
              first_relu=True,
              padding="SAME",
              force2d=True,
              name="sep_conv_block")
          y = common_layers.pool(y, (3, 3), "MAX", "SAME", strides=(2, 1))
          return y + common_layers.conv_block(
              x,
              filters, [((1, 1), (1, 1))],
              padding="SAME",
              strides=(2, 1),
              first_relu=res_relu,
              force2d=True,
              name="res_conv0")

      # Bitcast back from int32
      x = tf.bitcast(inputs, tf.float32)
      x.set_shape([None, None, None, 1])
      for i in xrange(self._model_hparams.audio_compression):
        x = xnet_resblock(x, 2**(i + 1), True, "compress_block_%d" % i)
      return xnet_resblock(x, self._body_input_depth, False,
                           "compress_block_final")


@registry.register_class_label_modality("2d")
class ClassLabelModality(modality.Modality):
  """Used for label data; if is2d=True, uses Xception flow to logits."""

  def __init__(self, model_hparams, vocab_size, is2d=True):
    super(ClassLabelModality, self).__init__(model_hparams, vocab_size)
    self._is_2d = is2d
    self._kernel = (3, 3) if is2d else (5, 1)
    self._strides = (2, 2) if is2d else (4, 1)
    self._padding = "SAME" if is2d else "LEFT"

  @property
  def name(self):
    return "class_label_modality_%d_%d" % (self._vocab_size,
                                           self._body_input_depth)

  @property
  def top_dimensionality(self):
    return self._vocab_size

  def bottom(self, x):
    with tf.variable_scope(self.name):
      return common_layers.embedding(
          x,
          self._vocab_size,
          self._body_input_depth,
          multiplier=self._body_input_depth**0.5 if
          self._model_hparams.multiply_embedding_mode == "sqrt_depth" else 1.0)

  def targets_bottom(self, x):
    with tf.variable_scope(self.name):
      return tf.zeros([tf.shape(x)[0], 1, 1, self._body_input_depth])

  def top(self, body_output, _):
    """Transform inputs from model space to target space.

    If instantiated with is2d=True, perform the Xception "Exit flow", consisting
    of a single residual block and two separable convolutional upscalings
    followed by global spatial average pooling.

    Otherwise, a single linear layer to logits.

    Args:
      body_output: A Tensor with shape [batch, ?, ?, body_output_size].
    Returns:
      a Tensors, each with shape [batch_size, ?, ?, vocab_size]
    """
    with tf.variable_scope(self.name):
      x = body_output

      # Assume input is a square with self._body_input_depth channels.
      if self._is_2d:
        length_float = tf.to_float(tf.shape(x)[1])
        length_float *= tf.to_float(tf.shape(x)[2])
        spatial_dim_float = tf.sqrt(length_float)
        spatial_dim = tf.to_int32(spatial_dim_float)
        x_depth = int(x.get_shape()[3])
        x = tf.reshape(x, [-1, spatial_dim, spatial_dim, x_depth])
        x = common_layers.conv_block_downsample(x, self._kernel, self._strides,
                                                self._padding)
        x = tf.nn.relu(x)
        x = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)

      res = tf.layers.dense(x, self._vocab_size)
      return tf.expand_dims(res, 3)

  def loss(self, top_out, targets, weights_fn=common_layers.weights_all):
    # Call the default implementation, but weight 1.0 on 0s by default.
    # (Since we're processing images and so have no padding and some pixel 0s.)
    return super(ClassLabelModality, self).loss(
        top_out, targets, weights_fn=weights_fn)


@registry.register_class_label_modality("default")
class ClassLabel1DModality(ClassLabelModality):
  """Used for label data."""

  def __init__(self, model_hparams, vocab_size):
    super(ClassLabel1DModality, self).__init__(
        model_hparams=model_hparams, vocab_size=vocab_size, is2d=False)


@registry.register_generic_modality("default")
@registry.register_audio_modality("identity")
@registry.register_image_modality("identity")
@registry.register_symbol_modality("identity")
@registry.register_class_label_modality("identity")
@registry.register_real_modality("identity")
class IdentityModality(modality.Modality):
  """Does nothing."""

  @property
  def targets_dimensionality(self):
    return self._vocab_size

  def bottom(self, x):
    return tf.to_float(x)

  def top(self, body_output, _):
    return body_output


class RealModality(modality.Modality):
  """Base class for real (i.e. float) vectors.

  * Bottom is a linear projection layer to hparams.hidden_size.
  * Top is a linear projection layer to vocab_size.
  """

  def bottom(self, x):
    with tf.variable_scope("real"):
      return tf.layers.dense(x, self._body_input_depth)

  def top(self, body_output, _):
    with tf.variable_scope("real"):
      return tf.layers.dense(body_output, self._vocab_size)

  def loss(self, top_out, targets, weights_fn=common_layers.weights_all):
    raise NotImplementedError()


@registry.register_real_modality("default")
@registry.register_real_modality("l2_loss")
class RealL2LossModality(RealModality):
  """Modality for real (i.e. float) vectors with L2 (Gaussian) loss."""

  def loss(self, top_out, targets, weights_fn=common_layers.weights_all):
    predictions = top_out
    with tf.name_scope("l2"):
      weights = weights_fn(targets)
      l2 = tf.pow(predictions - targets, 2)
      return tf.reduce_sum(l2 * weights), tf.reduce_sum(weights)


@registry.register_real_modality("log_poisson_loss")
class RealLogPoissonLossModality(RealL2LossModality):
  """Modality for real (i.e. float) vectors with log Poisson regression loss.
  """

  def bottom(self, x):
    return x

  def loss(self, top_out, targets, weights_fn=common_layers.weights_all):
    predictions = top_out
    with tf.name_scope("log_possion"):
      weights = weights_fn(targets)

      lp_loss = tf.nn.log_poisson_loss(targets, predictions)
      return tf.reduce_sum(lp_loss * weights), tf.reduce_sum(weights)


@registry.register_image_modality("identity_no_pad")
class IdentityModalityNoPad(modality.Modality):
  """Does nothing except making sure that there is no padding in cross-ent."""

  @property
  def top_dimensionality(self):
    return 256

  @property
  def targets_dimensionality(self):
    return self._vocab_size

  def bottom(self, x):
    return tf.to_float(x)

  def top(self, body_output, _):
    return body_output

  def loss(self, top_out, targets, weights_fn=common_layers.weights_all):
    # Call the default implementation, but weight 1.0 on 0s by default.
    # (Since we're processing images and so have no padding and some pixel 0s.)
    return super(IdentityModalityNoPad, self).loss(
        top_out, targets, weights_fn=weights_fn)
