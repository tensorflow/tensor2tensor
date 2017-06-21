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

"""Modalities define the bottom and top of the model (not the body)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.models import common_layers
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
    ret = eu.ConvertGradientToTensor(ret)
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
    if self._model_hparams.shared_embedding_and_softmax_weights:
      return self.bottom_simple(x, "shared", reuse=None)
    else:
      return self.bottom_simple(x, "input_emb", reuse=None)

  def targets_bottom(self, x):
    if self._model_hparams.shared_embedding_and_softmax_weights:
      return self.bottom_simple(x, "shared", reuse=True)
    else:
      return self.bottom_simple(x, "target_emb", reuse=None)

  def top(self, body_output, targets):
    """Generate logits.

    Args:
      body_output: A Tensor with shape [batch, p0, p1, body_input_depth]
      targets: A Tensor with shape [batch, p0, p1, 1]
    Returns:
      logits: A Tensor with shape  [batch, p0, p1, ?, vocab_size].
    """
    if self._model_hparams.shared_embedding_and_softmax_weights:
      scope_name = "shared"
      reuse = True
    else:
      scope_name = "softmax"
      reuse = False
    with tf.variable_scope(scope_name, reuse=reuse):
      var = self._get_weights()
      shape = tf.shape(body_output)[:-1]
      body_output = tf.reshape(body_output, [-1, self._body_input_depth])
      logits = tf.matmul(body_output, var, transpose_b=True)
      logits = tf.reshape(logits, tf.concat([shape, [self._vocab_size]], 0))
      # insert a channels dimension
      return tf.expand_dims(logits, 3)


@registry.register_image_modality
class SmallImageModality(modality.Modality):
  """Performs strided conv compressions for small image data."""

  @property
  def top_dimensionality(self):
    return 256

  def bottom(self, inputs):
    with tf.variable_scope(self.name):
      inputs = common_layers.standardize_images(inputs)
      # TODO(lukaszkaiser): summaries here don't work in multi-problem case yet.
      # tf.summary.image("inputs", inputs, max_outputs=2)
      if self._model_hparams.compress_steps > 0:
        strides = (2, 2)
      else:
        strides = (1, 1)
      return common_layers.conv_block(
          inputs,
          self._body_input_depth, [((1, 1), (3, 3))],
          first_relu=False,
          strides=strides,
          padding="SAME",
          force2d=True,
          name="small_image_conv")

  def targets_bottom(self, inputs):
    with tf.variable_scope(self.name):
      # Reshape inputs to 2-d tensor and embed the RGB pixel values.
      inputs = common_layers.flatten4d3d(inputs)
      ret = common_layers.embedding(
          inputs,
          self.top_dimensionality,
          self._body_input_depth,
          name="input_rgb_embedding")
      if self._model_hparams.multiply_embedding_mode == "sqrt_depth":
        ret *= self._body_input_depth**0.5
      return ret

  def top(self, body_output, _):
    with tf.variable_scope("rgb_softmax"):
      var = tf.get_variable(
          "output_rgb_embedding",
          [self.top_dimensionality, self._body_input_depth],
          initializer=tf.random_normal_initializer(0.0, self._body_input_depth
                                                   **-0.5))
      body_output = tf.reshape(body_output, [-1, self._body_input_depth])
      logits = tf.matmul(body_output, var, transpose_b=True)
      # Reshape logits to conform to CIFAR image shapes (32 by 32 by 3)
      logits = tf.reshape(logits, [-1, 32, 32, 3, 256])

      return logits

  def top_sharded(self,
                  sharded_body_output,
                  sharded_targets,
                  data_parallelism,
                  weights_fn=common_layers.weights_all):
    # Call the default implementation, but weight 1.0 on 0s by default.
    # (Since we're processing images and so have no padding and some pixel 0s.)
    return super(SmallImageModality, self).top_sharded(
        sharded_body_output,
        sharded_targets,
        data_parallelism,
        weights_fn=weights_fn)


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


@registry.register_class_label_modality("default")
class ClassLabelModality(modality.Modality):
  """Used for label data."""

  def __init__(self, model_hparams, vocab_size, is2d=False):
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

    Perform the Xception "Exit flow", consisting of a single residual block and
    two separable convolutional upscalings followed by global spatial average
    pooling.

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
        spatial_dim_float = tf.sqrt(length_float)
        spatial_dim = tf.to_int32(spatial_dim_float)
        x = tf.reshape(x,
                       [-1, spatial_dim, spatial_dim, self._body_input_depth])
      x = common_layers.conv_block_downsample(x, self._kernel, self._strides,
                                              self._padding)
      x = tf.nn.relu(x)
      x = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
      res = common_layers.conv(x, self._vocab_size, (1, 1))
      return tf.expand_dims(res, 3)

  def top_sharded(self,
                  sharded_body_output,
                  sharded_targets,
                  data_parallelism,
                  weights_fn=common_layers.weights_all):
    # Call the default implementation, but weight 1.0 on 0s by default.
    # (Since we're processing images and so have no padding and some labels 0.)
    return super(ClassLabelModality, self).top_sharded(
        sharded_body_output,
        sharded_targets,
        data_parallelism,
        weights_fn=weights_fn)


@registry.register_class_label_modality("class_label_2d")
class ClassLabel2DModality(ClassLabelModality):
  """Used for label data."""

  def __init__(self, model_hparams, vocab_size):
    super(ClassLabel2DModality, self).__init__(
        model_hparams=model_hparams, vocab_size=vocab_size, is2d=True)


@registry.register_generic_modality("default")
@registry.register_audio_modality("identity")
@registry.register_image_modality("identity")
@registry.register_symbol_modality("identity")
@registry.register_class_label_modality("identity")
class IdentityModality(modality.Modality):
  """Does nothing."""

  @property
  def targets_dimensionality(self):
    return self._vocab_size

  def inputs_bottom_simple(self, inputs):
    return tf.to_float(inputs)

  def targets_top_simple(self, body_output, _):
    return body_output
