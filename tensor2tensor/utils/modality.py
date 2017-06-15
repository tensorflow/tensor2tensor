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

import re

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.models import common_layers
from tensor2tensor.utils import expert_utils as eu
import tensorflow as tf


class Modality(object):
  """Abstract Modality class for data transformations.

  An abstract class representing modalities for transforming data to a space
  interpretable by sequence models. It has 3 functions:
  * inputs_bottom:  called on inputs entering the model.
  * targets_bottom: called on targets entering the model (e.g., the decoder).
  * targets_top :   called on targets to generate predictions.

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

  def __init__(self, model_hparams):
    self._model_hparams = model_hparams

  @property
  def name(self):
    camelcase_name = type(self).__name__  # DeCamelCase for TF readability.
    return re.sub("([A-Z]+)", r"_\1", camelcase_name).lower()[1:]

  @property
  def targets_dimensionality(self):
    """Integer, the last dimension of the predictions (vocab size)."""
    raise NotImplementedError("Abstract Method")

  @property
  def _body_input_depth(self):
    return self._model_hparams.hidden_size

  def inputs_bottom_simple(self, x):
    """Transform one shard of input.

    Args:
      x: An int32 Tensor with shape [batch, p0, p1, input_channels]
    Returns:
      A float32 Tensor with shape [batch, p0, p1, body_input_depth]
    """
    raise NotImplementedError("Abstract Method")

  def inputs_bottom_sharded(self, xs, data_parallelism):
    """Transform the inputs.

    Args:
      xs: A list of num_datashards Tensors (one per shard)
        each with shape [batch, p0, p1, depth]
      data_parallelism: a expert_utils.Parallelism object
    Returns:
      shaded_body_input: A list of num_datashards Tensors, each with shape
        [batch, p0, p1, body_input_depth].
    """
    return data_parallelism(self.inputs_bottom_simple, xs)

  def targets_bottom_simple(self, x):
    """Transform one shard of targets.

    Args:
      x: An int32 Tensor with shape [batch, p0, p1, target_channels]
    Returns:
      A float32 Tensor with shape [batch, p0, p1, body_input_depth]
    """
    with tf.variable_scope("targets_bottom_simple"):
      return self.inputs_bottom_simple(x)

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
    return data_parallelism(self.targets_bottom_simple, xs)

  def targets_top_simple(self, body_output, targets):
    """Transform one shard of output.

    Most classes will override this function.

    Args:
      body_output: A Tensor with shape [batch, p0, p1, body_output_depth]
      targets: A Tensor with shape [batch, p0, p1, targets_channels,
        targets_dimensionality]
    Returns:
      A Tensor of class logits.
    """
    raise NotImplementedError("Abstract Method")

  def targets_top_sharded(self,
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
    sharded_logits = data_parallelism(self.targets_top_simple,
                                      sharded_body_output, sharded_targets)
    loss_num, loss_den = data_parallelism(
        common_layers.padded_cross_entropy,
        sharded_logits,
        sharded_targets,
        self._model_hparams.label_smoothing,
        weights_fn=weights_fn)
    loss = tf.add_n(loss_num) / tf.maximum(1.0, tf.add_n(loss_den))
    return sharded_logits, loss


class SymbolModality(Modality):
  """Modality for sets of discrete symbols.

  Input:
    Embedding.

  Output:
    Linear transformation + softmax.
  """

  def __init__(self, model_hparams, vocab_size):
    super(SymbolModality, self).__init__(model_hparams)
    self._vocab_size = vocab_size
    self._datashard_device_to_embedding = None
    self._datashard_device_to_softmax_weights = None

  @property
  def name(self):
    return "symbol_modality_%d_%d" % (self._vocab_size, self._body_input_depth)

  @property
  def targets_dimensionality(self):
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

  def inputs_bottom_simple(self, x):
    if self._model_hparams.shared_embedding_and_softmax_weights:
      return self.bottom_simple(x, "shared", reuse=None)
    else:
      return self.bottom_simple(x, "input_emb", reuse=None)

  def targets_bottom_simple(self, x):
    if self._model_hparams.shared_embedding_and_softmax_weights:
      return self.bottom_simple(x, "shared", reuse=True)
    else:
      return self.bottom_simple(x, "target_emb", reuse=None)

  def targets_top_simple(self, body_output, targets):
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


class SmallImageModality(Modality):
  """Performs strided conv compressions for small image data."""

  def __init__(self, model_hparams):
    super(SmallImageModality, self).__init__(model_hparams)

  @property
  def targets_dimensionality(self):
    return 256

  def inputs_bottom_simple(self, inputs):
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

  def targets_bottom_simple(self, inputs):
    with tf.variable_scope(self.name):
      inputs = common_layers.standardize_images(inputs)
      if self._model_hparams.compress_steps > 0:
        kernel, strides = (2, 2), (2, 2)  # Crucial to not leak!
      else:
        kernel, strides = (1, 1), (1, 1)
      return common_layers.conv_block(
          inputs,
          self._body_input_depth, [((1, 1), kernel)],
          first_relu=False,
          strides=strides,
          force2d=True,
          name="small_image_conv")

  def targets_top_simple(self, body_output, targets):
    with tf.variable_scope(self.name):
      if self._model_hparams.compress_steps == 0:
        targets_shape = tf.shape(targets)
        channels = targets.shape.as_list()[-1]
        outputs = tf.layers.dense(body_output, 256 * channels)
        return tf.reshape(outputs, [
            targets_shape[0], targets_shape[1], targets_shape[2], 3, 256
        ])
      dilations_kernels = [((1, 1), (3, 1)), ((2, 1), (3, 1)), ((4, 1), (3, 1))]
      return common_layers.decompress_seqcnn(
          body_output, targets, 256, dilations_kernels, 2, is_2d=True)

  def targets_top_sharded(self,
                          sharded_body_output,
                          sharded_targets,
                          data_parallelism,
                          weights_fn=common_layers.weights_all):
    # Call the default implementation, but weight 1.0 on 0s by default.
    # (Since we're processing images and so have no padding and some pixel 0s.)
    return super(SmallImageModality, self).targets_top_sharded(
        sharded_body_output,
        sharded_targets,
        data_parallelism,
        weights_fn=weights_fn)


class ImageModality(Modality):
  """Performs embedding and strided conv compressions for large image data."""

  def __init__(self, model_hparams):
    super(ImageModality, self).__init__(model_hparams)

  @property
  def targets_dimensionality(self):
    return 256

  def inputs_bottom_simple(self, inputs):
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

  def targets_top_simple(self, body_output, _):
    # TODO(lukaszkaiser): work on a better way to generate large images.
    with tf.variable_scope(self.name):
      decompressed_inputs = common_layers.deconv_stride2_multistep(
          body_output,
          self._model_hparams.compress_steps,
          body_output.get_shape()[-1],
          name="deconv")
      return common_layers.conv(
          decompressed_inputs, self._vocab_size, (1, 1), padding="SAME")


class AudioModality(Modality):
  """Performs strided conv compressions for audio data."""

  def __init__(self, model_hparams):
    super(AudioModality, self).__init__(model_hparams)

  def inputs_bottom_simple(self, inputs):
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


class AudioSpectralModality(Modality):
  """Performs strided conv compressions for audio spectral data."""

  def __init__(self, model_hparams):
    super(AudioSpectralModality, self).__init__(model_hparams)

  def inputs_bottom_simple(self, inputs):
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


class ClassLabelModality(Modality):
  """Used for label data."""

  def __init__(self, model_hparams, vocab_size, is2d=False):
    super(ClassLabelModality, self).__init__(model_hparams)
    self._vocab_size = vocab_size
    self._is_2d = is2d
    self._kernel = (3, 3) if is2d else (5, 1)
    self._strides = (2, 2) if is2d else (4, 1)
    self._padding = "SAME" if is2d else "LEFT"

  @property
  def name(self):
    return "class_label_modality_%d_%d" % (self._vocab_size,
                                           self._body_input_depth)

  @property
  def targets_dimensionality(self):
    return self._vocab_size

  def inputs_bottom_simple(self, x):
    with tf.variable_scope(self.name):
      return common_layers.embedding(
          x,
          self._vocab_size,
          self._body_input_depth,
          multiplier=self._body_input_depth**0.5 if
          self._model_hparams.multiply_embedding_mode == "sqrt_depth" else 1.0)

  def targets_bottom_simple(self, x):
    with tf.variable_scope(self.name):
      return tf.zeros([tf.shape(x)[0], 1, 1, self._body_input_depth])

  def targets_top_simple(self, body_output, _):
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
        x = tf.reshape(x, [-1, spatial_dim, spatial_dim,
                           self._body_input_depth])
      x = common_layers.conv_block_downsample(x, self._kernel, self._strides,
                                              self._padding)
      x = tf.nn.relu(x)
      x = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
      res = common_layers.conv(x, self._vocab_size, (1, 1))
      return tf.expand_dims(res, 3)

  def targets_top_sharded(self,
                          sharded_body_output,
                          sharded_targets,
                          data_parallelism,
                          weights_fn=common_layers.weights_all):
    # Call the default implementation, but weight 1.0 on 0s by default.
    # (Since we're processing images and so have no padding and some labels 0.)
    return super(ClassLabelModality, self).targets_top_sharded(
        sharded_body_output,
        sharded_targets,
        data_parallelism,
        weights_fn=weights_fn)
