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

from tensorflow.python.eager import context


# TODO(noam): remove this function after TPUs do gather faster.
def tpu_gather(params, indices):
  vocab_size = params.get_shape().as_list()[0]
  indices_flat = tf.reshape(indices, [-1])
  out = tf.matmul(tf.one_hot(indices_flat, vocab_size), params)
  out = eu.reshape_like(out, tf.expand_dims(indices, -1))
  return out


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
  def top_is_pointwise(self):
    return True

  @property
  def targets_weights_fn(self):
    weights_fn = common_layers.weights_nonzero

    hp = self._model_hparams
    if hp and hp.prepend_mode != "none":
      assert (hp.prepend_mode == "prepend_inputs_masked_attention" or
              hp.prepend_mode == "prepend_inputs_full_attention")
      weights_fn = common_layers.weights_prepend_inputs_to_targets

    return weights_fn

  def _get_weights(self, hidden_dim=None):
    """Create or get concatenated embedding or softmax variable.

    Args:
      hidden_dim: dim of the variable. Defaults to self._body_input_depth

    Returns:
       a list of self._num_shards Tensors.
    """
    if hidden_dim is None:
      hidden_dim = self._body_input_depth
    num_shards = self._model_hparams.symbol_modality_num_shards
    shards = []
    for i in xrange(num_shards):
      shard_size = (self._vocab_size // num_shards) + (
          1 if i < self._vocab_size % num_shards else 0)
      var_name = "weights_%d" % i
      shards.append(
          tf.get_variable(
              var_name, [shard_size, hidden_dim],
              initializer=tf.random_normal_initializer(0.0, hidden_dim**-0.5)))
    if num_shards == 1:
      ret = shards[0]
    else:
      ret = tf.concat(shards, 0)
    # Convert ret to tensor.
    if not context.in_eager_mode():
      ret = eu.convert_gradient_to_tensor(ret)
    return ret

  def bottom_simple(self, x, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
      # Squeeze out the channels dimension.
      x = tf.squeeze(x, axis=3)
      var = self._get_weights()
      ret = (tpu_gather(var, x) if self._model_hparams.use_tpu
             else tf.gather(var, x))
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
    if self._model_hparams.symbol_modality_skip_top:
      return tf.expand_dims(body_output, 3)

    if self._model_hparams.shared_embedding_and_softmax_weights:
      scope_name = "shared"
      reuse = True
    else:
      scope_name = "softmax"
      reuse = False

    with tf.variable_scope(scope_name, reuse=reuse):
      body_output_shape = common_layers.shape_list(body_output)
      var = self._get_weights(body_output_shape[-1])
      if (self._model_hparams.factored_logits and
          self._model_hparams.mode == tf.estimator.ModeKeys.TRAIN):
        # insert channels dimension
        body_output = tf.expand_dims(body_output, 3)
        return common_layers.FactoredTensor(body_output, var)
      else:
        body_output = tf.reshape(body_output, [-1, body_output_shape[-1]])
        logits = tf.matmul(body_output, var, transpose_b=True)
        if (self._model_hparams.use_tpu and
            self._model_hparams.mode == tf.estimator.ModeKeys.TRAIN):
          # TPU does not react kindly to extra dimensions.
          # TODO(noam): remove this once TPU is more forgiving of extra dims.
          return logits
        else:
          return tf.reshape(
              logits, body_output_shape[:-1] + [1, self._vocab_size])


@registry.register_symbol_modality("ctc")
class CTCSymbolModality(SymbolModality):
  """SymbolModality that uses CTC loss."""

  def loss(self, logits, targets):
    """Compute the CTC loss."""
    with tf.name_scope("ctc_loss", [logits, targets]):
      # For CTC we assume targets are 1d, [batch, length, 1, 1] here.
      targets_shape = targets.get_shape().as_list()
      assert len(targets_shape) == 4
      assert targets_shape[2] == 1
      assert targets_shape[3] == 1
      targets = tf.squeeze(targets, axis=[2, 3])
      logits = tf.squeeze(logits, axis=[2, 3])
      targets_mask = 1 - tf.to_int32(tf.equal(targets, 0))
      targets_lengths = tf.reduce_sum(targets_mask, axis=1)
      sparse_targets = tf.keras.backend.ctc_label_dense_to_sparse(
          targets, targets_lengths)
      xent = tf.nn.ctc_loss(
          sparse_targets,
          logits,
          targets_lengths,
          time_major=False,
          preprocess_collapse_repeated=False,
          ctc_merge_repeated=False)
      weights = self.targets_weights_fn(targets)
      return tf.reduce_sum(xent), tf.reduce_sum(weights)


@registry.register_image_modality("default")
class ImageModality(modality.Modality):
  """Modality for images."""
  NUM_CHANNELS = 3

  def bottom(self, inputs):
    with tf.variable_scope(self.name):
      inputs = common_layers.standardize_images(inputs)
      if not context.in_eager_mode():
        tf.summary.image("inputs", inputs, max_outputs=2)
      return tf.to_float(inputs)

  def targets_bottom(self, inputs):
    with tf.variable_scope(self.name):
      # Reshape inputs to 2-d tensor and embed the RGB pixel values.
      ret = common_layers.embedding(
          tf.to_int32(common_layers.flatten4d3d(inputs)),
          self.top_dimensionality,
          self._body_input_depth,
          name="input_rgb_embedding")
      if self._model_hparams.multiply_embedding_mode == "sqrt_depth":
        ret *= self._body_input_depth**0.5

      reshape_shape = common_layers.shape_list(inputs)[:3]
      reshape_shape.append(self._body_input_depth * 3)
      ret = tf.reshape(ret, reshape_shape)
      return tf.layers.dense(ret, self._body_input_depth)

  def top(self, body_output, _):
    with tf.variable_scope("rgb_softmax"):

      body_output_shape = common_layers.shape_list(body_output)
      reshape_shape = body_output_shape[:3]
      dim = body_output_shape[-1] // 3
      reshape_shape.extend([self.NUM_CHANNELS, dim])

      out = tf.reshape(body_output, reshape_shape)
      res = tf.layers.dense(out, self.top_dimensionality)
      if not tf.get_variable_scope().reuse:
        res_argmax = tf.cast(tf.argmax(res, axis=-1), tf.uint8)
        tf.summary.image("result", res_argmax, max_outputs=1)
      return res


@registry.register_image_modality("image_identity_compress")
class ImageIdentityCompressModality(modality.Modality):
  """Modality for images used in generation."""

  def bottom_compress(self, inputs, name="bottom"):
    """Transform input from data space to model space.

    Perform conversion of RGB pixel values to a real number and combine values
    for each pixel to form representation of image_length x image_length dims.

    Args:
      inputs: A Tensor with shape [batch, ...]
      name: string, scope.
    Returns:
      body_input: A Tensor with shape [batch, ?, ?, body_input_depth].
    """
    with tf.variable_scope(name):
      inputs = common_layers.convert_rgb_to_real(inputs)
      ishape = common_layers.shape_list(inputs)
      inputs = tf.reshape(inputs, [-1, ishape[1], ishape[2] * ishape[3], 1])
      inputs.set_shape([None, None, None, 1])
      # We compress RGB intensities for each pixel using a conv.
      x = common_layers.conv_block(
          inputs,
          self._body_input_depth, [((1, 1), (1, 3))],
          first_relu=False,
          padding="VALID",
          strides=(1, 3),
          force2d=True,
          name="conv_input")
      return x

  def bottom(self, inputs):
    return self.bottom_compress(inputs, "input_bottom")

  def targets_bottom(self, inputs):
    return self.bottom_compress(inputs, "output_bottom")

  def top(self, body_output, _):
    with tf.variable_scope(self.name):
      hidden_dim = self._model_hparams.hidden_size
      img_len = self._model_hparams.img_len
      channels = self._model_hparams.num_channels
      batch = common_layers.shape_list(body_output)[0]
      x = common_layers.conv(
          body_output,
          hidden_dim * channels, (1, 1),
          padding="VALID",
          activation=tf.nn.relu,
          name="decompress_conv")
      x = tf.reshape(x, [batch, img_len, img_len * channels, hidden_dim])
      x.set_shape([None, None, None, hidden_dim])
      x = common_layers.conv(
          x, self.top_dimensionality, (1, 1), name="output_conv")
      x = tf.reshape(x,
                     [-1, img_len, img_len, channels, self.top_dimensionality])
      return x


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

  @property
  def name(self):
    return "class_label_modality_%d_%d" % (self._vocab_size,
                                           self._body_input_depth)

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
      return tf.zeros(
          [common_layers.shape_list(x)[0], 1, 1, self._body_input_depth])

  def top(self, body_output, _):
    """Transform inputs from model space to target space.

    Average over inner dims and a linear layer to logits.

    Args:
      body_output: A Tensor with shape [batch, ?, ?, body_output_size].

    Returns:
      a Tensors, each with shape [batch_size, ?, ?, vocab_size]
    """
    with tf.variable_scope(self.name):
      x = body_output
      x = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
      res = tf.layers.dense(x, self._vocab_size)
      return tf.expand_dims(res, 3)


@registry.register_generic_modality("default")
@registry.register_audio_modality("identity")
@registry.register_image_modality("identity")
@registry.register_class_label_modality("identity")
@registry.register_real_modality("identity")
class IdentityModality(modality.Modality):
  """Does nothing."""

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

  def loss(self, top_out, targets):
    raise NotImplementedError()


@registry.register_real_modality("default")
@registry.register_real_modality("l2_loss")
class RealL2LossModality(RealModality):
  """Modality for real (i.e. float) vectors with L2 (Gaussian) loss."""

  def loss(self, top_out, targets):
    predictions = top_out
    with tf.name_scope("l2"):
      weights = self.targets_weights_fn(targets)
      l2 = tf.pow(predictions - targets, 2)
      return tf.reduce_sum(l2 * weights), tf.reduce_sum(weights)


@registry.register_real_modality("log_poisson_loss")
class RealLogPoissonLossModality(RealModality):
  """Modality for real (i.e. float) vectors with log Poisson regression loss."""

  def loss(self, top_out, targets):
    predictions = top_out
    with tf.name_scope("log_possion"):
      weights = self.targets_weights_fn(targets)

      lp_loss = tf.nn.log_poisson_loss(targets, predictions)
      return tf.reduce_sum(lp_loss * weights), tf.reduce_sum(weights)


@registry.register_generic_modality("zero_loss")
@registry.register_audio_modality("zero_loss")
@registry.register_image_modality("zero_loss")
@registry.register_symbol_modality("zero_loss")
@registry.register_class_label_modality("zero_loss")
@registry.register_real_modality("zero_loss")
class IdentityZeroLossModality(IdentityModality):
  """Identity with 0 loss."""

  def loss(self, top_out, targets):
    return tf.constant(0., tf.float32), tf.constant(0., tf.float32)


@registry.register_symbol_modality("identity")
class IdentitySymbolModality(SymbolModality):
  """Symbol modality with identity top and bottom transformations.

  Uses the weights_fn from SymbolModality so that loss/metrics ignore padding.
  """

  def bottom(self, x):
    return tf.to_float(x)

  def top(self, body_output, _):
    return body_output
