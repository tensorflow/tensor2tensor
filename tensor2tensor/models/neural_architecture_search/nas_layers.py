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

"""Bank of layers for Translation NAS searches.

All encoder layers are registered in the global LayerRegistry ENCODER_LAYERS.
All decoder layers are registered on the global LayerRegistry DECODER_LAYERS.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

from tensor2tensor.layers import common_attention

import tensorflow.compat.v1 as tf

# Registry layer keys.
ATTEND_TO_ENCODER_REGISTRY_KEY = "attend_to_encoder"
ATTENTION_32_HEADS_REGISTRY_KEY = "attention_32_heads"
ATTENTION_16_HEADS_REGISTRY_KEY = "attention_16_heads"
ATTENTION_4_HEADS_REGISTRY_KEY = "attention_4_heads"
DEPTHWISE_CONV_3X1_REGISTRY_KEY = "depthwise_conv_3x1"
DEPTHWISE_CONV_5X1_REGISTRY_KEY = "depthwise_conv_5x1"
DEPTHWISE_CONV_7X1_REGISTRY_KEY = "depthwise_conv_7x1"
DILATED_CONV_3X1_REGISTRY_KEY = "dilated_conv_3x1"
DILATED_CONV_5X1_REGISTRY_KEY = "dilated_conv_5x1"
GATED_LINEAR_UNIT_REGISTRY_KEY = "gated_linear_unit"
IDENTITY_REGISTRY_KEY = "identity"
# Lightweight convolution naming convention uses "R_X" where X is the variable
# reduction factor.
LIGHTWEIGHT_CONV_3X1_R_1_REGISTRY_KEY = "lightweight_conv_3x1_r_1"
LIGHTWEIGHT_CONV_3X1_R_4_REGISTRY_KEY = "lightweight_conv_3x1_r_4"
LIGHTWEIGHT_CONV_3X1_R_16_REGISTRY_KEY = "lightweight_conv_3x1_r_16"
LIGHTWEIGHT_CONV_5X1_R_1_REGISTRY_KEY = "lightweight_conv_5x1_r_1"
LIGHTWEIGHT_CONV_5X1_R_4_REGISTRY_KEY = "lightweight_conv_5x1_r_4"
LIGHTWEIGHT_CONV_5X1_R_16_REGISTRY_KEY = "lightweight_conv_5x1_r_16"
LIGHTWEIGHT_CONV_7X1_R_1_REGISTRY_KEY = "lightweight_conv_7x1_r_1"
LIGHTWEIGHT_CONV_7X1_R_4_REGISTRY_KEY = "lightweight_conv_7x1_r_4"
LIGHTWEIGHT_CONV_7X1_R_16_REGISTRY_KEY = "lightweight_conv_7x1_r_16"
LIGHTWEIGHT_CONV_15X1_R_1_REGISTRY_KEY = "lightweight_conv_15x1_r_1"
LIGHTWEIGHT_CONV_15X1_R_4_REGISTRY_KEY = "lightweight_conv_15x1_r_4"
LIGHTWEIGHT_CONV_15X1_R_16_REGISTRY_KEY = "lightweight_conv_15x1_r_16"
SEPARABLE_CONV_3X1_REGISTRY_KEY = "separable_conv_3x1"
SEPARABLE_CONV_5X1_REGISTRY_KEY = "separable_conv_5x1"
SEPARABLE_CONV_7X1_REGISTRY_KEY = "separable_conv_7x1"
SEPARABLE_CONV_9X1_REGISTRY_KEY = "separable_conv_9x1"
SEPARABLE_CONV_11X1_REGISTRY_KEY = "separable_conv_11x1"
SEPARABLE_CONV_13X1_REGISTRY_KEY = "separable_conv_13x1"
SEPARABLE_CONV_15X1_REGISTRY_KEY = "separable_conv_15x1"
STANDARD_CONV_1X1_REGISTRY_KEY = "standard_conv_1x1"
STANDARD_CONV_3X1_REGISTRY_KEY = "standard_conv_3x1"
STANDARD_CONV_5X1_REGISTRY_KEY = "standard_conv_5x1"
STANDARD_ATTENTION_REGISTRY_KEY = "standard_attention"


class TranslationLayer(object):
  """Interface for the layers used in the Translation search space."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def _apply_logic(self, input_tensor, output_depth, hparams, var_scope_suffix,
                   nonpadding, mask_future, **kwargs):
    """Applies the layer specific logic to the `input_tensor`.

    This is called by `apply_layer()` to apply the subclass specific logic to
    the preprocessed `input_tensor`.

    Args:
      input_tensor: [batch_size, batch time_steps, embedding_depth] tensor.
      output_depth: Depth of the output tensor.
      hparams: Hyperparameters for the layer.
      var_scope_suffix: Suffix appended to the end of the variable scope.
      nonpadding: a [batch_size, batch time_steps] tensor with 1 where each
        batch member has sequence information and 0 everywhere else. This is
        used to mask out the irrelevant padded portions of the input.
      mask_future: Boolean. If False, information moves across the
        spatial/temporal dimension freely. If True, each timestep can only
        process the information that has come before it.
      **kwargs: Subclass-specific arguments.

    Returns:
      logic_output: [batch_size, batch time_steps, output_depth] tensor output
                    of the logic.
    """

  def apply_layer(self,
                  input_tensor,
                  residual_tensor,
                  output_depth,
                  activation,
                  hparams,
                  var_scope_suffix,
                  nonpadding,
                  mask_future,
                  layer_preprocess_fn=None,
                  postprocess_dropout=True,
                  **kwargs):
    """Applies the layer to the input.

    Also applies pad masking, preprocessing, postprocessing, and nonlinearity.

    Args:
      input_tensor: [batch_size, batch time_steps, embedding_depth] tensor.
      residual_tensor: Tensor that gets added to the output residually if
        `layer_postprocess` is True.
      output_depth: Depth of the output tensor.
      activation: Activation to be applied to the `layer_output`. If None, no
        activation will be applied.
      hparams: Hyperparameters for the layer.
      var_scope_suffix: Suffix appended to the end of the variable scope.
      nonpadding: a [batch_size, batch time_steps] tensor with 1 where each
        batch member has sequence information and 0 everywhere else. This is
        used to mask out the irrelevant padded portions of the input.
      mask_future: Boolean. If False, information moves across the
        spatial/temporal dimension freely. If True, each timestep can only
        process the information that has come before it.
      layer_preprocess_fn: Preprocess function applied to the input.
      postprocess_dropout: Whether or not to apply dropout.
      **kwargs: Arguments used by specific TranslationLayers.

    Returns:
      layer_output: The output of the layer.
    """
    input_depth = input_tensor.shape.as_list()[-1]
    layer_output = input_tensor
    if nonpadding is not None:
      nonpadding_input_tiled = tf.tile(
          tf.expand_dims(nonpadding, 2), [1, 1, input_depth])
      layer_output *= nonpadding_input_tiled

    if layer_preprocess_fn:
      layer_output = layer_preprocess_fn(layer_output)
      if nonpadding is not None:
        layer_output *= nonpadding_input_tiled

    layer_output = self._apply_logic(layer_output, output_depth, hparams,
                                     var_scope_suffix, nonpadding, mask_future,
                                     **kwargs)

    if activation:
      layer_output = activation(layer_output)

    if postprocess_dropout:
      layer_output = tf.nn.dropout(layer_output, 1 - hparams.relu_dropout)

    if residual_tensor is not None:
      layer_output += residual_tensor

    # Remove the output padding items.
    if nonpadding is not None:
      nonpadding_output_tiled = tf.tile(
          tf.expand_dims(nonpadding, 2), [1, 1, output_depth])
      layer_output *= nonpadding_output_tiled

    return layer_output

  @abc.abstractmethod
  def num_params(self, input_depth, output_depth, **kwargs):
    """Returns num_params in the layer for the given input and output depths.

    NOTE: This does not include layer norm parameters that appear in
      layer_preprocess or layer_postprocess!

    Args:
      input_depth: The depth of the input.
      output_depth: The depth of the output.
      **kwargs: TranslationLayer specific arguments.
    """


class LayerRegisteredError(Exception):
  """Layer name is already used in LayerRegistry."""


class LayerRegistry(object):
  """Registry of TranslationLayers.

  The registry is a mapping of string names to TranslationLayers. Layers can be
  added to the registry via `registry_layer()` and can be accessed via `get()`.
  """

  def __init__(self):
    self._layers = {}

  def register_layer(self, name, translation_layer):
    """Register a TranslationLayer under the key `name`."""
    if name in self._layers and self._layers[name] != translation_layer:
      raise LayerRegisteredError(
          "Already registered %s in layer registry with a different object!" %
          name)

    self._layers[name] = translation_layer

  def get(self, name):
    return self._layers[name]

  def get_layer_names(self):
    return sorted(six.iterkeys(self._layers))


DECODER_LAYERS = LayerRegistry()
ENCODER_LAYERS = LayerRegistry()


class ConvLayerBase(TranslationLayer):
  """Convolution TranslationLayer base class."""

  def __init__(self, conv_type, conv_width, dilation_rate):
    self._conv_type = conv_type
    self._conv_width = conv_width
    self._dilation_rate = dilation_rate

  def _conv_function(self, input_tensor, output_depth, padding):
    """Conv function that will be applied to the input tensor."""
    raise NotImplementedError()

  def _apply_logic(self, input_tensor, output_depth, hparams, var_scope_suffix,
                   nonpadding, mask_future, **unused_kwargs):
    """Applies conv logic to `input_tensor`."""
    with tf.variable_scope("%s_conv_%s" % (self._conv_type, var_scope_suffix)):
      if mask_future:
        # Pad shift the inputs so that temporal information does not leak. This
        # must be used in tandem with VALID padding.
        pad_amount = int(self._conv_width - 1) * self._dilation_rate
        logic_output = tf.pad(
            input_tensor, paddings=[[0, 0], [pad_amount, 0], [0, 0]])
        padding = "VALID"
      else:
        logic_output = input_tensor
        padding = "SAME"

      logic_output = tf.expand_dims(logic_output, 2)
      logic_output = self._conv_function(logic_output, output_depth, padding)

      logic_output = tf.squeeze(logic_output, 2)
    return logic_output


class SeparableConvLayer(ConvLayerBase):
  """Separable convolution TranslationLayer base class."""

  def __init__(self, conv_width):
    super(SeparableConvLayer, self).__init__("separable", conv_width, 1)

  def _conv_function(self, input_tensor, output_depth, padding):
    conv_output = tf.squeeze(input_tensor, 2)
    separable_conv_1d = tf.layers.SeparableConv1D(
        output_depth,
        self._conv_width,
        padding=padding,
        name="separable_conv_%sx1" % self._conv_width)
    conv_output = separable_conv_1d.apply(conv_output)
    return tf.expand_dims(conv_output, 2)

  def num_params(self, input_depth, output_depth, **unused_kwargs):
    return (self._conv_width * input_depth + input_depth * output_depth +
            output_depth)


class StandardConvLayer(ConvLayerBase):
  """Standard convolutional TranslationLayer base class."""

  def __init__(self, conv_width):
    super(StandardConvLayer, self).__init__("standard", conv_width, 1)

  def _conv_function(self, input_tensor, output_depth, padding):
    return tf.layers.conv2d(
        input_tensor,
        output_depth, [self._conv_width, 1],
        padding=padding,
        name="conv_%sx1" % self._conv_width)

  def num_params(self, input_depth, output_depth, **unused_kwargs):
    return self._conv_width * input_depth * output_depth + output_depth


def calculate_depthwise_channel_multiplier(input_depth, output_depth):
  """Calculates channel multiplier for depthwise convolution."""
  # Check to see if the output_depth >= input_depth
  # and output_depth % input_depth == 0. If this is the case then we
  # can satify the output_depth constraint, so the channel multiplier
  # will be set accordingly.
  if output_depth >= input_depth and output_depth % input_depth == 0:
    return output_depth // input_depth
  return 1


class DepthwiseConvLayer(ConvLayerBase):
  """Depthwise convolution TranslationLayer base class."""

  def __init__(self, conv_width):
    super(DepthwiseConvLayer, self).__init__("depthwise", conv_width, 1)

  def _conv_function(self, input_tensor, output_depth, padding):
    input_depth = input_tensor.shape.as_list()[-1]
    if not ((output_depth >= input_depth) and
            (output_depth % input_depth == 0)):
      raise ValueError(
          "Depthwise layer output_depth (%s) must be greater or equal to and "
          "a multiple of the depth of the "
          "input tensor (%s)." % (output_depth, input_depth))
    channel_multiplier = calculate_depthwise_channel_multiplier(
        input_depth, output_depth)
    kernel = tf.get_variable(
        "kernel", [self._conv_width, 1, input_depth, channel_multiplier])
    return tf.nn.depthwise_conv2d(
        input_tensor,
        kernel, [1, 1, 1, 1],
        padding=padding,
        name="depthwise_conv_%sx1" % str(self._conv_width))

  def num_params(self, input_depth, output_depth, **unused_kwargs):
    channel_multiplier = calculate_depthwise_channel_multiplier(
        input_depth, output_depth)
    return self._conv_width * input_depth * channel_multiplier


class LightweightConvLayer(ConvLayerBase):
  """Lightweight convolution TranslationLayer base class."""

  def __init__(self, conv_width, num_repeat):
    super(LightweightConvLayer, self).__init__("depthwise", conv_width, 1)
    self._num_repeat = num_repeat

  def _conv_function(self, input_tensor, output_depth, padding):
    input_depth = input_tensor.shape.as_list()[-1]
    if not ((output_depth >= input_depth) and
            (output_depth % input_depth == 0)):
      raise ValueError(
          "Depthwise layer output_depth (%s) must be greater or equal to and "
          "a multiple of the depth of the "
          "input tensor (%s)." % (output_depth, input_depth))
    channel_multiplier = calculate_depthwise_channel_multiplier(
        input_depth, output_depth)

    num_input_variables = input_depth // self._num_repeat
    kernel_base = tf.get_variable(
        "kernel_base",
        [self._conv_width, 1, num_input_variables, channel_multiplier])
    kernel = tf.concat([kernel_base] * self._num_repeat, axis=2)

    num_nonrepeated_variables = input_depth % self._num_repeat
    if num_nonrepeated_variables:
      nonrepeated_variables = tf.get_variable(
          "nonrepeated_kernel_variables",
          [self._conv_width, 1, num_nonrepeated_variables, channel_multiplier])
      kernel = tf.concat([kernel, nonrepeated_variables], axis=2)

    kernel = tf.nn.softmax(kernel, axis=0)
    return tf.nn.depthwise_conv2d(
        input_tensor,
        kernel, [1, 1, 1, 1],
        padding=padding,
        name="lightweight_conv_%sx1_r_%s" % (str(self._conv_width),
                                             str(self._num_repeat)))

  def num_params(self, input_depth, output_depth, **unused_kwargs):
    channel_multiplier = calculate_depthwise_channel_multiplier(
        input_depth, output_depth)
    return self._conv_width * (input_depth // self._num_repeat + (
        input_depth % self._num_repeat)) * channel_multiplier


class DilatedConvLayer(ConvLayerBase):
  """Dilated convolution TranslationLayer base class."""

  def __init__(self, conv_width):
    super(DilatedConvLayer, self).__init__("dilated", conv_width, 2)

  def _conv_function(self, input_tensor, output_depth, padding):
    input_depth = input_tensor.shape.as_list()[-1]
    kernel = tf.get_variable("kernel",
                             [self._conv_width, 1, input_depth, output_depth])
    return tf.nn.atrous_conv2d(
        input_tensor,
        kernel,
        self._dilation_rate,
        padding=padding,
        name="dilated_conv_%sx1" % str(self._conv_width))

  def num_params(self, input_depth, output_depth, **unused_kwargs):
    return self._conv_width * input_depth * output_depth


class AttentionLayer(TranslationLayer):
  """Attention layer base class."""

  def __init__(self,
               hidden_dim_multiplier,
               project_q,
               project_k,
               project_v,
               num_heads=None):
    self._hidden_dim_multiplier = hidden_dim_multiplier
    self._project_q = project_q
    self._project_k = project_k
    self._project_v = project_v
    self._num_heads = num_heads

  def _apply_logic(self,
                   input_tensor,
                   output_depth,
                   hparams,
                   var_scope_suffix,
                   nonpadding,
                   mask_future,
                   decoder_self_attention_bias=None,
                   attention_dropout_broadcast_dims=None,
                   **kwargs):
    """Applies attention logic to `input_tensor`."""
    with tf.variable_scope("standard_attention_layer_" + var_scope_suffix):
      hidden_depth = int(
          input_tensor.shape.as_list()[-1] * self._hidden_dim_multiplier)

      attention_bias = decoder_self_attention_bias

      # TODO(davidso): This dropout rate differs from the other layers. This
      #                should be fixed so that they all use the same dropout
      #                rate.
      num_heads = self._num_heads
      if num_heads is None:
        num_heads = hparams.num_heads
      logic_output = common_attention.multihead_attention(
          input_tensor,
          None,
          attention_bias,
          hidden_depth,
          hidden_depth,
          output_depth,
          num_heads,
          hparams.attention_dropout,
          attention_type=hparams.self_attention_type,
          max_relative_position=hparams.max_relative_position,
          dropout_broadcast_dims=attention_dropout_broadcast_dims)
    return logic_output

  def num_params(self, input_depth, output_depth, **unused_kwargs):
    # First account for the hidden to output projection params.
    hidden_depth = input_depth * self._hidden_dim_multiplier
    output_params = hidden_depth * output_depth

    # Next account for all the hidden projections.
    num_projections = sum([self._project_q, self._project_k, self._project_v])
    return input_depth * hidden_depth * num_projections + output_params


class AttendToEncoderLayerBase(TranslationLayer):
  """Attend to encoder base, with configurable encoder attend points."""

  def _determine_encoder_cell_index(self, cell_number, num_encoder_cells):
    """Determine the encoder cell index to attend to."""
    raise NotImplementedError()

  def _apply_logic(self,
                   input_tensor,
                   output_depth,
                   hparams,
                   var_scope_suffix,
                   nonpadding,
                   mask_future,
                   encoder_decoder_attention_bias,
                   encoder_cell_outputs,
                   cell_number,
                   attention_dropout_broadcast_dims=None,
                   **unused_kwargs):
    """Applies attention logic to `input_tensor`."""
    with tf.variable_scope("attend_to_encoder_layer_" + var_scope_suffix):
      hidden_depth = int(input_tensor.shape.as_list()[-1])
      num_encoder_cells = len(encoder_cell_outputs)
      encoder_cell_index = self._determine_encoder_cell_index(
          cell_number, num_encoder_cells)
      encoder_layer = encoder_cell_outputs[encoder_cell_index]

      # TODO(davidso): This dropout rate differs from the other layers. This
      #                should be fixed so that they all use the same dropout
      #                rate.
      logic_output = common_attention.multihead_attention(
          input_tensor,
          encoder_layer,
          encoder_decoder_attention_bias,
          hidden_depth,
          hidden_depth,
          output_depth,
          hparams.num_heads,
          hparams.attention_dropout,
          attention_type=hparams.self_attention_type,
          max_relative_position=hparams.max_relative_position,
          dropout_broadcast_dims=attention_dropout_broadcast_dims)

    return logic_output

  # Assumes uniform encoder output depths.
  def num_params(self, input_depth, output_depth, **kwargs):
    try:
      encoder_depth = kwargs["encoder_depth"]
    except KeyError:
      raise ValueError("`encoder_depth` must be in kwargs passed to "
                       "AttendToEncoder.num_params().")
    hidden_depth = input_depth

    # The number of params is comprised of the projection from the input tensor
    # to its hidden tensor, the two encoder tensor projects to its hidden
    # tensors, and the projection from the hidden concatenation to the output
    # tensor.
    return (input_depth * hidden_depth + 2 * encoder_depth * hidden_depth +
            hidden_depth * output_depth)


class AttendToEncoderTopDownLayer(AttendToEncoderLayerBase):
  """Attend to the encoder starting with the highest layer, then moving down.

    This allows the decoder to see higher level features first and then
    eventually move on to incorporate lower level information.
  """

  def __init__(self, delay, increment_step):
    self.delay = delay
    self.increment_step = increment_step

  def _determine_encoder_cell_index(self, cell_number, num_encoder_cells):
    """Attend to final encoder cell output first, then move down."""
    return max(
        0, num_encoder_cells -
        max(0, (cell_number - self.delay) * self.increment_step) - 1)


class GatedLinearUnitLayer(TranslationLayer):
  """Gated Linaer Unit Layer."""

  def __init__(self):
    pass

  def _apply_logic(self, input_tensor, output_depth, hparams, var_scope_suffix,
                   nonpadding, mask_future, **unused_kwargs):
    values = tf.layers.dense(input_tensor, output_depth)
    gates = tf.layers.dense(
        input_tensor, output_depth, activation=tf.nn.sigmoid)
    return values * gates

  def num_params(self, input_depth, output_depth, **unused_kwargs):
    return input_depth * output_depth * 2 + output_depth * 2


class IdentityLayer(TranslationLayer):
  """Identity TranslationLayer."""

  def _apply_logic(self, input_tensor, output_depth, hparams, var_scope_suffix,
                   nonpadding, mask_future, **unused_kwargs):
    input_depth = input_tensor.shape.as_list()[-1]
    if output_depth != input_depth:
      raise ValueError(
          "Identity layer output_depth (%s) must be equal to the depth of the "
          "input tensor (%s)." % (output_depth, input_depth))
    return input_tensor

  def num_params(self, input_depth, output_depth, **unused_kwargs):
    return 0


def register_encoder_decoder_layer(name, translation_layer):
  ENCODER_LAYERS.register_layer(name, translation_layer)
  DECODER_LAYERS.register_layer(name, translation_layer)


# Register all strictly decoder layers.
DECODER_LAYERS.register_layer(
    ATTEND_TO_ENCODER_REGISTRY_KEY,
    AttendToEncoderTopDownLayer(delay=0, increment_step=0))

# Register all encoder and decoder layers.
register_encoder_decoder_layer(IDENTITY_REGISTRY_KEY, IdentityLayer())

register_encoder_decoder_layer(SEPARABLE_CONV_3X1_REGISTRY_KEY,
                               SeparableConvLayer(conv_width=3))
register_encoder_decoder_layer(SEPARABLE_CONV_5X1_REGISTRY_KEY,
                               SeparableConvLayer(conv_width=5))
register_encoder_decoder_layer(SEPARABLE_CONV_7X1_REGISTRY_KEY,
                               SeparableConvLayer(conv_width=7))
register_encoder_decoder_layer(SEPARABLE_CONV_9X1_REGISTRY_KEY,
                               SeparableConvLayer(conv_width=9))
register_encoder_decoder_layer(SEPARABLE_CONV_11X1_REGISTRY_KEY,
                               SeparableConvLayer(conv_width=11))
register_encoder_decoder_layer(SEPARABLE_CONV_13X1_REGISTRY_KEY,
                               SeparableConvLayer(conv_width=13))
register_encoder_decoder_layer(SEPARABLE_CONV_15X1_REGISTRY_KEY,
                               SeparableConvLayer(conv_width=15))
register_encoder_decoder_layer(STANDARD_CONV_1X1_REGISTRY_KEY,
                               StandardConvLayer(conv_width=1))
register_encoder_decoder_layer(STANDARD_CONV_3X1_REGISTRY_KEY,
                               StandardConvLayer(conv_width=3))
register_encoder_decoder_layer(STANDARD_CONV_5X1_REGISTRY_KEY,
                               StandardConvLayer(conv_width=5))
register_encoder_decoder_layer(DEPTHWISE_CONV_3X1_REGISTRY_KEY,
                               DepthwiseConvLayer(conv_width=3))
register_encoder_decoder_layer(DEPTHWISE_CONV_5X1_REGISTRY_KEY,
                               DepthwiseConvLayer(conv_width=5))
register_encoder_decoder_layer(DEPTHWISE_CONV_7X1_REGISTRY_KEY,
                               DepthwiseConvLayer(conv_width=7))
register_encoder_decoder_layer(DILATED_CONV_3X1_REGISTRY_KEY,
                               DilatedConvLayer(conv_width=3))
register_encoder_decoder_layer(DILATED_CONV_5X1_REGISTRY_KEY,
                               DilatedConvLayer(conv_width=5))


register_encoder_decoder_layer(LIGHTWEIGHT_CONV_3X1_R_1_REGISTRY_KEY,
                               LightweightConvLayer(conv_width=3, num_repeat=1))
register_encoder_decoder_layer(LIGHTWEIGHT_CONV_3X1_R_4_REGISTRY_KEY,
                               LightweightConvLayer(conv_width=3, num_repeat=4))
register_encoder_decoder_layer(
    LIGHTWEIGHT_CONV_3X1_R_16_REGISTRY_KEY,
    LightweightConvLayer(conv_width=3, num_repeat=16))
register_encoder_decoder_layer(LIGHTWEIGHT_CONV_5X1_R_1_REGISTRY_KEY,
                               LightweightConvLayer(conv_width=5, num_repeat=1))
register_encoder_decoder_layer(LIGHTWEIGHT_CONV_5X1_R_4_REGISTRY_KEY,
                               LightweightConvLayer(conv_width=5, num_repeat=4))
register_encoder_decoder_layer(
    LIGHTWEIGHT_CONV_5X1_R_16_REGISTRY_KEY,
    LightweightConvLayer(conv_width=5, num_repeat=16))
register_encoder_decoder_layer(LIGHTWEIGHT_CONV_7X1_R_1_REGISTRY_KEY,
                               LightweightConvLayer(conv_width=7, num_repeat=1))
register_encoder_decoder_layer(LIGHTWEIGHT_CONV_7X1_R_4_REGISTRY_KEY,
                               LightweightConvLayer(conv_width=7, num_repeat=4))
register_encoder_decoder_layer(
    LIGHTWEIGHT_CONV_7X1_R_16_REGISTRY_KEY,
    LightweightConvLayer(conv_width=7, num_repeat=16))
register_encoder_decoder_layer(
    LIGHTWEIGHT_CONV_15X1_R_1_REGISTRY_KEY,
    LightweightConvLayer(conv_width=15, num_repeat=1))
register_encoder_decoder_layer(
    LIGHTWEIGHT_CONV_15X1_R_4_REGISTRY_KEY,
    LightweightConvLayer(conv_width=15, num_repeat=4))
register_encoder_decoder_layer(
    LIGHTWEIGHT_CONV_15X1_R_16_REGISTRY_KEY,
    LightweightConvLayer(conv_width=15, num_repeat=16))

register_encoder_decoder_layer(
    GATED_LINEAR_UNIT_REGISTRY_KEY,
    GatedLinearUnitLayer())


register_encoder_decoder_layer(
    STANDARD_ATTENTION_REGISTRY_KEY,
    AttentionLayer(
        hidden_dim_multiplier=1, project_q=True, project_k=True,
        project_v=True))
register_encoder_decoder_layer(
    ATTENTION_16_HEADS_REGISTRY_KEY,
    AttentionLayer(
        hidden_dim_multiplier=1,
        project_q=True,
        project_k=True,
        project_v=True,
        num_heads=16))
register_encoder_decoder_layer(
    ATTENTION_32_HEADS_REGISTRY_KEY,
    AttentionLayer(
        hidden_dim_multiplier=1,
        project_q=True,
        project_k=True,
        project_v=True,
        num_heads=32))
register_encoder_decoder_layer(
    ATTENTION_4_HEADS_REGISTRY_KEY,
    AttentionLayer(
        hidden_dim_multiplier=1,
        project_q=True,
        project_k=True,
        project_v=True,
        num_heads=4))
