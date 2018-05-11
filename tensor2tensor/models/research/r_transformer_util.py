# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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
"""Utilities for R-Transformer.


R-Transformer learns a function (for instance the transformer multi-head
attention plus a feed-forward unit) and uses this function over n-steps to
process the input.
In other words, we can describe this as having a vanilla transformer, in which
the weights in the layers are shared and we have a module(the recurrency module)
next to this transformer that controls how steps communicate with each other in
depth.

For instance, the recurrency module, can be a simple identity function
which passes the output of a step as the input to next step (applying one layer
of transformer n times on the input in a row --> lead to a better
generalization!). Or as another example, the recurrent module can be an LSTM,
(filliped vertically) next to the transformer which controls how state of the
model changes in depth, Or even a grit transformer (a transformer which learns
the attention over steps of an R-Transformer)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools

# Dependency imports

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import expert_utils

import tensorflow as tf


def r_transformer_encoder(encoder_input,
                          encoder_self_attention_bias,
                          hparams,
                          name="encoder",
                          nonpadding=None,
                          save_weights_to=None,
                          make_image_summary=True):
  """R_transformer_encoder function.

  Prepares all the arguments and the inputs and passes it to a
  r_transformer_layer to encode the encoder_input.

  Args:
    encoder_input: a Tensor
    encoder_self_attention_bias: bias Tensor for self-attention
       (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This must either be
      passed in, which we do for "packed" datasets, or inferred from
      encoder_self_attention_bias.  The knowledge about padding is used
      for pad_remover(efficiency) and to mask out padding in convoltutional
      layers.
    save_weights_to: an optional dictionary to capture attention weights
      for vizualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.

  Returns:
    y: a Tensors as the output of the encoder
    extra_output: which can be used to pass extra information to the body
  """

  x = encoder_input
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))
  with tf.variable_scope(name):
    if nonpadding is not None:
      padding = 1.0 - nonpadding
    else:
      padding = common_attention.attention_bias_to_padding(
          encoder_self_attention_bias)
      nonpadding = 1.0 - padding
    pad_remover = None
    if hparams.use_pad_remover and not common_layers.is_on_tpu():
      pad_remover = expert_utils.PadRemover(padding)

    ffn_unit = functools.partial(
        transformer_encoder_ffn_unit,
        hparams=hparams,
        nonpadding_mask=nonpadding,
        pad_remover=pad_remover)

    attention_unit = functools.partial(
        transformer_encoder_attention_unit,
        hparams=hparams,
        encoder_self_attention_bias=encoder_self_attention_bias,
        attention_dropout_broadcast_dims=attention_dropout_broadcast_dims,
        save_weights_to=save_weights_to,
        make_image_summary=make_image_summary)

    x, extra_output = r_transformer_layer(
        x, hparams, ffn_unit, attention_unit, pad_remover=pad_remover)

    if hparams.get("use_memory_as_last_state", False):
      x = extra_output  # which is memory
    return common_layers.layer_preprocess(x, hparams), extra_output


def r_transformer_decoder(decoder_input,
                          encoder_output,
                          decoder_self_attention_bias,
                          encoder_decoder_attention_bias,
                          hparams,
                          name="decoder",
                          nonpadding=None,
                          save_weights_to=None,
                          make_image_summary=True):
  """R_transformer decoder function.

  Prepares all the arguments and the inputs and passes it to a
  core_r_transformer_layer to decoder.

  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention
      (see common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This is used
      to mask out padding in convoltutional layers.  We generally only
      need this mask for "packed" datasets, because for ordinary datasets,
      no padding is ever followed by nonpadding.
    save_weights_to: an optional dictionary to capture attention weights
      for vizualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.

  Returns:
    y: the output Tensors
    extra_output: which can be used to pass extra information to the body
  """
  x = decoder_input
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))
  with tf.variable_scope(name):
    ffn_unit = functools.partial(
        transformer_decoder_ffn_unit,
        hparams=hparams,
        nonpadding_mask=nonpadding)

    attention_unit = functools.partial(
        transformer_decoder_attention_unit,
        hparams=hparams,
        encoder_output=encoder_output,
        decoder_self_attention_bias=decoder_self_attention_bias,
        encoder_decoder_attention_bias=encoder_decoder_attention_bias,
        attention_dropout_broadcast_dims=attention_dropout_broadcast_dims,
        save_weights_to=save_weights_to,
        make_image_summary=make_image_summary)

    x, extra_output = r_transformer_layer(x, hparams, ffn_unit, attention_unit)

    return common_layers.layer_preprocess(x, hparams), extra_output


def r_transformer_layer(x, hparams, ffn_unit, attention_unit, pad_remover=None):
  """Core function applying the r-transforemr layer.

  Args:
    x: input
    hparams: model hyper-parameters
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit
    pad_remover: to mask out padding in convolutional layers (efficiency).

  Returns:
    the output tensor,  extra output (can be memory, ponder time, etc.)

  Raises:
    ValueError: Unknown recurrence type
  """

  def add_vanilla_transformer_layer(x, num_layers):
    """Passes the input through num_layers of vanilla transformer layers.

    Args:
     x: input
     num_layers: number of layers

    Returns:
       output of vanilla_transformer_layer
    """

    if hparams.add_position_timing_signal:
      # In case of add_position_timing_signal=true, we set  hparams.pos=None
      # and add position timing signal at the beginning of each step, so for
      # the vanilla transformer, we need to add timing signal here.
      x = common_attention.add_timing_signal_1d(x)
    for layer in xrange(num_layers):
      with tf.variable_scope("layer_%d" % layer):
        x = ffn_unit(attention_unit(x))
    return x

  with tf.variable_scope("r_transformer_%s" % hparams.recurrence_type):

    if hparams.mix_with_transformer == "before_rt":
      x = add_vanilla_transformer_layer(x, hparams.num_mixedin_layers)

    if hparams.recurrence_type == "act":
      return r_transformer_act(x, hparams, ffn_unit, attention_unit)

    else:  # for all the other recurrency types with fixed number of steps
      rt_function, initializer = get_rt_layer(x, hparams, ffn_unit,
                                              attention_unit, pad_remover)

      output, _, extra_output = tf.foldl(
          rt_function, tf.range(hparams.num_rec_steps), initializer=initializer)

      # This can be the if we use r_transformer_lstm layer.
      if hparams.get("use_memory_as_final_state", False):
        output = extra_output

    if hparams.mix_with_transformer == "after_rt":
      output = add_vanilla_transformer_layer(output, hparams.num_mixedin_layers)

    return output, extra_output


def get_rt_layer(x, hparams, ffn_unit, attention_unit, pad_remover=None):
  """provides the function that is used in r-transforemr steps.

  Args:
    x: input
    hparams: model hyper-parameters
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit
    pad_remover: to mask out padding in convolutional layers (efficiency).

  Returns:
    rt_function and the rt_initializer

  Raises:
    ValueError: Unknown recurrence type
  """

  if hparams.recurrence_type == "basic":
    rt_initializer = (x, x, x)  # (state, input, memory)
    rt_function = functools.partial(
        r_transformer_basic,
        hparams=hparams,
        ffn_unit=ffn_unit,
        attention_unit=attention_unit)

  elif hparams.recurrence_type == "highway":
    rt_initializer = (x, x, x)  # (state, input, memory)
    rt_function = functools.partial(
        r_transformer_highway,
        hparams=hparams,
        ffn_unit=ffn_unit,
        attention_unit=attention_unit,
        pad_remover=pad_remover)

  elif hparams.recurrence_type == "skip":
    rt_initializer = (x, x, x)  # (state, input, memory)
    rt_function = functools.partial(
        r_transformer_skip,
        hparams=hparams,
        ffn_unit=ffn_unit,
        attention_unit=attention_unit,
        pad_remover=pad_remover)

  elif hparams.recurrence_type == "dwa":
    # memory contains the original input + all the states
    memory_size = hparams.num_rec_steps + 1

    # prepare initializer:
    memory_empty = tf.zeros([memory_size] + common_layers.shape_list(x))

    # filling the first slot with the original input
    memory = fill_memory_slot(memory_empty, x, 0)

    rt_initializer = (x, x, memory)  # (state, input, memory)
    rt_function = functools.partial(
        r_transformer_depthwise_attention,
        hparams=hparams,
        ffn_unit=ffn_unit,
        attention_unit=attention_unit)

  elif hparams.recurrence_type == "rnn":
    rt_initializer = (x, x, x)  # (state, input, memory)
    rt_function = functools.partial(
        r_transformer_rnn,
        hparams=hparams,
        ffn_unit=ffn_unit,
        attention_unit=attention_unit,
        pad_remover=pad_remover)

  elif hparams.recurrence_type == "gru":
    rt_initializer = (x, x, x)  # (state, input, memory)
    rt_function = functools.partial(
        r_transformer_gru,
        hparams=hparams,
        attention_unit=attention_unit,
        pad_remover=pad_remover)

  elif hparams.recurrence_type == "lstm":
    memory = tf.zeros(common_layers.shape_list(x))
    rt_initializer = (x, x, memory)  # (state, input, memory)
    rt_function = functools.partial(
        r_transformer_lstm,
        hparams=hparams,
        attention_unit=attention_unit,
        pad_remover=pad_remover)

  else:
    raise ValueError("Unknown recurrence type: %s" % hparams.recurrence_type)

  return rt_function, rt_initializer


def transformer_encoder_ffn_unit(x,
                                 hparams,
                                 nonpadding_mask=None,
                                 pad_remover=None):
  """Applies a feed-forward function which is parametrised for encoding.

  Args:
    x: input
    hparams: model hyper-parameters
    nonpadding_mask: optional Tensor with shape [batch_size, encoder_length]
    indicating what positions are not padding.  This is used
    to mask out padding in convoltutional layers.  We generally only
    need this mask for "packed" datasets, because for ordinary datasets,
    no padding is ever followed by nonpadding.
    pad_remover: to mask out padding in convolutional layers (efficiency).

  Returns:
    the output tensor
  """

  with tf.variable_scope("ffn"):
    if hparams.transformer_ffn_type == "fc":
      y = transformer.transformer_ffn_layer(
          common_layers.layer_preprocess(x, hparams),
          hparams,
          pad_remover,
          conv_padding="SAME",
          nonpadding_mask=nonpadding_mask)

    if hparams.transformer_ffn_type == "sepconv":
      assert nonpadding_mask is not None, (
          "The nonpadding_mask should be provided, otherwise the model uses "
          "the leaked padding information to estimate the length!")
      y = common_layers.sepconv_relu_sepconv(
          common_layers.layer_preprocess(x, hparams),
          filter_size=hparams.filter_size,
          output_size=hparams.hidden_size,
          padding="SAME",
          nonpadding_mask=nonpadding_mask,
          dropout=hparams.relu_dropout)

    x = common_layers.layer_postprocess(x, y, hparams)

  return x


def transformer_encoder_attention_unit(x,
                                       hparams,
                                       encoder_self_attention_bias,
                                       attention_dropout_broadcast_dims,
                                       save_weights_to=None,
                                       make_image_summary=True):
  """Applies multihead attention function which is parametrised for encoding.

  Args:
    x: input
    hparams: model hyper-parameters
    encoder_self_attention_bias: a bias tensor for use in encoder self-attention
    attention_dropout_broadcast_dims: Fpr noise broadcasting in the dropout
      layers to save memory during training
    save_weights_to: an optional dictionary to capture attention weights for
      visualization; the weights tensor will be appended there under a string
      key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.

  Returns:
    the output tensor

  """

  with tf.variable_scope("self_attention"):
    y = common_attention.multihead_attention(
        common_layers.layer_preprocess(x, hparams),
        None,
        encoder_self_attention_bias,
        hparams.attention_key_channels or hparams.hidden_size,
        hparams.attention_value_channels or hparams.hidden_size,
        hparams.hidden_size,
        hparams.num_heads,
        hparams.attention_dropout,
        attention_type=hparams.self_attention_type,
        save_weights_to=save_weights_to,
        max_relative_position=hparams.max_relative_position,
        make_image_summary=make_image_summary,
        dropout_broadcast_dims=attention_dropout_broadcast_dims)
    x = common_layers.layer_postprocess(x, y, hparams)
  return x


def transformer_decoder_ffn_unit(x, hparams, nonpadding_mask=None):
  """Applies a feed-forward function which is parametrised for decoding.

  Args:
    x: input
    hparams: model hyper-parameters
    nonpadding_mask: optional Tensor with shape [batch_size, encoder_length]
    indicating what positions are not padding.  This is used
    to mask out padding in convoltutional layers.  We generally only
    need this mask for "packed" datasets, because for ordinary datasets,
    no padding is ever followed by nonpadding.

  Returns:
    the output tensor

  """

  with tf.variable_scope("ffn"):
    if hparams.transformer_ffn_type == "fc":
      y = transformer.transformer_ffn_layer(
          common_layers.layer_preprocess(x, hparams),
          hparams,
          conv_padding="LEFT",
          nonpadding_mask=nonpadding_mask)

    if hparams.transformer_ffn_type == "sepconv":
      y = common_layers.sepconv_relu_sepconv(
          common_layers.layer_preprocess(x, hparams),
          filter_size=hparams.filter_size,
          output_size=hparams.hidden_size,
          padding="LEFT",
          nonpadding_mask=nonpadding_mask,
          dropout=hparams.relu_dropout)

    x = common_layers.layer_postprocess(x, y, hparams)

  return x


def transformer_decoder_attention_unit(x,
                                       hparams,
                                       encoder_output,
                                       decoder_self_attention_bias,
                                       encoder_decoder_attention_bias,
                                       attention_dropout_broadcast_dims,
                                       save_weights_to=None,
                                       make_image_summary=True):
  """Applies multihead attention function which is parametrised for decoding.

  Args:
    x: input (decoder input)
    hparams: model hyper-parameters
    encoder_output: Encoder representation. [batch_size, input_length,
      hidden_dim]
    decoder_self_attention_bias: Bias and mask weights for decoder
      self-attention. [batch_size, decoder_length]
    encoder_decoder_attention_bias: Bias and mask weights for encoder-decoder
      attention. [batch_size, input_length]
    attention_dropout_broadcast_dims: Fpr noise broadcasting in the dropout
      layers to save memory during training
    save_weights_to: an optional dictionary to capture attention weights for
      visualization; the weights tensor will be appended there under a string
      key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.

  Returns:
    The output tensor
  """

  with tf.variable_scope("self_attention"):
    y = common_attention.multihead_attention(
        common_layers.layer_preprocess(x, hparams),
        None,
        decoder_self_attention_bias,
        hparams.attention_key_channels or hparams.hidden_size,
        hparams.attention_value_channels or hparams.hidden_size,
        hparams.hidden_size,
        hparams.num_heads,
        hparams.attention_dropout,
        attention_type=hparams.self_attention_type,
        save_weights_to=save_weights_to,
        max_relative_position=hparams.max_relative_position,
        cache=None,
        make_image_summary=make_image_summary,
        dropout_broadcast_dims=attention_dropout_broadcast_dims)
    x = common_layers.layer_postprocess(x, y, hparams)
  if encoder_output is not None:
    with tf.variable_scope("encdec_attention"):
      y = common_attention.multihead_attention(
          common_layers.layer_preprocess(x, hparams),
          encoder_output,
          encoder_decoder_attention_bias,
          hparams.attention_key_channels or hparams.hidden_size,
          hparams.attention_value_channels or hparams.hidden_size,
          hparams.hidden_size,
          hparams.num_heads,
          hparams.attention_dropout,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          dropout_broadcast_dims=attention_dropout_broadcast_dims)
      x = common_layers.layer_postprocess(x, y, hparams)
  return x


def r_transformer_basic(layer_inputs, step, hparams, ffn_unit, attention_unit):
  """Basic r_transformer.

  This is in fact vanilla transformer in which weights are shared between
  layers. For some tasks, this simple idea brings a generalization that is not
  achievable by playing with the size of the model or drop_out parameters in
  the vanilla transformer.

  Args:
    layer_inputs:
        - state: state
    step: indicating number of steps take so far
    hparams: model hyper-parameters
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit

  Returns:
    layer_output:
         new_state: new state
  """
  state, inputs, memory = layer_inputs
  state = step_preprocess(state, step, hparams)

  new_state = ffn_unit(attention_unit(state))

  return new_state, inputs, memory


def r_transformer_highway(layer_inputs,
                          step,
                          hparams,
                          ffn_unit,
                          attention_unit,
                          pad_remover=None):
  """R_transformer with highway connection.


  It transforms the state using attention and ffn and wrap this transformation
  with a highway connection. (the new state is a combination of the state and
  the transformed-state based on cary/transform gates.)

  Interesting observation:
    Controlling the cary/transform gate with the original inputs works usually
    better (i.e. hparams.gates_inputs="i")

  Args:
    layer_inputs:
      - state: state
      - inputs: the original embedded inputs (= inputs to the first step)
    step: indicating number of steps take so far
    hparams: model hyper-parameters.
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit
    pad_remover: to mask out padding in convolutional layers (efficiency).

  Returns:
    layer_output:
        new_state: new state
        inputs: the original embedded inputs (= inputs to the first step)

  """

  state, inputs, memory = layer_inputs
  state = step_preprocess(state, step, hparams)

  transformed_state = ffn_unit(attention_unit(state))
  state.get_shape().assert_is_compatible_with(state.get_shape())

  gate_inputs = []
  if "s" in hparams.gates_inputs:
    gate_inputs.append(state)

  if "t" in hparams.gates_inputs:
    gate_inputs.append(transformed_state)

  if "i" in hparams.gates_inputs:
    gate_inputs.append(inputs)

  gate_ffn_layer = hparams.gate_ffn_layer

  transform_gate = _ffn_layer_multi_inputs(
      gate_inputs,
      hparams,
      ffn_layer_type=gate_ffn_layer,
      name="transform",
      bias_initializer=tf.constant_initializer(hparams.transform_bias_init),
      activation=tf.sigmoid,
      pad_remover=pad_remover,
      preprocess=True,
      postprocess=True)

  if hparams.couple_carry_transform_gates:
    carry_gate = tf.subtract(1.0, transform_gate, name="carry")

  else:
    carry_gate = _ffn_layer_multi_inputs(
        gate_inputs,
        hparams,
        ffn_layer_type=gate_ffn_layer,
        name="carry",
        bias_initializer=tf.constant_initializer(-hparams.transform_bias_init),
        activation=tf.sigmoid,
        pad_remover=pad_remover,
        preprocess=True,
        postprocess=True)

  new_state = state * carry_gate + transformed_state * transform_gate

  tf.contrib.summary.scalar("highway_transform_gate_layer",
                            tf.reduce_mean(transform_gate))

  tf.contrib.summary.scalar("highway_carry_gate_layer",
                            tf.reduce_mean(carry_gate))

  return new_state, inputs, memory


def r_transformer_skip(layer_inputs,
                       step,
                       hparams,
                       ffn_unit,
                       attention_unit,
                       pad_remover=None):
  """R_transformer with highway connection.


  It transforms the state using attention and ffn and wrap this transformation
  with a skip-all connection. (the new state is a combination of the state and
  the inputs (original inputs) based on cary/transform gates.)

  Observation:
    Controlling the cary/transform gate with the original inputs works usually
    better (i.e. hparams.gates_inputs="i")

  Args:
    layer_inputs:
      - state: state
      - inputs: the original embedded inputs (= inputs to the first step)
    step: indicating number of steps take so far
    hparams: model hyper-parameters.
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit
    pad_remover: to mask out padding in convolutional layers (efficiency).


  Returns:
    layer_output:
         new_state: new state
        inputs: the original embedded inputs (= inputs to the first step)
  """

  state, inputs, memory = layer_inputs
  state = step_preprocess(state, step, hparams)

  transformed_state = ffn_unit(attention_unit(state))

  inputs.get_shape().assert_is_compatible_with(state.get_shape())

  gate_inputs = []
  if "s" in hparams.gates_inputs:
    gate_inputs.append(state)

  if "t" in hparams.gates_inputs:
    gate_inputs.append(transformed_state)

  if "i" in hparams.gates_inputs:
    gate_inputs.append(inputs)

  gate_ffn_layer = hparams.gate_ffn_layer

  transform_gate = _ffn_layer_multi_inputs(
      gate_inputs,
      hparams,
      ffn_layer_type=gate_ffn_layer,
      name="transform",
      bias_initializer=tf.constant_initializer(hparams.transform_bias_init),
      activation=tf.sigmoid,
      pad_remover=pad_remover,
      preprocess=True,
      postprocess=True)

  if hparams.couple_carry_transform_gates:
    carry_gate = tf.subtract(1.0, transform_gate, name="carry")

  else:
    carry_gate = _ffn_layer_multi_inputs(
        gate_inputs,
        hparams,
        ffn_layer_type=gate_ffn_layer,
        name="carry",
        bias_initializer=tf.constant_initializer(-hparams.transform_bias_init),
        activation=tf.sigmoid,
        pad_remover=pad_remover,
        preprocess=True,
        postprocess=True)

  tf.contrib.summary.scalar("skip_transform_gate_layer",
                            tf.reduce_mean(transform_gate))

  tf.contrib.summary.scalar("skip_carry_gate_layer", tf.reduce_mean(carry_gate))

  new_state = inputs * carry_gate + transformed_state * transform_gate
  return new_state, inputs, memory


def r_transformer_depthwise_attention(layer_inputs, step, hparams, ffn_unit,
                                      attention_unit):
  """R_transformer with depth-wise attention.

  It uses an attention mechanism-flipped vertically-
  over all the states from previous steps to generate the new_state.

  Args:
    layer_inputs:
      - state: state
      - memory: contains states from all the previous steps.
    step: indicating number of steps take so far
    hparams: model hyper-parameters.
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit


  Returns:
    layer_output:
        new_state: new state
        memory: contains states from all the previous steps.

  """
  _, inputs, memory = layer_inputs
  all_states = memory

  # add depth signal
  if hparams.depth_embedding:
    all_states = add_depth_embedding(all_states)

  # get the states up to the current step (non-zero part of the memory)
  states_so_far = all_states[:step, :, :, :]

  states_so_far_weights = tf.nn.softmax(
      common_layers.dense(
          states_so_far, (hparams.hidden_size if hparams.dwa_elements else 1),
          activation=None,
          use_bias=True),
      axis=-1)

  # prepare the state tensor that will be transformed
  state_to_be_transformed = tf.reduce_sum(
      (states_so_far * states_so_far_weights), axis=0)

  state_to_be_transformed = step_preprocess(state_to_be_transformed, step,
                                            hparams)

  new_state = ffn_unit(attention_unit(state_to_be_transformed))

  # add the new state to the memory
  memory = fill_memory_slot(memory, new_state, step + 1)

  return new_state, inputs, memory


def r_transformer_rnn(layer_inputs,
                      step,
                      hparams,
                      ffn_unit,
                      attention_unit,
                      pad_remover=None):
  """The RT layer which models recurencey similar to basic RNN cell.

    It's an R-transformer with an RNN applied over the stats on depth.

  Args:
    layer_inputs:
      - state: state
      - inputs: the original embedded inputs (= inputs to the first step)
    step: indicating number of steps take so far
    hparams: model hyper-parameters.
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit
    pad_remover: to mask out padding in convolutional layers (efficiency).

  Returns:
    layer_output:
      new_state: new state
      inputs: the original embedded inputs (= inputs to the first step)

  Raises:
    ValueError: Unknown inputs_states_combination type

  """

  state, inputs, memory = layer_inputs
  state = step_preprocess(state, step, hparams)

  # TODO(dehghani) keep only the meaningful cases:
  if hparams.inputs_states_combination == "mh_attention_ffn_add":
    state.get_shape().assert_is_compatible_with(inputs.get_shape())
    state = ffn_unit(attention_unit(state))
    new_state = state + inputs

  elif hparams.inputs_states_combination == "add_mh_attention_ffn":
    state.get_shape().assert_is_compatible_with(inputs.get_shape())
    state += inputs
    new_state = ffn_unit(attention_unit(state))

  elif hparams.inputs_states_combination == "dense_mh_attention":
    state = _ffn_layer_multi_inputs(
        [state, inputs],
        hparams=hparams,
        ffn_layer_type="dense_relu_dense",
        name="rnn",
        activation=tf.tanh,
        pad_remover=pad_remover)

    new_state = attention_unit(state)

  elif hparams.inputs_states_combination == "mh_attention_dense":
    state = attention_unit(state)
    new_state = _ffn_layer_multi_inputs(
        [state, inputs],
        hparams=hparams,
        ffn_layer_type="dense_relu_dense",
        name="rnn",
        activation=tf.tanh,
        pad_remover=pad_remover)

  else:
    raise ValueError("Unknown inputs_states_combination type: %s" %
                     hparams.inputs_states_combination)

  return new_state, inputs, memory


def r_transformer_gru(layer_inputs,
                      step,
                      hparams,
                      attention_unit,
                      pad_remover=None):
  """The RT layer which models recurencey similar to GRU cell.

    It's an R-transformer with a gru applied over the stats on depth.
    Based on GRU paper: http://arxiv.org/abs/1406.1078

  Args:
    layer_inputs:
      - state: state
      - inputs: the original embedded inputs (= inputs to the first step)
    step: indicating number of steps take so far
    hparams: model hyper-parameters.
    attention_unit: multi-head attention unit
    pad_remover: to mask out padding in convolutional layers (efficiency).


  Returns:
    layer_output:
      new_state: new state
      inputs: the original embedded inputs (= inputs to the first step)
  """

  state, inputs, memory = layer_inputs
  state = step_preprocess(state, step, hparams)

  # TODO(dehghani): do we need preprocess here?
  state = common_layers.layer_preprocess(state, hparams)
  inputs = common_layers.layer_preprocess(inputs, hparams)

  update_gate = _ffn_layer_multi_inputs(
      [inputs, state],
      hparams,
      name="update",
      bias_initializer=tf.constant_initializer(1.0),
      activation=tf.sigmoid,
      pad_remover=pad_remover)

  reset_gate = _ffn_layer_multi_inputs(
      [inputs, state],
      hparams,
      name="reset",
      bias_initializer=tf.constant_initializer(1.0),
      activation=tf.sigmoid,
      pad_remover=pad_remover)

  reset_state = reset_gate * state

  candidate = _ffn_layer_multi_inputs(
      [inputs, reset_state],
      hparams,
      name="candidate",
      bias_initializer=tf.zeros_initializer(),
      activation=tf.tanh,
      pad_remover=pad_remover)

  if "candidate_transformation" in hparams.gru_transformation:
    candidate = attention_unit(candidate)

  if "state_transformation" in hparams.gru_transformation:
    state = attention_unit(state)

  state = update_gate * state + (1 - update_gate) * candidate

  if "state_transformation" in hparams.gru_transformation:
    state = attention_unit(state)
  # normalization on the output
  new_state = common_layers.layer_preprocess(state, hparams)

  return new_state, inputs, memory


def r_transformer_lstm(layer_inputs,
                       step,
                       hparams,
                       attention_unit,
                       pad_remover=None):
  """The RT layer which models recurencey similar to GRU cell.

  It's an R-transformer with a gru applied over the stats on depth.
  based on LSTM paper: https://arxiv.org/pdf/1409.2329.pdf

  Args:
    layer_inputs:
      - state: state
      - inputs: the original embedded inputs (= inputs to the first step)
      - memory: memory used in lstm.
    step: indicating number of steps take so far
    hparams: model hyper-parameters.
    attention_unit: multi-head attention unit
    pad_remover: to mask out padding in convolutional layers (efficiency).

  Returns:
    layer_output:
        new_state: new state
        inputs: the original embedded inputs (= inputs to the first step)
        memory: contains states from all the previous steps.
  """
  state, inputs, memory = layer_inputs
  state = step_preprocess(state, step, hparams)

  state = common_layers.layer_preprocess(state, hparams)
  inputs = common_layers.layer_preprocess(inputs, hparams)

  input_gate = _ffn_layer_multi_inputs(
      [inputs, state],
      hparams,
      name="input_g",
      bias_initializer=tf.zeros_initializer(),
      activation=tf.sigmoid,
      pad_remover=pad_remover)

  forget_gate = _ffn_layer_multi_inputs(
      [inputs, state],
      hparams,
      name="forget_g",
      bias_initializer=tf.zeros_initializer(),
      activation=None,
      pad_remover=pad_remover)

  output_gate = _ffn_layer_multi_inputs(
      [inputs, state],
      hparams,
      name="output_g",
      bias_initializer=tf.zeros_initializer(),
      activation=tf.sigmoid,
      pad_remover=pad_remover)

  input_modulation = _ffn_layer_multi_inputs(
      [inputs, state],
      hparams,
      name="input_modulation",
      bias_initializer=tf.zeros_initializer(),
      activation=tf.tanh,
      pad_remover=pad_remover)

  forget_bias_tensor = tf.constant(hparams.lstm_forget_bias)
  forget_gate = tf.sigmoid(forget_gate + forget_bias_tensor)

  if "modulated_input_transformation" in hparams.lstm_transformation:
    input_modulation = attention_unit(input_modulation)

  memory = memory * forget_gate + input_gate * input_modulation

  if "memory_transformation" in hparams.lstm_transformation:
    memory = attention_unit(memory)

  new_state = tf.tanh(memory) * output_gate

  if "state_transformation" in hparams.lstm_transformation:
    new_state = attention_unit(new_state)

  return new_state, inputs, memory


def r_transformer_act(x, hparams, ffn_unit, attention_unit):
  """ACT based models.

  Implementations of all act models are based on craffel@'s cl/160711592.

  Args:
    x: input
    hparams: model hyper-parameters
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit

  Returns:
    the output tensor,  (ponder_times, remainders)

  Raises:
    ValueError: Unknown act type

  """

  if hparams.act_type == "basic":
    return r_transformer_act_basic(x, hparams, ffn_unit, attention_unit)

  elif hparams.act_type == "accumulated":
    return r_transformer_act_accumulated(x, hparams, ffn_unit, attention_unit)

  elif hparams.act_type == "global":
    return r_transformer_act_global(x, hparams, ffn_unit, attention_unit)

  elif hparams.act_type == "random":
    return r_transformer_act_random(x, hparams, ffn_unit, attention_unit)

  else:
    raise ValueError("Unknown act type: %s" % hparams.act_type)


def r_transformer_act_basic(x, hparams, ffn_unit, attention_unit):
  """Basic r_transformer with ACT based on remainder-distribution ACT.

  Args:
    x: input
    hparams: model hyper-parameters
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit

  Returns:
    the output tensor,  (ponder_times, remainders)

  """

  state = x
  act_max_steps = hparams.act_max_steps
  threshold = 1.0 - hparams.act_epsilon

  batch_size = tf.shape(state)[0]
  length = tf.shape(state)[1]

  # Halting probabilities (p_t^n in the paper)
  halting_probability = tf.zeros(
      (
          batch_size,
          length,
      ), name="halting_probability")
  # Remainders (R(t) in the paper)
  remainders = tf.zeros(
      (
          batch_size,
          length,
      ), name="remainder")
  # Number of updates performed (N(t) in the paper)
  n_updates = tf.zeros(
      (
          batch_size,
          length,
      ), name="n_updates")

  # Previous cell states (s_t in the paper)
  previous_state = tf.zeros_like(state, name="previous_state")
  step = tf.constant(0, dtype=tf.int32)

  def rt_function(state, step, halting_probability, remainders, n_updates,
                  previous_state):
    """implements act (position-wise halting).

    Args:
      state: 3-D Tensor: [batch_size, length, channel]
      step: indicating number of steps take so far
      halting_probability: halting probability
      remainders: act remainders
      n_updates: act n_updates
      previous_state: previous state

    Returns:
      transformed_state: transformed state
      step: step+1
      halting_probability: halting probability
      remainders: act remainders
      n_updates: act n_updates
      new_state: new state
    """
    state_shape = state.get_shape()
    state = step_preprocess(state, step, hparams)

    with tf.variable_scope("sigmoid_activation_for_pondering"):
      p = common_layers.dense(
          state,
          1,
          activation=tf.nn.sigmoid,
          use_bias=True,
          bias_initializer=tf.constant_initializer(
              hparams.act_halting_bias_init))
      p = tf.squeeze(p)

    # Mask for inputs which have not halted yet
    still_running = tf.cast(tf.less(halting_probability, 1.0), tf.float32)

    # Mask of inputs which halted at this step
    new_halted = tf.cast(
        tf.greater(halting_probability + p * still_running, threshold),
        tf.float32) * still_running

    # Mask of inputs which haven't halted, and didn't halt this step
    still_running = tf.cast(
        tf.less_equal(halting_probability + p * still_running, threshold),
        tf.float32) * still_running

    # Add the halting probability for this step to the halting
    # probabilities for those input which haven't halted yet
    halting_probability += p * still_running

    # Compute remainders for the inputs which halted at this step
    remainders += new_halted * (1 - halting_probability)

    # Add the remainders to those inputs which halted at this step
    halting_probability += new_halted * remainders

    # Increment n_updates for all inputs which are still running
    n_updates += still_running + new_halted

    # Compute the weight to be applied to the new state and output
    # 0 when the input has already halted
    # p when the input hasn't halted yet
    # the remainders when it halted this step
    update_weights = tf.expand_dims(p * still_running + new_halted * remainders,
                                    -1)

    # apply transformation on the state
    transformed_state = ffn_unit(attention_unit(state))

    # update running part in the weighted state and keep the rest
    new_state = ((transformed_state * update_weights) +
                 (previous_state * 1 - update_weights))

    # remind TensorFlow of everything's shape
    transformed_state.set_shape(state_shape)
    for x in [halting_probability, remainders, n_updates]:
      x.set_shape([
          state_shape[0],
          state_shape[1],
      ])
      new_state.set_shape(state_shape)
    step += 1
    return (transformed_state, step, halting_probability, remainders, n_updates,
            new_state)

  # While loop stops when this predicate is FALSE.
  # Ie all (probability < 1-eps AND counter < N) are false.
  def should_continue(u0, u1, halting_probability, u2, n_updates, u3):
    del u0, u1, u2, u3
    return tf.reduce_any(
        tf.logical_and(
            tf.less(halting_probability, threshold),
            tf.less(n_updates, act_max_steps)))

  # Do while loop iterations until predicate above is false.
  (_, _, _, remainder, n_updates, new_state) = tf.while_loop(
      should_continue, rt_function,
      (state, step, halting_probability, remainders, n_updates, previous_state))

  ponder_times = n_updates
  remainders = remainder

  tf.contrib.summary.scalar("ponder_times", tf.reduce_mean(ponder_times))

  return new_state, (ponder_times, remainders)


def r_transformer_act_accumulated(x, hparams, ffn_unit, attention_unit):
  """The RTAct layer where the final state is accumulation of all states.

    (similar to the main ACT paper: --> check the issue of differentiability)

  Args:
    x: input
    hparams: model hyper-parameters
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit

  Returns:
    the output tensor,  (ponder_times, remainders)

  """
  state = x
  act_max_steps = hparams.act_max_steps
  threshold = 1.0 - hparams.act_epsilon

  batch_size = tf.shape(state)[0]
  length = tf.shape(state)[1]

  # Halting probabilities (p_t^n in the paper)
  halting_probability = tf.zeros(
      (
          batch_size,
          length,
      ), name="halting_probability")
  # Remainders (R(t) in the paper)
  remainders = tf.zeros(
      (
          batch_size,
          length,
      ), name="remainder")
  # Number of updates performed (N(t) in the paper)
  n_updates = tf.zeros(
      (
          batch_size,
          length,
      ), name="n_updates")

  # Accumulated cell states (s_t in the paper)
  accumulated_state = tf.zeros_like(state, name="previous_state")
  step = tf.constant(0, dtype=tf.int32)

  def rt_function(state, step, halting_probability, remainders, n_updates,
                  accumulated_state):
    """Position-wise act.

    Args:
      state: 3-D Tensor: [batch_size, length, channel]
      step: indicating number of steps take so far
      halting_probability: halting probability
      remainders: act remainders
      n_updates: act n_updates
      accumulated_state: accumulated state

    Returns:
      transformed_state: transformed state
      step: step+1
      halting_probability: halting probability
      remainders: act remainders
      n_updates: act n_updates
      accumulated_state: accumulated state
    """
    state_shape = state.get_shape()
    state = step_preprocess(state, step, hparams)

    with tf.variable_scope("sigmoid_activation_for_pondering"):
      p = common_layers.dense(
          state,
          1,
          activation=tf.nn.sigmoid,
          use_bias=True,
          bias_initializer=tf.constant_initializer(
              hparams.act_halting_bias_init))
      p = tf.squeeze(p)

    # Mask for inputs which have not halted yet
    still_running = tf.cast(tf.less(halting_probability, 1.0), tf.float32)

    # Mask of inputs which halted at this step
    new_halted = tf.cast(
        tf.greater(halting_probability + p * still_running, threshold),
        tf.float32) * still_running

    # Mask of inputs which haven't halted, and didn't halt this step
    still_running = tf.cast(
        tf.less_equal(halting_probability + p * still_running, threshold),
        tf.float32) * still_running

    # Add the halting probability for this step to the halting
    # probabilities for those input which haven't halted yet
    halting_probability += p * still_running

    # Compute remainders for the inputs which halted at this step
    remainders += new_halted * (1 - halting_probability)

    # Add the remainders to those inputs which halted at this step
    halting_probability += new_halted * remainders

    # Increment n_updates for all inputs which are still running
    n_updates += still_running + new_halted

    # Compute the weight to be applied to the new state and output
    # 0 when the input has already halted
    # p when the input hasn't halted yet
    # the remainders when it halted this step
    update_weights = tf.expand_dims(p * still_running + new_halted * remainders,
                                    -1)

    # apply transformation on the state
    transformed_state = ffn_unit(attention_unit(state))

    # Add in the weighted state
    accumulated_state = (transformed_state * update_weights) + accumulated_state

    # Remind TensorFlow of everything's shape
    state.set_shape(state_shape)
    for x in [halting_probability, remainders, n_updates]:
      x.set_shape([
          state_shape[0],
          state_shape[1],
      ])
    accumulated_state.set_shape(state_shape)
    step += 1
    return (transformed_state, step, halting_probability, remainders, n_updates,
            accumulated_state)

  # While loop stops when this predicate is FALSE.
  # Ie all (probability < 1-eps AND counter < N) are false.
  def should_continue(u0, u1, halting_probability, u2, n_updates, u3):
    del u0, u1, u2, u3
    return tf.reduce_any(
        tf.logical_and(
            tf.less(halting_probability, threshold),
            tf.less(n_updates, act_max_steps)))

  # Do while loop iterations until predicate above is false.
  (_, _, _, remainder, n_updates, accumulated_state) = tf.while_loop(
      should_continue, rt_function, (state, step, halting_probability,
                                     remainders, n_updates, accumulated_state))

  ponder_times = n_updates
  remainders = remainder

  tf.contrib.summary.scalar("ponder_times", tf.reduce_mean(ponder_times))

  return accumulated_state, (ponder_times, remainders)


def r_transformer_act_global(x, hparams, ffn_unit, attention_unit):
  """The RTAct  with global halting probability (not position-wise).

  Args:
    x: input
    hparams: model hyper-parameters
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit

  Returns:
    the output tensor,  (ponder_times, remainders)

  """
  state = x
  act_max_steps = hparams.act_max_steps
  threshold = 1.0 - hparams.act_epsilon
  act_max_steps = hparams.act_max_steps
  batch_size = tf.shape(state)[0]
  state_shape = state.get_shape()

  # Halting probabilities (p_t^n in the paper)
  halting_probability = tf.zeros((batch_size,), name="halting_probability")
  # Remainders (R(t) in the paper)
  remainders = tf.zeros((batch_size,), name="remainder")
  # Number of updates performed (N(t) in the paper)
  n_updates = tf.zeros((batch_size,), name="n_updates")
  # Previous cell states (s_t in the paper)
  previous_state = tf.zeros_like(state, name="previous_state")
  step = tf.constant(0, dtype=tf.int32)

  def rt_function(state, step, halting_probability, remainders, n_updates,
                  previous_state):
    """implements act (global halting).

    Args:
      state: 3-D Tensor: [batch_size, length, channel]
      step: indicating number of steps take so far
      halting_probability: halting probability
      remainders: act remainders
      n_updates: act n_updates
      previous_state: previous state

    Returns:
      transformed_state: transformed state
      step: step+1
      halting_probability: halting probability
      remainders: act remainders
      n_updates: act n_updates
      new_state: new state

    """

    state = step_preprocess(state, step, hparams)

    with tf.variable_scope("sigmoid_activation_for_pondering"):
      p = common_layers.dense(
          state,
          1,
          activation=tf.nn.sigmoid,
          use_bias=True,
          bias_initializer=tf.constant_initializer(
              hparams.act_halting_bias_init))
      # average over all positions (as a global halting prob)
      p = tf.reduce_mean(p, axis=1)
      p = tf.squeeze(p)

    # Mask for inputs which have not halted yet
    still_running = tf.cast(tf.less(halting_probability, 1.0), tf.float32)

    # Mask of inputs which halted at this step
    new_halted = tf.cast(
        tf.greater(halting_probability + p * still_running, threshold),
        tf.float32) * still_running

    # Mask of inputs which haven't halted, and didn't halt this step
    still_running = tf.cast(
        tf.less_equal(halting_probability + p * still_running, threshold),
        tf.float32) * still_running

    # Add the halting probability for this step to the halting
    # probabilities for those input which haven't halted yet
    halting_probability += p * still_running

    # Compute remainders for the inputs which halted at this step
    remainders += new_halted * (1 - halting_probability)

    # Add the remainders to those inputs which halted at this step
    halting_probability += new_halted * remainders

    # Increment n_updates for all inputs which are still running
    n_updates += still_running + new_halted

    # Compute the weight to be applied to the new state and output
    # 0 when the input has already halted
    # p when the input hasn't halted yet
    # the remainders when it halted this step
    update_weights = tf.expand_dims(
        tf.expand_dims(p * still_running + new_halted * remainders, -1), -1)

    # apply transformation on the state
    transformed_state = ffn_unit(attention_unit(state))

    # Add in the weighted state
    new_state = ((transformed_state * update_weights) +
                 (previous_state * 1 - update_weights))

    # Remind TensorFlow of everything's shape
    state.set_shape(state_shape)
    for x in [halting_probability, remainders, n_updates]:
      x.set_shape([
          state_shape[0],
      ])
      new_state.set_shape(state_shape)

    step += 1
    return [
        transformed_state, step, halting_probability, remainders, n_updates,
        new_state
    ]

  # While loop stops when this predicate is FALSE.
  # Ie all (probability < 1-eps AND counter < N) are false.
  def should_continue(u0, u1, halting_probability, u2, n_updates, u3):
    del u0, u1, u2, u3
    return tf.reduce_any(
        tf.logical_and(
            tf.less(halting_probability, threshold),
            tf.less(n_updates, act_max_steps)))

  # Do while loop iterations until predicate above is false.
  (_, _, _, remainder, n_updates, new_state) = tf.while_loop(
      should_continue, rt_function,
      (state, step, halting_probability, remainders, n_updates, previous_state))

  ponder_times = n_updates
  remainders = remainder

  tf.contrib.summary.scalar("ponder_times", tf.reduce_mean(ponder_times))

  return new_state, (ponder_times, remainders)


def r_transformer_act_random(x, hparams, ffn_unit, attention_unit):
  """r_transformer with ACT with random halting probability.

  Args:
    x: input
    hparams: model hyper-parameters
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit

  Returns:
    the output tensor,  (ponder_times, remainders)

  """

  state = x
  act_max_steps = hparams.act_max_steps
  threshold = 1.0 - hparams.act_epsilon

  batch_size = tf.shape(state)[0]
  length = tf.shape(state)[1]

  # Halting probabilities (p_t^n in the paper)
  halting_probability = tf.zeros(
      (
          batch_size,
          length,
      ), name="halting_probability")
  # Remainders (R(t) in the paper)
  remainders = tf.zeros(
      (
          batch_size,
          length,
      ), name="remainder")
  # Number of updates performed (N(t) in the paper)
  n_updates = tf.zeros(
      (
          batch_size,
          length,
      ), name="n_updates")

  # Previous cell states (s_t in the paper)
  previous_state = tf.zeros_like(state, name="previous_state")
  step = tf.constant(0, dtype=tf.int32)

  def rt_function(state, step, halting_probability, remainders, n_updates,
                  previous_state):
    """Implements act (position-wise halting).

    Args:
      state: 3-D Tensor: [batch_size, length, channel]
      step: indicating number of steps take so far
      halting_probability: halting probability
      remainders: act remainders
      n_updates: act n_updates
      previous_state: previous state

    Returns:
      transformed_state: transformed state
      step: step+1
      halting_probability: halting probability
      remainders: act remainders
      n_updates: act n_updates
      new_state: new state

    """
    state_shape = state.get_shape()
    state = step_preprocess(state, step, hparams)

    # random as halting probability
    p = tf.random_uniform(shape=common_layers.shape_list(halting_probability))

    # Mask for inputs which have not halted yet
    still_running = tf.cast(tf.less(halting_probability, 1.0), tf.float32)

    # Mask of inputs which halted at this step
    new_halted = tf.cast(
        tf.greater(halting_probability + p * still_running, threshold),
        tf.float32) * still_running

    # Mask of inputs which haven't halted, and didn't halt this step
    still_running = tf.cast(
        tf.less_equal(halting_probability + p * still_running, threshold),
        tf.float32) * still_running

    # Add the halting probability for this step to the halting
    # probabilities for those input which haven't halted yet
    halting_probability += p * still_running

    # Compute remainders for the inputs which halted at this step
    remainders += new_halted * (1 - halting_probability)

    # Add the remainders to those inputs which halted at this step
    halting_probability += new_halted * remainders

    # Increment n_updates for all inputs which are still running
    n_updates += still_running + new_halted

    # Compute the weight to be applied to the new state and output
    # 0 when the input has already halted
    # p when the input hasn't halted yet
    # the remainders when it halted this step
    update_weights = tf.expand_dims(p * still_running + new_halted * remainders,
                                    -1)

    # apply transformation on the state
    transformed_state = ffn_unit(attention_unit(state))

    # update running part in the weighted state and keep the rest
    new_state = ((transformed_state * update_weights) +
                 (previous_state * 1 - update_weights))

    # remind TensorFlow of everything's shape
    transformed_state.set_shape(state_shape)
    for x in [halting_probability, remainders, n_updates]:
      x.set_shape([
          state_shape[0],
          state_shape[1],
      ])
      new_state.set_shape(state_shape)
    step += 1
    return [
        transformed_state, step, halting_probability, remainders, n_updates,
        new_state
    ]

  # While loop stops when this predicate is FALSE.
  # Ie all (probability < 1-eps AND counter < N) are false.
  def should_continue(u0, u1, halting_probability, u2, n_updates, u3):
    del u0, u1, u2, u3
    return tf.reduce_any(
        tf.logical_and(
            tf.less(halting_probability, threshold),
            tf.less(n_updates, act_max_steps)))

  # Do while loop iterations until predicate above is false.
  (_, _, _, remainder, n_updates, new_state) = tf.while_loop(
      should_continue, rt_function,
      (state, step, halting_probability, remainders, n_updates, previous_state))

  ponder_times = n_updates
  remainders = remainder

  tf.contrib.summary.scalar("ponder_times", tf.reduce_mean(ponder_times))

  return new_state, (ponder_times, remainders)


def _ffn_layer_multi_inputs(inputs_list,
                            hparams,
                            ffn_layer_type="dense",
                            name="ffn",
                            kernel_initializer=None,
                            bias_initializer=None,
                            activation=None,
                            pad_remover=None,
                            preprocess=True,
                            postprocess=True):
  """Implements a Feed-forward layer with multiple inputs, pad-removing, etc.

  Args:
    inputs_list: list of input tensors
    hparams: hyper-parameters
    ffn_layer_type: dense / dense_dropconnect/ dense_relu_dense
    name: name
    kernel_initializer: kernel initializer
    bias_initializer: bias initializer
    activation: activation function
    pad_remover: pad remover
    preprocess: if preprocess the input
    postprocess: if postprocess the output

  Returns:
    a tensor
  Raises:
    ValueError: Unknown ffn_layer type.

  """

  # need at least one inputs
  num_inputs = len(inputs_list)
  assert num_inputs > 0

  if preprocess and num_inputs == 1:
    inputs_list[0] = common_layers.layer_preprocess(inputs_list[0], hparams)

  if postprocess:
    original_inputs = inputs_list[0]

  # the output size is the hidden size of the main inputs
  main_input = inputs_list[0]
  original_shape = common_layers.shape_list(main_input)
  assert hparams.hidden_size == common_layers.shape_list(main_input)[-1]

  # all the inputs are in the same shape with main inputs
  for inputs in inputs_list:
    main_input.get_shape().assert_is_compatible_with(inputs.get_shape())

  def remove_pads(x):
    original_shape = common_layers.shape_list(x)
    # Collapse `x` across examples, and remove padding positions.
    x = tf.reshape(x, tf.concat([[-1], original_shape[2:]], axis=0))
    x = tf.expand_dims(pad_remover.remove(x), axis=0)
    return x

  if pad_remover:
    for i, inputs in enumerate(inputs_list):
      inputs_list[i] = remove_pads(inputs)

  ffn_inputs = (
      inputs_list[0]
      if len(inputs_list) == 1 else tf.concat(inputs_list, axis=-1))

  if ffn_layer_type == "dense":
    output = common_layers.dense(
        ffn_inputs,
        hparams.hidden_size,
        name=name,
        activation=activation,
        use_bias=True,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer)

  elif ffn_layer_type == "dense_dropconnect":
    output = common_layers.dense_dropconnect(
        ffn_inputs,
        hparams.hidden_size,
        name=name,
        dropconnect_dropout=hparams.dropconnect_dropout,
        output_activation=activation)
    postprocess = False  # no dropout on the output unit

  elif ffn_layer_type == "dense_relu_dense":
    output = common_layers.dense_relu_dense(
        ffn_inputs,
        hparams.filter_size,
        hparams.hidden_size,
        name=name,
        dropout=hparams.relu_dropout,
        output_activation=activation,
    )

  else:
    raise ValueError("Unknown ffn_layer type: %s" % ffn_layer_type)

  if pad_remover:
    # Restore `output` to the original shape of `x`, including padding.
    output = tf.reshape(
        pad_remover.restore(tf.squeeze(output, axis=0)), original_shape)

  if postprocess:
    if num_inputs == 1:
      output = common_layers.layer_postprocess(original_inputs, output, hparams)
    else:  # only dropout (no residual)x
      hp = copy.copy(hparams)
      hp.layer_postprocess_sequence = hp.layer_postprocess_sequence.replace(
          "a", "")
      output = common_layers.layer_postprocess(original_inputs, output, hp)

  return output


def fill_memory_slot(memory, value, index):
  """Fills the memory slot at a particular index with the given value.

  Args:
    memory: a 4-d tensor [memory_size, batch, length, channel] containing
      the state of all steps
    value: a 3-d tensor [batch, length, channel] as the sate
    index: integer in [0, memory_size)

  Returns:
    filled memory

  """
  mask = tf.to_float(
      tf.one_hot(index,
                 tf.shape(memory)[0])[:, None, None, None])
  fill_memory = (1 - mask) * memory + mask * value[None, ...]
  return fill_memory


def add_depth_embedding(x):
  """Add n-dimensional embedding as the depth embedding (timing signal).

  Adds embeddings to represent the position of the step in the recurrent
  tower.

  Args:
    x: a tensor with shape [max_step, batch, length, depth]

  Returns:
    a Tensor the same shape as x.
  """
  x_shape = common_layers.shape_list(x)
  depth = x_shape[-1]
  num_steps = x_shape[0]
  shape = [num_steps, 1, 1, depth]
  depth_embedding = (
      tf.get_variable(
          "depth_embedding",
          shape,
          initializer=tf.random_normal_initializer(0, depth**-0.5)) * (depth**
                                                                       0.5))

  x += depth_embedding
  return x


def step_preprocess(x, step, hparams):
  """Preprocess the input at the beginning of each step.

  Args:
    x: input tensor
    step: step
    hparams: model hyper-parameters

  Returns:
    preprocessed input.

  """
  original_channel_size = common_layers.shape_list(x)[-1]

  if hparams.add_position_timing_signal:
    x = add_position_timing_signal(x, step, hparams)

  if hparams.add_step_timing_signal:
    x = add_step_timing_signal(x, step, hparams)

  if ((hparams.add_position_timing_signal or hparams.add_position_timing_signal)
      and hparams.add_or_concat_timing_signal == "concat"):
    # linear projection to the original dimension of x
    x = common_layers.dense(
        x, original_channel_size, activation=None, use_bias=False)

  if hparams.add_sru:
    x = common_layers.sru(x)

  return x


def add_position_timing_signal(x, step, hparams):
  """Add n-dimensional embedding as the position (horizontal) timing signal.

  Args:
    x: a tensor with shape [batch, length, depth]
    step: step
    hparams: model hyper parameters

  Returns:
    a Tensor with the same shape as x.

  """

  if not hparams.position_start_index:
    index = 0

  elif hparams.position_start_index == "random":
    # Shift all positions randomly
    # TODO(dehghani): What would be reasonable for max number of shift?
    index = tf.random_uniform(
        [], maxval=common_layers.shape_list(x)[1], dtype=tf.int32)

  elif hparams.position_start_index == "step":
    # Shift positions based on the step
    num_steps = (
        hparams.act_max_steps
        if hparams.recurrence_type == "act" else hparams.num_rec_steps)
    index = tf.cast(
        common_layers.shape_list(x)[1] * step / num_steps, dtype=tf.int32)

  # No need for the timing signal in the encoder/decoder input preparation
  assert hparams.pos is None

  length = common_layers.shape_list(x)[1]
  channels = common_layers.shape_list(x)[2]
  signal = common_attention.get_timing_signal_1d(
      length, channels, start_index=index)

  if hparams.add_or_concat_timing_signal == "add":
    x_with_timing = x + signal

  elif hparams.add_or_concat_timing_signal == "concat":
    batch_size = common_layers.shape_list(x)[0]
    signal_tiled = tf.tile(signal, [batch_size, 1, 1])
    x_with_timing = tf.concat((x, signal_tiled), axis=-1)

  return x_with_timing


def add_step_timing_signal(x, step, hparams):
  """Add n-dimensional embedding as the step (vertical) timing signal.

  Args:
    x: a tensor with shape [batch, length, depth]
    step: step
    hparams: model hyper parameters

  Returns:
    a Tensor with the same shape as x.

  """
  num_steps = (
      hparams.act_max_steps
      if hparams.recurrence_type == "act" else hparams.num_rec_steps)
  channels = common_layers.shape_list(x)[-1]

  if hparams.step_timing_signal_type == "learned":
    signal = common_attention.get_layer_timing_signal_learned_1d(
        channels, step, num_steps)

  elif hparams.step_timing_signal_type == "sinusoid":
    signal = common_attention.get_layer_timing_signal_sinusoid_1d(
        channels, step, num_steps)

  if hparams.add_or_concat_timing_signal == "add":
    x_with_timing = x + signal

  elif hparams.add_or_concat_timing_signal == "concat":
    batch_size = common_layers.shape_list(x)[0]
    length = common_layers.shape_list(x)[1]
    signal_tiled = tf.tile(signal, [batch_size, length, 1])
    x_with_timing = tf.concat((x, signal_tiled), axis=-1)

  return x_with_timing
