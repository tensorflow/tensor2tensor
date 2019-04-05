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

"""Utilities for Universal Transformer.

The Universal Transformer is based on the popular encoder-decoder architecture.
However, as opposed to a fixed stack of distinct layers (as is usually the case
for most popular neural sequence models), the Universal Transformer is
recurrent "in depth", and repeatedly applies the same series of functions with
the same parameters to all elements of the sequence in parallel, revising their
representations with every step. The encoder and decoder have the same
recurrent structure, but the decoder additionally consumes the final encoder
representations for each position. Like the Transformer, the Universal
Transformer is autoregressive. Trained using teacher-forcing, at generation
time it produces its output one position at a time, with the decoder consuming
the previously produced output positions.

Given an input sequence of length m, we start with a matrix whose rows are the
d-dimensional embeddings of the symbols at each position of the sequence.
The Universal Transformer then iteratively computes representation of the input
at each step by applying the multiheaded dot-product self-attention mechanism,
followed by a recurrent transition function. We also add residual connections
around each of these function blocks and apply dropout and layer normalization.

The recurrent transition function in fact controls how steps communicate with
each other in depth. For instance, the recurrent transition, can be a simple
identity function which passes the output of a step as the input to next step.
Or it can be an LSTM (filliped vertically) next to the transformer which
controls how state of the model changes in depth.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools

from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import expert_utils

import tensorflow as tf


def universal_transformer_encoder(encoder_input,
                                  encoder_self_attention_bias,
                                  hparams,
                                  name="encoder",
                                  nonpadding=None,
                                  save_weights_to=None,
                                  make_image_summary=True):
  """Universal Transformer encoder function.

  Prepares all the arguments and the inputs and passes it to a
  universal_transformer_layer to encode the encoder_input.

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
    if hparams.use_pad_remover and not common_layers.is_xla_compiled():
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

    x, extra_output = universal_transformer_layer(
        x, hparams, ffn_unit, attention_unit, pad_remover=pad_remover)

    return common_layers.layer_preprocess(x, hparams), extra_output


def universal_transformer_decoder(decoder_input,
                                  encoder_output,
                                  decoder_self_attention_bias,
                                  encoder_decoder_attention_bias,
                                  hparams,
                                  name="decoder",
                                  nonpadding=None,
                                  save_weights_to=None,
                                  make_image_summary=True):
  """Universal Transformer decoder function.

  Prepares all the arguments and the inputs and passes it to a
  core_universal_transformer_layer to decoder.

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

    x, extra_output = universal_transformer_layer(
        x, hparams, ffn_unit, attention_unit)

    return common_layers.layer_preprocess(x, hparams), extra_output


def universal_transformer_layer(x,
                                hparams,
                                ffn_unit,
                                attention_unit,
                                pad_remover=None):
  """Core function applying the universal transformer layer.

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

  def add_vanilla_transformer_layer(x, num_layers, name):
    """Passes the input through num_layers of vanilla transformer layers.

    Args:
     x: input
     num_layers: number of layers
     name: string, prefix of layer names

    Returns:
       output of vanilla_transformer_layer
    """
    if hparams.add_position_timing_signal:
      # In case of add_position_timing_signal=true, we set  hparams.pos=None
      # and add position timing signal at the beginning of each step, so for
      # the vanilla transformer, we need to add timing signal here.
      x = common_attention.add_timing_signal_1d(x)
    for layer in range(num_layers):
      with tf.variable_scope(name + "layer_%d" % layer):
        x = ffn_unit(attention_unit(x))
    return x

  with tf.variable_scope("universal_transformer_%s" % hparams.recurrence_type):
    if (hparams.mix_with_transformer and
        "before_ut" in hparams.mix_with_transformer):
      x = add_vanilla_transformer_layer(x, hparams.num_mixedin_layers,
                                        "before_ut_")

    if hparams.recurrence_type == "act":
      output, extra_output = universal_transformer_act(
          x, hparams, ffn_unit, attention_unit)

    else:  # for all the other recurrency types with fixed number of steps

      ut_function, initializer = get_ut_layer(x, hparams, ffn_unit,
                                              attention_unit, pad_remover)

      output, _, extra_output = tf.foldl(
          ut_function, tf.range(hparams.num_rec_steps),
          initializer=initializer)

      # Right now, this is only possible when the transition function is an lstm
      if (hparams.recurrence_type == "lstm" and
          hparams.get("use_memory_as_final_state", False)):
        output = extra_output

    if (hparams.mix_with_transformer and
        "after_ut" in hparams.mix_with_transformer):
      output = add_vanilla_transformer_layer(output, hparams.num_mixedin_layers,
                                             "after_ut_")

    return output, extra_output


def get_ut_layer(x,
                 hparams,
                 ffn_unit,
                 attention_unit,
                 pad_remover=None):
  """Provides the function that is used in universal transforemr steps.

  Args:
    x: input
    hparams: model hyper-parameters
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit
    pad_remover: to mask out padding in convolutional layers (efficiency).

  Returns:
    ut_function and the ut_initializer

  Raises:
    ValueError: Unknown recurrence type
  """

  if hparams.recurrence_type == "basic":
    ut_initializer = (x, x, x)  # (state, input, memory)
    ut_function = functools.partial(
        universal_transformer_basic,
        hparams=hparams,
        ffn_unit=ffn_unit,
        attention_unit=attention_unit)

  elif hparams.recurrence_type == "highway":
    ut_initializer = (x, x, x)  # (state, input, memory)
    ut_function = functools.partial(
        universal_transformer_highway,
        hparams=hparams,
        ffn_unit=ffn_unit,
        attention_unit=attention_unit,
        pad_remover=pad_remover)

  elif hparams.recurrence_type == "skip":
    ut_initializer = (x, x, x)  # (state, input, memory)
    ut_function = functools.partial(
        universal_transformer_skip,
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

    ut_initializer = (x, x, memory)  # (state, input, memory)
    ut_function = functools.partial(
        universal_transformer_depthwise_attention,
        hparams=hparams,
        ffn_unit=ffn_unit,
        attention_unit=attention_unit)

  elif hparams.recurrence_type == "gru":
    ut_initializer = (x, x, x)  # (state, input, memory)
    ut_function = functools.partial(
        universal_transformer_with_gru_as_transition_function,
        hparams=hparams,
        ffn_unit=ffn_unit,
        attention_unit=attention_unit,
        pad_remover=pad_remover)

  elif hparams.recurrence_type == "lstm":
    memory = tf.zeros(common_layers.shape_list(x))
    ut_initializer = (x, x, memory)  # (state, input, memory)
    ut_function = functools.partial(
        universal_transformer_with_lstm_as_transition_function,
        hparams=hparams,
        ffn_unit=ffn_unit,
        attention_unit=attention_unit,
        pad_remover=pad_remover)

  else:
    raise ValueError("Unknown recurrence type: %s" % hparams.recurrence_type)

  return ut_function, ut_initializer


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
          first_kernel_size=(3, 1),
          second_kernel_size=(5, 1),
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
        dropout_broadcast_dims=attention_dropout_broadcast_dims,
        hard_attention_k=hparams.hard_attention_k)
    x = common_layers.layer_postprocess(x, y, hparams)
  return x


def transformer_decoder_ffn_unit(x,
                                 hparams,
                                 nonpadding_mask=None):
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
          first_kernel_size=(3, 1),
          second_kernel_size=(5, 1),
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
        dropout_broadcast_dims=attention_dropout_broadcast_dims,
        hard_attention_k=hparams.hard_attention_k)
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
          dropout_broadcast_dims=attention_dropout_broadcast_dims,
          hard_attention_k=hparams.hard_attention_k)
      x = common_layers.layer_postprocess(x, y, hparams)
  return x


def universal_transformer_basic(layer_inputs,
                                step, hparams,
                                ffn_unit,
                                attention_unit):
  """Basic Universal Transformer.

  This model is pretty similar to the vanilla transformer in which weights are
  shared between layers. For some tasks, this simple idea brings a
  generalization that is not achievable by playing with the size of the model
  or drop_out parameters in the vanilla transformer.

  Args:
    layer_inputs:
        - state: state
    step: indicates number of steps taken so far
    hparams: model hyper-parameters
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit

  Returns:
    layer_output:
         new_state: new state
  """
  state, inputs, memory = tf.unstack(layer_inputs, num=None, axis=0,
                                     name="unstack")
  new_state = step_preprocess(state, step, hparams)

  for i in range(hparams.num_inrecurrence_layers):
    with tf.variable_scope("rec_layer_%d" % i):
      new_state = ffn_unit(attention_unit(new_state))

  return new_state, inputs, memory


def universal_transformer_highway(layer_inputs,
                                  step, hparams,
                                  ffn_unit,
                                  attention_unit,
                                  pad_remover=None):
  """Universal Transformer with highway connection.


  It transforms the state using a block contaaining sel-attention and transition
  function  and wrap the whole block with a highway connection.
  (the new state is a combination of the state and the transformed-state
  based on cary/transform gates.)

  Interesting observation:
    Controlling the cary/transform gate with the original inputs works usually
    better (i.e. hparams.gates_inputs="i")

  Args:
    layer_inputs:
      - state: state
      - inputs: the original embedded inputs (= inputs to the first step)
    step: indicates number of steps taken so far
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
  new_state = step_preprocess(state, step, hparams)

  for i in range(hparams.num_inrecurrence_layers):
    with tf.variable_scope("rec_layer_%d" % i):
      new_state = ffn_unit(attention_unit(new_state))

  transformed_state = new_state

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


def universal_transformer_skip(layer_inputs,
                               step,
                               hparams,
                               ffn_unit,
                               attention_unit,
                               pad_remover=None):
  """Universal Transformer with highway connection.


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
    step: indicates number of steps taken so far
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
  new_state = step_preprocess(state, step, hparams)

  for i in range(hparams.num_inrecurrence_layers):
    with tf.variable_scope("rec_layer_%d" % i):
      new_state = ffn_unit(attention_unit(new_state))

  transformed_state = new_state

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


def universal_transformer_depthwise_attention(layer_inputs,
                                              step, hparams,
                                              ffn_unit,
                                              attention_unit):
  """universal_transformer with depth-wise attention.

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

  new_state = step_preprocess(state_to_be_transformed, step, hparams)

  for i in range(hparams.num_inrecurrence_layers):
    with tf.variable_scope("rec_layer_%d" % i):
      new_state = ffn_unit(attention_unit(new_state))

  # add the new state to the memory
  memory = fill_memory_slot(memory, new_state, step + 1)

  return new_state, inputs, memory


def universal_transformer_with_gru_as_transition_function(
    layer_inputs, step, hparams, ffn_unit, attention_unit, pad_remover=None):
  """Universal Transformer which uses a gru as transition function.

  It's kind of like having a gru, filliped vertically next to the Universal
  Transformer that controls the flow of the information in depth,
  over different steps of the Universal Transformer.

  Args:
    layer_inputs:
      - state: state
      - inputs: not used here
      - memory: not used here
    step: indicates number of steps taken so far
    hparams: model hyper-parameters.
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit
    pad_remover: to mask out padding in convolutional layers (efficiency).
  Returns:
    layer_output:
        new_state: new state
        inputs: not uesed
        memory: not used
  """

  state, unused_inputs, unused_memory = tf.unstack(
      layer_inputs, num=None, axis=0, name="unstack")

  # state (ut_state): output of the gru in the previous step

  # Multi_head_attention:
  assert not hparams.add_step_timing_signal   # Let gru count for us!
  mh_attention_input = step_preprocess(state, step, hparams)
  transition_function_input = attention_unit(mh_attention_input)

  # Transition Function:
  if hparams.add_ffn_unit_to_the_transition_function:
    transition_function_input = ffn_unit(transition_function_input)

  transition_function_input = common_layers.layer_preprocess(
      transition_function_input, hparams)
  with tf.variable_scope("gru"):
    # gru update gate: z_t = sigmoid(W_z.x_t + U_z.h_{t-1})
    transition_function_update_gate = _ffn_layer_multi_inputs(
        [transition_function_input, state],
        hparams,
        name="update",
        bias_initializer=tf.constant_initializer(1.0),
        activation=tf.sigmoid,
        pad_remover=pad_remover,
        preprocess=False,
        postprocess=False)

    tf.contrib.summary.scalar("gru_update_gate",
                              tf.reduce_mean(transition_function_update_gate))

    # gru reset gate: r_t = sigmoid(W_r.x_t + U_r.h_{t-1})
    transition_function_reset_gate = _ffn_layer_multi_inputs(
        [transition_function_input, state],
        hparams,
        name="reset",
        bias_initializer=tf.constant_initializer(1.0),
        activation=tf.sigmoid,
        pad_remover=pad_remover,
        preprocess=False,
        postprocess=False)

    tf.contrib.summary.scalar("gru_reset_gate",
                              tf.reduce_mean(transition_function_reset_gate))
    reset_state = transition_function_reset_gate * state

    # gru_candidate_activation: h' = tanh(W_{x_t} + U (r_t h_{t-1})
    transition_function_candidate = _ffn_layer_multi_inputs(
        [transition_function_input, reset_state],
        hparams,
        name="candidate",
        bias_initializer=tf.zeros_initializer(),
        activation=tf.tanh,
        pad_remover=pad_remover,
        preprocess=False,
        postprocess=False)

    transition_function_output = (
        (1 - transition_function_update_gate) * transition_function_input +
        transition_function_update_gate * transition_function_candidate)

  transition_function_output = common_layers.layer_preprocess(
      transition_function_output, hparams)

  return transition_function_output, unused_inputs, unused_memory


def universal_transformer_with_lstm_as_transition_function(
    layer_inputs, step, hparams, ffn_unit, attention_unit, pad_remover=None):
  """Universal Transformer which uses a lstm as transition function.

  It's kind of like having a lstm, filliped vertically next to the Universal
  Transformer that controls the flow of the  information in depth,
  over different steps of the Universal Transformer.

  Args:
    layer_inputs:
      - state: state
      - inputs: the original embedded inputs (= inputs to the first step)
      - memory: memory used in lstm.
    step: indicates number of steps taken so far
    hparams: model hyper-parameters.
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit
    pad_remover: to mask out padding in convolutional layers (efficiency).
  Returns:
    layer_output:
        new_state: new state
        inputs: the original embedded inputs (= inputs to the first step)
        memory: contains information of state from all the previous steps.
  """

  state, unused_inputs, memory = tf.unstack(
      layer_inputs, num=None, axis=0, name="unstack")
  # NOTE:
  # state (ut_state): output of the lstm in the previous step
  # inputs (ut_input): original input --> we don't use it here
  # memory: lstm memory

  # Multi_head_attention:
  assert not hparams.add_step_timing_signal  # Let lstm count for us!
  mh_attention_input = step_preprocess(state, step, hparams)
  transition_function_input = attention_unit(mh_attention_input)

  # Transition Function:
  if hparams.add_ffn_unit_to_the_transition_function:
    transition_function_input = ffn_unit(transition_function_input)

  transition_function_input = common_layers.layer_preprocess(
      transition_function_input, hparams)
  with tf.variable_scope("lstm"):
    # lstm input gate: i_t = sigmoid(W_i.x_t + U_i.h_{t-1})
    transition_function_input_gate = _ffn_layer_multi_inputs(
        [transition_function_input, state],
        hparams,
        name="input",
        bias_initializer=tf.zeros_initializer(),
        activation=tf.sigmoid,
        pad_remover=pad_remover,
        preprocess=False,
        postprocess=False)

    tf.contrib.summary.scalar("lstm_input_gate",
                              tf.reduce_mean(transition_function_input_gate))

    # lstm forget gate: f_t = sigmoid(W_f.x_t + U_f.h_{t-1})
    transition_function_forget_gate = _ffn_layer_multi_inputs(
        [transition_function_input, state],
        hparams,
        name="forget",
        bias_initializer=tf.zeros_initializer(),
        activation=None,
        pad_remover=pad_remover,
        preprocess=False,
        postprocess=False)
    forget_bias_tensor = tf.constant(hparams.lstm_forget_bias)
    transition_function_forget_gate = tf.sigmoid(
        transition_function_forget_gate + forget_bias_tensor)

    tf.contrib.summary.scalar("lstm_forget_gate",
                              tf.reduce_mean(transition_function_forget_gate))

    # lstm output gate: o_t = sigmoid(W_o.x_t + U_o.h_{t-1})
    transition_function_output_gate = _ffn_layer_multi_inputs(
        [transition_function_input, state],
        hparams,
        name="output",
        bias_initializer=tf.zeros_initializer(),
        activation=tf.sigmoid,
        pad_remover=pad_remover,
        preprocess=False,
        postprocess=False)

    tf.contrib.summary.scalar("lstm_output_gate",
                              tf.reduce_mean(transition_function_output_gate))

    # lstm input modulation
    transition_function_input_modulation = _ffn_layer_multi_inputs(
        [transition_function_input, state],
        hparams,
        name="input_modulation",
        bias_initializer=tf.zeros_initializer(),
        activation=tf.tanh,
        pad_remover=pad_remover,
        preprocess=False,
        postprocess=False)

    transition_function_memory = (
        memory * transition_function_forget_gate +
        transition_function_input_gate * transition_function_input_modulation)

    transition_function_output = (
        tf.tanh(transition_function_memory) * transition_function_output_gate)

  transition_function_output = common_layers.layer_preprocess(
      transition_function_output, hparams)

  return transition_function_output, unused_inputs, transition_function_memory


def universal_transformer_act(x, hparams, ffn_unit, attention_unit):
  """ACT based models.

  Implementations of all act models are based on craffel@'s cl/160711592.

  (1) Basic AUT based on remainder-distribution ACT (position-wise).
  (2) AUT with global halting probability (not position-wise).
  (3) AUT with random halting probability (not position-wise).
  (4) AUT with final state as accumulation of all states.

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
  if hparams.act_type not in ["basic", "global", "random", "accumulated"]:
    raise ValueError("Unknown act type: %s" % hparams.act_type)

  state = x
  act_max_steps = hparams.act_max_steps
  threshold = 1.0 - hparams.act_epsilon
  state_shape_static = state.get_shape()

  state_slice = slice(0, 2)
  if hparams.act_type == "global":
    state_slice = slice(0, 1)

  # Dynamic shape for update tensors below
  update_shape = tf.shape(state)[state_slice]

  # Halting probabilities (p_t^n in the paper)
  halting_probability = tf.zeros(update_shape, name="halting_probability")

  # Remainders (R(t) in the paper)
  remainders = tf.zeros(update_shape, name="remainder")

  # Number of updates performed (N(t) in the paper)
  n_updates = tf.zeros(update_shape, name="n_updates")

  # Previous cell states (s_t in the paper)
  previous_state = tf.zeros_like(state, name="previous_state")
  step = tf.constant(0, dtype=tf.int32)

  def ut_function(state, step, halting_probability, remainders, n_updates,
                  previous_state):
    """implements act (position-wise halting).

    Args:
      state: 3-D Tensor: [batch_size, length, channel]
      step: indicates number of steps taken so far
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

    if hparams.act_type == "random":
      # random as halting probability
      p = tf.random_uniform(
          shape=common_layers.shape_list(halting_probability))
    else:
      with tf.variable_scope("sigmoid_activation_for_pondering"):
        p = common_layers.dense(
            state,
            1,
            activation=tf.nn.sigmoid,
            use_bias=True,
            bias_initializer=tf.constant_initializer(
                hparams.act_halting_bias_init))

        if hparams.act_type == "global":
          # average over all positions (as a global halting prob)
          p = tf.reduce_mean(p, axis=1)
          p = tf.squeeze(p)
        else:
          # maintain position-wise probabilities
          p = tf.squeeze(p, axis=-1)

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
        p * still_running + new_halted * remainders, -1)
    if hparams.act_type == "global":
      update_weights = tf.expand_dims(update_weights, -1)

    # apply transformation on the state
    transformed_state = state
    for i in range(hparams.num_inrecurrence_layers):
      with tf.variable_scope("rec_layer_%d" % i):
        transformed_state = ffn_unit(attention_unit(transformed_state))

    # update running part in the weighted state and keep the rest
    new_state = ((transformed_state * update_weights) +
                 (previous_state * (1 - update_weights)))

    if hparams.act_type == "accumulated":
      # Add in the weighted state
      new_state = (transformed_state * update_weights) + previous_state

    # remind TensorFlow of everything's shape
    transformed_state.set_shape(state_shape_static)
    for x in [halting_probability, remainders, n_updates]:
      x.set_shape(state_shape_static[state_slice])
    new_state.set_shape(state_shape_static)
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
      should_continue, ut_function,
      (state, step, halting_probability, remainders, n_updates, previous_state),
      maximum_iterations=act_max_steps + 1)

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
                            preprocess=False,
                            postprocess=False):
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

  ffn_inputs = inputs_list[0]
  if len(inputs_list) != 1:
    ffn_inputs = tf.concat(inputs_list, axis=-1)

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
    if hparams.recurrence_type == "act":
      num_steps = hparams.act_max_steps
    else:
      num_steps = hparams.num_rec_steps
    index = tf.cast(
        common_layers.shape_list(x)[1] * step / num_steps, dtype=tf.int32)

  # No need for the timing signal in the encoder/decoder input preparation
  assert hparams.pos is None

  length = common_layers.shape_list(x)[1]
  channels = common_layers.shape_list(x)[2]
  signal = common_attention.get_timing_signal_1d(
      length, channels, start_index=index)

  if hparams.add_or_concat_timing_signal == "add":
    x_with_timing = x + common_layers.cast_like(signal, x)

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
  if hparams.recurrence_type == "act":
    num_steps = hparams.act_max_steps
  else:
    num_steps = hparams.num_rec_steps
  channels = common_layers.shape_list(x)[-1]

  if hparams.step_timing_signal_type == "learned":
    signal = common_attention.get_layer_timing_signal_learned_1d(
        channels, step, num_steps)

  elif hparams.step_timing_signal_type == "sinusoid":
    signal = common_attention.get_layer_timing_signal_sinusoid_1d(
        channels, step, num_steps)

  if hparams.add_or_concat_timing_signal == "add":
    x_with_timing = x + common_layers.cast_like(signal, x)

  elif hparams.add_or_concat_timing_signal == "concat":
    batch_size = common_layers.shape_list(x)[0]
    length = common_layers.shape_list(x)[1]
    signal_tiled = tf.tile(signal, [batch_size, length, 1])
    x_with_timing = tf.concat((x, signal_tiled), axis=-1)

  return x_with_timing
