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

"""Transformer model from "Attention Is All You Need".

The Transformer model consists of an encoder and a decoder. Both are stacks
of self-attention layers followed by feed-forward layers. This model yields
good results on a number of problems, especially in NLP and machine translation.

See "Attention Is All You Need" (https://arxiv.org/abs/1706.03762) for the full
description of the model and the results obtained with its early version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import librispeech
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
from tensor2tensor.layers import transformer_layers
from tensor2tensor.layers import transformer_memory
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import inplace_ops
from tensorflow.python.util import nest
# pylint: enable=g-direct-tensorflow-import

# Alias some commonly reused layers, here and elsewhere.
transformer_prepare_encoder = transformer_layers.transformer_prepare_encoder
transformer_encoder = transformer_layers.transformer_encoder
transformer_ffn_layer = transformer_layers.transformer_ffn_layer


def transformer_encode(encoder_function, inputs, target_space, hparams,
                       attention_weights=None, features=None, losses=None,
                       **kwargs):
  """Encode transformer inputs.

  Args:
    encoder_function: the encoder function
    inputs: Transformer inputs [batch_size, input_length, 1, hidden_dim] which
      will be flattened along the two spatial dimensions.
    target_space: scalar, target space ID.
    hparams: hyperparameters for model.
    attention_weights: weight to store attention to.
    features: optionally pass the entire features dictionary as well. This is
      needed now for "packed" datasets.
    losses: optional list onto which to append extra training losses
    **kwargs: additional arguments to pass to encoder_function

  Returns:
    Tuple of:
        encoder_output: Encoder representation.
            [batch_size, input_length, hidden_dim]
        encoder_decoder_attention_bias: Bias and mask weights for
            encoder-decoder attention. [batch_size, input_length]
  """
  inputs = common_layers.flatten4d3d(inputs)

  encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
      transformer_prepare_encoder(
          inputs, target_space, hparams, features=features))

  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_LAYER_POSTPROCESS_DROPOUT,
      value=hparams.layer_prepostprocess_dropout,
      hparams=hparams)

  encoder_input = tf.nn.dropout(encoder_input,
                                1.0 - hparams.layer_prepostprocess_dropout)

  attn_bias_for_padding = None
  # Otherwise the encoder will just use encoder_self_attention_bias.
  if hparams.unidirectional_encoder:
    attn_bias_for_padding = encoder_decoder_attention_bias

  encoder_output = encoder_function(
      encoder_input,
      self_attention_bias,
      hparams,
      nonpadding=features_to_nonpadding(features, "inputs"),
      save_weights_to=attention_weights,
      make_image_summary=not common_layers.is_xla_compiled(),
      losses=losses,
      attn_bias_for_padding=attn_bias_for_padding,
      **kwargs)

  return encoder_output, encoder_decoder_attention_bias


def transformer_decode(decoder_function,
                       decoder_input,
                       encoder_output,
                       encoder_decoder_attention_bias,
                       decoder_self_attention_bias,
                       hparams,
                       attention_weights=None,
                       cache=None,
                       decode_loop_step=None,
                       nonpadding=None,
                       losses=None,
                       **kwargs):
  """Decode Transformer outputs from encoder representation.

  Args:
    decoder_function: the decoder function
    decoder_input: inputs to bottom of the model. [batch_size, decoder_length,
      hidden_dim]
    encoder_output: Encoder representation. [batch_size, input_length,
      hidden_dim]
    encoder_decoder_attention_bias: Bias and mask weights for encoder-decoder
      attention. [batch_size, input_length]
    decoder_self_attention_bias: Bias and mask weights for decoder
      self-attention. [batch_size, decoder_length]
    hparams: hyperparameters for model.
    attention_weights: weight to store attention to.
    cache: dict, containing tensors which are the results of previous
      attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop. Only used
      for inference on TPU.
    nonpadding: optional Tensor with shape [batch_size, decoder_length]
    losses: optional list onto which to append extra training losses
    **kwargs: additional arguments to pass to decoder_function

  Returns:
    Final decoder representation. [batch_size, decoder_length, hidden_dim]
  """
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_LAYER_POSTPROCESS_DROPOUT,
      value=hparams.layer_prepostprocess_dropout,
      hparams=hparams)
  decoder_input = tf.nn.dropout(decoder_input,
                                1.0 - hparams.layer_prepostprocess_dropout)

  decoder_output = decoder_function(
      decoder_input,
      encoder_output,
      decoder_self_attention_bias,
      encoder_decoder_attention_bias,
      hparams,
      cache=cache,
      decode_loop_step=decode_loop_step,
      nonpadding=nonpadding,
      save_weights_to=attention_weights,
      losses=losses,
      **kwargs)

  if (common_layers.is_xla_compiled() and
      hparams.mode == tf.estimator.ModeKeys.TRAIN):
    # TPU does not react kindly to extra dimensions.
    # TODO(noam): remove this once TPU is more forgiving of extra dims.
    return decoder_output
  else:
    # Expand since t2t expects 4d tensors.
    return tf.expand_dims(decoder_output, axis=2)


@registry.register_model
class Transformer(t2t_model.T2TModel):
  """Attention net.  See file docstring."""

  def __init__(self, *args, **kwargs):
    super(Transformer, self).__init__(*args, **kwargs)
    self.attention_weights = {}  # For visualizing attention heads.
    self.recurrent_memory_by_layer = None  # Override to enable recurrent memory
    self._encoder_function = transformer_encoder
    self._decoder_function = transformer_decoder

  def encode(self, inputs, target_space, hparams, features=None, losses=None):
    """Encode transformer inputs, see transformer_encode."""
    return transformer_encode(
        self._encoder_function, inputs, target_space, hparams,
        attention_weights=self.attention_weights,
        features=features, losses=losses)

  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_self_attention_bias,
             hparams,
             cache=None,
             decode_loop_step=None,
             nonpadding=None,
             losses=None,
             **kwargs):
    """Decode Transformer outputs, see transformer_decode."""
    return transformer_decode(
        self._decoder_function, decoder_input, encoder_output,
        encoder_decoder_attention_bias, decoder_self_attention_bias,
        hparams, attention_weights=self.attention_weights, cache=cache,
        decode_loop_step=decode_loop_step, nonpadding=nonpadding, losses=losses,
        **kwargs)

  def body(self, features):
    """Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs. [batch_size, input_length, 1,
            hidden_dim].
          "targets": Target decoder outputs. [batch_size, decoder_length, 1,
            hidden_dim]
          "target_space_id": A scalar int from data_generators.problem.SpaceID.

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams

    losses = []

    if self.has_input:
      inputs = features["inputs"]
      target_space = features["target_space_id"]
      encoder_output, encoder_decoder_attention_bias = self.encode(
          inputs, target_space, hparams, features=features, losses=losses)
    else:
      encoder_output, encoder_decoder_attention_bias = (None, None)

    targets = features["targets"]
    targets_shape = common_layers.shape_list(targets)
    targets = common_layers.flatten4d3d(targets)
    decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(
        targets, hparams, features=features)

    # Not all subclasses of Transformer support keyword arguments related to
    # recurrent memory, so only pass these arguments if memory is enabled.
    decode_kwargs = {}
    if self.recurrent_memory_by_layer is not None:
      # TODO(kitaev): The chunk_number feature currently has the same shape as
      # "targets", but this is only for the purposes of sharing sharding code.
      # In fact every token within the batch must have the same chunk number.
      chunk_number_each_token = tf.squeeze(features["chunk_number"], (-1, -2))
      chunk_number_each_batch = chunk_number_each_token[:, 0]
      # Uncomment the code below to verify that tokens within a batch share the
      # same chunk number:
      # with tf.control_dependencies([
      #     tf.assert_equal(chunk_number_each_token,
      #                     chunk_number_each_batch[:, None])
      # ]):
      #   chunk_number_each_batch = tf.identity(chunk_number_each_batch)
      decode_kwargs = dict(
          recurrent_memory_by_layer=self.recurrent_memory_by_layer,
          chunk_number=chunk_number_each_batch,
          )

    decoder_output = self.decode(
        decoder_input,
        encoder_output,
        encoder_decoder_attention_bias,
        decoder_self_attention_bias,
        hparams,
        nonpadding=features_to_nonpadding(features, "targets"),
        losses=losses,
        **decode_kwargs
        )

    expected_attentions = features.get("expected_attentions")
    if expected_attentions is not None:
      attention_loss = common_attention.encoder_decoder_attention_loss(
          expected_attentions, self.attention_weights,
          hparams.expected_attention_loss_type,
          hparams.expected_attention_loss_multiplier)
      return decoder_output, {"attention_loss": attention_loss}

    ret = tf.reshape(decoder_output, targets_shape)
    if losses:
      return ret, {"extra_loss": tf.add_n(losses)}
    else:
      return ret

  def _greedy_infer(self, features, decode_length, use_tpu=False):
    """Fast version of greedy decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      use_tpu: A bool. Whether to build the inference graph for TPU.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    # For real-valued modalities use the slow decode path for now.
    if (self._target_modality_is_real or
        self._hparams.self_attention_type != "dot_product"):
      return super(Transformer, self)._greedy_infer(features, decode_length)
    with tf.variable_scope(self.name):
      if use_tpu:
        return self._fast_decode_tpu(features, decode_length)
      return self._fast_decode(features, decode_length)

  def _beam_decode(self,
                   features,
                   decode_length,
                   beam_size,
                   top_beams,
                   alpha,
                   use_tpu=False):
    """Beam search decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
      use_tpu: A bool, whether to do beam decode on TPU.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }
    """
    if (self._hparams.self_attention_type not in [
        "dot_product", "dot_product_relative"
    ]):
      # Caching is not guaranteed to work with attention types other than
      # dot_product.
      # TODO(petershaw): Support fast decoding when using relative
      # position representations, i.e. "dot_product_relative" attention.
      return self._beam_decode_slow(features, decode_length, beam_size,
                                    top_beams, alpha, use_tpu)
    with tf.variable_scope(self.name):
      if use_tpu:
        return self._fast_decode_tpu(features, decode_length, beam_size,
                                     top_beams, alpha)
      return self._fast_decode(features, decode_length, beam_size, top_beams,
                               alpha)

  def _fast_decode_tpu(self,
                       features,
                       decode_length,
                       beam_size=1,
                       top_beams=1,
                       alpha=1.0):
    """Fast decoding.

    Implements both greedy and beam search decoding on TPU, uses beam search
    iff beam_size > 1, otherwise beam search related arguments are ignored.

    Args:
      features: A map of string to model features.
      decode_length: An integer, how many additional timesteps to decode.
      beam_size: An integer, number of beams.
      top_beams: An integer, how many of the beams to return.
      alpha: A float that controls the length penalty. Larger the alpha,
        stronger the preference for longer translations.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }.

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    if self._num_datashards != 1:
      raise NotImplementedError("Fast decoding only supports a single shard.")
    if "targets_segmentation" in features:
      raise NotImplementedError(
          "Decoding not supported on packed datasets "
          " If you want to decode from a dataset, use the non-packed version"
          " of the dataset when decoding.")
    dp = self._data_parallelism
    hparams = self._hparams
    target_modality = self._problem_hparams.modality["targets"]
    target_vocab_size = self._problem_hparams.vocab_size["targets"]
    if target_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
      target_vocab_size += (-target_vocab_size) % hparams.vocab_divisor

    if self.has_input:
      inputs = features["inputs"]
      if target_modality == modalities.ModalityType.CLASS_LABEL:
        decode_length = 1
      else:
        decode_length = (
            common_layers.shape_list(inputs)[1] + features.get(
                "decode_length", decode_length))

      # TODO(llion): Clean up this reshaping logic.
      inputs = tf.expand_dims(inputs, axis=1)
      if len(inputs.shape) < 5:
        inputs = tf.expand_dims(inputs, axis=4)
      s = common_layers.shape_list(inputs)
      batch_size = s[0]
      inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
      # _shard_features called to ensure that the variable names match
      inputs = self._shard_features({"inputs": inputs})["inputs"]
      input_modality = self._problem_hparams.modality["inputs"]
      input_vocab_size = self._problem_hparams.vocab_size["inputs"]
      if input_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
        input_vocab_size += (-input_vocab_size) % hparams.vocab_divisor
      modality_name = hparams.name.get(
          "inputs",
          modalities.get_name(input_modality))(hparams, input_vocab_size)
      with tf.variable_scope(modality_name):
        bottom = hparams.bottom.get("inputs",
                                    modalities.get_bottom(input_modality))
        inputs = dp(bottom, inputs, hparams, input_vocab_size)
      with tf.variable_scope("body"):
        encoder_output, encoder_decoder_attention_bias = dp(
            self.encode,
            inputs,
            features["target_space_id"],
            hparams,
            features=features)
      encoder_output = encoder_output[0]
      encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
      partial_targets = None
    else:
      # The problem has no inputs.
      encoder_output = None
      encoder_decoder_attention_bias = None

      # Prepare partial targets.
      # In either features["inputs"] or features["targets"].
      # We force the outputs to begin with these sequences.
      partial_targets = features.get("inputs")
      if partial_targets is None:
        partial_targets = features["targets"]
      assert partial_targets is not None
      partial_targets = common_layers.expand_squeeze_to_nd(partial_targets, 2)
      partial_targets = tf.to_int64(partial_targets)
      partial_targets_shape = common_layers.shape_list(partial_targets)
      partial_targets_length = partial_targets_shape[1]
      decode_length = (
          partial_targets_length + features.get("decode_length", decode_length))
      batch_size = partial_targets_shape[0]

    if hparams.pos == "timing":
      positional_encoding = common_attention.get_timing_signal_1d(
          decode_length + 1, hparams.hidden_size)
    elif hparams.pos == "emb":
      positional_encoding = common_attention.add_positional_embedding(
          tf.zeros([1, decode_length + 1, hparams.hidden_size]),
          hparams.max_length, "body/targets_positional_embedding", None)
    else:
      positional_encoding = None

    def preprocess_targets(targets, i):
      """Performs preprocessing steps on the targets to prepare for the decoder.

      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.

      Args:
        targets: A tensor, inputs ids to the decoder. [batch_size, 1].
        i: An integer, Step number of the decoding loop.

      Returns:
        A tensor, processed targets [batch_size, 1, hidden_dim].
      """
      # _shard_features called to ensure that the variable names match
      targets = self._shard_features({"targets": targets})["targets"]
      modality_name = hparams.name.get(
          "targets",
          modalities.get_name(target_modality))(hparams, target_vocab_size)
      with tf.variable_scope(modality_name):
        bottom = hparams.bottom.get(
            "targets", modalities.get_targets_bottom(target_modality))
        targets = dp(bottom, targets, hparams, target_vocab_size)[0]
      targets = common_layers.flatten4d3d(targets)

      # GO embeddings are all zero, this is because transformer_prepare_decoder
      # Shifts the targets along by one for the input which pads with zeros.
      # If the modality already maps GO to the zero embeddings this is not
      # needed.
      targets = tf.cond(
          tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

      if positional_encoding is not None:
        positional_encoding_shape = positional_encoding.shape.as_list()
        targets += tf.slice(
            positional_encoding, [0, i, 0],
            [positional_encoding_shape[0], 1, positional_encoding_shape[2]])
      return targets

    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(decode_length))
    if hparams.proximity_bias:
      decoder_self_attention_bias += common_attention.attention_bias_proximal(
          decode_length)

    def symbols_to_logits_tpu_fn(ids, i, cache):
      """Go from ids to logits for next symbol on TPU.

      Args:
        ids: A tensor, symbol IDs.
        i: An integer, step number of the decoding loop. Only used for inference
          on TPU.
        cache: A dict, containing tensors which are the results of previous
          attentions, used for fast decoding.

      Returns:
        ret: A tensor, computed logits.
        cache: A dict, containing tensors which are the results of previous
            attentions, used for fast decoding.
      """
      ids = ids[:, -1:]
      targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      targets = preprocess_targets(targets, i)

      bias_shape = decoder_self_attention_bias.shape.as_list()
      bias = tf.slice(decoder_self_attention_bias, [0, 0, i, 0],
                      [bias_shape[0], bias_shape[1], 1, bias_shape[3]])

      with tf.variable_scope("body"):
        body_outputs = dp(
            self.decode,
            targets,
            cache.get("encoder_output"),
            cache.get("encoder_decoder_attention_bias"),
            bias,
            hparams,
            cache,
            i,
            nonpadding=features_to_nonpadding(features, "targets"))
      modality_name = hparams.name.get(
          "targets",
          modalities.get_name(target_modality))(hparams, target_vocab_size)
      with tf.variable_scope(modality_name):
        top = hparams.top.get("targets",
                              modalities.get_top(target_modality))
        logits = dp(top, body_outputs, None, hparams, target_vocab_size)[0]

      ret = tf.squeeze(logits, axis=[1, 2, 3])
      if partial_targets is not None:
        # If the position is within the given partial targets, we alter the
        # logits to always return those values.
        # A faster approach would be to process the partial targets in one
        # iteration in order to fill the corresponding parts of the cache.
        # This would require broader changes, though.
        vocab_size = tf.shape(ret)[1]

        def forced_logits():
          return tf.one_hot(
              tf.tile(
                  tf.slice(partial_targets, [0, i],
                           [partial_targets.shape.as_list()[0], 1]),
                  [beam_size]), vocab_size, 0.0, -1e9)

        ret = tf.cond(
            tf.less(i, partial_targets_length), forced_logits, lambda: ret)
      return ret, cache

    ret = fast_decode_tpu(
        encoder_output=encoder_output,
        encoder_decoder_attention_bias=encoder_decoder_attention_bias,
        symbols_to_logits_fn=symbols_to_logits_tpu_fn,
        hparams=hparams,
        decode_length=decode_length,
        vocab_size=target_vocab_size,
        beam_size=beam_size,
        top_beams=top_beams,
        alpha=alpha,
        batch_size=batch_size,
        force_decode_length=self._decode_hparams.force_decode_length)
    if partial_targets is not None:
      if beam_size <= 1 or top_beams <= 1:
        ret["outputs"] = ret["outputs"][:, partial_targets_length:]
      else:
        ret["outputs"] = ret["outputs"][:, :, partial_targets_length:]
    return ret

  def _fast_decode(self,
                   features,
                   decode_length,
                   beam_size=1,
                   top_beams=1,
                   alpha=1.0):
    """Fast decoding.

    Implements both greedy and beam search decoding, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.

    Args:
      features: a map of string to model  features.
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    if self._num_datashards != 1:
      raise NotImplementedError("Fast decoding only supports a single shard.")
    dp = self._data_parallelism
    hparams = self._hparams
    target_modality = self._problem_hparams.modality["targets"]
    target_vocab_size = self._problem_hparams.vocab_size["targets"]
    if target_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
      target_vocab_size += (-target_vocab_size) % hparams.vocab_divisor
    if "targets_segmentation" in features:
      raise NotImplementedError(
          "Decoding not supported on packed datasets "
          " If you want to decode from a dataset, use the non-packed version"
          " of the dataset when decoding.")
    if self.has_input:
      inputs = features["inputs"]
      if target_modality == modalities.ModalityType.CLASS_LABEL:
        decode_length = 1
      else:
        decode_length = (
            common_layers.shape_list(inputs)[1] + features.get(
                "decode_length", decode_length))

      # TODO(llion): Clean up this reshaping logic.
      inputs = tf.expand_dims(inputs, axis=1)
      if len(inputs.shape) < 5:
        inputs = tf.expand_dims(inputs, axis=4)
      s = common_layers.shape_list(inputs)
      batch_size = s[0]
      inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
      # _shard_features called to ensure that the variable names match
      inputs = self._shard_features({"inputs": inputs})["inputs"]
      input_modality = self._problem_hparams.modality["inputs"]
      input_vocab_size = self._problem_hparams.vocab_size["inputs"]
      if input_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
        input_vocab_size += (-input_vocab_size) % hparams.vocab_divisor
      modality_name = hparams.name.get(
          "inputs",
          modalities.get_name(input_modality))(hparams, input_vocab_size)
      with tf.variable_scope(modality_name):
        bottom = hparams.bottom.get("inputs",
                                    modalities.get_bottom(input_modality))
        inputs = dp(bottom, inputs, hparams, input_vocab_size)
      with tf.variable_scope("body"):
        encoder_output, encoder_decoder_attention_bias = dp(
            self.encode,
            inputs,
            features["target_space_id"],
            hparams,
            features=features)
      encoder_output = encoder_output[0]
      encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
      partial_targets = None
    else:
      # The problem has no inputs.
      encoder_output = None
      encoder_decoder_attention_bias = None

      # Prepare partial targets.
      # In either features["inputs"] or features["targets"].
      # We force the outputs to begin with these sequences.
      partial_targets = features.get("inputs")
      if partial_targets is None:
        partial_targets = features["targets"]
      assert partial_targets is not None
      partial_targets = common_layers.expand_squeeze_to_nd(partial_targets, 2)
      partial_targets = tf.to_int64(partial_targets)
      partial_targets_shape = common_layers.shape_list(partial_targets)
      partial_targets_length = partial_targets_shape[1]
      decode_length = (
          partial_targets_length + features.get("decode_length", decode_length))
      batch_size = partial_targets_shape[0]

    if hparams.pos == "timing":
      positional_encoding = common_attention.get_timing_signal_1d(
          decode_length + 1, hparams.hidden_size)
    elif hparams.pos == "emb":
      positional_encoding = common_attention.add_positional_embedding(
          tf.zeros([1, decode_length, hparams.hidden_size]), hparams.max_length,
          "body/targets_positional_embedding", None)
    else:
      positional_encoding = None

    def preprocess_targets(targets, i):
      """Performs preprocessing steps on the targets to prepare for the decoder.

      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.

      Args:
        targets: inputs ids to the decoder. [batch_size, 1]
        i: scalar, Step number of the decoding loop.

      Returns:
        Processed targets [batch_size, 1, hidden_dim]
      """
      # _shard_features called to ensure that the variable names match
      targets = self._shard_features({"targets": targets})["targets"]
      modality_name = hparams.name.get(
          "targets",
          modalities.get_name(target_modality))(hparams, target_vocab_size)
      with tf.variable_scope(modality_name):
        bottom = hparams.bottom.get(
            "targets", modalities.get_targets_bottom(target_modality))
        targets = dp(bottom, targets, hparams, target_vocab_size)[0]
      targets = common_layers.flatten4d3d(targets)

      # GO embeddings are all zero, this is because transformer_prepare_decoder
      # Shifts the targets along by one for the input which pads with zeros.
      # If the modality already maps GO to the zero embeddings this is not
      # needed.
      targets = tf.cond(
          tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

      if positional_encoding is not None:
        targets += positional_encoding[:, i:i + 1]
      return targets

    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(decode_length))
    if hparams.proximity_bias:
      decoder_self_attention_bias += common_attention.attention_bias_proximal(
          decode_length)

    def symbols_to_logits_fn(ids, i, cache):
      """Go from ids to logits for next symbol."""
      ids = ids[:, -1:]
      targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      targets = preprocess_targets(targets, i)

      bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      with tf.variable_scope("body"):
        body_outputs = dp(
            self.decode,
            targets,
            cache.get("encoder_output"),
            cache.get("encoder_decoder_attention_bias"),
            bias,
            hparams,
            cache,
            nonpadding=features_to_nonpadding(features, "targets"))

      modality_name = hparams.name.get(
          "targets",
          modalities.get_name(target_modality))(hparams, target_vocab_size)
      with tf.variable_scope(modality_name):
        top = hparams.top.get("targets", modalities.get_top(target_modality))
        logits = dp(top, body_outputs, None, hparams, target_vocab_size)[0]

      ret = tf.squeeze(logits, axis=[1, 2, 3])
      if partial_targets is not None:
        # If the position is within the given partial targets, we alter the
        # logits to always return those values.
        # A faster approach would be to process the partial targets in one
        # iteration in order to fill the corresponding parts of the cache.
        # This would require broader changes, though.
        vocab_size = tf.shape(ret)[1]

        def forced_logits():
          return tf.one_hot(
              tf.tile(partial_targets[:, i], [beam_size]), vocab_size, 0.0,
              -1e9)

        ret = tf.cond(
            tf.less(i, partial_targets_length), forced_logits, lambda: ret)
      return ret, cache

    ret = fast_decode(
        encoder_output=encoder_output,
        encoder_decoder_attention_bias=encoder_decoder_attention_bias,
        symbols_to_logits_fn=symbols_to_logits_fn,
        hparams=hparams,
        decode_length=decode_length,
        vocab_size=target_vocab_size,
        beam_size=beam_size,
        top_beams=top_beams,
        alpha=alpha,
        batch_size=batch_size,
        force_decode_length=self._decode_hparams.force_decode_length)
    if partial_targets is not None:
      if beam_size <= 1 or top_beams <= 1:
        ret["outputs"] = ret["outputs"][:, partial_targets_length:]
      else:
        ret["outputs"] = ret["outputs"][:, :, partial_targets_length:]
    return ret


def fast_decode_tpu(encoder_output,
                    encoder_decoder_attention_bias,
                    symbols_to_logits_fn,
                    hparams,
                    decode_length,
                    vocab_size,
                    beam_size=1,
                    top_beams=1,
                    alpha=1.0,
                    sos_id=0,
                    eos_id=beam_search.EOS_ID,
                    batch_size=None,
                    force_decode_length=False,
                    scope_prefix="body/",
                    use_top_k_with_unique=True):
  """Given encoder output and a symbols to logits function, does fast decoding.

  Implements both greedy and beam search decoding for TPU, uses beam search iff
  beam_size > 1, otherwise beam search related arguments are ignored.

  Args:
    encoder_output: A tensor, output from encoder.
    encoder_decoder_attention_bias: A tensor, bias for use in encoder-decoder
      attention.
    symbols_to_logits_fn: Incremental decoding, function mapping triple `(ids,
      step, cache)` to symbol logits.
    hparams: Run hyperparameters.
    decode_length: An integer, how many additional timesteps to decode.
    vocab_size: Output vocabulary size.
    beam_size: An integer, number of beams.
    top_beams: An integer, how many of the beams to return.
    alpha: A float that controls the length penalty. Larger the alpha, stronger
      the preference for longer translations.
    sos_id: Start-of-sequence symbol.
    eos_id: End-of-sequence symbol.
    batch_size: An integer, must be passed if there is no input.
    force_decode_length: A bool, whether to force the full decode length, or if
      False, stop when all beams hit eos_id.
    scope_prefix: str, prefix for decoder layer variable scopes.
    use_top_k_with_unique: bool, whether to use a fast (but decreased precision)
      top_k during beam search.

  Returns:
    A dict of decoding results {
        "outputs": integer `Tensor` of decoded ids of shape
            [batch_size, <= decode_length] if top_beams == 1 or
            [batch_size, top_beams, <= decode_length] otherwise
        "scores": decoding log probs from the beam search,
            None if using greedy decoding (beam_size=1)
    }.

  Raises:
    NotImplementedError: If beam size > 1 with partial targets.
  """
  if encoder_output is not None:
    batch_size = common_layers.shape_list(encoder_output)[0]

  key_channels = hparams.attention_key_channels or hparams.hidden_size
  value_channels = hparams.attention_value_channels or hparams.hidden_size
  num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers
  vars_3d_num_heads = (
      hparams.num_heads if hparams.get("attention_variables_3d") else 0)

  cache = {
      "layer_%d" % layer: {  # pylint: disable=g-complex-comprehension
          "k":
          common_attention.split_heads(
              tf.zeros([batch_size, decode_length, key_channels]),
              hparams.num_heads),
          "v":
          common_attention.split_heads(
              tf.zeros([batch_size, decode_length, value_channels]),
              hparams.num_heads),
      } for layer in range(num_layers)
  }

  # If `ffn_layer` is in `["dense_relu_dense" or "conv_hidden_relu"]`, then the
  # cache key "f" won't be used, which means that the` shape of cache["f"]`
  # won't be changed to
  # `[beamsize*batch_size, decode_length, hparams.hidden_size]` and may cause
  # error when applying `nest.map reshape function` on it.
  if hparams.ffn_layer not in ["dense_relu_dense", "conv_hidden_relu"]:
    for layer in range(num_layers):
      cache["layer_%d" % layer]["f"] = tf.zeros(
          [batch_size, 0, hparams.hidden_size])

  if encoder_output is not None:
    for layer in range(num_layers):
      layer_name = "layer_%d" % layer
      with tf.variable_scope("%sdecoder/%s/encdec_attention/multihead_attention"
                             % (scope_prefix, layer_name)):
        k_encdec = common_attention.compute_attention_component(
            encoder_output,
            key_channels,
            name="k",
            vars_3d_num_heads=vars_3d_num_heads)
        k_encdec = common_attention.split_heads(k_encdec, hparams.num_heads)
        v_encdec = common_attention.compute_attention_component(
            encoder_output,
            value_channels,
            name="v",
            vars_3d_num_heads=vars_3d_num_heads)
        v_encdec = common_attention.split_heads(v_encdec, hparams.num_heads)
      cache[layer_name]["k_encdec"] = k_encdec
      cache[layer_name]["v_encdec"] = v_encdec

    cache["encoder_output"] = encoder_output
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_SEQ_BEAM_SEARCH,
      value={
          "vocab_size": vocab_size,
          "batch_size": batch_size,
          "beam_size": beam_size,
          "alpha": alpha,
          "max_decode_length": decode_length
      },
      hparams=hparams)
  if beam_size > 1:  # Beam Search
    initial_ids = sos_id * tf.ones([batch_size], dtype=tf.int32)
    decoded_ids, scores, _ = beam_search.beam_search(
        symbols_to_logits_fn,
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        alpha,
        states=cache,
        eos_id=eos_id,
        stop_early=(top_beams == 1),
        use_tpu=True,
        use_top_k_with_unique=use_top_k_with_unique)

    if top_beams == 1:
      decoded_ids = decoded_ids[:, 0, 1:]
      scores = scores[:, 0]
    else:
      decoded_ids = decoded_ids[:, :top_beams, 1:]
      scores = scores[:, :top_beams]
  else:  # Greedy

    def inner_loop(i, hit_eos, next_id, decoded_ids, cache, log_prob):
      """One step of greedy decoding."""
      logits, cache = symbols_to_logits_fn(next_id, i, cache)
      log_probs = common_layers.log_prob_from_logits(logits)
      temperature = getattr(hparams, "sampling_temp", 0.0)
      keep_top = getattr(hparams, "sampling_keep_top_k", -1)
      if hparams.sampling_method == "argmax":
        temperature = 0.0
      next_id = common_layers.sample_with_temperature(
          logits, temperature, keep_top)

      hit_eos |= tf.equal(next_id, eos_id)

      log_prob_indices = tf.stack([tf.range(tf.to_int64(batch_size)), next_id],
                                  axis=1)
      log_prob += tf.gather_nd(log_probs, log_prob_indices)

      next_id = tf.expand_dims(next_id, axis=1)
      decoded_ids = tf.transpose(decoded_ids)
      decoded_ids = inplace_ops.alias_inplace_update(
          decoded_ids, i, tf.squeeze(next_id, axis=1))
      decoded_ids = tf.transpose(decoded_ids)
      return i + 1, hit_eos, next_id, decoded_ids, cache, log_prob

    def is_not_finished(i, hit_eos, *_):
      finished = i >= decode_length
      if not force_decode_length:
        finished |= tf.reduce_all(hit_eos)
      return tf.logical_not(finished)

    decoded_ids = tf.zeros([batch_size, decode_length], dtype=tf.int64)
    hit_eos = tf.fill([batch_size], False)
    next_id = sos_id * tf.ones([batch_size, 1], dtype=tf.int64)
    initial_log_prob = tf.zeros([batch_size], dtype=tf.float32)

    def compute_cache_shape_invariants(tensor):
      return tf.TensorShape(tensor.shape.as_list())

    _, _, _, decoded_ids, _, log_prob = tf.while_loop(
        is_not_finished,
        inner_loop, [
            tf.constant(0), hit_eos, next_id, decoded_ids, cache,
            initial_log_prob
        ],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([batch_size]),
            tf.TensorShape([batch_size, 1]),
            tf.TensorShape([batch_size, decode_length]),
            nest.map_structure(compute_cache_shape_invariants, cache),
            tf.TensorShape([batch_size]),
        ])
    scores = log_prob

  return {"outputs": decoded_ids, "scores": scores}


def fast_decode(encoder_output,
                encoder_decoder_attention_bias,
                symbols_to_logits_fn,
                hparams,
                decode_length,
                vocab_size,
                beam_size=1,
                top_beams=1,
                alpha=1.0,
                sos_id=0,
                eos_id=beam_search.EOS_ID,
                batch_size=None,
                force_decode_length=False,
                scope_prefix="body/",
                cache=None):
  """Given encoder output and a symbols to logits function, does fast decoding.

  Implements both greedy and beam search decoding, uses beam search iff
  beam_size > 1, otherwise beam search related arguments are ignored.

  Args:
    encoder_output: Output from encoder.
    encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
      attention
    symbols_to_logits_fn: Incremental decoding; function mapping triple `(ids,
      step, cache)` to symbol logits.
    hparams: run hyperparameters
    decode_length: an integer.  How many additional timesteps to decode.
    vocab_size: Output vocabulary size.
    beam_size: number of beams.
    top_beams: an integer. How many of the beams to return.
    alpha: Float that controls the length penalty. larger the alpha, stronger
      the preference for longer translations.
    sos_id: End-of-sequence symbol in beam search.
    eos_id: End-of-sequence symbol in beam search.
    batch_size: an integer scalar - must be passed if there is no input
    force_decode_length: bool, whether to force the full decode length, or if
      False, stop when all beams hit eos_id.
    scope_prefix: str, prefix for decoder layer variable scopes.
    cache: cache dictionary for additional predictions.

  Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if top_beams == 1 or
              [batch_size, top_beams, <= decode_length] otherwise
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If beam size > 1 with partial targets.
  """
  if encoder_output is not None:
    batch_size = common_layers.shape_list(encoder_output)[0]

  key_channels = hparams.attention_key_channels or hparams.hidden_size
  value_channels = hparams.attention_value_channels or hparams.hidden_size
  num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers
  vars_3d_num_heads = (
      hparams.num_heads if hparams.get("attention_variables_3d") else 0)

  if cache is None:
    cache = {}
  cache.update({
      "layer_%d" % layer: {  # pylint: disable=g-complex-comprehension
          "k":
              common_attention.split_heads(
                  tf.zeros([batch_size, 0, key_channels]), hparams.num_heads),
          "v":
              common_attention.split_heads(
                  tf.zeros([batch_size, 0, value_channels]), hparams.num_heads),
      } for layer in range(num_layers)
  })

  # If `ffn_layer` is in `["dense_relu_dense" or "conv_hidden_relu"]`, then the
  # cache key "f" won't be used, which means that the` shape of cache["f"]`
  # won't be changed to
  # `[beamsize*batch_size, decode_length, hparams.hidden_size]` and may cause
  # error when applying `nest.map reshape function` on it.
  if hparams.ffn_layer not in ["dense_relu_dense", "conv_hidden_relu"]:
    for layer in range(num_layers):
      cache["layer_%d" % layer]["f"] = tf.zeros(
          [batch_size, 0, hparams.hidden_size])

  if encoder_output is not None:
    for layer in range(num_layers):
      layer_name = "layer_%d" % layer
      with tf.variable_scope("%sdecoder/%s/encdec_attention/multihead_attention"
                             % (scope_prefix, layer_name)):
        k_encdec = common_attention.compute_attention_component(
            encoder_output,
            key_channels,
            name="k",
            vars_3d_num_heads=vars_3d_num_heads)
        k_encdec = common_attention.split_heads(k_encdec, hparams.num_heads)
        v_encdec = common_attention.compute_attention_component(
            encoder_output,
            value_channels,
            name="v",
            vars_3d_num_heads=vars_3d_num_heads)
        v_encdec = common_attention.split_heads(v_encdec, hparams.num_heads)
      cache[layer_name]["k_encdec"] = k_encdec
      cache[layer_name]["v_encdec"] = v_encdec

    cache["encoder_output"] = encoder_output
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

  if beam_size > 1:  # Beam Search
    initial_ids = sos_id * tf.ones([batch_size], dtype=tf.int32)
    decoded_ids, scores, cache = beam_search.beam_search(
        symbols_to_logits_fn,
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        alpha,
        states=cache,
        eos_id=eos_id,
        stop_early=(top_beams == 1))

    if top_beams == 1:
      decoded_ids = decoded_ids[:, 0, 1:]
      scores = scores[:, 0]
    else:
      decoded_ids = decoded_ids[:, :top_beams, 1:]
      scores = scores[:, :top_beams]
  else:  # Greedy

    def inner_loop(i, hit_eos, next_id, decoded_ids, cache, log_prob):
      """One step of greedy decoding."""
      logits, cache = symbols_to_logits_fn(next_id, i, cache)
      log_probs = common_layers.log_prob_from_logits(logits)
      temperature = getattr(hparams, "sampling_temp", 0.0)
      keep_top = getattr(hparams, "sampling_keep_top_k", -1)
      if hparams.sampling_method == "argmax":
        temperature = 0.0
      next_id = common_layers.sample_with_temperature(
          logits, temperature, keep_top)
      hit_eos |= tf.equal(next_id, eos_id)

      log_prob_indices = tf.stack([tf.range(tf.to_int64(batch_size)), next_id],
                                  axis=1)
      log_prob += tf.gather_nd(log_probs, log_prob_indices)

      next_id = tf.expand_dims(next_id, axis=1)
      decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
      return i + 1, hit_eos, next_id, decoded_ids, cache, log_prob

    def is_not_finished(i, hit_eos, *_):
      finished = i >= decode_length
      if not force_decode_length:
        finished |= tf.reduce_all(hit_eos)
      return tf.logical_not(finished)

    decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int64)
    hit_eos = tf.fill([batch_size], False)
    next_id = sos_id * tf.ones([batch_size, 1], dtype=tf.int64)
    initial_log_prob = tf.zeros([batch_size], dtype=tf.float32)
    _, _, _, decoded_ids, cache, log_prob = tf.while_loop(
        is_not_finished,
        inner_loop, [
            tf.constant(0), hit_eos, next_id, decoded_ids, cache,
            initial_log_prob
        ],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            nest.map_structure(beam_search.get_state_shape_invariants, cache),
            tf.TensorShape([None]),
        ])
    scores = log_prob

  return {"outputs": decoded_ids, "scores": scores, "cache": cache}


@registry.register_model
class TransformerScorer(Transformer):
  """Transformer model, but only scores in PREDICT mode.

  Checkpoints between Transformer and TransformerScorer are interchangeable.
  """

  def __init__(self, *args, **kwargs):
    super(TransformerScorer, self).__init__(*args, **kwargs)
    self._name = "transformer"
    self._base_name = "transformer"

  def infer(self,
            features=None,
            decode_length=50,
            beam_size=1,
            top_beams=1,
            alpha=0.0,
            use_tpu=False):
    """Returns the targets and their log probabilities."""
    del decode_length, beam_size, top_beams, alpha, use_tpu
    assert features is not None

    # Run the model
    self.hparams.force_full_predict = True
    with tf.variable_scope(self.name):
      logits, _ = self.model_fn(features)
    assert len(logits.shape) == 5  # [batch, time, 1, 1, vocab]
    logits = tf.squeeze(logits, [2, 3])

    # Compute the log probabilities
    log_probs = common_layers.log_prob_from_logits(logits)

    targets = features["targets"]
    assert len(targets.shape) == 4  # [batch, time, 1, 1]
    targets = tf.squeeze(targets, [2, 3])

    # Slice out the log_probs of the targets
    log_probs = common_layers.index_last_dim_with_indices(log_probs, targets)

    # Sum over time to get the log_prob of the sequence
    scores = tf.reduce_sum(log_probs, axis=1)

    return {"outputs": targets, "scores": scores}


@registry.register_model
class TransformerEncoder(t2t_model.T2TModel):
  """Transformer, encoder only."""

  def body(self, features):
    hparams = self._hparams
    inputs = features["inputs"]
    target_space = features["target_space_id"]

    inputs = common_layers.flatten4d3d(inputs)

    (encoder_input, encoder_self_attention_bias, _) = (
        transformer_prepare_encoder(inputs, target_space, hparams))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)
    encoder_output = transformer_encoder(
        encoder_input,
        encoder_self_attention_bias,
        hparams,
        nonpadding=features_to_nonpadding(features, "inputs"))
    encoder_output = tf.expand_dims(encoder_output, 2)

    return encoder_output


@registry.register_model
class TransformerRegressor(TransformerEncoder):
  """Transformer inheriting from Encoder, for the regression problem.

  Final result is a tensor that has a shape of (?, 1, 1, 1).
  """

  def top(self, body_output, features):
    """Computes single scalar value from body_output."""

    with tf.variable_scope("reg_top_ffn"):
      x = body_output
      x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
      res = tf.layers.dense(x, 1, name="model_top")
      return res


def features_to_nonpadding(features, inputs_or_targets="inputs"):
  key = inputs_or_targets + "_segmentation"
  if features and key in features:
    return tf.minimum(tf.to_float(features[key]), 1.0)
  return None


def transformer_prepare_decoder(targets, hparams, features=None):
  """Prepare one shard of the model for the decoder.

  Args:
    targets: a Tensor.
    hparams: run hyperparameters
    features: optionally pass the entire features dictionary as well. This is
      needed now for "packed" datasets.

  Returns:
    decoder_input: a Tensor, bottom of decoder stack
    decoder_self_attention_bias: a bias tensor for use in decoder self-attention
  """
  if hparams.causal_decoder_self_attention:
    # Causal attention.
    if hparams.prepend_mode == "prepend_inputs_full_attention":
      decoder_self_attention_bias = (
          common_attention.attention_bias_prepend_inputs_full_attention(
              common_attention.embedding_to_padding(targets)))
    else:
      decoder_self_attention_bias = (
          common_attention.attention_bias_lower_triangle(
              common_layers.shape_list(targets)[1]))
  else:
    # Full attention.
    decoder_padding = common_attention.embedding_to_padding(targets)
    decoder_self_attention_bias = (
        common_attention.attention_bias_ignore_padding(decoder_padding))

  if features and "targets_segmentation" in features:
    # "Packed" dataset - keep the examples from seeing each other.
    targets_segmentation = features["targets_segmentation"]
    targets_position = features["targets_position"]
    decoder_self_attention_bias += common_attention.attention_bias_same_segment(
        targets_segmentation, targets_segmentation)
  else:
    targets_position = None
  if hparams.proximity_bias:
    decoder_self_attention_bias += common_attention.attention_bias_proximal(
        common_layers.shape_list(targets)[1])
  decoder_input = common_layers.shift_right_3d(targets)
  if hparams.pos == "timing":
    if targets_position is not None:
      decoder_input = common_attention.add_timing_signal_1d_given_position(
          decoder_input, targets_position)
    else:
      decoder_input = common_attention.add_timing_signal_1d(decoder_input)
  elif hparams.pos == "emb":
    decoder_input = common_attention.add_positional_embedding(
        decoder_input, hparams.max_length, "targets_positional_embedding",
        targets_position)

  if hparams.activation_dtype == "bfloat16":
    decoder_self_attention_bias = tf.cast(decoder_self_attention_bias,
                                          tf.bfloat16)
  return (decoder_input, decoder_self_attention_bias)


def transformer_decoder(decoder_input,
                        encoder_output,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        hparams,
                        cache=None,
                        decode_loop_step=None,
                        name="decoder",
                        nonpadding=None,
                        save_weights_to=None,
                        make_image_summary=True,
                        losses=None,
                        layer_collection=None,
                        recurrent_memory_by_layer=None,
                        chunk_number=None,
                        ):
  """A stack of transformer layers.

  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention (see
      common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    cache: dict, containing tensors which are the results of previous
      attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop. Only used
      for inference on TPU.
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This is used to mask out
      padding in convolutional layers.  We generally only need this mask for
      "packed" datasets, because for ordinary datasets, no padding is ever
      followed by nonpadding.
    save_weights_to: an optional dictionary to capture attention weights for
      visualization; the weights tensor will be appended there under a string
      key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    losses: optional list onto which to append extra training losses
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.
    recurrent_memory_by_layer: Optional dict, mapping layer names to instances
      of transformer_memory.RecurrentMemory. Default is None.
    chunk_number: an optional integer Tensor with shape [batch] used to operate
      the recurrent_memory.

  Returns:
    y: a Tensors
  """
  x = decoder_input
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))

  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_NUM_HIDDEN_LAYERS,
      value=hparams.num_decoder_layers or hparams.num_hidden_layers,
      hparams=hparams)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DROPOUT,
      value=hparams.attention_dropout,
      hparams=hparams)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DENSE,
      value={
          "use_bias": "false",
          "num_heads": hparams.num_heads,
          "hidden_size": hparams.hidden_size
      },
      hparams=hparams)

  with tf.variable_scope(name):
    for layer in range(hparams.num_decoder_layers or hparams.num_hidden_layers):
      layer_name = "layer_%d" % layer
      layer_cache = cache[layer_name] if cache is not None else None
      if recurrent_memory_by_layer is not None:
        recurrent_memory = recurrent_memory_by_layer[layer_name]
      else:
        recurrent_memory = None
      with tf.variable_scope(layer_name):
        with tf.variable_scope("self_attention"):
          y = common_attention.multihead_attention(
              common_layers.layer_preprocess(
                  x, hparams, layer_collection=layer_collection),
              None,
              decoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.self_attention_type,
              max_relative_position=hparams.max_relative_position,
              heads_share_relative_embedding=(
                  hparams.heads_share_relative_embedding),
              add_relative_to_values=hparams.add_relative_to_values,
              save_weights_to=save_weights_to,
              cache=layer_cache,
              make_image_summary=make_image_summary,
              dropout_broadcast_dims=attention_dropout_broadcast_dims,
              max_length=hparams.get("max_length"),
              decode_loop_step=decode_loop_step,
              vars_3d=hparams.get("attention_variables_3d"),
              activation_dtype=hparams.get("activation_dtype", "float32"),
              weight_dtype=hparams.get("weight_dtype", "float32"),
              layer_collection=layer_collection,
              recurrent_memory=recurrent_memory,
              chunk_number=chunk_number,
              hard_attention_k=hparams.get("hard_attention_k", 0)
              )
          x = common_layers.layer_postprocess(x, y, hparams)
        if encoder_output is not None:
          with tf.variable_scope("encdec_attention"):
            y = common_attention.multihead_attention(
                common_layers.layer_preprocess(
                    x, hparams, layer_collection=layer_collection),
                encoder_output,
                encoder_decoder_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,
                max_relative_position=hparams.max_relative_position,
                heads_share_relative_embedding=(
                    hparams.heads_share_relative_embedding),
                add_relative_to_values=hparams.add_relative_to_values,
                save_weights_to=save_weights_to,
                cache=layer_cache,
                make_image_summary=make_image_summary,
                dropout_broadcast_dims=attention_dropout_broadcast_dims,
                max_length=hparams.get("max_length"),
                vars_3d=hparams.get("attention_variables_3d"),
                activation_dtype=hparams.get("activation_dtype", "float32"),
                weight_dtype=hparams.get("weight_dtype", "float32"),
                layer_collection=layer_collection,
                hard_attention_k=hparams.get("hard_attention_k", 0))
            x = common_layers.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(
              common_layers.layer_preprocess(
                  x, hparams, layer_collection=layer_collection),
              hparams,
              conv_padding="LEFT",
              nonpadding_mask=nonpadding,
              losses=losses,
              cache=layer_cache,
              decode_loop_step=decode_loop_step,
              layer_collection=layer_collection)
          x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_NORM,
        value={"hidden_size": hparams.hidden_size})
    return common_layers.layer_preprocess(
        x, hparams, layer_collection=layer_collection)


@registry.register_model
class TransformerMemory(Transformer):
  """Transformer language model with memory across chunks."""

  # TODO(kitaev): consider overriding set_mode to swap out recurrent memory when
  # switching between training and evaluation.

  def __init__(self, *args, **kwargs):
    super(TransformerMemory, self).__init__(*args, **kwargs)

    hparams = self._hparams
    self.recurrent_memory_by_layer = {}
    for layer in range(hparams.num_decoder_layers or hparams.num_hidden_layers):
      layer_name = "layer_%d" % layer
      if hparams.memory_type == "neural_memory":
        memory = transformer_memory.TransformerMemory(
            batch_size=int(hparams.batch_size / hparams.max_length),
            key_depth=hparams.hidden_size,
            val_depth=hparams.hidden_size,
            memory_size=hparams.split_targets_chunk_length,
            sharpen_factor=1.,
            name=layer_name + "/recurrent_memory")
      elif hparams.memory_type == "transformer_xl":
        memory = transformer_memory.RecentTokensMemory(
            layer_name + "/recurrent_memory", hparams)
      else:
        raise ValueError("Unsupported memory type: %s" % hparams.memory_type)
      self.recurrent_memory_by_layer[layer_name] = memory

  @property
  def has_input(self):
    if hasattr(self._hparams, "unconditional") and self._hparams.unconditional:
      return False
    return super(TransformerMemory, self).has_input

  def _beam_decode(self, features, decode_length, beam_size, top_beams, alpha,
                   use_tpu=False):
    """Overriding beam search because for now only the slow version works with
    memory
    """
    return self._beam_decode_slow(features, decode_length, beam_size,
                                  top_beams, alpha, use_tpu)


@registry.register_hparams
def transformer_base_v1():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.norm_type = "layer"
  hparams.hidden_size = 512
  hparams.batch_size = 4096
  hparams.max_length = 256
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_schedule = "legacy"
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 4000
  hparams.initializer_gain = 1.0
  hparams.num_hidden_layers = 6
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.num_sampled_classes = 0
  hparams.label_smoothing = 0.1
  hparams.shared_embedding_and_softmax_weights = True
  hparams.symbol_modality_num_shards = 16

  # Add new ones like this.
  hparams.add_hparam("filter_size", 2048)
  # Layer-related flags. If zero, these fall back on hparams.num_hidden_layers.
  hparams.add_hparam("num_encoder_layers", 0)
  hparams.add_hparam("num_decoder_layers", 0)
  # Attention-related flags.
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  hparams.add_hparam("ffn_layer", "dense_relu_dense")
  hparams.add_hparam("parameter_attention_key_channels", 0)
  hparams.add_hparam("parameter_attention_value_channels", 0)
  # All hyperparameters ending in "dropout" are automatically set to 0.0
  # when not in training mode.
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("attention_dropout_broadcast_dims", "")
  hparams.add_hparam("relu_dropout", 0.0)
  hparams.add_hparam("relu_dropout_broadcast_dims", "")
  hparams.add_hparam("pos", "timing")  # timing, none
  hparams.add_hparam("nbr_decoder_problems", 1)
  hparams.add_hparam("proximity_bias", False)
  hparams.add_hparam("causal_decoder_self_attention", True)
  hparams.add_hparam("use_pad_remover", True)
  hparams.add_hparam("self_attention_type", "dot_product")
  hparams.add_hparam("conv_first_kernel", 3)
  hparams.add_hparam("attention_variables_3d", False)
  hparams.add_hparam("use_target_space_embedding", True)
  # These parameters are only used when ffn_layer=="local_moe_tpu"
  hparams.add_hparam("moe_overhead_train", 1.0)
  hparams.add_hparam("moe_overhead_eval", 2.0)
  hparams.moe_num_experts = 16
  hparams.moe_loss_coef = 1e-3
  # If specified, use this value instead of problem name in metrics.py.
  # This is useful for programs that can automatically compare experiments side
  #   by side based on the same metric names.
  hparams.add_hparam("overload_eval_metric_name", "")
  # For making a transformer encoder unidirectional by using masked
  # attention.
  hparams.add_hparam("unidirectional_encoder", False)
  # For hard attention.
  hparams.add_hparam("hard_attention_k", 0)
  return hparams


@registry.register_hparams
def transformer_base_v2():
  """Set of hyperparameters."""
  hparams = transformer_base_v1()
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1
  hparams.learning_rate_warmup_steps = 8000
  hparams.learning_rate = 0.2
  return hparams


@registry.register_hparams
def transformer_base_vq_ada_32ex_packed():
  """Set of hyperparameters for lm1b packed following tpu params."""
  hparams = transformer_base_v2()
  expert_utils.update_hparams_for_vq_gating(hparams)
  hparams.moe_num_experts = 32
  hparams.gating_type = "vq"
  # this gives us a batch size of 16 because each seq is len 256
  hparams.batch_size = 5072
  hparams.ffn_layer = "local_moe"
  hparams.shared_embedding_and_softmax_weights = False
  hparams.learning_rate_warmup_steps = 10000
  # one epoch for languagemodel_lm1b32k_packed = 27200 steps w/ bsize 128
  hparams.learning_rate_decay_steps = 27200
  hparams.num_heads = 4
  hparams.num_blocks = 1
  hparams.moe_k = 1
  hparams.num_decoder_layers = 6
  hparams.label_smoothing = 0.
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.layer_postprocess_sequence = "dan"
  hparams.layer_preprocess_sequence = "none"
  hparams.weight_decay = 1e-06
  hparams.attention_dropout = 0.1
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_schedule = "linear_warmup*rsqrt_decay*linear_decay"
  hparams.activation_dtype = "float32"
  hparams.learning_rate = 0.1
  hparams.learning_rate_constant = 1.0
  return hparams


@registry.register_hparams
def transformer_topk_16_packed():
  hparams = transformer_base_vq_ada_32ex_packed()
  hparams.gating_type = "topk"
  hparams.moe_num_experts = 16
  hparams.moe_k = 2
  return hparams


@registry.register_hparams
def transformer_base_vq1_16_nb1_packed_nda_b01_scales():
  """Set of hyperparameters."""
  hparams = transformer_base_vq_ada_32ex_packed()
  hparams.use_scales = int(True)
  hparams.moe_num_experts = 16
  hparams.moe_k = 1
  hparams.beta = 0.1
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.ema = False
  return hparams


@registry.register_hparams
def transformer_base_vq1_16_nb1_packed_dan_b01_scales():
  """Set of hyperparameters."""
  hparams = transformer_base_vq_ada_32ex_packed()
  hparams.use_scales = int(True)
  hparams.moe_num_experts = 16
  hparams.moe_k = 1
  hparams.beta = 0.1
  hparams.ema = False
  return hparams


@registry.register_hparams
def transformer_base_vq1_16_nb1_packed_nda_b01_scales_dialog():
  """Set of hyperparameters."""
  hparams = transformer_base_vq1_16_nb1_packed_nda_b01_scales()
  hparams.batch_size = 2048
  hparams.max_length = 1024
  hparams.filter_size = 3072
  return hparams


@registry.register_hparams
def transformer_ada_lmpackedbase():
  """Set of hyperparameters."""
  hparams = transformer_base_vq_ada_32ex_packed()
  hparams.ffn_layer = "dense_relu_dense"
  return hparams


@registry.register_hparams
def transformer_ada_lmpackedbase_dialog():
  """Set of hyperparameters."""
  hparams = transformer_base_vq_ada_32ex_packed()
  hparams.max_length = 1024
  hparams.ffn_layer = "dense_relu_dense"
  hparams.batch_size = 4096
  return hparams


@registry.register_hparams
def transformer_ada_lmpackedbase_relative():
  """Set of hyperparameters."""
  hparams = transformer_base_vq_ada_32ex_packed()
  hparams.ffn_layer = "dense_relu_dense"
  return hparams


@registry.register_hparams
def transformer_base_v3():
  """Base parameters for Transformer model."""
  # Update parameters here, then occasionally cut a versioned set, e.g.
  # transformer_base_v2.
  hparams = transformer_base_v2()
  hparams.optimizer_adam_beta2 = 0.997
  # New way of specifying learning rate schedule.
  # Equivalent to previous version.
  hparams.learning_rate_schedule = (
      "constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size")
  hparams.learning_rate_constant = 2.0
  return hparams


@registry.register_hparams
def transformer_base():
  """Base parameters for Transformer model."""
  hparams = transformer_base_v3()
  return hparams


@registry.register_hparams
def transformer_big():
  """HParams for transformer big model on WMT."""
  hparams = transformer_base()
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  # Reduce batch size to 2048 from 4096 to be able to train the model on a GPU
  # with 12 GB memory. For example, NVIDIA TITAN V GPU.
  hparams.batch_size = 2048
  hparams.num_heads = 16
  hparams.layer_prepostprocess_dropout = 0.3
  return hparams


@registry.register_hparams
def transformer_tall():
  """Hparams for transformer on LM for pretraining/finetuning/mixing."""
  hparams = transformer_base()
  hparams.batch_size = 2048
  hparams.hidden_size = 768
  hparams.filter_size = 3072
  hparams.num_hidden_layers = 12
  hparams.num_heads = 12
  hparams.label_smoothing = 0.0
  hparams.max_length = 1024
  hparams.eval_drop_long_sequences = True
  hparams.multiproblem_mixing_schedule = "pretrain"
  hparams.multiproblem_vocab_size = 65536
  hparams.clip_grad_norm = 1.0
  return hparams


@registry.register_hparams
def transformer_tall_finetune_tied():
  """Tied means fine-tune CNN/DM summarization as LM."""
  hparams = transformer_tall()
  hparams.multiproblem_max_input_length = 750
  hparams.multiproblem_max_target_length = 100
  hparams.multiproblem_schedule_max_examples = 0
  hparams.learning_rate_schedule = ("linear_warmup*constant*cosdecay")
  hparams.learning_rate_constant = 5e-5
  hparams.learning_rate_warmup_steps = 100
  # Set train steps to learning_rate_decay_steps or less
  hparams.learning_rate_decay_steps = 80000
  hparams.multiproblem_target_eval_only = True
  hparams.multiproblem_reweight_label_loss = True
  hparams.multiproblem_label_weight = 1.0
  hparams.optimizer = "true_adam"
  return hparams


@registry.register_hparams
def transformer_tall_train_tied():
  """Tied means train CNN/DM summarization as LM."""
  hparams = transformer_tall()
  hparams.multiproblem_max_input_length = 750
  hparams.multiproblem_max_target_length = 100
  hparams.multiproblem_schedule_max_examples = 0
  hparams.learning_rate_schedule = ("linear_warmup*constant*cosdecay")
  hparams.learning_rate_constant = 2e-4
  hparams.learning_rate_warmup_steps = 8000
  # Set train steps to learning_rate_decay_steps or less
  hparams.learning_rate_decay_steps = 150000
  hparams.multiproblem_target_eval_only = True
  hparams.multiproblem_reweight_label_loss = True
  hparams.multiproblem_label_weight = 1.0
  hparams.optimizer = "true_adam"
  return hparams


@registry.register_hparams
def transformer_tall_finetune_uniencdec():
  """Fine-tune CNN/DM with a unidirectional encoder and decoder."""
  hparams = transformer_tall()
  hparams.max_input_seq_length = 750
  hparams.max_target_seq_length = 100
  hparams.optimizer = "true_adam"
  hparams.learning_rate_schedule = ("linear_warmup*constant*cosdecay")
  hparams.learning_rate_decay_steps = 80000
  hparams.learning_rate_constant = 5e-5
  hparams.learning_rate_warmup_steps = 100
  hparams.unidirectional_encoder = True
  return hparams


@registry.register_hparams
def transformer_tall_train_uniencdec():
  """Train CNN/DM with a unidirectional encoder and decoder."""
  hparams = transformer_tall()
  hparams.max_input_seq_length = 750
  hparams.max_target_seq_length = 100
  hparams.optimizer = "true_adam"
  hparams.learning_rate_schedule = ("linear_warmup*constant*cosdecay")
  hparams.learning_rate_decay_steps = 150000
  hparams.learning_rate_constant = 2e-4
  hparams.unidirectional_encoder = True
  return hparams


@registry.register_hparams
def transformer_tall_finetune_textclass():
  """Hparams for transformer on LM for finetuning on text class problems."""
  hparams = transformer_tall()
  hparams.learning_rate_constant = 6.25e-5
  hparams.learning_rate_schedule = ("linear_warmup*constant*linear_decay")
  hparams.multiproblem_schedule_max_examples = 0
  hparams.multiproblem_target_eval_only = True
  hparams.learning_rate_warmup_steps = 50
  # Set train steps to learning_rate_decay_steps or less
  hparams.learning_rate_decay_steps = 25000
  hparams.multiproblem_reweight_label_loss = True
  hparams.multiproblem_label_weight = 0.95
  return hparams


@registry.register_hparams
def transformer_tall_pretrain_lm():
  """Hparams for transformer on LM pretraining (with 64k vocab)."""
  hparams = transformer_tall()
  hparams.learning_rate_constant = 2e-4
  hparams.learning_rate_schedule = ("linear_warmup*constant*cosdecay")
  hparams.optimizer = "adam_w"
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.999
  hparams.optimizer_adam_epsilon = 1e-8
  # Set max examples to something big when pretraining only the LM, definitely
  # something an order of magnitude bigger than number of train steps.
  hparams.multiproblem_schedule_max_examples = 5e8
  # Set train steps to learning_rate_decay_steps or less
  hparams.learning_rate_decay_steps = 5000000
  return hparams


@registry.register_hparams
def transformer_tall_pretrain_lm_tpu_adafactor():
  """Hparams for transformer on LM pretraining (with 64k vocab) on TPU."""
  hparams = transformer_tall_pretrain_lm()
  update_hparams_for_tpu(hparams)
  hparams.max_length = 1024
  # For multi-problem on TPU we need it in absolute examples.
  hparams.batch_size = 8
  hparams.multiproblem_vocab_size = 2**16
  return hparams


@registry.register_hparams
def transformer_tall_pretrain_lm_tpu_adafactor_large():
  """Hparams for transformer on LM pretraining on TPU, large model."""
  hparams = transformer_tall_pretrain_lm_tpu_adafactor()
  hparams.hidden_size = 1024
  hparams.num_heads = 16
  hparams.filter_size = 32768  # max fitting in 16G memory is 49152, batch 2
  hparams.batch_size = 4
  hparams.multiproblem_mixing_schedule = "constant"
  # Task order: lm/en-de/en-fr/en-ro/de-en/fr-en/ro-en/cnndm/mnli/squad.
  hparams.multiproblem_per_task_threshold = "320,80,160,1,80,160,2,20,10,5"
  return hparams


@registry.register_hparams
def transformer_tall_pretrain_lm_tpu():
  """Hparams for transformer on LM pretraining on TPU with AdamW."""
  hparams = transformer_tall_pretrain_lm_tpu_adafactor()
  # Optimizer gets reset in update_hparams_for_tpu so we set it again here.
  hparams.learning_rate_constant = 2e-4
  hparams.learning_rate_schedule = ("linear_warmup * constant * cosdecay")
  hparams.optimizer = "adam_w"
  return hparams


@registry.register_hparams
def transformer_tall_big():
  """Hparams for transformer on LM+MNLI."""
  hparams = transformer_tall()
  hparams.num_hidden_layers = 18
  return hparams


@registry.register_hparams
def transformer_big_single_gpu():
  """HParams for transformer big model for single GPU."""
  hparams = transformer_big()
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.learning_rate_warmup_steps = 16000
  return hparams


@registry.register_hparams
def transformer_base_single_gpu():
  """HParams for transformer base model for single GPU."""
  hparams = transformer_base()
  hparams.batch_size = 1024
  hparams.learning_rate_schedule = "constant*linear_warmup*rsqrt_decay"
  hparams.learning_rate_constant = 0.1
  hparams.learning_rate_warmup_steps = 16000
  return hparams


@registry.register_hparams
def transformer_base_multistep8():
  """HParams for simulating 8 GPUs with MultistepAdam optimizer."""
  hparams = transformer_base()
  hparams.optimizer = "MultistepAdam"
  hparams.optimizer_multistep_accumulate_steps = 8
  return hparams


@registry.register_hparams
def transformer_parsing_base():
  """HParams for parsing on WSJ only."""
  hparams = transformer_base()
  hparams.attention_dropout = 0.2
  hparams.layer_prepostprocess_dropout = 0.2
  hparams.max_length = 512
  hparams.learning_rate_warmup_steps = 16000
  hparams.hidden_size = 1024
  hparams.learning_rate = 0.05
  hparams.shared_embedding_and_softmax_weights = False
  return hparams


@registry.register_hparams
def transformer_parsing_big():
  """HParams for parsing on WSJ semi-supervised."""
  hparams = transformer_big()
  hparams.max_length = 512
  hparams.shared_source_target_embedding = False
  hparams.learning_rate_warmup_steps = 4000
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.batch_size = 2048
  hparams.learning_rate = 0.05
  return hparams


@registry.register_hparams
def transformer_parsing_ice():
  """HParams for parsing and tagging Icelandic text."""
  hparams = transformer_base_single_gpu()
  hparams.batch_size = 4096
  hparams.shared_embedding_and_softmax_weights = False
  return hparams


@registry.register_hparams
def transformer_tiny():
  hparams = transformer_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 128
  hparams.filter_size = 512
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def transformer_test():
  hparams = transformer_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 16
  hparams.filter_size = 8
  hparams.num_heads = 2
  return hparams


@registry.register_hparams
def transformer_small():
  hparams = transformer_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 256
  hparams.filter_size = 1024
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def transformer_l2():
  hparams = transformer_base()
  hparams.num_hidden_layers = 2
  return hparams


@registry.register_hparams
def transformer_l4():
  hparams = transformer_base()
  hparams.num_hidden_layers = 4
  return hparams


@registry.register_hparams
def transformer_l8():
  hparams = transformer_base()
  hparams.num_hidden_layers = 8
  return hparams


@registry.register_hparams
def transformer_l10():
  hparams = transformer_base()
  hparams.num_hidden_layers = 10
  return hparams


@registry.register_hparams
def transformer_h1():
  hparams = transformer_base()
  hparams.num_heads = 1
  return hparams


@registry.register_hparams
def transformer_h4():
  hparams = transformer_base()
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def transformer_h16():
  hparams = transformer_base()
  hparams.num_heads = 16
  return hparams


@registry.register_hparams
def transformer_h32():
  hparams = transformer_base()
  hparams.num_heads = 32
  return hparams


@registry.register_hparams
def transformer_k128():
  hparams = transformer_base()
  hparams.attention_key_channels = 128
  return hparams


@registry.register_hparams
def transformer_k256():
  hparams = transformer_base()
  hparams.attention_key_channels = 256
  return hparams


@registry.register_hparams
def transformer_ff1024():
  hparams = transformer_base()
  hparams.filter_size = 1024
  return hparams


@registry.register_hparams
def transformer_ff4096():
  hparams = transformer_base()
  hparams.filter_size = 4096
  return hparams


@registry.register_hparams
def transformer_dr0():
  hparams = transformer_base()
  hparams.layer_prepostprocess_dropout = 0.0
  return hparams


@registry.register_hparams
def transformer_dr2():
  hparams = transformer_base()
  hparams.layer_prepostprocess_dropout = 0.2
  return hparams


@registry.register_hparams
def transformer_ls0():
  hparams = transformer_base()
  hparams.label_smoothing = 0.0
  return hparams


@registry.register_hparams
def transformer_ls2():
  hparams = transformer_base()
  hparams.label_smoothing = 0.2
  return hparams


@registry.register_hparams
def transformer_hs256():
  hparams = transformer_base()
  hparams.hidden_size = 256
  return hparams


@registry.register_hparams
def transformer_hs1024():
  hparams = transformer_base()
  hparams.hidden_size = 1024
  return hparams


@registry.register_hparams
def transformer_big_dr1():
  hparams = transformer_base()
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  hparams.num_heads = 16
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams


@registry.register_hparams
def transformer_big_enfr():
  hparams = transformer_big_dr1()
  hparams.shared_embedding_and_softmax_weights = False
  hparams.filter_size = 8192
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams


@registry.register_hparams
def transformer_big_enfr_tpu():
  hparams = transformer_big_enfr()
  # For performance, use fewer heads so that matrix dimensions are at least 128
  hparams.num_heads = 8
  update_hparams_for_tpu(hparams)
  return hparams


@registry.register_hparams
def transformer_big_dr2():
  hparams = transformer_big_dr1()
  hparams.layer_prepostprocess_dropout = 0.2
  return hparams


@registry.register_hparams
def transformer_parameter_attention_a():
  hparams = transformer_base()
  hparams.ffn_layer = "parameter_attention"
  hparams.filter_size = 1536
  return hparams


@registry.register_hparams
def transformer_parameter_attention_b():
  hparams = transformer_base()
  hparams.ffn_layer = "parameter_attention"
  hparams.filter_size = 512
  hparams.parameter_attention_key_channels = 1024
  hparams.parameter_attention_value_channels = 1024
  hparams.num_heads = 16
  return hparams


@registry.register_hparams
def transformer_prepend_v2():
  hparams = transformer_base_v2()
  hparams.prepend_mode = "prepend_inputs_masked_attention"
  hparams.max_length = 0
  return hparams


@registry.register_hparams
def transformer_prepend_v1():
  hparams = transformer_base_v1()
  hparams.prepend_mode = "prepend_inputs_masked_attention"
  hparams.max_length = 0
  return hparams


@registry.register_hparams
def transformer_prepend():
  return transformer_prepend_v2()


@registry.register_ranged_hparams
def transformer_base_range(rhp):
  """Small range of hyperparameters."""
  # After starting from base, set intervals for some parameters.
  rhp.set_float("learning_rate", 0.3, 3.0, scale=rhp.LOG_SCALE)
  rhp.set_discrete("learning_rate_warmup_steps",
                   [1000, 2000, 4000, 8000, 16000])
  rhp.set_float("initializer_gain", 0.5, 2.0)
  rhp.set_float("optimizer_adam_beta1", 0.85, 0.95)
  rhp.set_float("optimizer_adam_beta2", 0.97, 0.99)
  rhp.set_float("weight_decay", 0.0, 1e-4)


@registry.register_hparams
def transformer_relative():
  """Use relative position embeddings instead of absolute position encodings."""
  hparams = transformer_base()
  hparams.pos = None
  hparams.self_attention_type = "dot_product_relative"
  hparams.max_relative_position = 20
  return hparams


@registry.register_hparams
def transformer_relative_tiny():
  hparams = transformer_relative()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 128
  hparams.filter_size = 512
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def transformer_relative_big():
  hparams = transformer_big()
  hparams.pos = None
  hparams.self_attention_type = "dot_product_relative"
  hparams.max_relative_position = 20
  return hparams


@registry.register_hparams
def transformer_timeseries():
  hparams = transformer_small()
  hparams.batch_size = 256
  hparams.learning_rate_warmup_steps = 2000
  return hparams


@registry.register_hparams
def transformer_mlperf_tpu():
  """HParams for Transformer model on TPU for MLPerf on TPU 2x2."""
  hparams = transformer_base_v3()
  hparams.mlperf_mode = True
  hparams.symbol_modality_num_shards = 1
  hparams.max_length = 256  # ignored when using "_packed" problems
  hparams.batch_size = 2048  # per-chip batch size matches the reference model
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  hparams.num_heads = 16
  hparams.attention_dropout_broadcast_dims = "0,1"  # batch, heads
  hparams.relu_dropout_broadcast_dims = "1"  # length
  hparams.layer_prepostprocess_dropout_broadcast_dims = "1"  # length
  return hparams


def update_hparams_for_tpu(hparams):
  """Change hparams to be compatible with TPU training."""

  # Adafactor uses less memory than Adam.
  # switch to Adafactor with its recommended learning rate scheme.
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_schedule = "rsqrt_decay"
  hparams.learning_rate_warmup_steps = 10000

  # Avoid an expensive concat on TPU.
  # >1 shards helps with faster parameter distribution on multi-GPU machines
  hparams.symbol_modality_num_shards = 1

  # Adaptive batch sizes and sequence lengths are not supported on TPU.
  # Instead, every batch has the same sequence length and the same batch size.
  # Longer sequences are dropped and shorter ones are padded.
  #
  # It is therefore suggested to use a problem where examples have been combined
  # to a longer length, e.g. the "_packed" problems.
  #
  # For problems with variable sequence lengths, this parameter controls the
  # maximum sequence length.  Shorter sequences are dropped and longer ones
  # are padded.
  #
  # For problems with fixed sequence lengths - e.g. the "_packed" problems,
  # this hyperparameter is ignored.
  hparams.max_length = 64

  # TPUs have less memory than GPUs, so decrease the batch size
  hparams.batch_size = 2048

  # Using noise broadcast in the dropout layers saves memory during training.
  hparams.attention_dropout_broadcast_dims = "0,1"  # batch, heads
  hparams.relu_dropout_broadcast_dims = "1"  # length
  hparams.layer_prepostprocess_dropout_broadcast_dims = "1"  # length
  return hparams


@registry.register_hparams
def transformer_tpu():
  """HParams for Transformer model on TPU."""
  hparams = transformer_base()
  update_hparams_for_tpu(hparams)
  return hparams


@registry.register_hparams
def transformer_timeseries_tpu():
  """HParams for running Transformer model on timeseries on TPU."""
  hparams = transformer_timeseries()
  update_hparams_for_tpu(hparams)
  hparams.batch_size = 256  # revert to value set in transformer_timeseries
  return hparams


@registry.register_hparams
def transformer_tpu_bf16_activation():
  """HParams for Transformer model with BF16 activation on TPU."""
  hparams = transformer_tpu()
  hparams.activation_dtype = "bfloat16"
  return hparams


@registry.register_hparams
def transformer_fairseq_fp16_activation_big():
  """Hparams intended to mirror those used in arxiv.org/pdf/1806.00187.pdf."""
  hparams = transformer_big()
  hparams.activation_dtype = "float16"
  hparams.batch_size = 3584
  return hparams


@registry.register_hparams
def transformer_packed_tpu():
  """Deprecated alias for transformer_tpu()."""
  return transformer_tpu()


@registry.register_hparams
def transformer_big_tpu():
  hparams = transformer_big()
  update_hparams_for_tpu(hparams)
  return hparams


@registry.register_hparams
def transformer_tiny_tpu():
  hparams = transformer_tiny()
  update_hparams_for_tpu(hparams)
  return hparams


@registry.register_ranged_hparams
def transformer_tiny_tpu_range(rhp):
  """Small range of hyperparameters."""
  rhp.set_float("learning_rate", 0.3, 3.0, scale=rhp.LOG_SCALE)
  rhp.set_float("weight_decay", 0.0, 2.0)


@registry.register_ranged_hparams
def transformer_tpu_range(rhp):
  """Small range of hyperparameters."""
  # After starting from base, set intervals for some parameters.
  rhp.set_float("learning_rate", 0.3, 3.0, scale=rhp.LOG_SCALE)
  rhp.set_discrete("learning_rate_warmup_steps",
                   [1000, 2000, 4000, 8000, 16000])
  rhp.set_float("initializer_gain", 0.5, 2.0)
  rhp.set_float("optimizer_adam_beta1", 0.85, 0.95)
  rhp.set_float("optimizer_adam_beta2", 0.97, 0.99)
  rhp.set_float("weight_decay", 0.0, 2.0)


@registry.register_hparams
def transformer_small_tpu():
  """TPU-friendly version of transformer_small.

  Returns:
    an hparams object.
  """
  hparams = transformer_small()
  update_hparams_for_tpu(hparams)
  return hparams


@registry.register_hparams
def transformer_clean():
  """No dropout, label smoothing, max_length."""
  hparams = transformer_base_v2()
  hparams.label_smoothing = 0.0
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.attention_dropout = 0.0
  hparams.relu_dropout = 0.0
  hparams.max_length = 0
  return hparams


@registry.register_hparams
def transformer_clean_big():
  hparams = transformer_clean()
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  return hparams


@registry.register_hparams
def transformer_clean_big_tpu():
  hparams = transformer_clean_big()
  update_hparams_for_tpu(hparams)
  return hparams


@registry.register_hparams
def transformer_tpu_with_conv():
  """Cut down on the number of heads, and use convs instead."""
  hparams = transformer_tpu()
  hparams.num_heads = 4  # Heads are expensive on TPUs.
  hparams.ffn_layer = "conv_relu_conv"
  return hparams


@registry.register_hparams
def transformer_lm_tpu_0():
  """HParams for training languagemodel_lm1b8k on tpu.  92M Params."""
  hparams = transformer_clean_big()
  update_hparams_for_tpu(hparams)
  hparams.num_heads = 4  # Heads are expensive on TPUs.
  hparams.batch_size = 4096
  hparams.shared_embedding_and_softmax_weights = False
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams


@registry.register_hparams
def transformer_lm_tpu_1():
  """HParams for training languagemodel_lm1b8k on tpu.  335M Params."""
  hparams = transformer_lm_tpu_0()
  hparams.hidden_size = 2048
  hparams.filter_size = 8192
  return hparams


@registry.register_hparams
def transformer_librispeech_v1():
  """HParams for training ASR model on LibriSpeech V1."""
  hparams = transformer_base()

  hparams.num_heads = 4
  hparams.filter_size = 1024
  hparams.hidden_size = 256
  hparams.num_encoder_layers = 5
  hparams.num_decoder_layers = 3
  hparams.learning_rate = 0.15
  hparams.batch_size = 6000000

  librispeech.set_librispeech_length_hparams(hparams)
  return hparams


@registry.register_hparams
def transformer_librispeech_v2():
  """HParams for training ASR model on LibriSpeech V2."""
  hparams = transformer_base()

  hparams.max_length = 1240000
  hparams.max_input_seq_length = 1550
  hparams.max_target_seq_length = 350
  hparams.batch_size = 16
  hparams.num_decoder_layers = 4
  hparams.num_encoder_layers = 6
  hparams.hidden_size = 384
  hparams.learning_rate = 0.15
  hparams.daisy_chain_variables = False
  hparams.filter_size = 1536
  hparams.num_heads = 2
  hparams.ffn_layer = "conv_relu_conv"
  hparams.conv_first_kernel = 9
  hparams.weight_decay = 0
  hparams.layer_prepostprocess_dropout = 0.2
  hparams.relu_dropout = 0.2

  return hparams


@registry.register_hparams
def transformer_librispeech_tpu_v1():
  """HParams for training ASR model on Librispeech on TPU v1."""
  hparams = transformer_librispeech_v1()
  update_hparams_for_tpu(hparams)

  hparams.batch_size = 16
  librispeech.set_librispeech_length_hparams(hparams)
  return hparams


@registry.register_hparams
def transformer_librispeech_tpu_v2():
  """HParams for training ASR model on Librispeech on TPU v2."""
  hparams = transformer_librispeech_v2()
  update_hparams_for_tpu(hparams)

  hparams.batch_size = 16
  librispeech.set_librispeech_length_hparams(hparams)
  return hparams


@registry.register_hparams
def transformer_librispeech():
  """HParams for training ASR model on Librispeech."""
  return transformer_librispeech_v2()


@registry.register_hparams
def transformer_librispeech_tpu():
  """HParams for training ASR model on Librispeech on TPU."""
  return transformer_librispeech_tpu_v2()


@registry.register_hparams
def transformer_common_voice():
  """HParams for training ASR model on Mozilla Common Voice."""
  return transformer_librispeech()


@registry.register_hparams
def transformer_common_voice_tpu():
  """HParams for training ASR model on Mozilla Common Voice on TPU."""
  hparams = transformer_librispeech_tpu()
  hparams.batch_size = 8
  return hparams


@registry.register_hparams
def transformer_supervised_attention():
  """HParams for supervised attention problems."""
  hparams = transformer_base()
  # Attention loss type (KL-divergence or MSE).
  hparams.add_hparam("expected_attention_loss_type", "kl_divergence")
  # Multiplier to the encoder-decoder expected attention loss.
  hparams.add_hparam("expected_attention_loss_multiplier", 1.0)
  return hparams


@registry.register_hparams
def transformer_tpu_1b():
  """Hparams for machine translation with ~1.1B parameters."""
  hparams = transformer_tpu()
  hparams.hidden_size = 2048
  hparams.filter_size = 8192
  hparams.num_hidden_layers = 8
  # smaller batch size to avoid OOM
  hparams.batch_size = 1024
  hparams.activation_dtype = "bfloat16"
  hparams.weight_dtype = "bfloat16"
  # maximize number of parameters relative to computation by not sharing.
  hparams.shared_embedding_and_softmax_weights = False
  return hparams


@registry.register_hparams
def transformer_wikitext103_l4k_v0():
  """HParams for training languagemodel_wikitext103_l4k."""
  hparams = transformer_big()

  # Adafactor uses less memory than Adam.
  # switch to Adafactor with its recommended learning rate scheme.
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_schedule = "rsqrt_decay"
  hparams.learning_rate_warmup_steps = 10000

  hparams.num_heads = 4
  hparams.max_length = 4096
  hparams.batch_size = 4096
  hparams.shared_embedding_and_softmax_weights = False

  hparams.num_hidden_layers = 8
  hparams.attention_dropout = 0.1
  hparams.layer_prepostprocess_dropout = 0.2
  hparams.relu_dropout = 0.1
  hparams.label_smoothing = 0.0

  # Using noise broadcast in the dropout layers saves memory during training.
  hparams.attention_dropout_broadcast_dims = "0,1"  # batch, heads
  hparams.relu_dropout_broadcast_dims = "1"  # length
  hparams.layer_prepostprocess_dropout_broadcast_dims = "1"  # length

  # Avoid an expensive concat on TPU.
  # >1 shards helps with faster parameter distribution on multi-GPU machines
  hparams.symbol_modality_num_shards = 1

  return hparams


@registry.register_hparams
def transformer_wikitext103_l4k_memory_v0():
  """HParams for training languagemodel_wikitext103_l4k with memory."""
  hparams = transformer_wikitext103_l4k_v0()

  hparams.split_targets_chunk_length = 64
  hparams.split_targets_max_chunks = 64
  hparams.add_hparam("memory_type", "transformer_xl")

  # The hparams specify batch size *before* chunking, but we want to have a
  # consistent 4K batch size *after* chunking to fully utilize the hardware.
  target_tokens_per_batch = 4096
  hparams.batch_size = int(target_tokens_per_batch * (
      hparams.max_length / hparams.split_targets_chunk_length))  # 262144

  hparams.pos = None
  hparams.self_attention_type = "dot_product_relative"
  hparams.max_relative_position = 2 * hparams.split_targets_chunk_length

  hparams.add_hparam("unconditional", True)
  hparams.add_hparam("recurrent_memory_batch_size", 0)  # 0 = try to guess
  # By default, cache one chunk only (like Transformer-XL)
  hparams.add_hparam("num_memory_items", hparams.split_targets_chunk_length)

  return hparams


@registry.register_hparams
def transformer_wikitext103_l16k_memory_v0():
  """HParams for training languagemodel_wikitext103_l16k with memory."""
  hparams = transformer_wikitext103_l4k_memory_v0()

  hparams.max_length = 16384
  hparams.split_targets_chunk_length = 64
  hparams.split_targets_max_chunks = int(
      hparams.max_length / hparams.split_targets_chunk_length)

  # The hparams specify batch size *before* chunking, but we want to have a
  # consistent 4K batch size *after* chunking to fully utilize the hardware.
  target_tokens_per_batch = 4096
  hparams.batch_size = int(target_tokens_per_batch * (
      hparams.max_length / hparams.split_targets_chunk_length))

  hparams.max_relative_position = 2 * hparams.split_targets_chunk_length

  return hparams


@registry.register_hparams
def transformer_cifar10_memory_v0():
  """HParams for training image_cifar10_plain_gen_flat_rev with memory."""
  hparams = transformer_wikitext103_l4k_memory_v0()

  hparams.num_hidden_layers = 6

  hparams.max_length = 32 * 32 * 3
  hparams.split_targets_chunk_length = 64 * 3
  hparams.split_targets_max_chunks = int(
      hparams.max_length / hparams.split_targets_chunk_length)
  hparams.num_memory_items = 128 * 3

  # Since this is an image problem, batch size refers to examples (not tokens)
  target_images_per_batch = 4
  hparams.batch_size = int(target_images_per_batch * (
      hparams.max_length / hparams.split_targets_chunk_length))

  # The recurrent memory needs to know the actual batch size (in sequences)
  hparams.recurrent_memory_batch_size = hparams.batch_size

  hparams.max_relative_position = (
      hparams.num_memory_items + hparams.split_targets_chunk_length)

  return hparams

