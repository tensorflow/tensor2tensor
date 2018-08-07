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
"""Temporary hack for decoding mtf_transformer models.

This is a transformer implementation in regular TensorFlow which is
checkpoint-compatible with MtfTransformer for eval/inference.

The purpose of this model is to run inference on MtfTransformer models.
We are working on native decoding in MtfTransformer which will be faster and
cleaner.

TODO(noam): Remove once we can decode in mtf.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

from tensorflow.python.util import nest


@registry.register_model
class MtfTransformerCompat(t2t_model.T2TModel):
  """Attention net.  See file docstring."""

  def __init__(self, *args, **kwargs):
    with tf.variable_scope("transformer"):
      self._top_scope = tf.get_variable_scope()
    kwargs["_scope"] = "transformer"
    super(MtfTransformerCompat, self).__init__(*args, **kwargs)
    self._name = "transformer"
    self._base_name = "transformer"

  @property
  def _targets_vocab_size(self):
    targets_vocab_size = self._problem_hparams.vocabulary["targets"].vocab_size
    targets_vocab_size += (-targets_vocab_size) % self._hparams.vocab_divisor
    return targets_vocab_size

  @property
  def _inputs_vocab_size(self):
    if not self.has_input:
      return None
    inputs_vocab_size = self._problem_hparams.vocabulary["inputs"].vocab_size
    inputs_vocab_size += (-inputs_vocab_size) % self._hparams.vocab_divisor
    return inputs_vocab_size

  @property
  def _embedding_and_softmax_var_names(self):
    """Figure out the variable names for the embedding and softmax variables.

    Equality between the returned names means that we should share the
    variables.

    Returns:
      inputs_embedding_name: a string or None
      targets_embedding_name: a string
      softmax_var_name: a string
    Raises:
      ValueError: if we try to share embeddings with different vocab sizes.
    """
    hparams = self._hparams
    inputs_embedding_name = "input_emb"
    targets_embedding_name = "target_emb"
    softmax_var_name = "softmax"
    if (self.has_input and
        (hparams.shared_embedding or
         hparams.shared_embedding_and_softmax_weights) and
        self._inputs_vocab_size != self._targets_vocab_size):
      raise ValueError(
          "hparams.shared_embedding_and_softmax_weights "
          " or hparams.shared_embedding require "
          "that input and target vocabulary sizes be equal %s vs %s"
          % (self._inputs_vocab_size, self._targets_vocab_size))
    if hparams.shared_embedding_and_softmax_weights:
      inputs_embedding_name = "shared"
      targets_embedding_name = "shared"
      softmax_var_name = "shared"
    elif hparams.shared_embedding:
      inputs_embedding_name = "shared"
      targets_embedding_name = "shared"
    targets_embedding_name = (
        "symbol_modality_%d_%d/%s/weights_0" %
        (self._targets_vocab_size, hparams.d_model, targets_embedding_name))
    softmax_var_name = (
        "symbol_modality_%d_%d/%s/weights_0" %
        (self._targets_vocab_size, hparams.d_model, softmax_var_name))
    if self.has_input:
      inputs_embedding_name = (
          "symbol_modality_%d_%d/%s/weights_0" %
          (self._inputs_vocab_size, hparams.d_model, inputs_embedding_name))
    else:
      inputs_embedding_name = None
    return inputs_embedding_name, targets_embedding_name, softmax_var_name

  @property
  def _get_targets_emb_var(self):
    with tf.variable_scope(self._top_scope, reuse=tf.AUTO_REUSE):
      return tf.get_variable(
          "targets_embedding",
          [self._targets_vocab_size, self._hparams.d_model])

  @property
  def _get_inputs_emb_var(self):
    if self._hparams.shared_embedding:
      return self._get_targets_emb_var
    with tf.variable_scope(self._top_scope, reuse=tf.AUTO_REUSE):
      return tf.get_variable(
          "inputs_embedding",
          [self._inputs_vocab_size, self._hparams.d_model])

  @property
  def _get_softmax_var(self):
    if self._hparams.shared_embedding_and_softmax_weights:
      return self._get_targets_emb_var * (self._hparams.d_model ** -0.5)
    with tf.variable_scope(self._top_scope, reuse=tf.AUTO_REUSE):
      return tf.get_variable(
          "softmax",
          [self._targets_vocab_size, self._hparams.d_model])

  def encode(self, inputs, hparams, features=None, losses=None):
    """Encode transformer inputs.

    Args:
      inputs: Transformer inputs [batch_size, input_length, input_height,
        hidden_dim] which will be flattened along the two spatial dimensions.
      hparams: hyperparameters for model.
      features: optionally pass the entire features dictionary as well.
        This is needed now for "packed" datasets.
      losses: optional list onto which to append extra training losses

    Returns:
      Tuple of:
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_decoder_attention_bias: Bias and mask weights for
              encoder-decoder attention. [batch_size, input_length]
    """
    tf.logging.info("Encode inputs = %s" % inputs)
    encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
        transformer_prepare_encoder(
            self._get_inputs_emb_var, inputs, hparams, features=features))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    encoder_output = transformer_encoder(
        encoder_input,
        self_attention_bias,
        hparams,
        losses=losses)

    return encoder_output, encoder_decoder_attention_bias

  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_self_attention_bias,
             hparams,
             cache=None,
             losses=None):
    """Decode Transformer outputs from encoder representation.

    Args:
      decoder_input: inputs to bottom of the model.
          [batch_size, decoder_length, hidden_dim]
      encoder_output: Encoder representation.
          [batch_size, input_length, hidden_dim]
      encoder_decoder_attention_bias: Bias and mask weights for
          encoder-decoder attention. [batch_size, input_length]
      decoder_self_attention_bias: Bias and mask weights for decoder
          self-attention. [batch_size, decoder_length]
      hparams: hyperparameters for model.
      cache: dict, containing tensors which are the results of previous
          attentions, used for fast decoding.
      losses: optional list onto which to append extra training losses

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    decoder_output = transformer_decoder(
        decoder_input,
        encoder_output,
        decoder_self_attention_bias,
        encoder_decoder_attention_bias,
        hparams,
        cache=cache,
        losses=losses)

    ret = tf.tensordot(decoder_output, self._get_softmax_var, axes=[[-1], [1]])
    ret = tf.expand_dims(tf.expand_dims(ret, 2), 3)
    return ret

  def body(self, features):
    """Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs [batch_size, input_length, hidden_dim]
          "targets": Target decoder outputs.
              [batch_size, decoder_length, hidden_dim]

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    with tf.variable_scope(self._top_scope):
      hparams = self._hparams
      losses = []

      if self.has_input:
        inputs = tf.squeeze(features["inputs_raw"], (2, 3))
        encoder_output, encoder_decoder_attention_bias = self.encode(
            inputs, hparams, features=features, losses=losses)
      else:
        encoder_output, encoder_decoder_attention_bias = (None, None)

      targets = tf.squeeze(features["targets_raw"], (2, 3))
      decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(
          self._get_targets_emb_var, targets, hparams, features=features)

      decoder_output = self.decode(
          decoder_input,
          encoder_output,
          encoder_decoder_attention_bias,
          decoder_self_attention_bias,
          hparams,
          losses=losses)

      if losses:
        return decoder_output, {"extra_loss": tf.add_n(losses)}
      else:
        return decoder_output

  def _greedy_infer(self, features, decode_length, use_tpu=False):
    """Fast version of greedy decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      use_tpu: a boolean

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
    with tf.variable_scope(self.name):
      return  self._fast_decode(features, decode_length)

  def _beam_decode(self, features, decode_length, beam_size, top_beams, alpha):
    """Beam search decoding.

    Args:
      features: an map of string to `Tensor`
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
    """
    with tf.variable_scope(self.name):
      return self._fast_decode(features, decode_length, beam_size, top_beams,
                               alpha)

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
    target_modality = self._problem_hparams.target_modality
    if "targets_segmentation" in features:
      raise NotImplementedError(
          "Decoding not supported on packed datasets "
          " If you want to decode from a dataset, use the non-packed version"
          " of the dataset when decoding.")
    if self.has_input:
      inputs = features["inputs"]
      if target_modality.is_class_modality:
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
      inputs = tf.squeeze(inputs, (2, 3))
      # _shard_features called to ensure that the variable names match
      inputs = self._shard_features({"inputs": inputs})["inputs"]

      # input_modality = self._problem_hparams.input_modality["inputs"]
      # with tf.variable_scope(input_modality.name):
      #   inputs = input_modality.bottom_sharded(inputs, dp)
      encoder_output, encoder_decoder_attention_bias = dp(
          self.encode,
          inputs,
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

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      positional_encoding = common_attention.add_positional_embedding(
          tf.zeros([1, decode_length, hparams.d_model]),
          hparams.max_length, "positional_embedding", None)

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
      targets_emb_var = self._get_targets_emb_var
      targets = tf.gather(targets_emb_var, targets)
      tf.logging.info("targets = %s" % targets)
      targets = tf.squeeze(targets, (2, 3))
      if positional_encoding is not None:
        targets += positional_encoding[:, i:i + 1]
      return targets

    def symbols_to_logits_fn(ids, i, cache):
      """Go from ids to logits for next symbol."""
      ids = ids[:, -1:]
      targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      targets = preprocess_targets(targets, i)

      bias = None  # decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      body_outputs = dp(
          self.decode,
          targets,
          cache.get("encoder_output"),
          cache.get("encoder_decoder_attention_bias"),
          bias,
          hparams,
          cache)

      logits = body_outputs[0]
      # with tf.variable_scope(target_modality.name):
      #   logits = target_modality.top_sharded(body_outputs, None, dp)[0]

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
        vocab_size=target_modality.top_dimensionality,
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


def fast_decode(encoder_output,
                encoder_decoder_attention_bias,
                symbols_to_logits_fn,
                hparams,
                decode_length,
                vocab_size,
                beam_size=1,
                top_beams=1,
                alpha=1.0,
                eos_id=beam_search.EOS_ID,
                batch_size=None,
                force_decode_length=False):
  """Given encoder output and a symbols to logits function, does fast decoding.

  Implements both greedy and beam search decoding, uses beam search iff
  beam_size > 1, otherwise beam search related arguments are ignored.

  Args:
    encoder_output: Output from encoder.
    encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
      attention
    symbols_to_logits_fn: Incremental decoding; function mapping triple
      `(ids, step, cache)` to symbol logits.
    hparams: run hyperparameters
    decode_length: an integer.  How many additional timesteps to decode.
    vocab_size: Output vocabulary size.
    beam_size: number of beams.
    top_beams: an integer. How many of the beams to return.
    alpha: Float that controls the length penalty. larger the alpha, stronger
      the preference for longer translations.
    eos_id: End-of-sequence symbol in beam search.
    batch_size: an integer scalar - must be passed if there is no input
    force_decode_length: bool, whether to force the full decode length, or if
      False, stop when all beams hit eos_id.

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

  num_layers = hparams.num_decoder_layers
  cache = {
      "layer_%d" % layer: {
          "k": tf.zeros([batch_size, hparams.num_heads,
                         0, hparams.d_kv]),
          "v": tf.zeros([batch_size, hparams.num_heads,
                         0, hparams.d_kv]),
      } for layer in range(num_layers)
  }

  if encoder_output is not None:
    for layer in range(num_layers):
      layer_name = "layer_%d" % layer
      with tf.variable_scope("decoder/%s" % layer_name):
        k_encdec, v_encdec = multihead_attention_compat(
            None,
            encoder_output,
            None,
            hparams.d_kv,
            hparams.num_heads,
            name="encdec_attention")
      cache[layer_name]["k_encdec"] = k_encdec
      cache[layer_name]["v_encdec"] = v_encdec

    cache["encoder_output"] = encoder_output
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

  if beam_size > 1:  # Beam Search
    initial_ids = tf.zeros([batch_size], dtype=tf.int32)
    decoded_ids, scores = beam_search.beam_search(
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
      temperature = (0.0 if hparams.sampling_method == "argmax" else
                     hparams.sampling_temp)
      next_id = common_layers.sample_with_temperature(logits, temperature)
      hit_eos |= tf.equal(next_id, eos_id)

      log_prob_indices = tf.stack(
          [tf.range(tf.to_int64(batch_size)), next_id], axis=1)
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
    next_id = tf.zeros([batch_size, 1], dtype=tf.int64)
    initial_log_prob = tf.zeros([batch_size], dtype=tf.float32)
    _, _, _, decoded_ids, _, log_prob = tf.while_loop(
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

  return {"outputs": decoded_ids, "scores": scores}


def transformer_prepare_encoder(
    inputs_emb_var, inputs, hparams, features=None):
  """Prepare one shard of the model for the encoder.

  Args:
    inputs_emb_var: a Tensor
    inputs: a Tensor.
    hparams: run hyperparameters
    features: optionally pass the entire features dictionary as well.
      This is needed now for "packed" datasets.

  Returns:
    encoder_input: a Tensor, bottom of encoder stack
    encoder_self_attention_bias: a bias tensor for use in encoder self-attention
    encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
      attention
  """
  encoder_input = tf.gather(inputs_emb_var, inputs)

  if features and "inputs_segmentation" in features:
    # Packed dataset.  Keep the examples from seeing each other.
    inputs_segmentation = features["inputs_segmentation"]
    inputs_position = features["inputs_position"]
    targets_segmentation = features["targets_segmentation"]
    encoder_self_attention_bias = common_attention.attention_bias_same_segment(
        inputs_segmentation, inputs_segmentation)
    encoder_decoder_attention_bias = (
        common_attention.attention_bias_same_segment(targets_segmentation,
                                                     inputs_segmentation))
  else:
    # Usual case - not a packed dataset.
    encoder_padding = tf.to_float(tf.equal(inputs, 0))
    ignore_padding = common_attention.attention_bias_ignore_padding(
        encoder_padding)
    encoder_self_attention_bias = ignore_padding
    encoder_decoder_attention_bias = ignore_padding
    inputs_position = None
  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    encoder_input = common_attention.add_positional_embedding(
        encoder_input, hparams.max_length, "positional_embedding",
        inputs_position)
  if hparams.activation_dtype == "bfloat16":
    encoder_self_attention_bias = tf.cast(encoder_self_attention_bias,
                                          tf.bfloat16)
    encoder_decoder_attention_bias = tf.cast(encoder_decoder_attention_bias,
                                             tf.bfloat16)
  return (encoder_input, encoder_self_attention_bias,
          encoder_decoder_attention_bias)


def transformer_prepare_decoder(
    targets_emb_var, targets, hparams, features=None):
  """Prepare one shard of the model for the decoder.

  Args:
    targets_emb_var: a Tensor
    targets: a Tensor.
    hparams: run hyperparameters
    features: optionally pass the entire features dictionary as well.
      This is needed now for "packed" datasets.

  Returns:
    decoder_input: a Tensor, bottom of decoder stack
    decoder_self_attention_bias: a bias tensor for use in decoder self-attention
  """
  decoder_self_attention_bias = (
      common_attention.attention_bias_lower_triangle(
          common_layers.shape_list(targets)[1]))

  if features and "targets_segmentation" in features:
    # "Packed" dataset - keep the examples from seeing each other.
    targets_segmentation = features["targets_segmentation"]
    targets_position = features["targets_position"]
    decoder_self_attention_bias += common_attention.attention_bias_same_segment(
        targets_segmentation, targets_segmentation)
  else:
    targets_position = None
  decoder_input = tf.gather(
      targets_emb_var, common_layers.shift_right_2d(targets))
  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    decoder_input = common_attention.add_positional_embedding(
        decoder_input, hparams.max_length, "positional_embedding",
        targets_position)

  if hparams.activation_dtype == "bfloat16":
    decoder_self_attention_bias = tf.cast(decoder_self_attention_bias,
                                          tf.bfloat16)
  return (decoder_input, decoder_self_attention_bias)


def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        hparams,
                        name="encoder",
                        losses=None):
  """A stack of transformer layers.

  Args:
    encoder_input: a Tensor
    encoder_self_attention_bias: bias Tensor for self-attention
       (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string
    losses: optional list onto which to append extra training losses

  Returns:
    y: a Tensors
  """
  x = encoder_input
  with tf.variable_scope(name):
    num_layer_norms = hparams.num_encoder_layers * 2 + 1
    layer_norm_combined_var = tf.get_variable(
        "layer_norm_scale", [num_layer_norms, hparams.d_model])
    layer_norm_vars = tf.unstack(layer_norm_combined_var, num_layer_norms)
    def normalize(x):
      scale = layer_norm_vars.pop(0)
      variance = tf.reduce_mean(tf.square(x), -1, keep_dims=True)
      return x * tf.rsqrt(variance + hparams.norm_epsilon) * scale
    for layer in range(hparams.num_encoder_layers):
      with tf.variable_scope("layer_%d" % layer):
        x += multihead_attention_compat(
            normalize(x),
            None,
            encoder_self_attention_bias,
            kv_channels=hparams.d_kv,
            heads=hparams.num_heads,
            name="self_attention")
        x += transformer_feedforward_layer(normalize(x), hparams, losses=losses)
    x = normalize(x)
    return x


def transformer_decoder(decoder_input,
                        encoder_output,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        hparams,
                        cache=None,
                        name="decoder",
                        losses=None):
  """A stack of transformer layers.

  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention
      (see common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    cache: dict, containing tensors which are the results of previous
        attentions, used for fast decoding.
    name: a string
    losses: optional list onto which to append extra training losses

  Returns:
    y: a Tensors
  """
  x = decoder_input
  with tf.variable_scope(name):
    num_layer_norms = (
        hparams.num_decoder_layers * (2 if encoder_output is None else 3) + 1)
    layer_norm_combined_var = tf.get_variable(
        "layer_norm_scale", [num_layer_norms, hparams.d_model])
    layer_norm_vars = tf.unstack(layer_norm_combined_var, num_layer_norms)
    def normalize(x):
      scale = layer_norm_vars.pop(0)
      variance = tf.reduce_mean(tf.square(x), -1, keep_dims=True)
      return x * tf.rsqrt(variance + hparams.norm_epsilon) * scale
    for layer in range(hparams.num_decoder_layers):
      layer_name = "layer_%d" % layer
      layer_cache = cache[layer_name] if cache is not None else None
      with tf.variable_scope(layer_name):
        x += multihead_attention_compat(
            normalize(x),
            None,
            decoder_self_attention_bias,
            kv_channels=hparams.d_kv,
            heads=hparams.num_heads,
            cache=layer_cache,
            name="self_attention")
        if encoder_output is not None:
          x += multihead_attention_compat(
              normalize(x),
              encoder_output,
              encoder_decoder_attention_bias,
              kv_channels=hparams.d_kv,
              heads=hparams.num_heads,
              cache=layer_cache,
              name="encdec_attention")
        x += transformer_feedforward_layer(normalize(x), hparams, losses=losses)
    x = normalize(x)
    return x


def transformer_feedforward_layer(x, hparams, losses=None):
  """Feed-forward layer in the transformer.

  Args:
    x: a Tensor of shape [batch_size, length, hparams.d_model]
    hparams: hyperparameters for model
    losses: an optional list

  Returns:
    a Tensor of shape [batch_size, length, hparams.d_model]

  Raises:
    ValueError: If losses arg is None, but layer generates extra losses.
  """
  del losses
  feedforward_layer = hparams.feedforward_layer
  if feedforward_layer == "dense_relu_dense":
    return dense_relu_dense_compat(x, hparams.d_ff)
  else:
    raise ValueError("Unknown hparams.feedforward_layer = %s"
                     % hparams.feedforward_layer)


def dense_relu_dense_compat(x, filter_depth, name=None):
  """Hidden layer with RELU activation followed by linear projection.

  Args:
    x: a Tensor
    filter_depth: integer
    name: an optional string

  Returns:
    a tf.Tensor
  """
  with tf.variable_scope(name, default_name="dense_relu_dense"):
    io_channels = x.shape.as_list()[-1]
    w = tf.get_variable("kernel", [2, io_channels, filter_depth])
    wi, wo = tf.unstack(w, num=2, axis=0)
    h = tf.nn.relu(tf.tensordot(x, wi, axes=[[-1], [0]]))
    return tf.tensordot(h, wo, axes=[[-1], [1]])


def multihead_attention_compat(query_antecedent,
                               memory_antecedent,
                               mask,
                               kv_channels,
                               heads,
                               cache=None,
                               name="multihead_attention"):
  """Multihead scaled-dot-product attention with input/output transformations.

  In order to use only one variable containing the four weight matrices
  packed together, we insist that the query and memory antecedents have the
  same dimensionality (io_channels) and that the keys and values have the
  same dimensionality (kv_channels).

  Args:
    query_antecedent: a Tensor with shape [batch, query_length, io_channels]
    memory_antecedent: a Tensor with shape
      [batch, memory_length, io_channels] (optional)
    mask: mask Tensor (see attention_mask())
    kv_channels: integer
    heads: integer
    cache: an optional dict
    name: an optional string.

  Returns:
    A Tensor with shape [batch, qlen, io_channels]

  Raises:
    ValueError: if the dimensions do not match.
  """
  memory_or_query_antecedent = (
      memory_antecedent if memory_antecedent is not None
      else query_antecedent)
  io_channels = memory_or_query_antecedent.shape.as_list()[-1]
  with tf.variable_scope(name,
                         default_name="multihead_attention",
                         values=[query_antecedent, memory_antecedent],
                         reuse=tf.AUTO_REUSE):
    var = tf.get_variable("qkvo", [4, heads, io_channels, kv_channels])
    q_var, k_var, v_var, o_var = tf.unstack(var, num=4, axis=0)
    if cache is None or memory_antecedent is None:
      k = tf.einsum("bmi,hik->bhmk", memory_or_query_antecedent, k_var)
      v = tf.einsum("bmi,hiv->bhmv", memory_or_query_antecedent, v_var)
      if query_antecedent is None:
        # we are computing the cache.
        return k, v
      q = tf.einsum("bqi,hik->bhqk", query_antecedent, q_var)
    if cache is not None:
      if memory_antecedent is not None:
        q = tf.einsum("bqi,hik->bhqk", query_antecedent, q_var)
        k = cache["k_encdec"]
        v = cache["v_encdec"]
      else:
        k = cache["k"] = tf.concat([cache["k"], k], axis=2)
        v = cache["v"] = tf.concat([cache["v"], v], axis=2)
    logits = tf.einsum("bhqk,bhmk->bhqm", q, k)
    if mask is not None:
      logits += mask
    weights = tf.nn.softmax(logits)
    o = tf.einsum("bhqm,bhmv->bhqv", weights, v)
    return tf.einsum("bhqv,hiv->bqi", o, o_var)
