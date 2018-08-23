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
"""Transformer model."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.mesh_tensorflow import mesh_tensorflow as mtf
from tensor2tensor.mesh_tensorflow import mtf_beam_search
from tensor2tensor.mesh_tensorflow import mtf_layers
from tensor2tensor.mesh_tensorflow import mtf_model
from tensor2tensor.mesh_tensorflow.research import moe
from tensor2tensor.utils import registry
import tensorflow as tf


@registry.register_model
class MtfTransformer(mtf_model.MtfModel):
  """Transformer in mesh_tensorflow."""

  @property
  def batch_dim(self):
    return mtf.Dimension("batch", self._hparams.batch_size)

  @property
  def inputs_vocab_dim(self):
    assert self.has_input
    return mtf.Dimension("vocab", self._inputs_vocab_size)

  @property
  def targets_vocab_dim(self):
    return mtf.Dimension("vocab", self._targets_vocab_size)

  @property
  def model_dim(self):
    return mtf.Dimension("d_model", self._hparams.d_model)

  @property
  def max_length_dim(self):
    return mtf.Dimension("max_length", self._hparams.max_length)

  @property
  def length_dim(self):
    return mtf.Dimension("length", self._hparams.max_length)

  @property
  def memory_length_dim(self):
    return mtf.Dimension("memory_length", self._hparams.max_length)

  @property
  def heads_dim(self):
    return mtf.Dimension("heads", self._hparams.num_heads)

  @property
  def kv_dim(self):
    return mtf.Dimension("d_kv", self._hparams.d_kv)

  @property
  def feedforward_dim(self):
    return mtf.Dimension("d_ff", self._hparams.d_ff)

  @property
  def activation_dtype(self):
    if self._hparams.activation_dtype == "float32":
      return tf.float32
    elif self._hparams.activation_dtype == "bfloat16":
      return tf.bfloat16
    else:
      raise ValueError(
          "unknown hparams.activation_dtype %s"
          % self._hparams.activation_dtype)

  def _import_to_batch_by_length(self, x, name, mesh, hparams):
    del hparams
    x = tf.reshape(x, [self.batch_dim.size, self.length_dim.size])
    return mtf.import_fully_replicated(
        mesh, x, mtf.Shape([self.batch_dim, self.length_dim]), name=name)

  def _embedding_and_softmax_vars(self, mesh):
    hparams = self._hparams
    targets_embedding_var = mtf.get_variable(
        mesh, "targets_embedding",
        mtf.Shape([self.targets_vocab_dim, self.model_dim]),
        initializer=tf.random_normal_initializer(),
        activation_dtype=self.activation_dtype)
    if self.has_input:
      if hparams.shared_embedding:
        inputs_embedding_var = targets_embedding_var
      else:
        inputs_embedding_var = mtf.get_variable(
            mesh, "inputs_embedding",
            mtf.Shape([self.inputs_vocab_dim, self.model_dim]),
            initializer=tf.random_normal_initializer(),
            activation_dtype=self.activation_dtype)
    else:
      inputs_embedding_var = None
    if hparams.shared_embedding_and_softmax_weights:
      softmax_var = targets_embedding_var * (self.model_dim.size ** -0.5)
    else:
      softmax_var = mtf.get_variable(
          mesh,
          "softmax",
          mtf.Shape([self.targets_vocab_dim, self.model_dim]),
          initializer=tf.random_normal_initializer(
              stddev=self.model_dim.size**-0.5),
          activation_dtype=self.activation_dtype)
    positional_embedding_var = mtf.get_variable(
        mesh, "positional_embedding",
        mtf.Shape([self.max_length_dim, self.model_dim]),
        initializer=tf.random_normal_initializer(),
        activation_dtype=self.activation_dtype)
    return (inputs_embedding_var, targets_embedding_var,
            softmax_var, positional_embedding_var)

  def _mtf_model_fn(self, features, mesh):
    features = copy.copy(features)
    hparams = self._hparams
    targets = tf.to_int32(features["targets"])
    if len(targets.get_shape()) > 2:
      tf.logging.info("targets = %s" % targets)
      targets = tf.squeeze(targets, [2, 3])
    # pad targets to max_length
    def pad_to_max_length(x):
      extra_length = hparams.max_length - tf.shape(x)[1]
      x = tf.pad(x, [[0, 0], [0, extra_length]])
      x = tf.reshape(x, [hparams.batch_size, hparams.max_length])
      return x
    targets = pad_to_max_length(targets)
    for key in ["targets_segmentation", "targets_position",
                "inputs_segmentation", "inputs_position"]:
      if key in features:
        features[key] = pad_to_max_length(features[key])
    shifted_targets = common_layers.shift_right_2d(targets)

    targets = self._import_to_batch_by_length(targets, "targets", mesh, hparams)
    shifted_targets = self._import_to_batch_by_length(
        shifted_targets, "shifted_targets", mesh, hparams)

    if "targets_segmentation" in features:
      # "Packed" dataset - keep the examples from seeing each other.
      targets_segmentation = self._import_to_batch_by_length(
          features["targets_segmentation"], "targets_segmentation",
          mesh, hparams)
      targets_position = self._import_to_batch_by_length(
          features["targets_position"], "targets_position",
          mesh, hparams)
      decoder_self_attention_mask = (
          mtf_layers.attention_mask_autoregressive(
              targets_position, dtype=self.activation_dtype) +
          mtf_layers.attention_mask_same_segment(
              targets_segmentation, dtype=self.activation_dtype))
    else:
      targets_position = mtf.range(mesh, self.length_dim, dtype=tf.int32)
      decoder_self_attention_mask = mtf_layers.attention_mask_autoregressive(
          targets_position, dtype=self.activation_dtype)

    def layer_prepostprocess_dropout(x):
      return mtf.dropout(
          x, keep_prob=1.0 - hparams.layer_prepostprocess_dropout,
          noise_shape=mtf.Shape([self.batch_dim, self.model_dim]))

    extra_losses = []
    (inputs_embedding_var,
     targets_embedding_var,
     softmax_var,
     positional_embedding_var) = self._embedding_and_softmax_vars(mesh)
    if self.has_input:
      inputs = tf.squeeze(tf.to_int32(features["inputs"]), [2, 3])
      inputs = pad_to_max_length(inputs)
      inputs = self._import_to_batch_by_length(inputs, "inputs", mesh, hparams)
      if "inputs_segmentation" in features:
        # "Packed" dataset - keep the examples from seeing each other.
        inputs_segmentation = self._import_to_batch_by_length(
            features["inputs_segmentation"], "inputs_segmentation",
            mesh, hparams)
        inputs_position = self._import_to_batch_by_length(
            features["inputs_position"], "inputs_position",
            mesh, hparams)
        encoder_self_attention_mask = (
            mtf_layers.attention_mask_same_segment(
                inputs_segmentation, dtype=self.activation_dtype))
        encoder_decoder_attention_mask = (
            mtf_layers.attention_mask_same_segment(
                targets_segmentation, inputs_segmentation,
                dtype=self.activation_dtype))
      else:
        inputs_position = mtf.range(mesh, self.length_dim, dtype=tf.int32)
        encoder_self_attention_mask = (
            mtf_layers.attention_mask_ignore_padding(
                inputs, dtype=self.activation_dtype))
        encoder_decoder_attention_mask = encoder_self_attention_mask

      x = (mtf.gather(inputs_embedding_var, inputs, self.inputs_vocab_dim) +
           mtf.gather(positional_embedding_var, inputs_position,
                      self.max_length_dim))
      x = layer_prepostprocess_dropout(x)
      with tf.variable_scope("encoder"):
        x = self._layer_stack(x,
                              hparams.num_encoder_layers,
                              self_attention_mask=encoder_self_attention_mask,
                              losses=extra_losses)
      encoder_output = mtf.rename_dimension(
          x, self.length_dim.name, self.memory_length_dim.name)
    else:
      encoder_output = None
      encoder_decoder_attention_mask = None

    # DECODER
    x = (mtf.gather(
        targets_embedding_var, shifted_targets, self.targets_vocab_dim) +
         mtf.gather(
             positional_embedding_var, targets_position, self.max_length_dim))
    x = layer_prepostprocess_dropout(x)

    # Decoder
    with tf.variable_scope("decoder"):
      x = self._layer_stack(
          x,
          hparams.num_decoder_layers,
          encoder_output=encoder_output,
          self_attention_mask=decoder_self_attention_mask,
          encdec_attention_mask=encoder_decoder_attention_mask,
          losses=extra_losses)
    logits = mtf.matmul(x, softmax_var)
    off_value = hparams.label_smoothing / self._targets_vocab_size
    on_value = 1.0 - hparams.label_smoothing + off_value
    soft_targets = mtf.one_hot(
        targets, self.targets_vocab_dim, on_value=on_value, off_value=off_value,
        dtype=self.activation_dtype)
    loss = mtf_layers.softmax_cross_entropy_with_logits(
        logits, soft_targets, self.targets_vocab_dim)
    weights = mtf_layers.weights_nonzero(
        targets, dtype=self.activation_dtype)
    loss = mtf.reduce_mean(loss * weights)
    for l in extra_losses:
      loss += l
    return logits, loss

  def mtf_model_fn(self, features, mesh):
    with tf.variable_scope("transformer"):
      return self._mtf_model_fn(features, mesh)

  @property
  def _targets_vocab_size(self):
    targets_vocab_size = self._problem_hparams.target_modality._vocab_size  # pylint: disable=protected-access
    targets_vocab_size += (-targets_vocab_size) % self._hparams.vocab_divisor
    return targets_vocab_size

  @property
  def _inputs_vocab_size(self):
    if not self.has_input:
      return None
    inputs_vocab_size = self._problem_hparams.input_modality[   # pylint: disable=protected-access
        "inputs"]._vocab_size
    inputs_vocab_size += (-inputs_vocab_size) % self._hparams.vocab_divisor
    return inputs_vocab_size

  def _feedforward_layer(self, x, losses=None):
    """Feed-forward layer.

    Args:
      x: a mtf.Tensor with shape [batch_dim, length_dim, model_dim]
      losses: a list to be appended-to
    Returns:
      a mtf.Tensor with shape [batch_dim, length_dim, model_dim]
    Raises:
      ValueError: if hparams make no sense
    """
    hparams = self._hparams
    feedforward_layer = hparams.feedforward_layer
    if feedforward_layer == "dense_relu_dense":
      return mtf_layers.dense_relu_dense(
          x, self.feedforward_dim, dropout=hparams.relu_dropout,
          dropout_broadcast_dims=[self.length_dim])
    elif feedforward_layer == "moe":
      output, loss = moe.transformer_moe_layer_v1(
          x,
          self.model_dim,
          hparams,
          hparams.mode == tf.estimator.ModeKeys.TRAIN)
      if losses is not None:
        losses.append(loss)
        return output
    else:
      raise ValueError(
          "hparams.feedforward_layer not recognized %s" % feedforward_layer)

  def _layer_stack(self,
                   x,
                   num_layers,
                   encoder_output=None,
                   self_attention_mask=None,
                   encdec_attention_mask=None,
                   losses=None):
    """Encoder or decoder stack.

    Args:
      x: a mtf.Tensor with shape [batch_dim, length_dim, model_dim]
      num_layers: an integer
      encoder_output: an optional mtf.Tensor with shape
        [batch_dim, encoder_length_dim, model_dim]
      self_attention_mask: an optional mtf.Tensor with shape
        [batch, length_dim, memory_length_dim] containing values 0 or -inf.
      encdec_attention_mask: an optional mtf.Tensor with shape
        [batch, length_dim, encoder_length_dim] containing values 0 or -inf.
      losses: a list to be appended-to
    Returns:
      a mtf.Tensor with shape [batch_dim, length_dim, model_dim]
    Raises:
      ValueError: if hparams make no sense
    """
    hparams = self._hparams

    def layer_prepostprocess_dropout(x):
      return mtf.dropout(
          x, keep_prob=1.0 - hparams.layer_prepostprocess_dropout,
          noise_shape=mtf.Shape([self.batch_dim, self.model_dim]))
    num_layer_norms = num_layers * (2 if encoder_output is None else 3) + 1
    layer_norms_dim = mtf.Dimension("layer_norms", num_layer_norms)
    layer_norm_combined_var = mtf.get_variable(
        x.mesh,
        "layer_norm_scale",
        mtf.Shape([layer_norms_dim, self.model_dim]),
        initializer=tf.ones_initializer(),
        activation_dtype=x.dtype)
    layer_norm_vars = mtf.unstack(layer_norm_combined_var, layer_norms_dim)
    def normalize(x):
      scale = layer_norm_vars.pop(0)
      variance = mtf.reduce_mean(mtf.square(x), reduced_dim=self.model_dim)
      return x * mtf.rsqrt(variance + hparams.norm_epsilon) * scale

    for layer in range(num_layers):
      with tf.variable_scope("layer_%d" % layer):
        # Self attention layer
        x += layer_prepostprocess_dropout(
            mtf_layers.multihead_attention(
                normalize(x), None,
                self_attention_mask, self.kv_dim, self.heads_dim,
                dropout=hparams.attention_dropout,
                dropout_broadcast_dims=[self.length_dim],
                name="self_attention"))
        if encoder_output is not None:
          # Encoder-Decoder attention layer
          x += layer_prepostprocess_dropout(
              mtf_layers.multihead_attention(
                  normalize(x), encoder_output,
                  encdec_attention_mask, self.kv_dim, self.heads_dim,
                  dropout=hparams.attention_dropout,
                  dropout_broadcast_dims=[self.length_dim],
                  name="encdec_attention"))
        # ffn layer
        x += layer_prepostprocess_dropout(
            self._feedforward_layer(normalize(x), losses=losses))
    x = layer_prepostprocess_dropout(normalize(x))
    assert not layer_norm_vars
    return x

  def sample(self, features, mesh):
    with tf.variable_scope("transformer"):
      return self._sample(features, mesh)

  def _sample(self, features, mesh):
    hparams = self._hparams
    (inputs_embedding_var,
     targets_embedding_var,
     softmax_var,
     positional_embedding_var) = self._embedding_and_softmax_vars(mesh)
    if self.has_input:
      inputs = features["inputs"]
      while len(inputs.shape.as_list()) > 2:
        inputs = tf.squeeze(inputs, axis=2)
      actual_batch_size = tf.shape(inputs)[0]
      actual_length = tf.shape(inputs)[1]
      inputs = tf.pad(
          inputs, [[0, hparams.batch_size - actual_batch_size],
                   [0, hparams.max_length - actual_length]])
      inputs = self._import_to_batch_by_length(
          inputs, "inputs", mesh, hparams)
      x = (mtf.gather(inputs_embedding_var, inputs, self.inputs_vocab_dim) +
           mtf.reshape(positional_embedding_var,
                       mtf.Shape([self.length_dim, self.model_dim])))
      encoder_attention_mask = (
          mtf_layers.attention_mask_ignore_padding(
              inputs, dtype=self.activation_dtype))
      with tf.variable_scope("encoder"):
        x = self._layer_stack(x,
                              hparams.num_encoder_layers,
                              self_attention_mask=encoder_attention_mask)
      encoder_output = mtf.rename_dimension(
          x, self.length_dim.name, self.memory_length_dim.name)
      encdec_tensors = []
      for layer_num in xrange(hparams.num_decoder_layers):
        with tf.variable_scope("decoder/layer_%d/encdec_attention" % layer_num):
          q_var, k_var, v_var, o_var = mtf_layers.multihead_attention_vars(
              mesh, self.heads_dim, self.model_dim,
              self.kv_dim, self.activation_dtype)
          k = mtf.einsum(
              [encoder_output, k_var],
              mtf.Shape(
                  [self.batch_dim, self.heads_dim,
                   self.memory_length_dim, self.kv_dim]))
          v = mtf.einsum(
              [encoder_output, v_var],
              mtf.Shape(
                  [self.batch_dim, self.heads_dim,
                   self.memory_length_dim, self.kv_dim]))
        encdec_tensors.append((q_var, o_var, k, v))
      partial_targets = None
    else:
      encdec_tensors = None
      encoder_output = None
      encoder_attention_mask = None
      # Prepare partial targets.
      # In either features["inputs"] or features["targets"].
      # We force the outputs to begin with these sequences.
      partial_targets = features.get("inputs", None)
      if partial_targets is None:
        partial_targets = features.get("targets", None)
      if partial_targets is not None:
        partial_targets = common_layers.expand_squeeze_to_nd(partial_targets, 2)
        partial_targets = tf.to_int32(partial_targets)
        partial_targets_batch = tf.shape(partial_targets)[0]
        partial_targets_length = tf.shape(partial_targets)[1]
        partial_targets = tf.pad(
            partial_targets, [[0, hparams.batch_size - partial_targets_batch],
                              [0, hparams.max_length - partial_targets_length]])
        partial_targets = self._import_to_batch_by_length(
            partial_targets, "partial_targets", mesh, hparams)

    if hparams.beam_size == 1:
      ids_shape = mtf.Shape([self.batch_dim, self.length_dim])
      kv_shape = mtf.Shape([self.batch_dim, self.heads_dim,
                            self.memory_length_dim, self.kv_dim])
    else:
      beam_dim = mtf.Dimension("beam", hparams.beam_size)
      ids_shape = mtf.Shape([self.batch_dim, beam_dim, self.length_dim])
      kv_shape = mtf.Shape([self.batch_dim, beam_dim, self.heads_dim,
                            self.memory_length_dim, self.kv_dim])

    initial_ids = mtf.constant(mesh, 0, ids_shape, dtype=tf.int32)
    initial_kv_states = (
        [mtf.zeros(mesh, kv_shape, dtype=self.activation_dtype)]
        * (2 * hparams.num_decoder_layers))
    def logits_fn(step_num, ids, states):
      """Produce logits for this step, and new states."""
      self_attention_k = states[:hparams.num_decoder_layers]
      self_attention_v = states[hparams.num_decoder_layers:]
      ids_this_step = mtf.gather(ids, step_num - 1, self.length_dim)
      x = (mtf.gather(targets_embedding_var, ids_this_step,
                      self.targets_vocab_dim) +
           mtf.gather(positional_embedding_var, step_num, self.max_length_dim))
      with tf.variable_scope("decoder"):
        x, new_self_attention_k, new_self_attention_v = (
            self._decoder_layer_stack_incremental(
                x,
                step_num,
                encdec_tensors,
                self_attention_k,
                self_attention_v,
                encdec_attention_mask=encoder_attention_mask))
      logits = mtf.matmul(x, softmax_var)
      return logits, new_self_attention_k + new_self_attention_v

    if hparams.beam_size == 1:
      temperature = (0.0 if hparams.sampling_method == "argmax"
                     else hparams.sampling_temp)
      return mtf_beam_search.greedy_decode(
          logits_fn,
          initial_ids,
          temperature=temperature,
          initial_states=initial_kv_states,
          forced_ids=partial_targets,
          use_tpu=hparams.use_tpu)
    else:
      if self.has_input:
        input_length = mtf.reduce_sum(
            mtf.to_float(mtf.cast(inputs, tf.bool)),
            reduced_dim=self.length_dim)
        max_input_length = mtf.reduce_max(input_length)
        decode_length = mtf.cast(
            max_input_length * hparams.decode_length_multiplier
            + hparams.decode_length_constant, tf.int32)
      else:
        decode_length = None
      beams, unused_scores = mtf_beam_search.beam_search(
          logits_fn,
          initial_ids,
          hparams.alpha,
          states=initial_kv_states,
          decode_length=decode_length,
          use_tpu=hparams.use_tpu)
      return mtf.gather(beams, mtf.constant(mesh, 0, dtype=tf.int32), beam_dim)

  def _decoder_layer_stack_incremental(self,
                                       x,
                                       step_num,
                                       encdec_tensors,
                                       self_attention_k,
                                       self_attention_v,
                                       encdec_attention_mask=None):
    """Decoder layer stack during inference.

    We are processing only one position at a time.

    The self-attention keys and values have already been computed for
    previous positions.  In addition to the decoder output, we need to
    produce the updated self-attention keys and values.

    If there is an encoder, then additional Tensors are supplied in
    encdec_tensors, which give us the keys and values for encoder-decoder
    attention as well as the weight matrices q_var and o_var.

    Args:
      x: a mtf.Tensor with shape [batch_dim, model_dim]
      step_num: an mtf integer Scalar
      encdec_tensors: an optional list of num_layers tuples, each of the form
        (q_var, o_var, k, v)
      self_attention_k: an optional list of num_layers Tensors each with shape
        [batch, heads, memory_length, kv_channels]
      self_attention_v: an optional list of num_layers Tensors each with shape
        [batch, heads, memory_length, kv_channels]
      encdec_attention_mask: an optional mtf.Tensor with shape
        [batch, length_dim, encoder_length_dim] containing values 0 or -inf.

    Returns:
      y: a mtf.Tensor with shape [batch_dim, model_dim]
      new_self_attention_k: a list of num_layers mtf.Tensors, with the same
        shapes as the elements of self_attention_k
      new_self_attention_v: a list of num_layers mtf.Tensors, with the same
        shapes as the elements of self_attention_v

    Raises:
      ValueError: if hparams make no sense
    """
    hparams = self._hparams
    num_layers = hparams.num_decoder_layers
    num_layer_norms = num_layers * (2 if encdec_tensors is None else 3) + 1
    layer_norms_dim = mtf.Dimension("layer_norms", num_layer_norms)
    layer_norm_combined_var = mtf.get_variable(
        x.mesh,
        "layer_norm_scale",
        mtf.Shape([layer_norms_dim, self.model_dim]),
        initializer=tf.ones_initializer(),
        activation_dtype=x.dtype)
    layer_norm_vars = mtf.unstack(layer_norm_combined_var, layer_norms_dim)
    def normalize(x):
      scale = layer_norm_vars.pop(0)
      variance = mtf.reduce_mean(mtf.square(x), reduced_dim=self.model_dim)
      return x * mtf.rsqrt(variance + hparams.norm_epsilon) * scale

    new_self_attention_k = []
    new_self_attention_v = []
    for layer in range(num_layers):
      with tf.variable_scope("layer_%d" % layer):
        # Self attention layer
        y, new_k, new_v = mtf_layers.multihead_self_attention_incremental(
            normalize(x),
            prev_k=self_attention_k[layer],
            prev_v=self_attention_v[layer],
            step_num=step_num,
            name="self_attention")
        new_self_attention_k.append(new_k)
        new_self_attention_v.append(new_v)
        x += y
        if encdec_tensors is not None:
          # Encoder-Decoder attention layer
          q_var, o_var, k, v = encdec_tensors[layer]
          x += mtf_layers.multihead_encdec_attention_incremental(
              normalize(x),
              q_var, o_var, k, v,
              encdec_attention_mask,
              name="encdec_attention")
        # ffn layer
        x += self._feedforward_layer(normalize(x), hparams)
    x = normalize(x)
    assert not layer_norm_vars
    return x, new_self_attention_k, new_self_attention_v


@registry.register_hparams
def mtf_transformer_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.no_data_parallelism = True
  hparams.use_fixed_batch_size = True
  hparams.add_hparam("mtf_mode", True)
  hparams.batch_size = 64
  hparams.max_length = 256
  hparams.add_hparam("d_model", 512)
  hparams.add_hparam("d_kv", 128)
  hparams.label_smoothing = 0.1
  # 8-way model-parallelism
  hparams.add_hparam("mesh_shape", "model:8")
  hparams.add_hparam("layout", "batch:batch;vocab:model;d_ff:model;heads:model")
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("d_ff", 2048)
  hparams.add_hparam("num_encoder_layers", 6)
  hparams.add_hparam("num_decoder_layers", 6)
  hparams.add_hparam("attention_dropout", 0.1)
  hparams.add_hparam("relu_dropout", 0.1)
  hparams.layer_prepostprocess_dropout = 0.1

  # round up vocab sizes to be a multiple of this value
  hparams.vocab_divisor = 128

  hparams.add_hparam("feedforward_layer", "dense_relu_dense")

  # Use targets_embedding_var * rsqrt(d_model) as softmax_var
  hparams.shared_embedding_and_softmax_weights = True
  # Reuse targets_embedding_var as inputs_embedding_var
  hparams.shared_embedding = True
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_schedule = "linear_warmup*rsqrt_decay*linear_decay"
  hparams.learning_rate_warmup_steps = 10000
  hparams.activation_dtype = "float32"

  # These parameters make Transformer model compatible with MtfTransformer
  # Do not override these, as mtf_transformer does not support other options.
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.target_modality = "symbol:identity"
  hparams.input_modalities = "inputs:symbol:identity"

  # Parameters for computing the maximum decode length in beam search.
  # Maximum decode length is:
  #    min(max_length,
  #        decode_length_multiplier * input_length + decode_length_constant)
  hparams.add_hparam("decode_length_multiplier", 1.5)
  hparams.add_hparam("decode_length_constant", 10.0)

  return hparams


@registry.register_hparams
def mtf_transformer_tiny():
  """Catch bugs locally..."""
  hparams = mtf_transformer_base()
  hparams.d_model = 128
  hparams.d_ff = 512
  hparams.batch_size = 4
  hparams.num_encoder_layers = 2
  hparams.num_decoder_layers = 2
  hparams.num_heads = 4
  # data parallelism and model-parallelism
  hparams.mesh_shape = "batch:2;model:2"
  return hparams


@registry.register_hparams
def mtf_transformer_single():
  hparams = mtf_transformer_tiny()
  hparams.mesh_shape = ""
  return hparams


@registry.register_hparams
def mtf_transformer_tiny_8gpu():
  hparams = mtf_transformer_tiny()
  hparams.mesh_shape = "model:8"
  return hparams


def mtf_transformer_paper_lm(size):
  """Config for language-model experiments.

  Train these on languagemodel_lm1b32k_packed for 136000 steps (10 epochs)

  The size parameter is an integer that controls the number of heads and the
  size of the size of the feedforward hidden layers.  Increasing size by 1
  doubles each of these.

  Results:
  size   params/10^9  log-ppl(per-token)
  -1     0.14         3.209
  0      0.22         3.119
  1      0.37         3.037
  2      0.67         2.969
  3      1.28         2.912
  4      2.48         2.874
  5      4.90         2.871

  (to get word-level log-ppl, multiply by 1.1078)

  Args:
    size: an integer
  Returns:
    a hparams object
  """
  n = 2 ** size
  hparams = mtf_transformer_base()
  hparams.label_smoothing = 0.0
  hparams.batch_size = 256
  hparams.d_model = 1024
  hparams.d_ff = int(8192 * n)
  hparams.d_kv = 256
  hparams.num_heads = int(8 * n)
  hparams.shared_embedding_and_softmax_weights = False
  # one epoch for languagemodel_lm1b32k_packed = 13600 steps
  hparams.learning_rate_decay_steps = 13600
  return hparams


@registry.register_hparams
def mtf_transformer_paper_lm_m1():
  hparams = mtf_transformer_paper_lm(-1)
  hparams.mesh_shape = "batch:32"
  return hparams


@registry.register_hparams
def mtf_transformer_paper_lm_0():
  hparams = mtf_transformer_paper_lm(0)
  hparams.mesh_shape = "batch:32"
  return hparams


@registry.register_hparams
def mtf_transformer_paper_lm_1():
  hparams = mtf_transformer_paper_lm(1)
  hparams.mesh_shape = "model:4;batch:8"
  return hparams


@registry.register_hparams
def mtf_transformer_paper_lm_2():
  hparams = mtf_transformer_paper_lm(2)
  hparams.mesh_shape = "model:4;batch:8"
  return hparams


@registry.register_hparams
def mtf_transformer_paper_lm_3():
  hparams = mtf_transformer_paper_lm(3)
  hparams.mesh_shape = "model:8;batch:16"
  return hparams


@registry.register_hparams
def mtf_transformer_paper_lm_4():
  hparams = mtf_transformer_paper_lm(4)
  hparams.mesh_shape = "batch:16;model:32"
  return hparams


@registry.register_hparams
def mtf_transformer_paper_lm_5():
  hparams = mtf_transformer_paper_lm(5)
  hparams.mesh_shape = "batch:16;model:32"
  return hparams


def mtf_transformer_paper_tr(size):
  """Config for translation experiments.

  Train these on translate_enfr_wmt32k_packed for 154000 steps (3 epochs)

  The size parameter is an integer that controls the number of heads and the
  size of the size of the feedforward hidden layers.  Increasing size by 1
  doubles each of these.

  Args:
    size: an integer
  Returns:
    a hparams object
  """
  n = 2 ** size
  hparams = mtf_transformer_base()
  hparams.label_smoothing = 0.1
  hparams.batch_size = 128
  hparams.d_model = 1024
  hparams.d_ff = int(4096 * n)
  hparams.num_heads = int(8 * n)
  hparams.shared_embedding_and_softmax_weights = False
  # one epoch for translate_enfr_wmt32k_packed = 51400 steps
  hparams.learning_rate_decay_steps = 51400
  return hparams


@registry.register_hparams
def mtf_transformer_paper_tr_m1():
  hparams = mtf_transformer_paper_tr(-1)
  hparams.mesh_shape = "batch:32"
  return hparams


@registry.register_hparams
def mtf_transformer_paper_tr_0():
  hparams = mtf_transformer_paper_tr(0)
  hparams.mesh_shape = "batch:32"
  return hparams


@registry.register_hparams
def mtf_transformer_paper_tr_1():
  hparams = mtf_transformer_paper_tr(1)
  hparams.mesh_shape = "model:4;batch:8"
  return hparams


@registry.register_hparams
def mtf_transformer_paper_tr_2():
  hparams = mtf_transformer_paper_tr(2)
  hparams.mesh_shape = "model:4;batch:8"
  return hparams


@registry.register_hparams
def mtf_transformer_paper_tr_3():
  hparams = mtf_transformer_paper_tr(3)
  hparams.mesh_shape = "model:8;batch:16"
  return hparams


@registry.register_hparams
def mtf_transformer_paper_tr_4():
  hparams = mtf_transformer_paper_tr(4)
  hparams.mesh_shape = "model:8;batch:16"
  return hparams


@registry.register_hparams
def mtf_transformer_paper_tr_0_mesh_8():
  hparams = mtf_transformer_paper_tr(0)
  hparams.mesh_shape = "batch:8"
  return hparams


@registry.register_hparams
def mtf_transformer_paper_tr_4_mesh_16_8():
  hparams = mtf_transformer_paper_tr(4)
  hparams.mesh_shape = "batch:8;model:16"
  return hparams


@registry.register_hparams
def mtf_transformer_paper_tr_6_mesh_64_8():
  # Note: This mesh shape does align well with physical [16, 16, 2] topology.
  hparams = mtf_transformer_paper_tr(6)
  hparams.mesh_shape = "model:64;batch:8"
  return hparams


@registry.register_hparams
def mtf_transformer_paper_tr_0_mesh_8_v2():
  hparams = mtf_transformer_paper_tr(0)
  hparams.batch_size = int(hparams.batch_size / 4)
  hparams.mesh_shape = "batch:8"
  return hparams


@registry.register_hparams
def mtf_transformer_paper_tr_0_mesh_128():
  hparams = mtf_transformer_paper_tr(0)
  hparams.batch_size = int(hparams.batch_size * 4)
  hparams.mesh_shape = "batch:128"
  return hparams


@registry.register_hparams
def mtf_transformer_paper_tr_0_mesh_512():
  hparams = mtf_transformer_paper_tr(0)
  hparams.batch_size = int(hparams.batch_size * 16)
  hparams.mesh_shape = "batch:512"
  return hparams


@registry.register_hparams
def mtf_transformer_lm_baseline():
  """Small language model to run on 1 TPU.

  Run this on 2x2 on languagemodel_lm1b32k_packed for 272000 steps (10 epochs)
  Results:
         params/10^9  log-ppl(per-token)
         0.14         3.202

  Returns:
    a hparams
  """
  hparams = mtf_transformer_paper_lm(-1)
  hparams.batch_size = 128
  hparams.learning_rate_decay_steps = 27200  # one epoch on lm1b
  hparams.mesh_shape = "batch:8"
  return hparams


