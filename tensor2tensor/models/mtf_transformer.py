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

"""Transformer model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import mesh_tensorflow as mtf
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
from tensor2tensor.models.research import moe
from tensor2tensor.utils import mtf_model
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_model
class MtfTransformer(mtf_model.MtfModel):
  """Transformer in mesh_tensorflow."""

  def __init__(self,
               hparams,
               mode=tf.estimator.ModeKeys.TRAIN,
               problem_hparams=None,
               data_parallelism=None,
               decode_hparams=None,
               **kwargs):
    """Init with assignments of hparams.encoder_layers / decoder_layers."""
    # Finalize encoder_layers, decoder_layers
    hparams.encoder_layers = (
        hparams.encoder_layers * hparams.encoder_replicate_factor)
    hparams.decoder_layers = (
        hparams.decoder_layers * hparams.decoder_replicate_factor)

    super(MtfTransformer, self).__init__(hparams,
                                         mode=mode,
                                         problem_hparams=problem_hparams,
                                         data_parallelism=data_parallelism,
                                         decode_hparams=decode_hparams,
                                         **kwargs)

  @property
  def batch_dims(self):
    hparams = self._hparams
    if hparams.outer_batch_size == 0:
      return [mtf.Dimension("batch", hparams.batch_size)]
    else:
      if hparams.batch_size % hparams.outer_batch_size != 0:
        raise ValueError(
            "hparams.outer_batch_size must divide hparams.batch_size")
      return [
          mtf.Dimension("outer_batch", hparams.outer_batch_size),
          mtf.Dimension("inner_batch",
                        hparams.batch_size // hparams.outer_batch_size)]

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
  def master_dtype(self):
    return tf.as_dtype(self._hparams.master_dtype)

  @property
  def slice_dtype(self):
    return tf.as_dtype(self._hparams.slice_dtype)

  @property
  def activation_dtype(self):
    return tf.as_dtype(self._hparams.activation_dtype)

  def _import_to_batch_by_length(self, x, name, mesh, hparams):
    del hparams
    mtf_shape = mtf.Shape(self.batch_dims + [self.length_dim])
    x = tf.reshape(x, mtf_shape.to_integer_list)
    return mtf.import_fully_replicated(mesh, x, mtf_shape, name=name)

  def _embedding_and_softmax_vars(self, mesh):
    hparams = self._hparams
    if hparams.transformer_type == "encoder":
      targets_embedding_var = None
    else:
      targets_embedding_var = mtf.get_variable(
          mesh, "targets_embedding",
          mtf.Shape([self.targets_vocab_dim, self.model_dim]),
          initializer=tf.random_normal_initializer(),
          master_dtype=self.master_dtype,
          slice_dtype=self.slice_dtype,
          activation_dtype=self.activation_dtype)
    if hparams.transformer_type == "decoder":
      inputs_embedding_var = None
    else:
      if hparams.shared_embedding and targets_embedding_var:
        inputs_embedding_var = targets_embedding_var
      else:
        inputs_embedding_var = mtf.get_variable(
            mesh, "inputs_embedding",
            mtf.Shape([self.inputs_vocab_dim, self.model_dim]),
            initializer=tf.random_normal_initializer(),
            master_dtype=self.master_dtype,
            slice_dtype=self.slice_dtype,
            activation_dtype=self.activation_dtype)
    if hparams.shared_embedding_and_softmax_weights:
      softmax_var = (targets_embedding_var or inputs_embedding_var) * (
          self.model_dim.size ** -0.5)
    else:
      softmax_var = mtf.get_variable(
          mesh,
          "softmax",
          mtf.Shape([self.targets_vocab_dim, self.model_dim]),
          initializer=tf.random_normal_initializer(
              stddev=self.model_dim.size**-0.5),
          master_dtype=self.master_dtype,
          slice_dtype=self.slice_dtype,
          activation_dtype=self.activation_dtype)
    positional_embedding_var = mtf.get_variable(
        mesh, "positional_embedding",
        mtf.Shape([self.max_length_dim, self.model_dim]),
        initializer=tf.random_normal_initializer(),
        activation_dtype=self.activation_dtype)
    return (inputs_embedding_var, targets_embedding_var,
            softmax_var, positional_embedding_var)

  def _noisy_targets_from_spec(self, targets, noising_spec, losses=None):
    if noising_spec["type"] == "mask":
      # Replace a randomly-chosen noising_spec["prob"] of input tokens with 0.
      return targets * mtf.cast(
          mtf.greater(mtf.random_uniform(targets.mesh, targets.shape),
                      noising_spec["prob"]), targets.dtype)
    elif noising_spec["type"] == "random_zipfian":
      # Replace a randomly-chosen noising_spec["prob"] of input tokens.
      # Rather than drawing the replacement tokens uniformly, we sample from
      #   a distribution favoring lower token-ids, assuming that the ids have
      #   been assigned in frequency order.  The probability of choosing an
      #   id is proportional to 1/(id+10)
      logits = mtf.log(1.0 / (mtf.range(
          targets.mesh, self.targets_vocab_dim, dtype=tf.float32) + 10.0))
      logits = mtf.broadcast(logits, new_shape=targets.shape + logits.shape)
      r = mtf.sample_with_temperature(logits, self.targets_vocab_dim)
      use_noise = mtf.less(
          mtf.random_uniform(targets.mesh, targets.shape), noising_spec["prob"])
      return mtf.where(use_noise, r, targets)
    elif noising_spec["type"] == "transformer":
      # Train a small transformer to fill in masked out values, then
      # sample from it.
      hparams = self._hparams
      if hparams.mode != tf.estimator.ModeKeys.TRAIN:
        raise NotImplementedError("Not implemented")
      noiser_hparams = copy.copy(self._hparams)
      noiser_hparams.del_hparam("mode")
      noiser_hparams.override_from_dict(noising_spec["overrides"])
      with tf.variable_scope("noiser"):
        noiser = MtfTransformer(
            noiser_hparams,
            mode=hparams.mode,
            problem_hparams=self._problem_hparams)
        logits, loss = noiser._mtf_model_fn(  # pylint: disable=protected-access
            self._original_features, targets.mesh)
        samples = mtf.sample_with_temperature(logits, self.targets_vocab_dim)
      losses.append(loss)
      return samples
    else:
      raise ValueError("unknown noising spec %s" % noising_spec)

  def _noisy_targets(self, targets, losses=None):
    """Generate noisy targets for denoising models.

    Args:
      targets: a Tensor
      losses: an optional list onto which to append traning losses
    Returns:
      a Tensor the same dtype and shape as Targets
    """
    hparams = self._hparams
    if hparams.mode == tf.estimator.ModeKeys.TRAIN:
      nt_train = self._noisy_targets_from_spec(
          targets, hparams.noising_spec_train, losses=losses)
      if hparams.noising_use_eval_during_train > 0:
        nt_eval = self._noisy_targets_from_spec(
            targets, hparams.noising_spec_eval)
        use_eval_noising = mtf.less(
            mtf.random_uniform(targets.mesh, targets.shape - self.length_dim),
            hparams.noising_use_eval_during_train)
        nt_train = mtf.where(use_eval_noising, nt_eval, nt_train)
      return nt_train
    else:
      return self._noisy_targets_from_spec(targets, hparams.noising_spec_eval)

  def _mtf_model_fn(self, features, mesh):
    self._original_features = features
    features = copy.copy(features)
    hparams = self._hparams
    extra_losses = []
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
    targets = self._import_to_batch_by_length(targets, "targets", mesh, hparams)
    for key in ["targets_segmentation", "targets_position",
                "inputs_segmentation", "inputs_position"]:
      if key in features:
        features[key] = pad_to_max_length(features[key])
    if hparams.decoder_type == "autoregressive":
      shifted_targets = mtf.shift(
          targets, offset=1, dim=self.length_dim, wrap=False)
    elif hparams.decoder_type == "denoising":
      shifted_targets = self._noisy_targets(targets, extra_losses)
    else:
      raise ValueError(
          "unknown hparams.decoder_type = %s" % hparams.decoder_type)

    if "targets_segmentation" in features:
      # "Packed" dataset - keep the examples from seeing each other.
      targets_segmentation = self._import_to_batch_by_length(
          features["targets_segmentation"], "targets_segmentation",
          mesh, hparams)
      targets_position = self._import_to_batch_by_length(
          features["targets_position"], "targets_position",
          mesh, hparams)
      decoder_self_attention_mask = mtf.layers.attention_mask_same_segment(
          targets_segmentation, dtype=self.activation_dtype)
      if hparams.decoder_type == "autoregressive":
        decoder_self_attention_mask += mtf.layers.attention_mask_autoregressive(
            targets_position, dtype=self.activation_dtype)
    else:
      targets_position = mtf.range(mesh, self.length_dim, dtype=tf.int32)
      if hparams.decoder_type == "autoregressive":
        decoder_self_attention_mask = mtf.layers.attention_mask_autoregressive(
            targets_position, dtype=self.activation_dtype)
      else:
        decoder_self_attention_mask = None

    def layer_prepostprocess_dropout(x):
      return mtf.dropout(
          x, keep_prob=1.0 - hparams.layer_prepostprocess_dropout,
          noise_shape=mtf.Shape(self.batch_dims + [self.model_dim]))

    (inputs_embedding_var,
     targets_embedding_var,
     softmax_var,
     positional_embedding_var) = self._embedding_and_softmax_vars(mesh)
    if hparams.transformer_type == "decoder":
      encoder_output = None
      encoder_decoder_attention_mask = None
    else:
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
            mtf.layers.attention_mask_same_segment(
                inputs_segmentation, dtype=self.activation_dtype))
      else:
        inputs_position = mtf.range(mesh, self.length_dim, dtype=tf.int32)
        encoder_self_attention_mask = (
            mtf.layers.attention_mask_ignore_padding(
                inputs, dtype=self.activation_dtype))

      x = (mtf.gather(inputs_embedding_var, inputs, self.inputs_vocab_dim) +
           mtf.gather(positional_embedding_var, inputs_position,
                      self.max_length_dim))
      x = layer_prepostprocess_dropout(x)
      with tf.variable_scope("encoder"):
        x = self._layer_stack(x,
                              hparams.encoder_layers,
                              self_attention_mask=encoder_self_attention_mask,
                              losses=extra_losses)

    if hparams.transformer_type == "encdec":
      if "inputs_segmentation" in features:
        encoder_decoder_attention_mask = (
            mtf.layers.attention_mask_same_segment(
                targets_segmentation, inputs_segmentation,
                dtype=self.activation_dtype))
      else:
        encoder_decoder_attention_mask = encoder_self_attention_mask
      encoder_output = mtf.rename_dimension(
          x, self.length_dim.name, self.memory_length_dim.name)

    if hparams.transformer_type != "encoder":
      # DECODER
      x = (mtf.gather(
          targets_embedding_var, shifted_targets, self.targets_vocab_dim) +
           mtf.gather(
               positional_embedding_var, targets_position, self.max_length_dim))
      x = layer_prepostprocess_dropout(x)
      with tf.variable_scope("decoder"):
        x = self._layer_stack(
            x,
            hparams.decoder_layers,
            encoder_output=encoder_output,
            self_attention_mask=decoder_self_attention_mask,
            encdec_attention_mask=encoder_decoder_attention_mask,
            losses=extra_losses)
    if (hparams.reshape_logits_hack and
        hparams.mode == tf.estimator.ModeKeys.TRAIN):
      # For some reason, the logits computation is extremely slow on TPU
      # in some cases where the batch size per core is 1.  Reshape the logits
      # and the targets to double the batch size and halve the length.
      # TODO(noam): file a bug.
      old_dims = self.batch_dims + [self.length_dim]
      new_dims = self.batch_dims[:-1] + [
          mtf.Dimension(self.batch_dims[-1].name,
                        self.batch_dims[-1].size * 2),
          mtf.Dimension(self.length_dim.name, self.length_dim.size // 2)]
      x = mtf.reshape(x, new_dims + [self.model_dim])
      targets = mtf.reshape(targets, new_dims)

    logits = mtf.matmul(x, softmax_var)
    if hparams.mode == tf.estimator.ModeKeys.TRAIN:
      logits = mtf.layers.multiplicative_jitter(logits, epsilon=1e-2)
    off_value = hparams.label_smoothing / self._targets_vocab_size
    on_value = 1.0 - hparams.label_smoothing + off_value
    soft_targets = mtf.one_hot(
        targets, self.targets_vocab_dim, on_value=on_value, off_value=off_value,
        dtype=self.activation_dtype)
    loss = mtf.layers.softmax_cross_entropy_with_logits(
        logits, soft_targets, self.targets_vocab_dim)
    weights = mtf.layers.weights_nonzero(targets, dtype=self.activation_dtype)
    loss = mtf.reduce_mean(loss * weights)
    for l in extra_losses:
      loss += l
    if (hparams.reshape_logits_hack and
        hparams.mode == tf.estimator.ModeKeys.TRAIN):
      logits = mtf.reshape(logits, old_dims + [self.targets_vocab_dim])
    logits = mtf.to_float(logits)
    return logits, loss

  def mtf_model_fn(self, features, mesh):
    with tf.variable_scope("transformer"):
      logits, loss = self._mtf_model_fn(features, mesh)
      # combine batch dims
      if len(self.batch_dims) > 1:
        combined_batch_dim = mtf.Dimension(
            self.batch_dims[0].name, mtf.Shape(self.batch_dims).size)
        logits = mtf.reshape(
            logits, [combined_batch_dim] + logits.shape.dims[-2:])
      return logits, loss

  @property
  def _targets_vocab_size(self):
    targets_vocab_size = self._problem_hparams.vocab_size["targets"]
    targets_vocab_size += (-targets_vocab_size) % self._hparams.vocab_divisor
    return targets_vocab_size

  @property
  def _inputs_vocab_size(self):
    inputs_vocab_size = self._problem_hparams.vocab_size["inputs"]
    inputs_vocab_size += (-inputs_vocab_size) % self._hparams.vocab_divisor
    return inputs_vocab_size

  def _feedforward_layer(self, x, layer_type, losses=None):
    """Feed-forward layer.

    Args:
      x: a mtf.Tensor with shape [<batch_dims>, length_dim, model_dim]
      layer_type: a string
      losses: a list to be appended-to
    Returns:
      a mtf.Tensor with shape [<batch_dims>, length_dim, model_dim]
    Raises:
      ValueError: if hparams make no sense
    """
    hparams = self._hparams

    if layer_type == "drd":
      return mtf.layers.dense_relu_dense(
          x, self.feedforward_dim, dropout=hparams.relu_dropout,
          dropout_broadcast_dims=[self.length_dim],
          master_dtype=self.master_dtype,
          slice_dtype=self.slice_dtype)
    elif layer_type == "none":
      return x
    elif layer_type == "moe":
      output, loss = moe.transformer_moe_layer_v1(
          x,
          self.model_dim,
          hparams,
          hparams.mode == tf.estimator.ModeKeys.TRAIN,
          master_dtype=self.master_dtype,
          slice_dtype=self.slice_dtype)
      if losses is not None:
        losses.append(loss)
      return output
    elif layer_type == "hmoe":
      output, loss = moe.transformer_moe_layer_v2(
          x,
          self.model_dim,
          hparams,
          hparams.mode == tf.estimator.ModeKeys.TRAIN,
          master_dtype=self.master_dtype,
          slice_dtype=self.slice_dtype)
      if losses is not None:
        losses.append(loss)
      return output
    else:
      raise ValueError("layer_type not recognized %s" % layer_type)

  def _layer_stack(self,
                   x,
                   layers,
                   encoder_output=None,
                   self_attention_mask=None,
                   encdec_attention_mask=None,
                   losses=None,
                   step_num=None,
                   encdec_tensors=None,
                   states=None):
    """Encoder or decoder stack.

    Args:
      x: a mtf.Tensor with shape [<batch_dims>, length_dim, model_dim]
      layers: an list of strings
      encoder_output: an optional mtf.Tensor with shape
        [<batch_dims>, encoder_length_dim, model_dim]
      self_attention_mask: an optional mtf.Tensor with shape
        [batch, length_dim, memory_length_dim] containing values 0 or -inf.
      encdec_attention_mask: an optional mtf.Tensor with shape
        [batch, length_dim, encoder_length_dim] containing values 0 or -inf.
      losses: a list to be appended-to
      step_num: an optional mtf integer Scalar (used in incrmenental mode)
      encdec_tensors: an optional list of num_layers tuples, each of the form
        (q_var, o_var, k, v), (used in incremental mode)
      states: an optional list of Tensors (used in incremental mode)
    Returns:
      a mtf.Tensor with shape [<batch_dims>, length_dim, model_dim]
    Raises:
      ValueError: if hparams make no sense
    """
    hparams = self._hparams
    is_incremental = (step_num is not None)
    def layer_prepostprocess_dropout(x):
      if is_incremental:
        return x
      return mtf.dropout(
          x, keep_prob=1.0 - hparams.layer_prepostprocess_dropout,
          noise_shape=mtf.Shape(self.batch_dims + [self.model_dim]))
    num_layers = len(layers)
    num_layer_norms = num_layers + 1
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

    if is_incremental:
      states = list(states)
      new_states = []
    tf.logging.info("states = %s" % (states,))

    for lnum, layer_type in enumerate(layers):
      with tf.variable_scope("%s_%d" % (layer_type, lnum)):
        if layer_type == "att":
          # Self attention layer
          if is_incremental:
            y, new_k, new_v = mtf.layers.multihead_self_attention_incremental(
                normalize(x),
                prev_k=states.pop(0),
                prev_v=states.pop(0),
                step_num=step_num,
                master_dtype=self.master_dtype,
                slice_dtype=self.slice_dtype,
                name="att")
            new_states.append(new_k)
            new_states.append(new_v)
            x += y
          else:
            x += layer_prepostprocess_dropout(
                mtf.layers.multihead_attention(
                    normalize(x), None,
                    self_attention_mask, self.kv_dim, self.heads_dim,
                    dropout=hparams.attention_dropout,
                    dropout_broadcast_dims=[self.length_dim],
                    master_dtype=self.master_dtype,
                    slice_dtype=self.slice_dtype,
                    name="att"))
        elif layer_type == "enc_att":
          # Encoder-Decoder attention layer
          if is_incremental:
            # Encoder-Decoder attention layer
            q_var, o_var, k, v = encdec_tensors[lnum]
            x += mtf.layers.multihead_encdec_attention_incremental(
                normalize(x),
                q_var, o_var, k, v,
                encdec_attention_mask,
                name="enc_att")
          else:
            x += layer_prepostprocess_dropout(
                mtf.layers.multihead_attention(
                    normalize(x), encoder_output,
                    encdec_attention_mask, self.kv_dim, self.heads_dim,
                    dropout=hparams.attention_dropout,
                    dropout_broadcast_dims=[self.length_dim],
                    master_dtype=self.master_dtype,
                    slice_dtype=self.slice_dtype,
                    name="enc_att"))
        elif layer_type == "local_att":
          if is_incremental:
            y, new_k, new_v = mtf.layers.masked_local_attention_1d_incremental(
                normalize(x),
                prev_k=states.pop(0),
                prev_v=states.pop(0),
                step_num=step_num,
                master_dtype=self.master_dtype,
                slice_dtype=self.slice_dtype,
                name="local_att")
            new_states.append(new_k)
            new_states.append(new_v)
            x += y
          else:
            x += layer_prepostprocess_dropout(
                mtf.layers.masked_local_attention_1d(
                    normalize(x),
                    self.kv_dim, self.heads_dim,
                    window_size=hparams.local_attention_window_size,
                    master_dtype=self.master_dtype,
                    slice_dtype=self.slice_dtype,
                    length_per_split=mtf.tensor_dim_to_size_per_split(
                        hparams.layout, hparams.mesh_shape,
                        self.max_length_dim),
                    name="local_att"))
        elif layer_type == "compressed_att":
          if is_incremental:
            raise ValueError("compressed_att incremental not implemented")
          else:
            x += layer_prepostprocess_dropout(
                mtf.layers.multihead_self_attention_memory_compressed(
                    normalize(x),
                    mask_right=True,
                    compression_factor=hparams.compression_factor,
                    kv_channels=self.kv_dim,
                    heads=self.heads_dim,
                    dropout=hparams.attention_dropout,
                    dropout_broadcast_dims=[self.length_dim],
                    master_dtype=self.master_dtype,
                    slice_dtype=self.slice_dtype,
                    name="compressed_att"))
        else:
          if is_incremental:
            # insert length dimension.
            x_shape = x.shape
            shape_with_length = mtf.Shape(
                x_shape.dims[:-1] + [mtf.Dimension("length", 1)]
                + x_shape.dims[-1:])
            x = mtf.reshape(x, shape_with_length)
          # ffn layer
          x += layer_prepostprocess_dropout(
              self._feedforward_layer(normalize(x), layer_type, losses=losses))
          if is_incremental:
            # remove length dimension
            x = mtf.reshape(x, x_shape)

    x = layer_prepostprocess_dropout(normalize(x))
    assert not layer_norm_vars
    if is_incremental:
      return x, new_states
    else:
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
    if hparams.transformer_type == "encdec":
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
          mtf.layers.attention_mask_ignore_padding(
              inputs, dtype=self.activation_dtype))
      with tf.variable_scope("encoder"):
        x = self._layer_stack(x,
                              hparams.encoder_layers,
                              self_attention_mask=encoder_attention_mask)
      encoder_output = mtf.rename_dimension(
          x, self.length_dim.name, self.memory_length_dim.name)
      encdec_tensors = []
      for layer_num, layer_type in enumerate(hparams.decoder_layers):
        if layer_type == "enc_att":
          with tf.variable_scope("decoder/enc_att_%d/enc_att" % layer_num):
            q_var, k_var, v_var, o_var = mtf.layers.multihead_attention_vars(
                mesh, self.heads_dim, self.model_dim,
                self.kv_dim, self.master_dtype, self.slice_dtype,
                self.activation_dtype)
            k = mtf.einsum(
                [encoder_output, k_var],
                mtf.Shape(
                    self.batch_dims + [self.heads_dim,
                                       self.memory_length_dim, self.kv_dim]))
            v = mtf.einsum(
                [encoder_output, v_var],
                mtf.Shape(
                    self.batch_dims + [self.heads_dim,
                                       self.memory_length_dim, self.kv_dim]))
          encdec_tensors.append((q_var, o_var, k, v))
        else:
          encdec_tensors.append(None)
      partial_targets = None
    elif hparams.transformer_type == "decoder":
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
    else:
      raise ValueError(
          "hparams.model_type = %s not yet supported"
          % hparams.transformer_type)

    local_attention_window = mtf.Dimension(
        "local_attention_window", hparams.local_attention_window_size)
    if hparams.beam_size == 1:
      ids_shape = mtf.Shape(self.batch_dims + [self.length_dim])
      kv_shape = mtf.Shape(self.batch_dims +
                           [self.heads_dim,
                            self.memory_length_dim, self.kv_dim])
      local_kv_shape = mtf.Shape(self.batch_dims +
                                 [self.heads_dim,
                                  local_attention_window, self.kv_dim])
    else:
      beam_dim = mtf.Dimension("beam", hparams.beam_size)
      ids_shape = mtf.Shape(self.batch_dims + [beam_dim, self.length_dim])
      kv_shape = mtf.Shape(self.batch_dims +
                           [beam_dim, self.heads_dim,
                            self.memory_length_dim, self.kv_dim])
      local_kv_shape = mtf.Shape(self.batch_dims +
                                 [beam_dim, self.heads_dim,
                                  local_attention_window, self.kv_dim])

    initial_ids = mtf.constant(mesh, 0, ids_shape, dtype=tf.int32)
    initial_states = []
    for layer in hparams.decoder_layers:
      if layer == "att":
        initial_states.extend(
            [mtf.zeros(mesh, kv_shape, dtype=self.activation_dtype)] * 2)
      elif layer == "local_att":
        initial_states.extend(
            [mtf.zeros(mesh, local_kv_shape, dtype=self.activation_dtype)] * 2)

    def logits_fn(step_num, ids, states):
      """Produce logits for this step, and new states."""
      ids_this_step = mtf.gather(ids, step_num - 1, self.length_dim)
      x = (mtf.gather(targets_embedding_var, ids_this_step,
                      self.targets_vocab_dim) +
           mtf.gather(positional_embedding_var, step_num, self.max_length_dim))
      with tf.variable_scope("decoder"):
        x, new_states = self._layer_stack(
            x,
            hparams.decoder_layers,
            encdec_attention_mask=encoder_attention_mask,
            step_num=step_num,
            encdec_tensors=encdec_tensors,
            states=states)
      logits = mtf.matmul(x, softmax_var)
      return logits, new_states

    if hparams.beam_size == 1:
      temperature = (0.0 if hparams.sampling_method == "argmax"
                     else hparams.sampling_temp)
      return mtf.beam_search.greedy_decode(
          logits_fn,
          initial_ids,
          temperature=temperature,
          initial_states=initial_states,
          forced_ids=partial_targets,
          use_tpu=hparams.use_tpu)
    else:
      if hparams.transformer_type == "encdec":
        input_length = mtf.reduce_sum(
            mtf.to_float(mtf.cast(inputs, tf.bool)),
            reduced_dim=self.length_dim)
        max_input_length = mtf.reduce_max(input_length)
        decode_length = mtf.cast(
            max_input_length * hparams.decode_length_multiplier
            + hparams.decode_length_constant, tf.int32)
      else:
        decode_length = None
      beams, unused_scores = mtf.beam_search.beam_search(
          logits_fn,
          initial_ids,
          hparams.alpha,
          states=initial_states,
          decode_length=decode_length,
          use_tpu=hparams.use_tpu,
          dtype=self.activation_dtype)
      return mtf.gather(beams, mtf.constant(mesh, 0, dtype=tf.int32), beam_dim)


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
  hparams.add_hparam("local_attention_window_size", 128)
  hparams.label_smoothing = 0.1
  # 8-way model-parallelism
  hparams.add_hparam("mesh_shape", "model:8")
  hparams.add_hparam("layout", "batch:batch;vocab:model;d_ff:model;heads:model")
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("d_ff", 2048)
  hparams.add_hparam("encoder_replicate_factor", 1)
  hparams.add_hparam("decoder_replicate_factor", 1)
  hparams.add_hparam("encoder_layers", ["att", "drd"] * 6)
  hparams.add_hparam("decoder_layers", ["att", "enc_att", "drd"] * 6)
  hparams.add_hparam("attention_dropout", 0.1)
  hparams.add_hparam("relu_dropout", 0.1)
  hparams.layer_prepostprocess_dropout = 0.1

  # Describes what model architecture:
  #   "encdec": encoder + autoregressive decoder
  #   "decoder": single-stack autoregressive sequence model.
  #   "encoder": single-stack non-autoregressive model
  #      with equal-length inputs and outputs.
  hparams.add_hparam("transformer_type", "encdec")

  # What does the decoder do:
  #   "autoregressive": Decoder left to right
  #   "denoising": Fills in masked-out values simultaneously
  hparams.add_hparam("decoder_type", "autoregressive")

  # Parameters describing the noising algorithm for denoising decoders
  hparams.add_hparam("noising_spec_train", {"type": "mask", "prob": 0.15})
  hparams.add_hparam("noising_spec_eval", {"type": "mask", "prob": 0.15})
  # during training, we use the eval noiser with this probability
  hparams.add_hparam("noising_use_eval_during_train", 0.1)

  # round up vocab sizes to be a multiple of this value
  hparams.vocab_divisor = 128

  # options are dense_relu_dense, moe, hmoe
  hparams.add_hparam("feedforward_layer", "drd")

  # If True, then reuse targets_embedding_var * rsqrt(d_model) as softmax_var
  # If hparams.transformer_type == "encoder", then there is no targets embedding
  # so we reuse the inputs embedding instead.
  hparams.shared_embedding_and_softmax_weights = True
  # Reuse targets_embedding_var as inputs_embedding_var
  # relevant only if hparams.transformer_type == "encdec"
  hparams.shared_embedding = True
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_schedule = "linear_warmup*rsqrt_decay*linear_decay"
  hparams.learning_rate_warmup_steps = 10000
  hparams.add_hparam("master_dtype", "bfloat16")
  hparams.add_hparam("slice_dtype", "float32")
  hparams.activation_dtype = "bfloat16"

  # These parameters make Transformer model compatible with MtfTransformer
  # Do not override these, as mtf_transformer does not support other options.
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.bottom = {
      "inputs": modalities.identity_bottom,
      "targets": modalities.identity_bottom,
  }
  hparams.top = {
      "targets": modalities.identity_top,
  }

  # Parameters for computing the maximum decode length in beam search.
  # Maximum decode length is:
  #    min(max_length,
  #        decode_length_multiplier * input_length + decode_length_constant)
  hparams.add_hparam("decode_length_multiplier", 1.5)
  hparams.add_hparam("decode_length_constant", 10.0)

  # If nonzero, we split the batch across two tensor-dimensions named
  # "outer_batch" and "inner_batch", allowing for splitting across two mesh
  # dimensions.  This is necessary for hierarchical mixture of experts.
  # The two tensor dimensions have sizes hparams.outer_batch_size and
  # hparams.batch_size // hparams.outer_batch_size.
  hparams.add_hparam("outer_batch_size", 0)

  # TODO(noam): file a bug
  hparams.add_hparam("reshape_logits_hack", False)
  hparams.add_hparam("compression_factor", 4)

  return hparams


@registry.register_hparams
def mtf_transformer_base_lm():
  hparams = mtf_transformer_base()
  hparams.decoder_layers = hparams.encoder_layers
  hparams.transformer_type = "decoder"
  hparams.label_smoothing = 0.0
  hparams.sampling_method = "random"
  return hparams


@registry.register_hparams
def mtf_transformer_tiny():
  """Catch bugs locally..."""
  hparams = mtf_transformer_base()
  hparams.d_model = 128
  hparams.d_ff = 512
  hparams.batch_size = 8
  hparams.encoder_layers = ["att", "drd"] * 2
  hparams.decoder_layers = ["att", "enc_att", "drd"] * 2
  hparams.num_heads = 8
  # data parallelism and model-parallelism
  hparams.mesh_shape = "batch:2;model:4"
  hparams.activation_dtype = "float32"
  return hparams


@registry.register_hparams
def mtf_transformer_tiny_lm():
  hparams = mtf_transformer_tiny()
  hparams.decoder_layers = hparams.encoder_layers
  hparams.transformer_type = "decoder"
  hparams.label_smoothing = 0.0
  hparams.sampling_method = "random"
  return hparams


@registry.register_hparams
def mtf_transformer_tiny_denoising():
  hparams = mtf_transformer_tiny_lm()
  hparams.decoder_type = "denoising"
  hparams.noising_spec_train = ("random_zipfian", 0.3)
  hparams.noising_use_eval_during_train = 0.5
  hparams.max_length = 1024
  return hparams


@registry.register_hparams
def mtf_transformer_single():
  hparams = mtf_transformer_tiny()
  hparams.mesh_shape = ""
  return hparams


@registry.register_hparams
def mtf_transformer_enc_single():
  hparams = mtf_transformer_single()
  hparams.transformer_type = "encoder"
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
  hparams = mtf_transformer_base_lm()
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
def mtf_transformer_paper_tr_0_a32():
  hparams = mtf_transformer_paper_tr_0()
  hparams.activation_dtype = "float32"
  return hparams


@registry.register_hparams
def mtf_transformer_paper_tr_0_nf():
  hparams = mtf_transformer_paper_tr_0()
  hparams.optimizer_adafactor_factored = False
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
