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

import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import moe
from mesh_tensorflow.transformer import transformer
from mesh_tensorflow.transformer import transformer_layers
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
from tensor2tensor.utils import mtf_model
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_model
class MtfUnitransformer(mtf_model.MtfModel):
  """Single-stack Transformer (Transformer Decoder) in mesh_tensorflow.

  Can optionally be autoregressive (language generation) or non-autoregressive
  like BERT.
  """

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

  def combine_batch_dims(self, x):
    if len(self.batch_dims) <= 1:
      return x
    return mtf.replace_dimensions(
        x, self.batch_dims, mtf.combined_dimension(self.batch_dims))

  @property
  def autoregressive(self):
    return self._hparams.autoregressive

  @property
  def variable_dtype(self):
    return mtf.VariableDType(
        tf.as_dtype(self._hparams.master_dtype),
        tf.as_dtype(self._hparams.slice_dtype),
        tf.as_dtype(self._hparams.activation_dtype))

  @property
  def length_dim(self):
    return mtf.Dimension(
        "length", self._hparams.length or self._hparams.max_length)

  def _import_to_batch_by_length(self, x, name, mesh):
    mtf_shape = mtf.Shape(self.batch_dims + [self.length_dim])
    x = tf.reshape(x, mtf_shape.to_integer_list)
    return mtf.import_fully_replicated(mesh, x, mtf_shape, name=name)

  def _import_feature(self, features, mesh, key):
    """Import a feature from the features dictionary into a mtf.Tensor.

    Args:
      features: a features dictionary
      mesh: a Mesh
      key: a string

    Returns:
      a mtf.Tensor with dtype int32 and shape self.batch_dims + self.length_dim
    """
    if key not in features:
      return None
    x = tf.to_int32(features[key])
    x = common_layers.expand_squeeze_to_nd(x, 2)
    batch_size = mtf.Shape(self.batch_dims).size
    x = x[:, :self.length_dim.size]
    extra_length = self.length_dim.size - tf.shape(x)[1]
    extra_batch = batch_size - tf.shape(x)[0]
    x = tf.pad(x, [[0, extra_batch], [0, extra_length]])
    mtf_shape = mtf.Shape(self.batch_dims + [self.length_dim])
    x = tf.reshape(x, mtf_shape.to_integer_list)
    return mtf.import_fully_replicated(mesh, x, mtf_shape, name=key)

  def model(self):
    hparams = self._hparams
    if hparams.label_smoothing != 0:
      raise NotImplementedError(
          "Label smoothing not implemented in unitransformer."
          "  Do you really want it?")
    layer_stack = layer_stack_from_hparams(hparams, "")
    if self.autoregressive:
      input_vocab_size = self._targets_vocab_size
    else:
      input_vocab_size = self._inputs_vocab_size
    return transformer.Unitransformer(
        layer_stack=layer_stack,
        d_model=hparams.d_model,
        input_vocab_size=input_vocab_size,
        output_vocab_size=self._targets_vocab_size,
        autoregressive=self.autoregressive,
        max_length=hparams.max_length,
        shared_embedding_and_softmax_weights=(
            hparams.shared_embedding_and_softmax_weights),
        z_loss=hparams.z_loss,
        layout=hparams.layout,
        mesh_shape=hparams.mesh_shape)

  def _mtf_model_fn(self, features, mesh):
    self._original_features = features
    hparams = self._hparams
    def import_feature(key):
      return self._import_feature(features, mesh, key)
    targets = import_feature("targets")
    sequence_id = import_feature("targets_segmentation")
    if hparams.use_global_position_in_packed_sequence:
      position = None
    else:
      position = import_feature("targets_position")
    if self.autoregressive:
      inputs = mtf.shift(
          targets, offset=1, dim=self.length_dim, wrap=False)
      # We should have a 0 at the beginning of each sequence rather than the
      # shifted EOS (1) from the previous sequence.
      inputs -= mtf.to_int32(mtf.equal(inputs, 1))
    else:
      inputs = import_feature("inputs")
      # TODO(noam): options for bert-style masking here?
    model = self.model()
    logits, loss = model.call_simple(
        inputs=inputs,
        targets=targets,
        compute_loss=True,
        mode=hparams.mode,
        variable_dtype=self.variable_dtype,
        sequence_id=sequence_id,
        position=position)
    return logits, loss

  def mtf_model_fn(self, features, mesh):
    logits, loss = self._mtf_model_fn(features, mesh)
    # combine batch dims
    logits = self.combine_batch_dims(logits)
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

  def sample(self, features, mesh):
    hparams = self._hparams
    model = self.model()
    def import_feature(key):
      return self._import_feature(features, mesh, key)

    if self.autoregressive:
      # Prepare partial targets.
      # In either features["inputs"] or features["targets"].
      # We force the outputs to begin with these sequences.
      partial_targets = import_feature("inputs")
      if partial_targets is None:
        partial_targets = import_feature("targets")
      if partial_targets:
        partial_targets *= mtf.cast(
            mtf.not_equal(partial_targets, 1), partial_targets.dtype)
      else:
        ids_shape = mtf.Shape(self.batch_dims + [self.length_dim])
        partial_targets = mtf.constant(mesh, 0, ids_shape, dtype=tf.int32)
      if hparams.beam_size > 1:
        raise NotImplementedError(
            "Beam search not implemented for unitransformer.")
      ret = model.sample_autoregressive(
          partial_targets,
          temperature=hparams.sampling_temp,
          variable_dtype=self.variable_dtype)
      return self.combine_batch_dims(ret)
    else:
      raise ValueError(
          "Don't know how to sample from non-autoregressive unitransformer")


@registry.register_model
class MtfBitransformer(MtfUnitransformer):
  """Encoder-Decoder Transformer in mesh_tensorflow."""

  def model(self):
    hparams = self._hparams
    encoder_layer_stack = layer_stack_from_hparams(hparams, "encoder_")
    decoder_layer_stack = layer_stack_from_hparams(hparams, "decoder_")
    encoder = transformer.Unitransformer(
        layer_stack=encoder_layer_stack,
        d_model=hparams.d_model,
        input_vocab_size=self._inputs_vocab_size,
        output_vocab_size=None,
        autoregressive=False,
        max_length=hparams.max_length,
        name="encoder",
        layout=hparams.layout,
        mesh_shape=hparams.mesh_shape,
    )
    decoder = transformer.Unitransformer(
        layer_stack=decoder_layer_stack,
        d_model=hparams.d_model,
        input_vocab_size=self._targets_vocab_size,
        output_vocab_size=self._targets_vocab_size,
        autoregressive=True,
        max_length=hparams.max_length,
        label_smoothing=hparams.label_smoothing,
        shared_embedding_and_softmax_weights=(
            hparams.shared_embedding_and_softmax_weights),
        z_loss=hparams.z_loss,
        name="decoder",
        layout=hparams.layout,
        mesh_shape=hparams.mesh_shape,
    )
    return transformer.Bitransformer(
        encoder, decoder, shared_embedding=hparams.shared_embedding)

  def _mtf_model_fn(self, features, mesh):
    self._original_features = features
    hparams = self._hparams
    def import_feature(key):
      return self._import_feature(features, mesh, key)
    targets = import_feature("targets")
    inputs = import_feature("inputs")
    if not inputs:
      raise ValueError("inputs feature is missing")
    encoder_sequence_id = import_feature("inputs_segmentation")
    if not encoder_sequence_id:
      encoder_sequence_id = mtf.to_int32(mtf.not_equal(inputs, 0))
    decoder_sequence_id = import_feature("targets_segmentation")
    if decoder_sequence_id is None:
      decoder_sequence_id = mtf.to_int32(mtf.not_equal(targets, 0))
    if hparams.use_global_position_in_packed_sequence:
      encoder_position = None
      decoder_position = None
    else:
      encoder_position = import_feature("inputs_position")
      decoder_position = import_feature("targets_position")
    model = self.model()
    logits, loss = model.call_simple(
        inputs=inputs,
        targets=targets,
        compute_loss=True,
        mode=hparams.mode,
        variable_dtype=self.variable_dtype,
        encoder_sequence_id=encoder_sequence_id,
        decoder_sequence_id=decoder_sequence_id,
        encoder_position=encoder_position,
        decoder_position=decoder_position)
    return logits, loss

  def sample(self, features, mesh):
    hparams = self._hparams
    model = self.model()
    inputs = self._import_feature(features, mesh, "inputs")
    ret = model.decode(
        inputs,
        self.variable_dtype,
        beam_size=hparams.beam_size,
        alpha=hparams.alpha,
        temperature=hparams.sampling_temp if hparams.beam_size == 1 else 0,
        decode_length_multiplier=hparams.decode_length_multiplier,
        decode_length_constant=hparams.decode_length_constant)
    return self.combine_batch_dims(ret)


layers_registry = registry.Registries.mtf_layers


# The following functions construct layers based on hyperparmeters
def attention_kwargs_from_hparams(hparams):
  return {
      "dropout_rate": hparams.attention_dropout,
      "extra_logit": 0.0 if hparams.extra_logit else None,
  }


@layers_registry.register("self_att")
def self_attention_layer(hparams, prefix):
  """Create self-attention layer based on hyperparameters."""
  return transformer_layers.SelfAttention(
      num_heads=hparams.get(prefix + "num_heads"),
      num_memory_heads=hparams.get(prefix + "num_memory_heads"),
      key_value_size=hparams.d_kv,
      shared_kv=hparams.get(prefix + "shared_kv", False),
      attention_kwargs=attention_kwargs_from_hparams(hparams))


@layers_registry.register("local_self_att")
def local_self_attention_layer(hparams, prefix):
  """Create self-attention layer based on hyperparameters."""
  return transformer_layers.LocalSelfAttention(
      num_heads=hparams.get(prefix + "num_heads"),
      num_memory_heads=hparams.get(prefix + "num_memory_heads"),
      radius=hparams.local_attention_radius,
      key_value_size=hparams.d_kv,
      shared_kv=hparams.get(prefix + "shared_kv", False),
      attention_kwargs=attention_kwargs_from_hparams(hparams))


@layers_registry.register("enc_att")
def enc_dec_attention_layer(hparams, prefix):
  return transformer_layers.EncDecAttention(
      num_heads=hparams.get(prefix + "num_heads"),
      num_memory_heads=hparams.get(prefix + "num_memory_heads"),
      key_value_size=hparams.d_kv,
      shared_kv=hparams.get(prefix + "shared_kv", False),
      attention_kwargs=attention_kwargs_from_hparams(hparams))


@layers_registry.register("drd")
def dense_relu_dense_layer(hparams, prefix):
  del prefix
  return transformer_layers.DenseReluDense(
      hidden_size=hparams.d_ff,
      dropout_rate=hparams.relu_dropout)


@layers_registry.register("moe_1d")
def moe_1d_layer(hparams, prefix):
  del prefix
  return moe.MoE1D(num_experts=hparams.moe_num_experts,
                   hidden_size=hparams.moe_hidden_size)


@layers_registry.register("moe_2d")
def moe_2d_layer(hparams, prefix):
  del prefix
  return moe.MoE2D(expert_x=hparams.moe_expert_x,
                   expert_y=hparams.moe_expert_y,
                   hidden_size=hparams.moe_hidden_size)


def layer_stack_from_hparams(hparams, prefix):
  """Create a layer stack based on the hyperparameter values."""
  layers = hparams.get(prefix + "layers")
  return transformer.LayerStack(
      [layers_registry[l](hparams, prefix) for l in layers],
      dropout_rate=hparams.layer_prepostprocess_dropout,
      norm_epsilon=hparams.norm_epsilon)


def mtf_transformer2_base():
  """Hyperparameters common to both unitransformer and bitransformer."""
  hparams = common_hparams.basic_params1()

  hparams.add_hparam("d_model", 1024)
  hparams.batch_size = 4
  hparams.max_length = 1024
  hparams.label_smoothing = 0.0
  # a small positive value - this seems important for stability when training
  # with bfloat16 activations.
  hparams.add_hparam("z_loss", 1e-4)

  # hparams applying to both encoder and decoder layer stacks.
  hparams.add_hparam("d_ff", 2048)
  hparams.add_hparam("d_kv", 128)
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("relu_dropout", 0.0)
  hparams.del_hparam("num_heads")
  hparams.del_hparam("num_hidden_layers")
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.add_hparam("extra_logit", False)
  # number of experts for moe_1d
  hparams.moe_num_experts = 32
  # number of experts for moe_2d = moe_expert_x * moe_expert_y
  hparams.add_hparam("moe_expert_x", 8)
  hparams.add_hparam("moe_expert_y", 4)
  hparams.add_hparam("moe_hidden_size", 32768)

  # round up vocab sizes to be a multiple of this value
  hparams.vocab_divisor = 128

  hparams.optimizer = "Adafactor"
  hparams.learning_rate_schedule = "rsqrt_decay*linear_decay"
  hparams.learning_rate_warmup_steps = 10000
  hparams.add_hparam("master_dtype", "bfloat16")
  hparams.add_hparam("slice_dtype", "float32")
  hparams.activation_dtype = "bfloat16"

  # 8-way model-parallelism
  hparams.add_hparam("mesh_shape", "model:8")
  hparams.add_hparam("layout", "batch:batch;vocab:model;d_ff:model;heads:model")

  # If nonzero, we split the batch across two tensor-dimensions named
  # "outer_batch" and "inner_batch", allowing for splitting across two mesh
  # dimensions.  This is necessary for hierarchical mixture of experts.
  # The two tensor dimensions have sizes hparams.outer_batch_size and
  # hparams.batch_size // hparams.outer_batch_size.
  hparams.add_hparam("outer_batch_size", 0)

  hparams.shared_embedding_and_softmax_weights = False
  # length for training or decoding - defaults to max_length
  hparams.add_hparam("length", 0)

  # These parameters make Transformer model compatible with mtf
  # Do not override these.
  hparams.no_data_parallelism = True
  hparams.use_fixed_batch_size = True
  hparams.add_hparam("mtf_mode", True)
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.bottom = {
      "inputs": modalities.identity_bottom,
      "targets": modalities.identity_bottom,
  }
  hparams.top = {
      "targets": modalities.identity_top,
  }
  hparams.add_hparam("beam_size", 1)

  # If this is True, then in a packed dataset (where exaples are concatenated
  # to form longer examples) we use the global position (within the concatenated
  # sequence) to compute the positional embedding, instead of the position
  # within the individual sequence.  This is counterintuitive, but for some
  # reason, it keeps the model from diverging.
  hparams.add_hparam("use_global_position_in_packed_sequence", True)

  return hparams


@registry.register_hparams
def mtf_unitransformer_base():
  """Hyperparameters for single-stack Transformer."""
  hparams = mtf_transformer2_base()
  hparams.add_hparam("autoregressive", True)
  # HYPERPARAMETERS FOR THE SINGLE LAYER STACK
  hparams.add_hparam("layers", ["self_att", "drd"] * 6)
  # number of heads in multihead attention
  hparams.add_hparam("num_heads", 8)
  # default of 0 for standard transformer behavior
  # 1 means a single set of keys and values that are read by all query heads
  hparams.add_hparam("num_memory_heads", 0)
  # share attention keys and values
  hparams.add_hparam("shared_kv", False)
  # if nonzero then use local attention
  hparams.add_hparam("local_attention_radius", 128)
  return hparams


@registry.register_hparams
def mtf_bitransformer_base():
  """Machine translation base configuration."""
  hparams = mtf_transformer2_base()
  hparams.max_length = 256
  hparams.shared_embedding = True
  # HYPERPARAMETERS FOR THE LAYER STACKS
  hparams.add_hparam("encoder_layers", ["self_att", "drd"] * 6)
  hparams.add_hparam("decoder_layers", ["self_att", "enc_att", "drd"] * 6)
  hparams.add_hparam("encoder_num_layers", 6)
  hparams.add_hparam("decoder_num_layers", 6)
  # number of heads in multihead attention
  hparams.add_hparam("encoder_num_heads", 8)
  hparams.add_hparam("decoder_num_heads", 8)
  hparams.add_hparam("local_attention_radius", 128)

  # default of 0 for standard transformer behavior
  # 1 means a single set of keys and values that are read by all query heads
  hparams.add_hparam("encoder_num_memory_heads", 0)
  hparams.add_hparam("decoder_num_memory_heads", 0)
  # share attention keys and values
  hparams.add_hparam("encoder_shared_kv", False)
  hparams.add_hparam("decoder_shared_kv", False)

  # Parameters for computing the maximum decode length in beam search.
  # Maximum decode length is:
  #    min(max_length,
  #        decode_length_multiplier * input_length + decode_length_constant)
  hparams.add_hparam("decode_length_multiplier", 1.5)
  hparams.add_hparam("decode_length_constant", 10.0)
  # used during decoding
  hparams.add_hparam("alpha", 0.6)
  hparams.sampling_temp = 0.0
  return hparams


@registry.register_hparams
def mtf_unitransformer_tiny():
  hparams = mtf_unitransformer_base()
  hparams.batch_size = 2
  hparams.mesh_shape = ""
  hparams.d_model = 128
  hparams.layers = ["self_att", "drd"] * 2
  hparams.num_heads = 4
  hparams.d_ff = 512
  return hparams


@registry.register_hparams
def mtf_bitransformer_tiny():
  """Small encoder-decoder model for testing."""
  hparams = mtf_bitransformer_base()
  hparams.batch_size = 2
  hparams.mesh_shape = ""
  hparams.d_model = 128
  hparams.encoder_layers = ["self_att", "drd"] * 2
  hparams.decoder_layers = ["self_att", "enc_att", "drd"] * 2
  hparams.num_heads = 4
  hparams.d_ff = 512
  return hparams


@registry.register_hparams
def mtf_unitransformer_all_layers_tiny():
  """Test out all the layers on local CPU."""
  hparams = mtf_unitransformer_tiny()
  hparams.moe_num_experts = 4
  hparams.moe_expert_x = 4
  hparams.moe_expert_y = 4
  hparams.moe_hidden_size = 512
  hparams.layers = ["self_att", "local_self_att", "moe_1d", "moe_2d", "drd"]
  return hparams


@registry.register_hparams
def mtf_bitransformer_all_layers_tiny():
  """Test out all the layers on local CPU."""
  hparams = mtf_bitransformer_tiny()
  hparams.moe_num_experts = 4
  hparams.moe_expert_x = 4
  hparams.moe_expert_y = 4
  hparams.moe_hidden_size = 512
  hparams.encoder_layers = [
      "self_att", "local_self_att", "moe_1d", "moe_2d", "drd"]
  hparams.decoder_layers = [
      "self_att", "local_self_att", "enc_att", "moe_1d", "moe_2d", "drd"]
  return hparams


@registry.register_hparams
def mtr_lm_dense(sz):
  """Series of architectures for language modeling.

  We assume infinite training data, so no dropout necessary.

  You can use languagemodel_wiki_noref_v32k_l1k.
  (1 epoch = ~46000 steps).
  TODO(noam): find a large enough dataset for these experiments.

  Args:
    sz: an integer

  Returns:
    a hparams
  """
  n = 2 ** sz
  hparams = mtf_unitransformer_base()
  hparams.d_model = 1024
  hparams.max_length = 1024
  hparams.batch_size = 128
  # Parameters for my_layer_stack()
  hparams.num_hidden_layers = 6
  hparams.d_ff = 8192 * n
  hparams.d_kv = 256
  hparams.num_heads = 8 * n
  hparams.learning_rate_decay_steps = 65536
  hparams.layout = "batch:batch;vocab:model;d_ff:model;heads:model"
  hparams.mesh_shape = "batch:32"
  return hparams


@registry.register_hparams
def mtr_lm_dense_0():
  return mtr_lm_dense(0)


@registry.register_hparams
def mtr_lm_dense_0_h1_16():
  hparams = mtr_lm_dense_0()
  hparams.decoder_num_heads = 16
  hparams.decoder_num_memory_heads = 1
  return hparams


@registry.register_hparams
def mtr_lm_dense_1():
  return mtr_lm_dense(1)


@registry.register_hparams
def mtr_lm_dense_2():
  hparams = mtr_lm_dense(2)
  hparams.mesh_shape = "model:4;batch:8"
  return hparams


@registry.register_hparams
def mtr_lm_dense_3():
  hparams = mtr_lm_dense(3)
  hparams.mesh_shape = "model:4;batch:8"
  return hparams


@registry.register_hparams
def mtr_lm_v1():
  """Model incorporating mixture-of-experts, local and global attention.

  ~6B parameters

  32 experts in 3 hierarchichal moe layers.

  Returns:
    a hparams
  """
  hparams = mtr_lm_dense(0)
  hparams.layers = (["local_self_att", "local_self_att", "drd",
                     "self_att", "drd", "local_self_att",
                     "local_self_att", "moe_2d"] * 4)[:-1]
  hparams.d_kv = 128
  hparams.moe_expert_x = 8
  hparams.moe_expert_y = 4
  hparams.moe_hidden_size = 32768
  hparams.d_ff = 2048
  hparams.num_memory_heads = 0
  hparams.mesh_shape = "b0:4;b1:8"
  hparams.layout = "outer_batch:b0;inner_batch:b1,expert_x:b1,expert_y:b0"
  hparams.outer_batch_size = 4
  return hparams


@registry.register_hparams
def mtr_lm_v1_h1_8():
  """Version for fast decoding."""
  hparams = mtr_lm_v1()
  hparams.num_memory_heads = 1
  return hparams


def mtr_tr_dense(sz):
  """Series of machine translation models.

  All models are trained on sequences of 256 tokens.

  You can use the dataset translate_enfr_wmt32k_packed.
  154000 steps = 3 epochs.

  Args:
    sz: an integer

  Returns:
    a hparams
  """
  n = 2 ** sz
  hparams = mtf_bitransformer_base()
  hparams.d_model = 1024
  hparams.max_length = 256
  hparams.batch_size = 128
  hparams.d_ff = int(4096 * n)
  hparams.d_kv = 128
  hparams.encoder_num_heads = int(8 * n)
  hparams.decoder_num_heads = int(8 * n)
  # one epoch for translate_enfr_wmt32k_packed = 51400 steps
  hparams.learning_rate_decay_steps = 51400
  hparams.layout = "batch:batch;vocab:model;d_ff:model;heads:model"
  hparams.mesh_shape = "batch:32"
  hparams.label_smoothing = 0.1
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1
  return hparams


@registry.register_hparams
def mtr_tr_dense_0():
  return mtr_tr_dense(0)


@registry.register_hparams
def mtr_tr_dense_1():
  return mtr_tr_dense(1)


@registry.register_hparams
def mtr_tr_dense_2():
  hparams = mtr_tr_dense(2)
  hparams.mesh_shape = "model:4;batch:8"
  return hparams


@registry.register_hparams
def mtr_tr_dense_3():
  hparams = mtr_tr_dense(3)
  hparams.mesh_shape = "model:4;batch:8"
  return hparams


@registry.register_hparams
def mtr_tr_dense_3_88():
  hparams = mtr_tr_dense(3)
  hparams.mesh_shape = "model:8;batch:16"
  return hparams


@registry.register_hparams
def mtr_tr_dense_3_fast():
  hparams = mtr_tr_dense_3()
  hparams.local_attention_radius = 32
  hparams.decoder_num_heads = 128
  hparams.decoder_num_memory_heads = 8
  return hparams


def mtr_tr_dense_local(sz):
  """With local self-attention in the decoder."""
  hparams = mtr_tr_dense(sz)
  hparams.decoder_layers = ["local_self_att", "enc_att", "drd"] * 6
  hparams.local_attention_radius = 32
  return hparams


@registry.register_hparams
def mtr_tr_dense_local_0():
  return mtr_tr_dense_local(0)


@registry.register_hparams
def mtr_tr_dense_local_0_w8():
  hparams = mtr_tr_dense_local_0()
  hparams.local_attention_radius = 8
  return hparams


@registry.register_hparams
def mtr_tr_dense_local_0_h1_16():
  hparams = mtr_tr_dense_local_0()
  hparams.decoder_num_heads = 16
  hparams.decoder_num_memory_heads = 1
  return hparams


@registry.register_hparams
def mtr_tr_dense_local_0_h1_16_shared():
  hparams = mtr_tr_dense_local_0_h1_16()
  hparams.shared_embedding_and_softmax_weights = True
  return hparams


@registry.register_hparams
def mtr_tr_dense_local_0_h1_8_kv256():
  hparams = mtr_tr_dense_local_0()
  hparams.decoder_num_heads = 8
  hparams.decoder_num_memory_heads = 1
  hparams.d_kv = 256
  return hparams


@registry.register_hparams
def mtr_tr_dense_local_0_h1_16_shared_kv():
  hparams = mtr_tr_dense_local_0_h1_16()
  hparams.decoder_shared_kv = True
  return hparams


@registry.register_hparams
def mtr_tr_dense_0_h4():
  hparams = mtr_tr_dense_0()
  hparams.decoder_num_heads = 4
  return hparams


@registry.register_hparams
def mtr_tr_dense_0_h16():
  hparams = mtr_tr_dense_0()
  hparams.decoder_num_heads = 16
  return hparams


@registry.register_hparams
def mtr_tr_dense_0_extra_logit():
  hparams = mtr_tr_dense_0()
  hparams.extra_logit = True
  return hparams


@registry.register_hparams
def mtr_tr_dense_0_h1_8():
  hparams = mtr_tr_dense_0()
  hparams.decoder_num_memory_heads = 1
  return hparams


@registry.register_hparams
def mtr_tr_dense_0_h1_1():
  hparams = mtr_tr_dense_0()
  hparams.decoder_num_heads = 1
  return hparams


@registry.register_hparams
def mtr_tr_dense_0_h1_16():
  hparams = mtr_tr_dense_0()
  hparams.decoder_num_heads = 16
  hparams.decoder_num_memory_heads = 1
  return hparams


@registry.register_hparams
def mtr_tr_dense_0_h2_16():
  hparams = mtr_tr_dense_0()
  hparams.decoder_num_heads = 16
  hparams.decoder_num_memory_heads = 2
  return hparams


@registry.register_hparams
def mtr_tr_dense_0_shared_kv():
  hparams = mtr_tr_dense_0()
  hparams.decoder_shared_kv = True
  return hparams


@registry.register_hparams
def mtr_tr_enfr_v0():
  # good parameters for wmt-en-fr
  hparams = mtr_tr_dense_local_0_h1_16()
  return hparams


@registry.register_hparams
def mtr_tr_ende_v0():
  # good parameters for wmt-en-de
  hparams = mtr_tr_dense_local_0_h1_16()
  hparams.learning_rate_decay_steps = 20000
  hparams.shared_embedding_and_softmax_weights = True
  hparams.layer_prepostprocess_dropout = 0.2
  return hparams


@registry.register_hparams
def mtr_tr_ende_deep():
  hparams = mtr_tr_ende_v0()
  hparams.decoder_num_heads = 8
  hparams.encoder_num_heads = 4
  hparams.d_ff = 2048
  hparams.encoder_num_layers = 12
  hparams.decoder_num_layers = 12
  return hparams
