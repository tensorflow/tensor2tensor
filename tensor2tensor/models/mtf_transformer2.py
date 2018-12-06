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
class MtfTransformer2(mtf_model.MtfModel):
  """Transformer in mesh_tensorflow."""

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

  def _import_to_batch_by_decode_length(self, x, name, mesh):
    mtf_shape = mtf.Shape(self.batch_dims + [self.length_dim])
    x = tf.reshape(x, mtf_shape.to_integer_list)
    return mtf.import_fully_replicated(mesh, x, mtf_shape, name=name)

  def model(self):
    hparams = self._hparams
    if isinstance(hparams.layer_stack, transformer.LayerStack):
      layer_stack = hparams.layer_stack
    else:
      # hparams.layer_stack is a function for creating a LayerStack
      layer_stack = hparams.layer_stack(hparams)
    return transformer.Transformer(
        layer_stack=layer_stack,
        d_model=hparams.d_model,
        input_vocab_size=self._targets_vocab_size,
        output_vocab_size=self._targets_vocab_size,
        autoregressive=hparams.decoder_type == "autoregressive",
        max_length=hparams.max_length)

  def _mtf_model_fn(self, features, mesh):
    self._original_features = features
    features = copy.copy(features)
    hparams = self._hparams
    targets = tf.to_int32(features["targets"])
    if len(targets.get_shape()) > 2:
      tf.logging.info("targets = %s" % targets)
      targets = tf.squeeze(targets, [2, 3])
    # pad targets to max_length
    def pad_to_length(x):
      extra_length = self.length_dim.size - tf.shape(x)[1]
      x = tf.pad(x, [[0, 0], [0, extra_length]])
      x = tf.reshape(x, [hparams.batch_size, self.length_dim.size])
      return x
    targets = pad_to_length(targets)
    targets = self._import_to_batch_by_length(targets, "targets", mesh)
    for key in ["targets_segmentation", "targets_position",
                "inputs_segmentation", "inputs_position"]:
      if key in features:
        features[key] = pad_to_length(features[key])
    if hparams.decoder_type == "autoregressive":
      shifted_targets = mtf.shift(
          targets, offset=1, dim=self.length_dim, wrap=False)
    else:
      raise ValueError(
          "unknown hparams.decoder_type = %s" % hparams.decoder_type)
    model = self.model()
    logits, loss = model.call_simple(
        inputs=shifted_targets,
        targets=targets,
        compute_loss=True,
        mode=hparams.mode,
        variable_dtype=self.variable_dtype)
    # mesh_shape=hparams.mesh_shape,
    # layout=hparams.layout,
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
    targets_vocab_size = self._problem_hparams.modality[
        "targets"].top_dimensionality
    targets_vocab_size += (-targets_vocab_size) % self._hparams.vocab_divisor
    return targets_vocab_size

  @property
  def _inputs_vocab_size(self):
    inputs_vocab_size = self._problem_hparams.modality[
        "inputs"].top_dimensionality
    inputs_vocab_size += (-inputs_vocab_size) % self._hparams.vocab_divisor
    return inputs_vocab_size

  def sample(self, features, mesh):
    hparams = self._hparams
    model = self.model()
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
                            [0, self.length_dim.size - partial_targets_length]])
      partial_targets = self._import_to_batch_by_length(
          partial_targets, "partial_targets", mesh)
      # strip EOS
      partial_targets *= mtf.to_int32(mtf.not_equal(partial_targets, 1))

    else:
      ids_shape = mtf.Shape(self.batch_dims + [self.length_dim])
      partial_targets = mtf.constant(mesh, 0, ids_shape, dtype=tf.int32)
    if hparams.beam_size == 1:
      pass
    else:
      raise NotImplementedError("not implemented")
      # beam_dim = mtf.Dimension("beam", hparams.beam_size)
      # ids_shape = mtf.Shape(self.batch_dims + [beam_dim, self.length_dim])

    partial_targets = mtf.Print(partial_targets, [partial_targets],
                                "Partial_Targets", summarize=1000)
    return model.sample_autoregressive(
        partial_targets,
        temperature=hparams.sampling_temp,
        variable_dtype=self.variable_dtype)


@registry.register_hparams
def mtf_transformer2_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.no_data_parallelism = True
  hparams.use_fixed_batch_size = True
  hparams.add_hparam("mtf_mode", True)
  hparams.batch_size = 4
  hparams.max_length = 1024
  hparams.add_hparam("d_model", 1024)
  hparams.label_smoothing = 0.0
  # 8-way model-parallelism
  hparams.add_hparam("mesh_shape", "model:8")
  hparams.add_hparam("layout", "batch:batch;vocab:model;d_ff:model;heads:model")

  # hparams.layer_stack should be either a transformer.LayerStack or a function
  # from hparams to transformer.LayerStack
  def my_layer_stack(hparams):
    return transformer.LayerStack(
        [transformer_layers.SelfAttention(
            num_heads=hparams.num_heads,
            key_value_size=hparams.d_kv,
            dropout_rate=hparams.attention_dropout),
         transformer_layers.DenseReluDense(
             hidden_size=hparams.d_ff,
             dropout_rate=hparams.layer_prepostprocess_dropout),
        ] * hparams.num_hidden_layers)
  hparams.layer_stack = my_layer_stack

  # These hyperparameters are used in the above default layer_stack function.
  # They may not be respected if hparams.layer_stack is changed.
  hparams.num_hidden_layers = 6
  hparams.add_hparam("d_ff", 2048)
  hparams.add_hparam("d_kv", 128)
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("relu_dropout", 0.0)
  hparams.layer_prepostprocess_dropout = 0.0

  # Describes what model architecture:
  #   "encdec": encoder + autoregressive decoder
  #   "decoder": single-stack autoregressive sequence model.
  #   "encoder": single-stack non-autoregressive model
  #      with equal-length inputs and outputs.
  # TODO(noam): implement different types of transformers.
  hparams.add_hparam("transformer_type", "decoder")

  # What does the decoder do:
  #   "autoregressive": Decoder left to right
  #   "denoising": Fills in masked-out values simultaneously
  # TODO(noam): only autoregressive is implemented so far.
  hparams.add_hparam("decoder_type", "autoregressive")

  # round up vocab sizes to be a multiple of this value
  hparams.vocab_divisor = 128

  hparams.optimizer = "Adafactor"
  hparams.learning_rate_schedule = "rsqrt_decay*linear_decay"
  hparams.learning_rate_warmup_steps = 10000
  hparams.add_hparam("master_dtype", "bfloat16")
  hparams.add_hparam("slice_dtype", "float32")
  hparams.activation_dtype = "bfloat16"

  # These parameters make Transformer model compatible with MtfTransformer2
  # Do not override these, as mtf_transformer does not support other options.
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.modality = {
      "inputs": modalities.IdentitySymbolModality,
      "targets": modalities.IdentitySymbolModality,
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

  # length for training or decoding - defaults to max_length
  hparams.add_hparam("length", 0)

  hparams.sampling_method = "random"
  return hparams


@registry.register_hparams
def mtf_transformer2_tiny():
  hparams = mtf_transformer2_base()
  hparams.batch_size = 2
  hparams.mesh_shape = ""
  hparams.d_model = 128
  hparams.num_hidden_layers = 2
  hparams.num_heads = 4
  hparams.d_ff = 512
  return hparams


@registry.register_hparams
def mtf_transformer2_all_layers_tiny():
  """Test out all the layers on local CPU."""
  hparams = mtf_transformer2_base()
  hparams.batch_size = 2
  hparams.mesh_shape = ""
  hparams.d_model = 128
  hparams.layer_stack = transformer.LayerStack(
      [transformer_layers.SelfAttention(num_heads=4),
       transformer_layers.LocalSelfAttention(num_heads=4),
       moe.MoE1D(num_experts=4, hidden_size=512),
       moe.MoE2D(expert_x=4, expert_y=4, hidden_size=512),
       transformer_layers.DenseReluDense(hidden_size=512)])
  return hparams


@registry.register_hparams
def mtr2_lm_dense(sz):
  """Series of architectural experiments on language modeling.

  Larger models than the ones above.

  All models are trained on sequences of 1024 tokens.

  We assume infinite training data, so no dropout necessary.
  We process 2^36 tokens in training = 524288 steps at batch size 128

  TODO(noam): find a large enough dataset for these experiments.

  You can use languagemodel_wiki_noref_v32k_l1k, but this is too small,
  (1 epoch = ~46000 steps) so training will cover about 11 epochs.

  Note: configurations and code are likely to change without notice.

  Run on TPU 4x4 for 524288 steps unless otherwise indicated.

  Args:
    sz: an integer

  Returns:
    a hparams
  """
  n = 2 ** sz
  hparams = mtf_transformer2_base()
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
def mtr2_lm_dense_0():
  return mtr2_lm_dense(0)


@registry.register_hparams
def mtr2_lm_dense_1():
  return mtr2_lm_dense(1)


@registry.register_hparams
def mtr2_lm_dense_2():
  return mtr2_lm_dense(2)


@registry.register_hparams
def mtr2_v1():
  """Model incorporating mixture-of-experts, local and global attention.

  ~6B parameters

  32 experts in 3 hierarchichal moe layers.

  Returns:
    a hparams
  """
  hparams = mtr2_lm_dense(0)
  local_att = transformer_layers.LocalSelfAttention(
      num_heads=4, key_value_size=128)
  att = transformer_layers.SelfAttention(num_heads=4, key_value_size=128)
  drd = transformer_layers.DenseReluDense(hidden_size=2048)
  hmoe = moe.MoE2D(expert_x=8, expert_y=4, hidden_size=32768)
  hparams.layer_stack = transformer.LayerStack(
      ([local_att, local_att, drd,
        att, drd, local_att, local_att, hmoe] * 4)[:-1])
  hparams.mesh_shape = "b0:4;b1:8"
  hparams.layout = "outer_batch:b0;inner_batch:b1,expert_x:b1,expert_y:b0"
  hparams.outer_batch_size = 4
  return hparams
