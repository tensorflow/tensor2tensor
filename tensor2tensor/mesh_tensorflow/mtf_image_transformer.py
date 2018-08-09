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
"""Image Transformer model with model and data parallelism using MTF.

Integration of Mesh tensorflow with Image Transformer to do model parallelism.
Currently, this supports unconditional image generation. Specify a particular
architecture layout in the hparams that specifies how different dimensions are
split or replicated along the mesh dimensions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.mesh_tensorflow import mesh_tensorflow as mtf
from tensor2tensor.mesh_tensorflow import mtf_layers
from tensor2tensor.mesh_tensorflow import mtf_model
from tensor2tensor.utils import registry
import tensorflow as tf


@registry.register_model
class MtfImageTransformer(mtf_model.MtfModel):
  """Transformer in mesh_tensorflow."""

  def set_activation_type(self):
    hparams = self._hparams
    if hparams.activation_dtype == "float32":
      activation_dtype = tf.float32
    elif hparams.activation_dtype == "float16":
      activation_dtype = tf.float16
    elif hparams.activation_dtype == "bfloat16":
      activation_dtype = tf.bfloat16
    else:
      raise ValueError(
          "unknown hparams.activation_dtype %s" % hparams.activation_dtype)
    return activation_dtype

  def mtf_model_fn(self, features, mesh):
    features = copy.copy(features)
    tf.logging.info("features = %s" % features)
    hparams = self._hparams
    activation_dtype = self.set_activation_type()

    # We assume fixed vocab size for targets
    targets_vocab_size = self._problem_hparams.target_modality._vocab_size  # pylint: disable=protected-access
    targets = tf.to_int32(features["targets"])

    # Image preprocessing, reshape into a 1D sequence and shift right.
    length = hparams.img_len*hparams.img_len*hparams.num_channels
    targets = tf.reshape(targets, [hparams.batch_size, length])
    shifted_targets = common_layers.shift_right_2d(targets)

    # Declare all the dimensions
    model_dim = mtf.Dimension("model", hparams.hidden_size)
    batch_dim = mtf.Dimension("batch", hparams.batch_size)
    length_dim = mtf.Dimension("length", length)
    filter_dim = mtf.Dimension("filter_size", hparams.filter_size)
    kv_channels = mtf.Dimension("kv_channels", hparams.d_kv)
    heads = mtf.Dimension("heads", hparams.num_heads)

    def import_to_batch_by_length(x, name):
      return mtf.import_tf_tensor(
          mesh, x, mtf.Shape([batch_dim, length_dim]), name=name)

    def layer_prepostprocess_dropout(x):
      return mtf.dropout(
          x, keep_prob=1.0 - hparams.layer_prepostprocess_dropout,
          noise_shape=mtf.Shape([batch_dim, model_dim]))

    targets = import_to_batch_by_length(targets, "targets")
    shifted_targets = import_to_batch_by_length(
        shifted_targets, "shifted_targets")

    extra_losses = []

    # TODO(nikip): Verify conditional.
    if self.has_input and not hparams.unconditional:
      vocab_size = hparams.num_classes
      inputs_vocab_dim = mtf.Dimension("vocab", vocab_size)
      inputs = tf.squeeze(tf.to_int32(features["inputs"]), [2, 3])
      inputs = import_to_batch_by_length(inputs, "inputs")

      # Input embeddings
      inputs, _ = mtf_layers.embedding(
          inputs, inputs_vocab_dim, model_dim,
          activation_dtype=activation_dtype,
          name="inputs_embedding")

    # Create targets content and position embeddings.
    targets_position = mtf.range(mesh, length_dim, dtype=tf.int32)
    targets_vocab_size = 256 * hparams.num_channels
    targets_vocab_dim = mtf.Dimension("vocab", targets_vocab_size)
    outputs_vocab_dim = mtf.Dimension("output_vocab", 256)

    # Create embedding var for targets and positions and do a gather.
    targets_embedding_var = mtf.get_variable(
        mesh, "targets_embedding",
        mtf.Shape([targets_vocab_dim, model_dim]),
        initializer=tf.random_normal_initializer(),
        activation_dtype=activation_dtype)

    positional_embedding_var = mtf.get_variable(
        mesh, "positional_embedding",
        mtf.Shape([targets_vocab_dim, model_dim]),
        initializer=tf.random_normal_initializer(),
        activation_dtype=activation_dtype)
    x = (mtf.gather(targets_embedding_var, shifted_targets, targets_vocab_dim) +
         mtf.gather(
             positional_embedding_var, targets_position, targets_vocab_dim))

    # Image Transformer Decoder
    # [ self attention - ffn - residual + dropout] x n
    for layer in range(hparams.num_decoder_layers):
      layer_name = "decoder_layer_%d" % layer
      with tf.variable_scope(layer_name):
        # Self attention layer
        x += layer_prepostprocess_dropout(
            mtf_layers.masked_local_attention_1d(
                mtf_layers.layer_norm(x, model_dim, name="layer_norm_self_att"),
                None,
                kv_channels,
                heads,
                block_length=hparams.block_length,
                name="self_att"))
        # ffn layer
        x += layer_prepostprocess_dropout(mtf_layers.dense_relu_dense(
            mtf_layers.layer_norm(x, model_dim, name="layer_norm_ffn"),
            filter_dim, hparams.dropout, dropout_broadcast_dims=[length_dim]))

    x = mtf_layers.layer_norm(x, model_dim, name="decoder_final_layer_norm")

    # Calculate the logits and loss.
    logits = mtf_layers.dense(x, outputs_vocab_dim, name="logits")
    soft_targets = mtf.one_hot(
        targets, outputs_vocab_dim, dtype=activation_dtype)
    loss = mtf_layers.softmax_cross_entropy_with_logits(
        logits, soft_targets, outputs_vocab_dim)

    loss = mtf.reduce_mean(loss)
    for l in extra_losses:
      loss += l
    return logits, loss


@registry.register_hparams
def mtf_image_transformer_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.no_data_parallelism = True
  hparams.use_fixed_batch_size = True
  hparams.batch_size = 1
  hparams.max_length = 256
  hparams.hidden_size = 256
  hparams.label_smoothing = 0.0
  # 8-way model-parallelism
  hparams.add_hparam("mesh_shape", "8")
  hparams.add_hparam("layout", "vocab:0;filter_size:0;heads:0")
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("filter_size", 512)
  hparams.add_hparam("num_encoder_layers", 0)
  hparams.add_hparam("num_decoder_layers", 6)
  hparams.add_hparam("attention_key_size", 256)
  hparams.add_hparam("attention_value_size", 256)
  # Share weights between input and target embeddings
  hparams.shared_embedding = True

  # mixture of experts hparams
  hparams.add_hparam("ffn_layer", "dense_relu_dense")
  hparams.add_hparam("moe_overhead_train", 1.0)
  hparams.add_hparam("moe_overhead_eval", 2.0)
  hparams.moe_num_experts = 16
  hparams.moe_loss_coef = 1e-3

  hparams.shared_embedding_and_softmax_weights = True
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_schedule = "rsqrt_decay"
  hparams.learning_rate_warmup_steps = 10000
  hparams.add_hparam("d_kv", 32)

  # Image related hparams
  hparams.add_hparam("img_len", 32)
  hparams.add_hparam("num_channels", 3)
  hparams.add_hparam("unconditional", True)
  hparams.add_hparam("block_length", 128)
  return hparams


@registry.register_hparams
def mtf_image_transformer_tiny():
  """Catch bugs locally..."""
  hparams = mtf_image_transformer_base()
  hparams.hidden_size = 128
  hparams.filter_size = 256
  hparams.batch_size = 4
  hparams.num_encoder_layers = 1
  hparams.num_decoder_layers = 1
  hparams.num_heads = 4
  hparams.attention_key_size = 128
  hparams.attention_value_size = 128
  # data parallelism and model-parallelism
  hparams.mesh_shape = "2.2"
  hparams.layout = "batch:0;filter_size:1"
  return hparams


@registry.register_hparams
def mtf_image_transformer_single():
  """Small single parameters."""
  hparams = mtf_image_transformer_tiny()
  hparams.mesh_shape = ""
  hparams.layout = ""
  hparams.hidden_size = 32
  hparams.filter_size = 32
  hparams.batch_size = 1
  hparams.num_encoder_layers = 1
  hparams.num_decoder_layers = 1
  hparams.num_heads = 2
  hparams.attention_key_size = 32
  hparams.attention_value_size = 32
  hparams.block_length = 16
  return hparams


@registry.register_hparams
def mtf_image_transformer_base_single():
  """Small single parameters."""
  hparams = mtf_image_transformer_base()
  hparams.num_decoder_layers = 6
  hparams.filter_size = 256
  hparams.block_length = 128
  hparams.mesh_shape = ""
  hparams.layout = ""
  return hparams


@registry.register_hparams
def mtf_image_transformer_tiny_moe():
  hparams = mtf_image_transformer_tiny()
  hparams.mesh_shape = "4"
  hparams.layout = "batch:0,experts:0"
  hparams.ffn_layer = "moe"
  return hparams


@registry.register_hparams
def mtf_image_transformer_tiny_8gpu():
  hparams = mtf_image_transformer_tiny()
  hparams.mesh_shape = "8"
  hparams.layout = "vocab:0;filter_size:0;heads:0"
  return hparams


@registry.register_hparams
def mtf_image_transformer_length_sharded():
  hparams = mtf_image_transformer_tiny()
  hparams.mesh_shape = "2"
  hparams.layout = "length:0"
  return hparams
