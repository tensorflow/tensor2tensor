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
import mesh_tensorflow as mtf

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import mtf_model
from tensor2tensor.utils import registry
import tensorflow as tf


@registry.register_model
class MtfImageTransformer(mtf_model.MtfModel):
  """Image Transformer in mesh_tensorflow."""

  @property
  def inputs_vocab_dim(self):
    assert self.has_input
    return mtf.Dimension("inputs_vocab", self._hparams.num_classes)

  @property
  def targets_vocab_dim(self):
    vocab_size = self._problem_hparams.vocab_size["targets"]
    if hasattr(self._hparams, "vocab_divisor"):
      vocab_size += (-vocab_size) % self._hparams.vocab_divisor
    return mtf.Dimension("vocab", vocab_size)

  @property
  def outputs_vocab_dim(self):
    return mtf.Dimension("output_vocab", 256)

  @property
  def pos_dim(self):
    return mtf.Dimension("pos", self._hparams.img_len)

  @property
  def rows_dim(self):
    return mtf.Dimension("rows", self._hparams.img_len)

  @property
  def cols_dim(self):
    return mtf.Dimension(
        "cols", self._hparams.img_len*self._hparams.num_channels)

  @property
  def orig_cols_dim(self):
    return mtf.Dimension("orig_cols", self._hparams.img_len)

  @property
  def channels_dim(self):
    return mtf.Dimension("channels", self._hparams.num_channels)

  @property
  def model_dim(self):
    return mtf.Dimension("d_model", self._hparams.hidden_size)

  @property
  def max_length_dim(self):
    return mtf.Dimension(
        "max_length",
        self._hparams.img_len*self._hparams.img_len*self._hparams.num_channels)

  @property
  def length_dim(self):
    return mtf.Dimension(
        "length",
        self._hparams.img_len*self._hparams.img_len*self._hparams.num_channels)

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
  def activation_type(self):
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

  def create_positional_emb_2d(self, targets):
    """Learned 2d positional embedding for images."""
    mesh = targets.mesh

    positional_emb_rows_var = mtf.get_variable(
        mesh, "positional_emb_rows",
        mtf.Shape([self.pos_dim, self.model_dim]),
        initializer=tf.random_normal_initializer(),
        activation_dtype=self.activation_type)
    positional_emb_cols_var = mtf.get_variable(
        mesh, "positional_emb_cols",
        mtf.Shape([self.pos_dim, self.model_dim]),
        initializer=tf.random_normal_initializer(),
        activation_dtype=self.activation_type)

    targets_position_x = mtf.range(mesh, self.rows_dim, dtype=tf.int32)
    targets_position_y = mtf.range(mesh, self.cols_dim, dtype=tf.int32)
    position_x = mtf.broadcast(
        mtf.gather(positional_emb_rows_var, targets_position_x,
                   self.pos_dim),
        mtf.Shape([self.rows_dim, self.cols_dim, self.model_dim]))

    position_y = mtf.broadcast(
        mtf.gather(positional_emb_cols_var, targets_position_y,
                   self.pos_dim),
        mtf.Shape([self.rows_dim, self.cols_dim, self.model_dim]))
    return position_x + position_y

  def mtf_model_fn(self, features, mesh):
    features = copy.copy(features)
    tf.logging.info("features = %s" % features)
    hparams = self._hparams
    activation_dtype = self.activation_type

    # We assume fixed vocab size for targets
    targets = tf.to_int32(features["targets"])

    # Image preprocessing, reshape into a 1D sequence and shift right.
    length = hparams.img_len*hparams.img_len*hparams.num_channels
    targets = tf.reshape(targets, [hparams.batch_size, length])
    shifted_targets = common_layers.shift_right_2d(targets)

    # Declare all the dimensions
    batch_dim = mtf.Dimension("batch", hparams.batch_size)

    def import_to_batch_by_length(x, name):
      return mtf.import_tf_tensor(
          mesh, x, mtf.Shape([batch_dim, self.length_dim]), name=name)

    targets = import_to_batch_by_length(targets, "targets")
    shifted_targets = import_to_batch_by_length(
        shifted_targets, "shifted_targets")

    extra_losses = []

    # Create targets content and position embeddings.
    # Create embedding var for targets and positions and do a gather.
    targets_embedding_var = mtf.get_variable(
        mesh, "targets_embedding",
        mtf.Shape([self.targets_vocab_dim, self.model_dim]),
        initializer=tf.random_normal_initializer(),
        activation_dtype=activation_dtype)

    x = mtf.gather(targets_embedding_var,
                   shifted_targets, self.targets_vocab_dim)

    # Add positional embeddings
    x += mtf.reshape(self.create_positional_emb_2d(targets),
                     [self.length_dim, self.model_dim])

    # If conditional and input is given, add the input embedding to the target.
    # TODO(nikip): Verify conditional.
    if self.has_input and not hparams.unconditional:
      inputs = tf.squeeze(tf.to_int32(features["inputs"]), [2, 3])
      inputs = import_to_batch_by_length(inputs, "inputs")

      # Input embeddings
      inputs_embedding_var = mtf.layers.embedding(
          mesh, "input_embedding",
          mtf.Shape([self.inputs_vocab_dim, self.model_dim]),
          activation_dtype=activation_dtype)
      inputs_emb = mtf.gather(
          inputs_embedding_var, inputs, self.inputs_vocab_dim)
      x += inputs_emb

    # Image Transformer Decoder
    # [ self attention - ffn - residual + dropout] x n
    if hparams.attention_type == "local1d_spatial":
      decoder_output = local_attention1d_spatial_decoder(
          x, self.kv_dim, self.heads_dim, self.feedforward_dim, hparams)
    elif hparams.attention_type == "local2d_spatial":
      decoder_output = local_attention2d_spatial_decoder(
          x, self.kv_dim, self.heads_dim, self.feedforward_dim, hparams)
    elif hparams.attention_type == "local1d":
      decoder_output = local_attention1d_masked_decoder(
          x, self.kv_dim, self.heads_dim, self.feedforward_dim, hparams)
    else:
      raise ValueError("Invalid attention type.")

    # Calculate the logits and loss.
    logits = mtf.layers.dense(
        decoder_output, self.outputs_vocab_dim, name="logits")
    # Need a reshape for logits
    logits = mtf.reshape(
        logits, mtf.Shape([batch_dim, self.length_dim, self.outputs_vocab_dim]))
    soft_targets = mtf.one_hot(
        targets, self.outputs_vocab_dim, dtype=activation_dtype)
    loss = mtf.layers.softmax_cross_entropy_with_logits(
        logits, soft_targets, self.outputs_vocab_dim)
    loss = mtf.reduce_mean(loss)
    for l in extra_losses:
      loss += l

    # Reshape logits to original target shape.
    logits = mtf.reshape(
        logits,
        mtf.Shape([batch_dim, self.rows_dim, self.orig_cols_dim,
                   self.channels_dim, self.outputs_vocab_dim]))

    return logits, loss


def layer_prepostprocess_dropout(x, hparams):
  batch_dim = x.shape.dims[0]
  model_dim = x.shape.dims[-1]
  return mtf.dropout(
      x,
      keep_prob=1.0 - hparams.layer_prepostprocess_dropout,
      noise_shape=mtf.Shape([batch_dim, model_dim]))


def local_attention1d_spatial_decoder(x, kv_dim, heads_dim,
                                      feedforward_dim, hparams):
  """Image Transformer decoder with local1D spatial layers."""
  batch_dim, length_dim, model_dim = x.shape.dims
  blocks_w_dim = mtf.Dimension("blocksw", hparams.block_length)
  num_w_blocks_dim = mtf.Dimension("num_wblocks",
                                   length_dim.size // blocks_w_dim.size)
  x = mtf.reshape(
      x, mtf.Shape([batch_dim, num_w_blocks_dim, blocks_w_dim, model_dim]))
  # [ self attention - ffn - residual + dropout] x n
  for layer in range(hparams.num_decoder_layers):
    layer_name = "decoder_layer_%d" % layer
    with tf.variable_scope(layer_name):
      # Self attention layer
      x += layer_prepostprocess_dropout(
          mtf.layers.local_self_attention_spatial_blocks(
              mtf.layers.layer_norm(x, model_dim, name="layer_norm_att"),
              kv_dim,
              heads_dim,
              memory_w_dim=blocks_w_dim,
              mask_right=True,
              name="self_att"), hparams)
      # ffn layer
      x += layer_prepostprocess_dropout(
          mtf.layers.dense_relu_dense(
              mtf.layers.layer_norm(x, model_dim, name="layer_norm_ffn"),
              feedforward_dim,
              hparams.dropout,
              dropout_broadcast_dims=[length_dim]), hparams)

  output = mtf.layers.layer_norm(x, model_dim, name="final_layer_norm")
  return output


def local_attention2d_spatial_decoder(x, kv_dim, heads_dim,
                                      feedforward_dim, hparams):
  """Image Transformer decoder with local2D spatial layers."""
  batch_dim, length_dim, model_dim = x.shape.dims
  blocks_h_dim = mtf.Dimension("blocksh", hparams.block_height)
  blocks_w_dim = mtf.Dimension("blocksw", hparams.block_width)
  num_h_blocks_dim = mtf.Dimension("num_h_blocks",
                                   hparams.img_len // hparams.block_height)
  num_w_blocks_dim = mtf.Dimension(
      "num_w_blocks",
      hparams.img_len * hparams.num_channels // hparams.block_width)
  x = mtf.transpose(
      mtf.reshape(
          x,
          mtf.Shape([
              batch_dim, num_h_blocks_dim, blocks_h_dim,
              num_w_blocks_dim, blocks_w_dim, model_dim
          ])),
      mtf.Shape([
          batch_dim, num_h_blocks_dim, num_w_blocks_dim,
          blocks_h_dim, blocks_w_dim, model_dim
      ]))
  # Image Transformer Decoder
  # [ self attention - ffn - residual + dropout] x n
  for layer in range(hparams.num_decoder_layers):
    layer_name = "decoder_layer_%d" % layer
    with tf.variable_scope(layer_name):
      # Self attention layer
      x += layer_prepostprocess_dropout(
          mtf.layers.local_2d_self_attention_spatial_blocks(
              mtf.layers.layer_norm(x, model_dim, name="layer_norm_att"),
              kv_dim,
              heads_dim,
              memory_h_dim=num_h_blocks_dim,
              memory_w_dim=num_w_blocks_dim,
              name="self_att"), hparams)
      # ffn layer
      x += layer_prepostprocess_dropout(
          mtf.layers.dense_relu_dense(
              mtf.layers.layer_norm(x, model_dim, name="layer_norm_ffn"),
              feedforward_dim,
              hparams.dropout,
              dropout_broadcast_dims=[length_dim]), hparams)

  output = mtf.layers.layer_norm(x, model_dim, name="final_layer_norm")
  return output


def local_attention1d_masked_decoder(x, kv_dim, heads_dim,
                                     feedforward_dim, hparams):
  """Image Transformer decoder with local1D masked layers."""
  print(x)
  _, length_dim, model_dim = x.shape.dims
  for layer in range(hparams.num_decoder_layers):
    layer_name = "decoder_layer_%d" % layer
    with tf.variable_scope(layer_name):
      # Self attention layer
      length_per_split = mtf.tensor_dim_to_size_per_split(
          hparams.layout, hparams.mesh_shape, length_dim)
      x += layer_prepostprocess_dropout(
          mtf.layers.masked_local_attention_1d(
              mtf.layers.layer_norm(x, model_dim, name="layer_norm_att"),
              kv_dim,
              heads_dim,
              window_size=hparams.block_length,
              length_per_split=length_per_split,
              name="self_att"), hparams)
      # ffn layer
      x += layer_prepostprocess_dropout(
          mtf.layers.dense_relu_dense(
              mtf.layers.layer_norm(x, model_dim, name="layer_norm_ffn"),
              feedforward_dim,
              hparams.dropout,
              dropout_broadcast_dims=[length_dim]), hparams)

  output = mtf.layers.layer_norm(x, model_dim, name="final_layer_norm")
  return output


@registry.register_hparams
def mtf_image_transformer_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.no_data_parallelism = True
  hparams.use_fixed_batch_size = True
  hparams.batch_size = 1
  hparams.max_length = 3072
  hparams.hidden_size = 256
  hparams.label_smoothing = 0.0
  # 8-way model-parallelism
  hparams.add_hparam("mesh_shape", "batch:8")
  hparams.add_hparam("layout", "batch:batch")
  hparams.add_hparam("mtf_mode", True)
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("filter_size", 1024)
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
  hparams.add_hparam("d_kv", 64)
  hparams.add_hparam("d_ff", 2048)

  # Image related hparams
  hparams.add_hparam("img_len", 32)
  hparams.add_hparam("num_channels", 3)
  hparams.add_hparam("unconditional", True)

  # Local Attention related params
  hparams.add_hparam("block_length", 128)
  hparams.add_hparam("block_height", 16)
  hparams.add_hparam("block_width", 16)
  hparams.add_hparam("attention_type", "local1d")
  return hparams


@registry.register_hparams
def mtf_image_transformer_tiny():
  """Catch bugs locally..."""
  hparams = mtf_image_transformer_base()
  hparams.hidden_size = 128
  hparams.d_ff = 256
  hparams.batch_size = 4
  hparams.num_encoder_layers = 1
  hparams.num_decoder_layers = 4
  hparams.num_heads = 4
  hparams.attention_key_size = 128
  hparams.attention_value_size = 128
  hparams.block_length = 32
  # data parallelism and model-parallelism
  hparams.mesh_shape = "batch:2"
  hparams.layout = "batch:batch"
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
def mtf_image_transformer_tiny_spatial1d():
  """Small single parameters."""
  hparams = mtf_image_transformer_tiny()
  hparams.num_decoder_layers = 6
  hparams.filter_size = 128
  hparams.block_height = 8
  hparams.block_width = 8
  hparams.attention_type = "local1d_spatial"
  hparams.mesh_shape = ""
  hparams.layout = ""
  return hparams


@registry.register_hparams
def mtf_image_transformer_tiny_spatial2d():
  """Small single parameters."""
  hparams = mtf_image_transformer_tiny()
  hparams.num_decoder_layers = 6
  hparams.filter_size = 128
  hparams.block_height = 8
  hparams.block_width = 8
  hparams.attention_type = "local2d_spatial"
  hparams.mesh_shape = "b1:2,b2:2"
  hparams.layout = "num_h_blocks:b1,num_wblocks:b2"
  return hparams


@registry.register_hparams
def mtf_image_transformer_base_cifar():
  """Data parallel CIFAR parameters."""
  hparams = mtf_image_transformer_base()
  hparams.mesh_shape = "batch:8"
  hparams.layout = "batch:batch"
  hparams.learning_rate_decay_steps = 13600  # one epoch
  hparams.batch_size = 32
  hparams.num_heads = 4
  hparams.num_decoder_layers = 12
  hparams.block_length = 256
  hparams.hidden_size = 512
  hparams.d_ff = 2048
  hparams.learning_rate = 0.5
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.layer_prepostprocess_dropout = 0.3
  hparams.unconditional = True
  return hparams


@registry.register_hparams
def mtf_image_transformer_cifar_4x():
  """Data parallel CIFAR parameters."""
  hparams = mtf_image_transformer_base_cifar()
  hparams.mesh_shape = "batch:32"
  hparams.layout = "batch:batch"
  hparams.batch_size = 128
  return hparams


@registry.register_hparams
def mtf_image_transformer_cifar_mp_4x():
  """Data parallel CIFAR parameters."""
  hparams = mtf_image_transformer_base_cifar()
  hparams.mesh_shape = "model:4;batch:8"
  hparams.layout = "batch:batch;d_ff:model;heads:model"
  hparams.batch_size = 32
  hparams.num_heads = 8
  hparams.d_ff = 8192
  return hparams


@registry.register_hparams
def mtf_image_transformer_base_imagenet():
  """Data parallel CIFAR parameters."""
  hparams = mtf_image_transformer_base_cifar()
  hparams.mesh_shape = "batch:32"
  hparams.layout = "batch:batch"
  hparams.batch_size = 128
  hparams.d_ff = 2048
  hparams.hidden_size = 512
  hparams.num_decoder_layers = 12
  hparams.learning_rate = 0.5
  hparams.learning_rate_warmup_steps = 31250
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.unconditional = True
  return hparams


@registry.register_hparams
def mtf_image_transformer_base_imagenet_mp():
  """Model parallel ImageNet parameters."""
  hparams = mtf_image_transformer_base_imagenet()
  hparams.mesh_shape = "model:4;batch:8"
  hparams.layout = "batch:batch;d_ff:model;heads:model"
  hparams.batch_size = 32
  hparams.num_heads = 8
  hparams.d_ff = 8192
  hparams.learning_rate_warmup_steps = 31250
  hparams.unconditional = True
  return hparams


@registry.register_hparams
def mtf_image_transformer_base_imagenet_mp128():
  """Model parallel ImageNet parameters."""
  hparams = mtf_image_transformer_base_imagenet()
  hparams.mesh_shape = "model:8;batch:4"
  hparams.layout = "batch:batch;d_ff:model;heads:model"
  hparams.batch_size = 8
  hparams.img_len = 128
  hparams.block_length = 128
  hparams.num_heads = 8
  hparams.num_decoder_layers = 4
  hparams.d_ff = 4096
  hparams.learning_rate_warmup_steps = 31250
  hparams.unconditional = True
  hparams.max_length = 256*256*3
  return hparams


@registry.register_hparams
def mtf_image_transformer_base_imagenet_mp_sp():
  """Model parallel ImageNet parameters."""
  hparams = mtf_image_transformer_base_imagenet_mp128()
  hparams.mesh_shape = "model:8;batch:4"
  hparams.layout = "batch:batch;d_ff:model;num_wblocks:model"
  hparams.batch_size = 8
  hparams.img_len = 128
  hparams.block_length = 128
  hparams.attention_type = "local1d_spatial"
  return hparams


@registry.register_hparams
def mtf_image_transformer_base_imagenet_mp64():
  """Model parallel ImageNet parameters."""
  hparams = mtf_image_transformer_base_imagenet()
  hparams.mesh_shape = "model:8;batch:4"
  hparams.layout = "batch:batch;d_ff:model;heads:model"
  hparams.batch_size = 8
  hparams.img_len = 64
  hparams.num_decoder_layers = 8
  return hparams


@registry.register_hparams
def mtf_image_transformer_tiny_8gpu():
  hparams = mtf_image_transformer_tiny()
  hparams.mesh_shape = "all:8"
  hparams.layout = "vocab:all;filter_size:all;heads:all"
  return hparams


@registry.register_hparams
def mtf_image_transformer_length_sharded():
  hparams = mtf_image_transformer_tiny()
  hparams.mesh_shape = "all:2"
  hparams.layout = "length:all"
  return hparams
