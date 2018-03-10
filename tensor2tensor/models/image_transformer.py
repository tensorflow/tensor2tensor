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

"""image generation with transformer (attention).

encoder: [Self-Attention, Feed-forward] x n
decoder: [Self-Attention, Source-Target-Attention, Feed-forward] x n

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

# Dependency imports

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_image_attention as cia
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


@registry.register_model
class Imagetransformer(t2t_model.T2TModel):
  """Conditional image generation with attention. See file docstring."""

  def body(self, features):
    hparams = copy.copy(self._hparams)
    inputs = features["inputs"]
    targets = features["targets"]
    if not (tf.get_variable_scope().reuse or
            hparams.mode == tf.contrib.learn.ModeKeys.INFER):
      tf.summary.image("targets", targets, max_outputs=1)

    # Prepare decoder inputs and bias.
    decoder_input, rows, cols = cia.prepare_decoder(targets, hparams)
    # Add class label to decoder input.
    if not hparams.unconditional:
      decoder_input += tf.reshape(
          inputs,
          [common_layers.shape_list(targets)[0], 1, 1, hparams.hidden_size])
    decoder_output = cia.transformer_decoder_layers(
        decoder_input,
        None,
        hparams.num_decoder_layers or hparams.num_hidden_layers,
        hparams,
        attention_type=hparams.dec_attention_type,
        name="decoder")
    output = cia.create_output(decoder_output, rows, cols, targets, hparams)
    return output


@registry.register_model
class ImagetransformerMoe(t2t_model.T2TModel):
  """Conditional image generation with attention and MoE."""

  @property
  def use_body_sharded(self):
    return True

  def body_sharded(self, sharded_features):
    dp = self._data_parallelism
    hparams = copy.copy(self._hparams)
    inputs = sharded_features["inputs"]
    targets = sharded_features["targets"]

    # Determine attention type and padding from hparams.
    q_padding, kv_padding = "VALID", "VALID"
    if hparams.q_filter_width > 1:
      q_padding = "LEFT"
    if hparams.kv_filter_width > 1:
      kv_padding = "LEFT"

    # Prepare decoder inputs and bias.
    decoder_input, rows, cols = dp(cia.prepare_decoder_inputs,
                                   inputs, targets, hparams)

    # Run decoder.
    decoder_output, extra_loss = cia.transformer_layers_sharded(
        dp,
        self._ps_devices,
        decoder_input,
        hparams.num_hidden_layers,
        hparams,
        self_attention_bias=None,
        enc_output=None,
        attention_type=hparams.dec_attention_type,
        q_padding=q_padding,
        kv_padding=kv_padding,
        name="decoder")

    output = dp(cia.create_output, decoder_output, rows, cols, targets, hparams)
    return output, extra_loss


@registry.register_hparams
def image_transformer_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.hidden_size = 512
  hparams.batch_size = 1
  hparams.max_length = 3075
  hparams.dropout = 0.0
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 4000
  hparams.initializer_gain = 0.2
  hparams.num_hidden_layers = 6
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.label_smoothing = 0.0
  hparams.target_modality = "image:identity"
  hparams.norm_type = "layer"
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.add_hparam("filter_size", 512)  # Add new ones like this.

  # attention-related flags
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  hparams.add_hparam("ffn_layer", "conv_hidden_relu")
  # All hyperparameters ending in "dropout" are automatically set to 0.0
  # when not in training mode.
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("relu_dropout", 0.0)
  hparams.add_hparam("pos", "timing")  # timing, none
  hparams.add_hparam("nbr_decoder_problems", 1)
  hparams.add_hparam("num_output_layers", 3)
  hparams.add_hparam("block_size", 1)

  # dilated attention based flags
  hparams.add_hparam("gap_sizes", [2, 4, 8, 16, 32, 64, 2, 4, 8, 16, 32, 64])

  # image size related flags
  # assuming that the image has same height and width
  hparams.add_hparam("img_len", 32)
  hparams.add_hparam("num_channels", 3)
  # Local attention params
  hparams.add_hparam("local_and_global_att", False)
  hparams.add_hparam("block_length", 256)
  hparams.add_hparam("block_width", 128)
  hparams.add_hparam("num_encoder_layers", 4)
  hparams.add_hparam("num_decoder_layers", 12)
  hparams.sep_rgb_embed = False
  hparams.add_hparam("dec_attention_type", cia.AttentionType.LOCAL_1D)
  hparams.add_hparam("block_rastor_scan", False)

  # multipos attention params
  hparams.add_hparam("q_filter_width", 1)
  hparams.add_hparam("kv_filter_width", 1)

  hparams.add_hparam("unconditional", False)  # unconditional generation

  return hparams


@registry.register_hparams
def imagetransformer_base():
  hparams = image_transformer_base()
  return hparams


@registry.register_hparams
def imagetransformer_sep_channels():
  """separate rgb embeddings."""
  hparams = imagetransformer_base()
  hparams.num_heads = 4
  hparams.attention_key_channels = hparams.attention_value_channels = 0
  hparams.hidden_size = 256
  hparams.filter_size = 512
  hparams.num_hidden_layers = 6
  return hparams


@registry.register_hparams
def imagetransformer_sep_channels_8l():
  """separate rgb embeddings."""
  hparams = imagetransformer_base()
  hparams.num_heads = 4
  hparams.attention_key_channels = hparams.attention_value_channels = 0
  hparams.hidden_size = 256
  hparams.filter_size = 256
  hparams.num_hidden_layers = 8
  hparams.sampling_method = "random"
  return hparams


@registry.register_hparams
def imagetransformer_sep_channels_8l_multipos3():
  """separate rgb embeddings."""
  hparams = imagetransformer_sep_channels_8l()
  hparams.q_filter_width = 3
  hparams.kv_filter_width = 3
  return hparams


@registry.register_hparams
def imagetransformer_sep_output_channels_8l():
  """separate rgb embeddings."""
  hparams = imagetransformer_sep_channels_8l()
  hparams.sep_rgb_embed = True
  hparams.sampling_method = "random"
  return hparams


@registry.register_hparams
def imagetransformer_base_8l_8h_big_cond_dr03_dan():
  """big 1d model for conditional image generation.2.99 on cifar10."""
  hparams = imagetransformer_sep_channels_8l()
  hparams.block_width = 256
  hparams.block_length = 256
  hparams.hidden_size = 512
  hparams.num_heads = 8
  hparams.filter_size = 2048
  hparams.batch_size = 4
  hparams.max_length = 3075
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.num_decoder_layers = 8
  hparams.layer_prepostprocess_dropout = 0.3
  return hparams


@registry.register_hparams
def imagetransformer_base_10l_8h_big_uncond_dr03_dan_64():
  """big 1d model for unconditional generation on imagenet."""
  hparams = imagetransformer_base_10l_8h_big_cond_dr03_dan()
  hparams.unconditional = True
  hparams.max_length = 14000
  hparams.batch_size = 1
  hparams.img_len = 64
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams


@registry.register_hparams
def imagetransformer_base_8l_8h_big_cond_dr03_dan_128():
  hparams = imagetransformer_base_8l_8h_big_cond_dr03_dan()
  hparams.block_width = 128
  hparams.block_length = 128
  return hparams


@registry.register_hparams
def imagetransformer_base_10l_8h_big_cond_dr03_dan():
  """Best conditional Cifar10 gen param."""
  hparams = imagetransformer_base_8l_8h_big_cond_dr03_dan()
  hparams.num_decoder_layers = 10
  return hparams


@registry.register_hparams
def imagetransformer_base_10l_8h_big_uncond_dr03_dan():
  """Best unconditional Cifar10 gen param."""
  hparams = imagetransformer_base_10l_8h_big_cond_dr03_dan()
  hparams.num_decoder_layers = 10
  return hparams


@registry.register_hparams
def imagetransformer_base_8l_8h_big_cond_dr03_dan_dilated():
  """Dilated hparams."""
  hparams = imagetransformer_base_8l_8h_big_cond_dr03_dan()
  hparams.gap_sizes = [0, 16, 64, 0, 16, 64, 128, 0]
  hparams.dec_attention_type = cia.AttentionType.DILATED
  hparams.block_length = 128
  hparams.block_width = 128
  hparams.add_hparam("num_memory_blocks", 1)
  return hparams


@registry.register_hparams
def imagetransformer_base_8l_8h_big_cond_dr03_dan_dilated_b():
  """Dilated hparams."""
  hparams = imagetransformer_base_8l_8h_big_cond_dr03_dan_dilated()
  hparams.block_width = 64
  hparams.num_memory_blocks = 2
  return hparams


@registry.register_hparams
def imagetransformer_base_8l_8h_big_cond_dr03_dan_dilated_c():
  """Dilated hparams."""
  hparams = imagetransformer_base_8l_8h_big_cond_dr03_dan_dilated()
  hparams.block_width = 32
  hparams.num_memory_blocks = 4
  return hparams


@registry.register_hparams
def imagetransformer_base_8l_8h_big_cond_dr03_dan_dilated_d():
  """Dilated hparams."""
  hparams = imagetransformer_base_8l_8h_big_cond_dr03_dan_dilated()
  hparams.gap_sizes = [0, 16, 64, 16, 64, 128, 256, 0]
  return hparams


@registry.register_hparams
def imagetransformer_base_12l_8h_big():
  """big 1d model for conditional image generation."""
  hparams = imagetransformer_sep_channels_8l_8h()
  hparams.filter_size = 1024
  hparams.num_decoder_layers = 12
  hparams.batch_size = 1
  hparams.hidden_size = 512
  hparams.learning_rate_warmup_steps = 4000
  hparams.sampling_method = "random"
  hparams.beam_size = 1
  hparams.block_width = 256
  return hparams


@registry.register_hparams
def imagetransformer1d_base_8l_64by64():
  """hparams fo 12 layer big 1d model for imagenet 64x64."""
  hparams = image_transformer_base()
  hparams.num_heads = 8
  hparams.hidden_size = 512
  hparams.filter_size = 2048
  hparams.num_decoder_layers = 8
  hparams.batch_size = 1
  hparams.block_length = 512
  hparams.block_width = 768
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.max_length = 14000
  hparams.unconditional = int(False)
  return hparams


@registry.register_hparams
def imagetransformer1d_base_12l_64by64():
  """hparams fo 12 layer big 1d model for imagenet 64x64."""
  hparams = image_transformer_base()
  hparams.num_heads = 8
  hparams.hidden_size = 512
  hparams.filter_size = 2048
  hparams.num_decoder_layers = 12
  hparams.batch_size = 1
  hparams.block_length = 512
  hparams.block_width = 768
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.max_length = 14000
  hparams.unconditional = int(False)
  return hparams


@registry.register_hparams
def imagetransformer_base_14l_8h_big():
  """big 1d model for conditional image generation."""
  hparams = imagetransformer_base_12l_8h_big()
  hparams.num_decoder_layers = 14
  return hparams


@registry.register_hparams
def imagetransformer_base_14l_8h_big_dr01():
  """big 1d model for conditional image generation."""
  hparams = imagetransformer_base_14l_8h_big()
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams


@registry.register_hparams
def imagetransformer_base_12l_8h_big_uncond():
  """big 1d model for conditional image generation."""
  hparams = imagetransformer_base_12l_8h_big()
  hparams.unconditional = True
  return hparams


@registry.register_hparams
def imagetransformer_base_14l_8h_big_uncond():
  """big 1d model for conditional image generation."""
  hparams = imagetransformer_base_12l_8h_big_uncond()
  hparams.num_decoder_layers = 14
  return hparams


@registry.register_hparams
def imagetransformer_base_14l_8h_big_uncond_dr01():
  """big 1d model for conditional image generation."""
  hparams = imagetransformer_base_14l_8h_big_uncond()
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams


@registry.register_hparams
def imagetransformer_sep_channels_12l_16h_imagenet_large():
  """separate rgb embeddings."""
  hparams = imagetransformer_sep_channels_8l_8h()
  hparams.num_hidden_layers = 12
  hparams.batch_size = 1
  hparams.filter_size = 2048
  hparams.num_heads = 16
  hparams.learning_rate_warmup_steps = 16000
  hparams.sampling_method = "random"
  hparams.learning_rate = 0.1
  return hparams


@registry.register_hparams
def imagetransformer_sep_channels_16l_16h_imgnet_lrg_loc():
  """separate rgb embeddings."""
  hparams = imagetransformer_sep_channels_12l_16h_imagenet_large()
  hparams.num_hidden_layers = 16
  hparams.local_attention = True
  hparams.batch_size = 1
  hparams.block_length = 256
  return hparams


@registry.register_hparams
def imagetransformer_sep_channels_16l_16h_imgnet_lrg_loc_128():
  """separate rgb embeddings."""
  hparams = imagetransformer_sep_channels_12l_16h_imagenet_large()
  hparams.num_hidden_layers = 16
  hparams.local_attention = True
  hparams.batch_size = 1
  hparams.block_length = 128
  return hparams


@registry.register_hparams
def imagetransformer_sep_output_channels_8l_local_and_global_att():
  """separate rgb embeddings."""
  hparams = imagetransformer_sep_channels_8l()
  hparams.sep_rgb_embed = True
  hparams.sampling_method = "random"
  hparams.local_and_global_att = True
  return hparams


@registry.register_hparams
def imagetransformer_base_10l_16h_big_uncond_dr01_imgnet():
  """big 1d model for conditional image generation."""
  hparams = imagetransformer_base_14l_8h_big_uncond_dr01()
  # num_hidden_layers
  hparams.num_decoder_layers = 10
  hparams.num_heads = 16
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  hparams.batch_size = 1
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams


@registry.register_hparams
def imagetransformer_base_10l_16h_big_dr01_imgnet():
  """big 1d model for conditional image generation."""
  hparams = imagetransformer_base_14l_8h_big_uncond_dr01()
  # num_hidden_layers
  hparams.num_decoder_layers = 10
  hparams.num_heads = 16
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  hparams.batch_size = 1
  hparams.unconditional = False
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams


@registry.register_hparams
def imagetransformer_sep_channels_8l_8h():
  """separate rgb embeddings."""
  hparams = imagetransformer_base()
  hparams.num_heads = 8
  hparams.batch_size = 1
  hparams.attention_key_channels = hparams.attention_value_channels = 0
  hparams.hidden_size = 512
  hparams.filter_size = 512
  hparams.num_hidden_layers = 8
  hparams.sampling_method = "random"
  return hparams


@registry.register_hparams
def imagetransformer_sep_channels_10l_8h():
  """separate rgb embeddings."""
  hparams = imagetransformer_sep_channels_8l_8h()
  hparams.num_hidden_layers = 8
  hparams.learning_rate_warmup_steps = 16000
  hparams.sampling_method = "random"
  return hparams


@registry.register_hparams
def imagetransformer_sep_channels_12l_8h():
  """separate rgb embeddings."""
  hparams = imagetransformer_sep_channels_8l_8h()
  hparams.num_hidden_layers = 12
  hparams.batch_size = 2
  hparams.learning_rate_warmup_steps = 16000
  hparams.sampling_method = "random"
  return hparams


@registry.register_hparams
def imagetransformer_sep_channels_12l_8h_nda():
  """separate rgb embeddings."""
  hparams = imagetransformer_sep_channels_8l_8h()
  hparams.num_hidden_layers = 12
  hparams.batch_size = 2
  hparams.learning_rate_warmup_steps = 16000
  hparams.sampling_method = "random"
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  return hparams


@registry.register_hparams
def imagetransformer_sep_channels_12l_8h_4k():
  """separate rgb embeddings."""
  hparams = imagetransformer_sep_channels_8l_8h()
  hparams.num_hidden_layers = 12
  hparams.batch_size = 2
  hparams.learning_rate_warmup_steps = 4000
  hparams.sampling_method = "random"
  return hparams


@registry.register_hparams
def imagetransformer_sep_channels_12l_8h_sep_rgb():
  """separate rgb embeddings."""
  hparams = imagetransformer_sep_channels_8l_8h()
  hparams.num_hidden_layers = 12
  hparams.batch_size = 2
  hparams.learning_rate_warmup_steps = 16000
  hparams.sep_rgb_embed = True
  hparams.sampling_method = "random"
  return hparams


@registry.register_hparams
def imagetransformer_sep_channels_8l_8h_local_and_global_att():
  """separate rgb embeddings."""
  hparams = imagetransformer_sep_channels_8l_8h()
  hparams.num_heads = 8
  hparams.batch_size = 1
  hparams.attention_key_channels = hparams.attention_value_channels = 0
  hparams.hidden_size = 256
  hparams.filter_size = 256
  hparams.num_hidden_layers = 4
  hparams.sampling_method = "random"
  hparams.local_and_global_att = True
  return hparams


@registry.register_hparams
def imagetransformer_sep_channels_8l_self_att_ffn():
  """separate rgb embeddings."""
  hparams = imagetransformer_sep_channels_8l()
  hparams.num_parts = 4
  hparams.ffn_layer = "self_attention_ffn"
  hparams.share_kv = True
  return hparams


@registry.register_hparams
def imagetransformer_sep_channels_8l_glu_ffn():
  """separate rgb embeddings."""
  hparams = imagetransformer_sep_channels_8l()
  hparams.ffn_layer = "glu_ffn"
  return hparams


@registry.register_hparams
def imagetransformer_bas8l_8h_big_uncond_dr03_imgnet():
  """big 1d model for conditional image generation."""
  hparams = imagetransformer_base_14l_8h_big_uncond_dr01()
  # num_hidden_layers
  hparams.num_decoder_layers = 8
  hparams.num_heads = 8
  hparams.hidden_size = 512
  hparams.filter_size = 2048
  hparams.layer_prepostprocess_dropout = 0.3
  return hparams


@registry.register_hparams
def imagetransformer_tiny():
  hparams = imagetransformer_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 64
  hparams.batch_size = 1
  return hparams


@registry.register_hparams
def imagetransformer_tiny_tpu():
  hparams = imagetransformer_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 16
  hparams.batch_size = 2
  hparams.num_heads = 2
  return hparams


@registry.register_hparams
def imagetransformer_base_10l_16h_big_dr01_moe_imgnet():
  """big 1d model for conditional image generation."""
  hparams = imagetransformer_base_10l_16h_big_dr01_imgnet()
  hparams.initializer = "orthogonal"
  hparams.learning_rate_warmup_steps = 16000
  hparams.add_hparam("moe_layers_decoder", "2,7")  # Which layer is MoE.
  hparams.moe_hidden_sizes = "4096"  # Hidden layer sizes (comma-separated).
  hparams.moe_num_experts = 64  # Number of experts in each MoE layer.
  hparams.moe_k = 4  # How many experts to use per batch element (try 2 or 4).
  hparams.moe_loss_coef = 3e-2  # MoE loss coefficient (1e-2 is usually ok).
  hparams.scheduled_sampling_prob = 0.1
  hparams.scheduled_sampling_warmup_steps = 200000
  return hparams


@registry.register_hparams
def imagetransformer_moe_tiny():
  """Set of hyperparameters for a very small imagetransformer with MoE."""
  hparams = imagetransformer_tiny()
  hparams.hidden_size = 64
  hparams.batch_size = 1
  hparams.num_hidden_layers = 3
  hparams.dec_attention_type = cia.AttentionType.MOE_LOCAL_1D
  hparams.add_hparam("moe_layers_decoder", "1")  # Which layer is MoE.
  hparams.moe_hidden_sizes = "1024"  # Hidden layer sizes (comma-separated).
  hparams.moe_num_experts = 16  # Number of experts in each MoE layer.
  hparams.moe_k = 2  # How many experts to use per batch element (try 2 or 4).
  hparams.moe_loss_coef = 1e-2  # MoE loss coefficient (1e-2 is usually ok).
  return hparams


def update_hparams_for_tpu(hparams):
  hparams.use_pad_remover = False  # where op not supported
  hparams.optimizer = "TrueAdam"
  hparams.batch_size = 4


@registry.register_hparams
def imagetransformer_base_tpu():
  hparams = imagetransformer_base()
  update_hparams_for_tpu(hparams)
  hparams.batch_size = 4
  hparams.num_heads = 4   # heads are expensive on tpu
  hparams.hidden_size = 256
  hparams.filter_size = 512
  hparams.num_hidden_layers = 8
  hparams.sampling_method = "random"
  return hparams


@registry.register_hparams
def imagetransformer_sep_channels_8l_tpu():
  """Hparams for training imagetransformer on tpu."""
  hparams = imagetransformer_sep_channels_8l()
  update_hparams_for_tpu(hparams)
  hparams.batch_size = 4
  hparams.num_heads = 4   # heads are expensive on tpu
  hparams.shared_embedding_and_softmax_weights = False
  return hparams


@registry.register_hparams
def imagetransformer_bas8l_8h_big_uncond_dr03_imgnet_tpu():
  hparams = imagetransformer_bas8l_8h_big_uncond_dr03_imgnet()
  update_hparams_for_tpu(hparams)
  hparams.batch_size = 1
  hparams.num_heads = 8   # heads are expensive on tpu
  return hparams
