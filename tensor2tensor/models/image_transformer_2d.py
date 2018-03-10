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
class Imagetransformer2d(t2t_model.T2TModel):
  """Conditional image generation with attention. See file docstring."""

  def body(self, features):
    hparams = copy.copy(self._hparams)
    inputs = features["inputs"]
    targets = features["targets"]
    targets_shape = common_layers.shape_list(targets)
    if not (tf.get_variable_scope().reuse or
            hparams.mode == tf.contrib.learn.ModeKeys.INFER):
      tf.summary.image("targets", targets, max_outputs=1)

    decoder_input, rows, cols = cia.prepare_decoder(
        targets, hparams)
    # Add class label to decoder input.
    if not hparams.unconditional:
      decoder_input += tf.reshape(inputs,
                                  [targets_shape[0], 1, 1, hparams.hidden_size])

    decoder_output = cia.transformer_decoder_layers(
        decoder_input, None,
        hparams.num_decoder_layers,
        hparams,
        attention_type=hparams.dec_attention_type,
        name="decoder")

    output = cia.create_output(decoder_output, rows, cols, targets, hparams)
    return output


@registry.register_model
class Img2imgTransformer(t2t_model.T2TModel):
  """Image 2 Image transformer net."""

  def body(self, features):
    hparams = copy.copy(self._hparams)
    targets = features["targets"]
    inputs = features["inputs"]
    if not (tf.get_variable_scope().reuse or
            hparams.mode == tf.contrib.learn.ModeKeys.INFER):
      tf.summary.image("inputs", inputs, max_outputs=1)
      tf.summary.image("targets", targets, max_outputs=1)

    encoder_input = cia.prepare_encoder(inputs, hparams)
    encoder_output = cia.transformer_encoder_layers(
        encoder_input,
        hparams.num_encoder_layers,
        hparams,
        attention_type=hparams.enc_attention_type,
        name="encoder")
    decoder_input, rows, cols = cia.prepare_decoder(
        targets, hparams)
    decoder_output = cia.transformer_decoder_layers(
        decoder_input,
        encoder_output,
        hparams.num_decoder_layers,
        hparams,
        attention_type=hparams.dec_attention_type,
        name="decoder")
    output = cia.create_output(decoder_output, rows, cols, targets, hparams)
    return output


@registry.register_hparams
def image_transformer2d_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.hidden_size = 512
  hparams.batch_size = 1
  hparams.max_length = 256
  hparams.dropout = 0.0
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 4000
  hparams.initializer_gain = 0.2
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

  # image size related flags
  # assuming that the image has same height and width
  hparams.add_hparam("img_len", 32)
  hparams.add_hparam("num_channels", 3)
  # Local attention params
  hparams.add_hparam("local_and_global_att", False)
  hparams.add_hparam("block_length", 256)
  hparams.add_hparam("block_width", 128)
  # Local 2D attention params
  hparams.add_hparam("query_shape", (16, 16))
  hparams.add_hparam("memory_flange", (16, 32))
  hparams.add_hparam("num_encoder_layers", 4)
  hparams.add_hparam("num_decoder_layers", 8)
  # attention type related params
  hparams.add_hparam("enc_attention_type", cia.AttentionType.GLOBAL)
  hparams.add_hparam("dec_attention_type", cia.AttentionType.LOCAL_2D)
  hparams.add_hparam("block_rastor_scan", False)

  # multipos attention params
  hparams.add_hparam("q_filter_width", 1)
  hparams.add_hparam("kv_filter_width", 1)

  hparams.add_hparam("unconditional", False)  # unconditional generation
  return hparams


@registry.register_hparams
def imagetransformer2d_base():
  hparams = image_transformer2d_base()
  hparams.dec_attention_type = cia.AttentionType.LOCAL_2D
  hparams.block_rastor_scan = True
  return hparams


@registry.register_hparams
def imagetransformer2d_base_8l_8_16():
  hparams = image_transformer2d_base()
  hparams.num_decoder_layers = 8
  hparams.batch_size = 1
  hparams.memory_flange = (8, 16)
  return hparams


@registry.register_hparams
def imagetransformer2d_base_8l_8_16_ls():
  hparams = image_transformer2d_base()
  hparams.num_decoder_layers = 8
  hparams.label_smoothing = 0.05
  hparams.batch_size = 1
  hparams.memory_flange = (8, 16)
  return hparams


@registry.register_hparams
def imagetransformer2d_base_8l_8_16_big():
  hparams = image_transformer2d_base()
  hparams.filter_size = 1024
  hparams.num_decoder_layers = 8
  hparams.batch_size = 1
  hparams.memory_flange = (8, 16)
  return hparams


@registry.register_hparams
def imagetransformer2d_base_12l_8_16_big():
  hparams = image_transformer2d_base()
  hparams.filter_size = 1024
  hparams.num_decoder_layers = 12
  hparams.batch_size = 1
  hparams.memory_flange = (8, 16)
  hparams.sampling_method = "random"
  hparams.beam_size = 1
  return hparams


@registry.register_hparams
def imagetransformer2d_base_8l_8_32_big():
  """hparams fo 8 layer big 2d model for cifar 10."""
  hparams = image_transformer2d_base()
  hparams.num_heads = 16
  hparams.hidden_size = 1024
  hparams.filter_size = 2048
  hparams.num_decoder_layers = 8
  hparams.batch_size = 1
  hparams.layer_prepostprocess_dropout = 0.3
  hparams.query_shape = (8, 16)
  hparams.memory_flange = (0, 32)
  hparams.unconditional = int(False)
  return hparams


@registry.register_hparams
def imagetransformer_base_10l_8h_big_uncond_dr03_dan_64_2d():
  """big 1d model for unconditional generation on imagenet."""
  hparams = image_transformer2d_base()
  hparams.unconditional = True
  hparams.hidden_size = 512
  hparams.batch_size = 1
  hparams.img_len = 64
  hparams.num_heads = 8
  hparams.filter_size = 2048
  hparams.batch_size = 1
  hparams.max_length = 3075
  hparams.max_length = 14000
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.dec_attention_type = cia.AttentionType.LOCAL_2D
  hparams.query_shape = (16, 16)
  hparams.memory_flange = (8, 8)
  return hparams


@registry.register_hparams
def imagetransformer2d_base_8l_8_64_64by64():
  """hparams fo 12 layer big 2d model for imagenet 64x64."""
  hparams = image_transformer2d_base()
  hparams.num_heads = 8
  hparams.hidden_size = 512
  hparams.filter_size = 2048
  hparams.num_decoder_layers = 8
  hparams.batch_size = 1
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.query_shape = (8, 64)
  hparams.memory_flange = (4, 32)
  hparams.unconditional = int(False)
  hparams.max_length = 14000
  return hparams


@registry.register_hparams
def imagetransformer2d_base_12l_8_64_64by64():
  """hparams fo 12 layer big 2d model for imagenet 64x64."""
  hparams = image_transformer2d_base()
  hparams.num_heads = 8
  hparams.hidden_size = 512
  hparams.filter_size = 2048
  hparams.num_decoder_layers = 12
  hparams.batch_size = 1
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.query_shape = (8, 64)
  hparams.memory_flange = (4, 32)
  hparams.unconditional = int(False)
  hparams.max_length = 14000
  return hparams


@registry.register_hparams
def imagetransformer2d_base_14l_8_16_big():
  hparams = image_transformer2d_base()
  hparams.filter_size = 1024
  hparams.num_decoder_layers = 14
  hparams.batch_size = 1
  hparams.memory_flange = (8, 16)
  return hparams


@registry.register_hparams
def imagetransformer2d_base_14l_8_16_big_uncond():
  hparams = imagetransformer2d_base_14l_8_16_big()
  hparams.unconditional = True
  return hparams


@registry.register_hparams
def imagetransformer2d_base_8l_8_16_big_16k():
  hparams = image_transformer2d_base()
  hparams.filter_size = 1024
  hparams.num_decoder_layers = 8
  hparams.batch_size = 1
  hparams.memory_flange = (8, 16)
  hparams.learning_rate_warmup_steps = 16000
  return hparams


@registry.register_hparams
def img2img_transformer2d_base():
  """Base params for img2img 2d attention."""
  hparams = image_transformer2d_base()
  # learning related flags
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  # This version seems to benefit from a higher learning rate.
  hparams.learning_rate = 0.2
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.learning_rate_warmup_steps = 12000
  hparams.filter_size = 2048
  hparams.num_encoder_layers = 4
  hparams.num_decoder_layers = 8
  hparams.dec_attention_type = cia.AttentionType.LOCAL_2D
  hparams.block_rastor_scan = True
  return hparams


@registry.register_hparams
def img2img_transformer2d_q1():
  hparams = img2img_transformer2d_base()
  hparams.batch_size = 2
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.query_shape = (16, 16)
  hparams.memory_flange = (16, 64)
  return hparams


@registry.register_hparams
def img2img_transformer2d_q2():
  hparams = img2img_transformer2d_q1()
  hparams.batch_size = 2
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.query_shape = (16, 16)
  hparams.memory_flange = (16, 32)
  return hparams


@registry.register_hparams
def img2img_transformer2d_q3():
  """Current best hparams for local 2d."""
  hparams = img2img_transformer2d_q1()
  hparams.batch_size = 2
  hparams.query_shape = (8, 16)
  hparams.memory_flange = (8, 32)
  return hparams


@registry.register_hparams
def img2img_transformer_base():
  """Base params for local1d attention."""
  hparams = image_transformer2d_base()
  # learning related flags
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  # This version seems to benefit from a higher learning rate.
  hparams.learning_rate = 0.2
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.learning_rate_warmup_steps = 12000
  hparams.filter_size = 2048
  hparams.num_encoder_layers = 4
  hparams.num_decoder_layers = 8
  hparams.block_length = 256
  hparams.block_width = 256
  hparams.dec_attention_type = cia.AttentionType.LOCAL_1D
  hparams.block_rastor_scan = False
  return hparams


@registry.register_hparams
def img2img_transformer_b1():
  hparams = img2img_transformer_base()
  hparams.batch_size = 2
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.block_length = 512
  return hparams


@registry.register_hparams
def img2img_transformer_b2():
  hparams = img2img_transformer_base()
  hparams.batch_size = 2
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.block_length = 256
  return hparams


@registry.register_hparams
def img2img_transformer_b3():
  """Current best hparams for local 1d."""
  hparams = img2img_transformer_base()
  hparams.batch_size = 2
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.block_length = 128
  hparams.sampling_temp = 0.9
  return hparams


@registry.register_hparams
def img2img_transformer_dilated():
  """Try dilated."""
  hparams = img2img_transformer_base()
  hparams.add_hparam("num_memory_blocks", 1)
  hparams.num_heads = 8
  hparams.attention_key_channels = hparams.attention_value_channels = 0
  hparams.hidden_size = 512
  hparams.filter_size = 2048
  hparams.num_decoder_layers = 8
  hparams.sampling_method = "random"
  hparams.gap_sizes = [0, 16, 64, 0, 16, 64, 128, 0]
  hparams.dec_attention_type = cia.AttentionType.DILATED
  hparams.img_len = 64
  hparams.block_length = 128
  hparams.block_width = 128
  return hparams


@registry.register_hparams
def imagetransformer2d_tiny():
  hparams = imagetransformer2d_base()
  hparams.num_decoder_layers = 2
  hparams.hidden_size = 64
  hparams.batch_size = 1
  return hparams


def update_hparams_for_tpu(hparams):
  hparams.use_pad_remover = False  # where op not supported
  hparams.optimizer = "TrueAdam"
  hparams.batch_size = 4


@registry.register_hparams
def img2img_transformer_base_tpu():
  """Hparams for training img2img_transformer on tpu."""
  hparams = img2img_transformer_base()
  update_hparams_for_tpu(hparams)
  hparams.batch_size = 2
  hparams.num_heads = 4   # heads are expensive on tpu
  hparams.num_decoder_layers = 8
  hparams.num_encoder_layers = 4
  hparams.shared_embedding_and_softmax_weights = False
  return hparams


@registry.register_hparams
def img2img_transformer_tiny_tpu():
  hparams = img2img_transformer_base_tpu()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 16
  hparams.batch_size = 2
  hparams.num_heads = 2
  return hparams


@registry.register_hparams
def img2img_transformer2d_n3():
  hparams = img2img_transformer2d_base()
  hparams.batch_size = 1
  hparams.num_encoder_layers = 4
  hparams.num_decoder_layers = 12
  hparams.query_shape = (16, 32)
  hparams.memory_flange = (16, 16)
  hparams.layer_prepostprocess_dropout = 0.0
  return hparams


@registry.register_hparams
def img2img_transformer2d_n31():
  """Set of hyperparameters."""
  hparams = img2img_transformer2d_base()
  hparams.batch_size = 1
  hparams.num_encoder_layers = 6
  hparams.num_decoder_layers = 12
  hparams.num_heads = 8
  hparams.query_shape = (16, 32)
  hparams.memory_flange = (16, 32)
  return hparams


@registry.register_hparams
def img2img_transformer2d_n24():
  hparams = img2img_transformer2d_base()
  hparams.batch_size = 1
  hparams.hidden_size = 1024
  hparams.filter_size = 2048
  hparams.layer_prepostprocess_dropout = 0.2
  hparams.num_decoder_layers = 8
  hparams.query_shape = (8, 16)
  hparams.memory_flange = (8, 32)
  return hparams


@registry.register_hparams
def img2img_transformer2d_n44():
  hparams = img2img_transformer2d_base()
  hparams.batch_size = 1
  hparams.num_decoder_layers = 8
  hparams.query_shape = (8, 16)
  hparams.memory_flange = (8, 32)
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams


@registry.register_hparams
def img2img_transformer2d_n103():
  """Best config for img2img."""
  hparams = img2img_transformer2d_base()
  hparams.batch_size = 1
  hparams.num_decoder_layers = 12
  hparams.num_encoder_layers = 6
  hparams.query_shape = (8, 32)
  hparams.memory_flange = (8, 64)
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams


@registry.register_hparams
def img2img_transformer2d_tiny():
  """Tiny params."""
  hparams = img2img_transformer2d_base()
  hparams.num_decoder_layers = 2
  hparams.hidden_size = 128
  hparams.batch_size = 4
  hparams.max_length = 128
  hparams.attention_key_channels = hparams.attention_value_channels = 0
  hparams.filter_size = 128
  hparams.num_heads = 4
  hparams.pos = "timing"
  hparams.img_len = 32
  return hparams


@registry.register_hparams
def img2img_transformer_tiny():
  """Tiny params."""
  hparams = img2img_transformer2d_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 128
  hparams.batch_size = 4
  hparams.max_length = 128
  hparams.attention_key_channels = hparams.attention_value_channels = 0
  hparams.filter_size = 128
  hparams.num_heads = 1
  hparams.pos = "timing"
  return hparams
