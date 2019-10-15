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

"""image generation with transformer (attention).

encoder: [Self-Attention, Feed-forward] x n
decoder: [Self-Attention, Source-Target-Attention, Feed-forward] x n

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_image_attention as cia
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


@registry.register_model
class Imagetransformer(t2t_model.T2TModel):
  """Conditional image generation with attention. See file docstring.

  The model admits either a Categorical or discretized mixture of logistic
  distributions (DMOL) as the likelihood. When using DMOL for training, double
  check that the evaluation metrics also use it.
  """

  def body(self, features):
    hparams = copy.copy(self._hparams)
    targets = features["targets"]
    if (hparams.likelihood == cia.DistributionType.DMOL and
        hparams.num_channels != 1):
      raise ValueError("When using DMOL for the likelihood, bottom function "
                       " must be identity and num_channels must be 1.")
    if (not tf.get_variable_scope().reuse and
        hparams.mode != tf.estimator.ModeKeys.PREDICT):
      tf.summary.image("targets", tf.to_float(targets), max_outputs=1)

    # Extra losses list if we want to use moe.
    losses = []
    # Prepare decoder inputs and bias.
    decoder_input, rows, cols = cia.prepare_decoder(targets, hparams)
    # Add class label to decoder input.
    if not hparams.unconditional:
      inputs = features["inputs"]
      decoder_input += tf.reshape(
          inputs,
          [common_layers.shape_list(targets)[0], 1, 1, hparams.hidden_size])
    decoder_output = cia.transformer_decoder_layers(
        decoder_input,
        None,
        hparams.num_decoder_layers or hparams.num_hidden_layers,
        hparams,
        attention_type=hparams.dec_attention_type,
        losses=losses,
        name="decoder")
    output = cia.create_output(decoder_output, rows, cols, targets, hparams)

    if losses:
      return output, {"extra_loss": tf.add_n(losses)}
    else:
      return output

  def loss(self, logits, features):
    if self._hparams.likelihood == cia.DistributionType.DMOL:
      return common_layers.dml_loss(logits, features["targets"])

    return super(Imagetransformer, self).loss(logits, features)

  def sample(self, features):
    """Run the model and extract samples.

    Args:
      features: an map of string to `Tensor`.

    Returns:
       samples: an integer `Tensor`.
       logits: a list of `Tensor`s, one per datashard.
       losses: a dictionary: {loss-name (string): floating point `Scalar`}.
    """
    if self._hparams.likelihood == cia.DistributionType.DMOL:
      logits, losses = self(features)  # pylint: disable=not-callable
      samples = common_layers.sample_from_discretized_mix_logistic(
          logits, seed=None)
      return samples, logits, losses

    return super(Imagetransformer, self).sample(features)

  def _slow_greedy_infer(self, features, decode_length):
    """A slow greedy inference method.

    Quadratic time in decode_length.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.

    Returns:
       samples: an integer `Tensor`.
       logits: `Tensor` of shape [batch_size, time, 1, 1, vocab_size].
       losses: a dictionary: {loss-name (string): floating point `Scalar`}
    """
    if self._hparams.likelihood == cia.DistributionType.DMOL:
      raise NotImplementedError("Decoding is not currently available for DMOL.")
    return super(Imagetransformer, self)._slow_greedy_infer(features,
                                                            decode_length)


@registry.register_model
class ImagetransformerMoe(t2t_model.T2TModel):
  """Conditional image generation with attention and MoE."""

  @staticmethod
  def use_body_sharded():
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
    # TODO(nikip): Use q_padding and kv_padding
    del q_padding, kv_padding
    decoder_output, extra_loss = cia.transformer_layers_sharded(
        dp,
        self._ps_devices,
        decoder_input,
        hparams.num_hidden_layers,
        hparams,
        self_attention_bias=None,
        enc_output=None,
        attention_type=hparams.dec_attention_type,
        name="decoder")

    output = dp(cia.create_output, decoder_output, rows, cols, targets, hparams)
    return output, extra_loss


@registry.register_hparams
def image_transformer_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.hidden_size = 512
  hparams.batch_size = 4
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
  hparams.bottom["targets"] = modalities.image_channel_embeddings_bottom
  hparams.top["targets"] = modalities.identity_top
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
  hparams.add_hparam("dec_attention_type", cia.AttentionType.LOCAL_1D)
  hparams.add_hparam("block_raster_scan", False)

  # multipos attention params
  hparams.add_hparam("q_filter_width", 1)
  hparams.add_hparam("kv_filter_width", 1)

  hparams.add_hparam("likelihood", cia.DistributionType.CAT)
  hparams.add_hparam("unconditional", False)  # unconditional generation

  # parameters of discretized mixture of logistics loss from pixel cnn++
  hparams.add_hparam("num_mixtures", 10)

  # These parameters are only used when ffn_layer=="local_moe_tpu"
  hparams.add_hparam("moe_overhead_train", 1.0)
  hparams.add_hparam("moe_overhead_eval", 2.0)
  hparams.moe_num_experts = 8
  hparams.moe_loss_coef = 1e-3

  # These parameters are for relative attention
  hparams.add_hparam("shared_rel", False)  # share relative embeddings
  return hparams


@registry.register_hparams
def imagetransformer_base():
  hparams = image_transformer_base()
  return hparams


@registry.register_hparams
def imagetransformer_cifar10_base():
  """Best config for 2.90 bits/dim on CIFAR10 using cross entropy."""
  hparams = image_transformer_base()
  hparams.batch_size = 4
  hparams.num_heads = 4
  hparams.num_decoder_layers = 12
  hparams.block_length = 256
  hparams.hidden_size = 512
  hparams.filter_size = 2048
  hparams.learning_rate = 0.5
  hparams.learning_rate_warmup_steps = 4000
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.layer_prepostprocess_dropout = 0.3
  hparams.unconditional = True
  return hparams


@registry.register_hparams
def imagetransformer_cifar10_base_dmol():
  """Best config for 2.90 bits/dim on CIFAR10 using DMOL."""
  hparams = image_transformer_base()
  hparams.likelihood = cia.DistributionType.DMOL
  hparams.num_channels = 1
  hparams.bottom["targets"] = modalities.image_channel_compress_targets_bottom
  hparams.top["targets"] = modalities.identity_top
  hparams.num_heads = 8
  hparams.batch_size = 8
  hparams.sampling_method = "random"
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.summarize_grads = True
  hparams.hidden_size = 256
  hparams.filter_size = 512
  hparams.attention_key_channels = 512
  hparams.attention_value_channels = 512
  hparams.num_decoder_layers = 12
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.learning_rate = 0.1
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.pos = "emb"
  hparams.unconditional = True
  return hparams


@registry.register_hparams
def imagetransformer_base_tpu():
  """Transformer base params for cifar-10."""
  hparams = imagetransformer_bas8l_8h_big_uncond_dr03_imgnet()
  update_hparams_for_tpu(hparams)
  hparams.batch_size = 4
  hparams.num_heads = 4   # heads are expensive on tpu
  hparams.num_decoder_layers = 12
  hparams.block_length = 128
  hparams.hidden_size = 512
  hparams.filter_size = 2048
  hparams.learning_rate = 0.2
  hparams.learning_rate_warmup_steps = 6000
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.layer_prepostprocess_dropout = 0.3
  return hparams


@registry.register_hparams
def imagetransformer_base_imagenet_tpu():
  """Transformer base params for cifar-10."""
  hparams = imagetransformer_base_tpu()
  hparams.batch_size = 4
  hparams.num_heads = 4   # heads are expensive on tpu
  hparams.num_decoder_layers = 12
  hparams.block_length = 128
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams


@registry.register_hparams
def imagetransformer_imagenet32_base():
  """Best config for ImageNet-32 with 3.77 bits/dim using cross entropy."""
  hparams = imagetransformer_cifar10_base()
  hparams.batch_size = 4
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams


@registry.register_hparams
def imagetransformer_base_rel():
  """Base with relative attention."""
  hparams = imagetransformer_base()
  hparams.dec_attention_type = cia.AttentionType.RELATIVE_LOCAL_1D
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
def imagetransformerpp_sep_channels_8l_8h():
  """separate rgb embeddings."""
  hparams = imagetransformer_base()
  hparams.likelihood = cia.DistributionType.DMOL
  hparams.num_channels = 1
  hparams.bottom["targets"] = modalities.image_channel_compress_targets_bottom
  hparams.top["targets"] = modalities.identity_top
  hparams.num_heads = 8
  hparams.batch_size = 4
  hparams.attention_key_channels = hparams.attention_value_channels = 0
  hparams.hidden_size = 512
  hparams.filter_size = 512
  hparams.num_hidden_layers = 8
  hparams.sampling_method = "random"
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.summarize_grads = True
  hparams.learning_rate = 0.1
  return hparams


@registry.register_hparams
def imagetransformerpp_base_8l_8h_big_cond_dr03_dan():
  """big 1d model for conditional image generation.2.99 on cifar10."""
  hparams = imagetransformerpp_sep_channels_8l_8h()
  hparams.hidden_size = 512
  hparams.num_heads = 8
  hparams.filter_size = 2048
  hparams.batch_size = 4
  hparams.max_length = 3075
  hparams.layer_prepostprocess_dropout = 0.3
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.summarize_grads = True
  hparams.learning_rate = 0.01
  return hparams


@registry.register_hparams
def imagetransformerpp_base_8l_8h_big_cond_dr03_dan_a():
  hparams = imagetransformerpp_base_8l_8h_big_cond_dr03_dan()
  hparams.learning_rate = 0.1
  return hparams


@registry.register_hparams
def imagetransformerpp_base_10l_8h_big_uncond_dr03_dan():
  hparams = imagetransformerpp_base_8l_8h_big_cond_dr03_dan_a()
  hparams.unconditional = True
  hparams.num_decoder_layers = 10
  return hparams


@registry.register_hparams
def imagetransformerpp_base_10l_8h_big_uncond_dr03_dan_a():
  hparams = imagetransformerpp_base_10l_8h_big_uncond_dr03_dan()
  hparams.learning_rate = 0.01
  return hparams


@registry.register_hparams
def imagetransformerpp_base_10l_8h_big_uncond_dr03_dan_b():
  hparams = imagetransformerpp_base_10l_8h_big_uncond_dr03_dan()
  hparams.learning_rate = 0.1
  hparams.hidden_size = 256
  hparams.attention_key_channels = 512
  hparams.attention_value_channels = 512
  hparams.filter_size = 1024
  return hparams


@registry.register_hparams
def imagetransformerpp_base_10l_8h_big_uncond_dr03_dan_g():
  hparams = imagetransformerpp_base_10l_8h_big_uncond_dr03_dan_b()
  hparams.filter_size = 512
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.learning_rate = 0.1
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.pos = "emb"
  return hparams


@registry.register_hparams
def imagetransformerpp_base_12l_8h_big_uncond_dr03_dan_k():
  hparams = imagetransformerpp_base_10l_8h_big_uncond_dr03_dan_g()
  hparams.num_decoder_layers = 12
  return hparams


@registry.register_hparams
def imagetransformerpp_base_12l_8h_big_uncond_dr03_dan_l():
  hparams = imagetransformerpp_base_10l_8h_big_uncond_dr03_dan_g()
  hparams.num_decoder_layers = 12
  hparams.clip_grad_norm = 40.
  return hparams


@registry.register_hparams
def imagetransformerpp_base_12l_8h_big_uncond_dr03_dan_m():
  hparams = imagetransformerpp_base_12l_8h_big_uncond_dr03_dan_k()
  hparams.batch_size = 8
  return hparams


@registry.register_hparams
def imagetransformerpp_base_12l_8h_big_uncond_dr03_dan_m_rel():
  hparams = imagetransformerpp_base_12l_8h_big_uncond_dr03_dan_k()
  hparams.batch_size = 8
  hparams.dec_attention_type = cia.AttentionType.RELATIVE_LOCAL_1D
  return hparams


@registry.register_hparams
def imagetransformerpp_base_12l_8h_big_uncond_dr03_dan_m_relsh():
  hparams = imagetransformerpp_base_12l_8h_big_uncond_dr03_dan_m_rel()
  hparams.shared_rel = True
  return hparams


@registry.register_hparams
def imagetransformerpp_base_14l_8h_big_uncond_dr03_dan_p():
  """Gets to 2.92 in just under 4 days on 8 p100s."""
  hparams = imagetransformerpp_base_12l_8h_big_uncond_dr03_dan_l()
  hparams.num_decoder_layers = 14
  hparams.batch_size = 8
  hparams.layer_prepostprocess_dropout = 0.2
  return hparams


@registry.register_hparams
def imagetransformerpp_base_12l_8h_big_uncond_dr03_dan_m_bs1():
  """For 128x128."""
  # TODO(trandustin): why are these running? max_length and img_len not set
  # 256x256 was also training without setting max_length
  hparams = imagetransformerpp_base_12l_8h_big_uncond_dr03_dan_m()
  hparams.batch_size = 1
  return hparams


@registry.register_hparams
def imagetransformerpp_base_14l_8h_big_uncond_dr03_dan_p_bs1():
  """For 128x128."""
  hparams = imagetransformerpp_base_14l_8h_big_uncond_dr03_dan_p()
  hparams.batch_size = 1
  return hparams


@registry.register_hparams
def imagetransformerpp_base_5l_8h_big_uncond_dr00_dan_g_bs1():
  """For 256x256."""
  hparams = imagetransformerpp_base_10l_8h_big_uncond_dr03_dan_g()
  # TODO(trandustin): I forgot to set this in the runs! Maybe it's not used in
  # image transformer training implementation?
  # hparams.img_len = 256
  hparams.max_length = 66000  # allow for 256x256
  hparams.batch_size = 1
  hparams.num_decoder_layers = 5
  hparams.hidden_size = 128
  hparams.filter_size = 128
  hparams.attention_key_channels = 64
  hparams.attention_value_channels = 64
  hparams.layer_prepostprocess_dropout = 0.0
  return hparams


@registry.register_hparams
def imagetransformerpp_base_5l_8h_dr00_dan_g_bs1_adafactor():
  """For 256x256."""
  hparams = imagetransformerpp_base_5l_8h_big_uncond_dr00_dan_g_bs1()
  # Use Adafactor which uses less memory than Adam, and its recommendations.
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_schedule = "rsqrt_decay"
  return hparams


@registry.register_hparams
def imagetransformerpp_base_6l_8h_dr00_dan_g_bs1_adafactor():
  """For 256x256."""
  hparams = imagetransformerpp_base_5l_8h_dr00_dan_g_bs1_adafactor()
  hparams.num_decoder_layers = 6
  return hparams


@registry.register_hparams
def imagetransformerpp_base_14l_8h_big_uncond_dr03_dan_eval():
  """Gets to 2.92 in just under 4 days on 8 p100s."""
  hparams = imagetransformerpp_base_12l_8h_big_uncond_dr03_dan_l()
  hparams.num_decoder_layers = 14
  hparams.batch_size = 8
  # hparams.layer_prepostprocess_dropout = 0.2
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
  hparams.sampling_method = "random"
  hparams.local_and_global_att = True
  return hparams


@registry.register_hparams
def imagetransformer_base_10l_16h_big_uncond_dr01_imgnet():
  """big 1d model for conditional image generation."""
  hparams = imagetransformer_base_14l_8h_big_dr01()
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
  hparams = imagetransformer_base_14l_8h_big_dr01()
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
def imagetransformer_bas8l_8h_big_uncond_dr03_imgnet():
  """big 1d model for conditional image generation."""
  hparams = imagetransformer_base_14l_8h_big_dr01()
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
  hparams.num_decoder_layers = 2
  hparams.hidden_size = 64
  hparams.batch_size = 1
  hparams.unconditional = True
  hparams.max_length = 66000  # allow for 256x256
  return hparams


@registry.register_hparams
def imagetransformerpp_tiny():
  hparams = imagetransformer_tiny()
  hparams.likelihood = cia.DistributionType.DMOL
  hparams.num_channels = 1
  hparams.bottom["targets"] = modalities.image_channel_compress_targets_bottom
  hparams.top["targets"] = modalities.identity_top
  return hparams


@registry.register_hparams
def imagetransformer_tiny_tpu():
  hparams = imagetransformer_tiny()
  update_hparams_for_tpu(hparams)
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
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_schedule = "rsqrt_decay"
  hparams.learning_rate_warmup_steps = 6000
  hparams.batch_size = 4


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
def imagetransformer_b10l_4h_big_uncond_dr03_tpu():
  """Small model for tpu cifar 10."""
  hparams = imagetransformer_bas8l_8h_big_uncond_dr03_imgnet()
  update_hparams_for_tpu(hparams)
  hparams.batch_size = 4
  hparams.num_heads = 4   # heads are expensive on tpu
  hparams.num_decoder_layers = 10
  hparams.block_length = 128
  hparams.hidden_size = 512
  hparams.filter_size = 1024
  hparams.learning_rate = 0.2
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  return hparams


@registry.register_hparams
def imagetransformer_b10l_dr03_moe_tpu():
  """Moe tpu params."""
  hparams = imagetransformer_b10l_4h_big_uncond_dr03_tpu()
  update_hparams_for_tpu(hparams)
  hparams.batch_size = 4
  hparams.num_heads = 4   # heads are expensive on tpu
  hparams.num_decoder_layers = 10
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.ffn_layer = "local_moe_tpu"
  return hparams


@registry.register_hparams
def imagetransformer_b10l_4h_big_uncond_dr03_lr025_tpu():
  """TPU related small model."""
  hparams = imagetransformer_bas8l_8h_big_uncond_dr03_imgnet()
  update_hparams_for_tpu(hparams)
  hparams.batch_size = 4
  hparams.num_heads = 4   # heads are expensive on tpu
  hparams.num_decoder_layers = 10
  hparams.learning_rate = 0.25
  hparams.learning_rate_warmup_steps = 8000
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  # hparams.unconditional = True
  return hparams


@registry.register_hparams
def imagetransformer_b12l_4h_big_uncond_dr03_tpu():
  """TPU 12 layer model."""
  hparams = imagetransformer_bas8l_8h_big_uncond_dr03_imgnet()
  update_hparams_for_tpu(hparams)
  hparams.batch_size = 4
  hparams.num_heads = 4   # heads are expensive on tpu
  hparams.num_decoder_layers = 12
  hparams.block_length = 128
  hparams.hidden_size = 512
  hparams.filter_size = 1024
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.layer_prepostprocess_dropout = 0.3
  return hparams


@registry.register_hparams
def imagetransformer_b12l_4h_big_uncond_dr03_lr025_tpu():
  hparams = imagetransformer_b12l_4h_big_uncond_dr03_tpu()
  update_hparams_for_tpu(hparams)
  hparams.learning_rate = 0.25
  hparams.learning_rate_warmup_steps = 5000
  return hparams


@registry.register_hparams
def imagetransformer_b12l_4h_b256_uncond_dr03_tpu():
  """works very well on 4x4."""
  hparams = imagetransformer_bas8l_8h_big_uncond_dr03_imgnet()
  update_hparams_for_tpu(hparams)
  hparams.batch_size = 4
  hparams.num_heads = 4   # heads are expensive on tpu
  hparams.num_decoder_layers = 12
  hparams.block_length = 256
  hparams.hidden_size = 512
  hparams.filter_size = 2048
  hparams.learning_rate = 0.5
  hparams.learning_rate_warmup_steps = 4000
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.layer_prepostprocess_dropout = 0.3
  hparams.unconditional = True
  return hparams


@registry.register_hparams
def imagetransformer_b12l_4h_b256_uncond_dr03_rel_tpu():
  """works very well on 4x4."""
  hparams = imagetransformer_b12l_4h_b256_uncond_dr03_tpu()
  hparams.shared_rel = True
  hparams.dec_attention_type = cia.AttentionType.RELATIVE_LOCAL_1D
  return hparams


@registry.register_ranged_hparams
def imagetransformer_cifar_tpu_range(rhp):
  """Range of hyperparameters for vizier."""
  # After starting from base, set intervals for some parameters.
  rhp.set_float("learning_rate", 0.01, 1.0, scale=rhp.LOG_SCALE)
  rhp.set_discrete("num_decoder_layers", [8, 10, 12, 14, 16])
  rhp.set_discrete("hidden_size", [256, 512, 1024])
  rhp.set_discrete("block_length", [128, 256, 512])
  rhp.set_categorical("dec_attention_type", [
      cia.AttentionType.RELATIVE_LOCAL_1D, cia.AttentionType.LOCAL_1D])


@registry.register_hparams
def imagetransformer_b12l_4h_b128_h512_uncond_dr03_tpu():
  """TPU related big model."""
  hparams = imagetransformer_bas8l_8h_big_uncond_dr03_imgnet()
  update_hparams_for_tpu(hparams)
  hparams.batch_size = 4
  hparams.num_heads = 4   # heads are expensive on tpu
  hparams.num_decoder_layers = 12
  hparams.block_length = 128
  hparams.hidden_size = 512
  hparams.filter_size = 2048
  hparams.learning_rate = 0.2
  hparams.learning_rate_warmup_steps = 6000
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.layer_prepostprocess_dropout = 0.3
  return hparams


@registry.register_hparams
def imagetransformer_b12l_4h_b128_h512_uncond_dr01_im():
  """TPU related imagenet model."""
  hparams = imagetransformer_b12l_4h_b256_uncond_dr03_tpu()
  update_hparams_for_tpu(hparams)
  hparams.batch_size = 4
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_schedule = "rsqrt_decay"
  hparams.learning_rate_warmup_steps = 6000
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams


@registry.register_hparams
def imagetransformer_b12l_4h_uncond_dr03_tpu():
  """TPU related small model."""
  hparams = imagetransformer_b12l_4h_b256_uncond_dr03_tpu()
  hparams.learning_rate = 0.2
  hparams.learning_rate_warmup_steps = 4000
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.layer_prepostprocess_dropout = 0.3
  return hparams


@registry.register_hparams
def imagetransformer_b12l_4h_b128_uncond_dr03_tpu():
  """TPU config for cifar 10."""
  hparams = imagetransformer_bas8l_8h_big_uncond_dr03_imgnet()
  update_hparams_for_tpu(hparams)
  hparams.batch_size = 2
  hparams.num_heads = 4   # heads are expensive on tpu
  hparams.num_decoder_layers = 12
  hparams.block_length = 128
  hparams.hidden_size = 256
  hparams.filter_size = 2048
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_schedule = "rsqrt_decay"
  hparams.learning_rate_warmup_steps = 10000
  return hparams


@registry.register_hparams
def imagetransformer_b12l_8h_b256_uncond_dr03_tpu():
  """TPU related 12 layer 8 heads model."""
  hparams = imagetransformer_bas8l_8h_big_uncond_dr03_imgnet()
  update_hparams_for_tpu(hparams)
  hparams.batch_size = 2
  hparams.num_heads = 8   # heads are expensive on tpu
  hparams.num_decoder_layers = 12
  hparams.block_length = 256
  hparams.hidden_size = 512
  hparams.filter_size = 2048
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.layer_prepostprocess_dropout = 0.3
  return hparams


@registry.register_hparams
def imagetransformer_b10l_4h_big_uncond_dr01_tpu():
  """big 1d model for conditional image generation."""
  hparams = imagetransformer_b12l_4h_big_uncond_dr03_tpu()
  # num_hidden_layers
  hparams.num_decoder_layers = 10
  hparams.num_heads = 4
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  hparams.batch_size = 1
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams
