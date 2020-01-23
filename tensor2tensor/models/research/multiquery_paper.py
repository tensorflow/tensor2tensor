# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""Experiments for Multiquery-Attention Paper.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.models import mtf_transformer2
from tensor2tensor.utils import registry


@registry.register_hparams
def mqp_ende_base():
  # params=211M
  hparams = mtf_transformer2.mtr_tr_dense_0()
  hparams.learning_rate_decay_steps = 20000
  hparams.shared_embedding_and_softmax_weights = True
  hparams.layer_prepostprocess_dropout = 0.2
  return hparams


@registry.register_hparams
def mqp_ende_local():
  hparams = mqp_ende_base()
  hparams.decoder_local_attention_radius = 32
  return hparams


@registry.register_hparams
def mqp_ende_mq8():
  # params=178M
  hparams = mqp_ende_base()
  hparams.decoder_num_heads = 8
  hparams.decoder_num_memory_heads = 1
  hparams.encoder_num_heads = 8
  hparams.encoder_num_memory_heads = 1
  return hparams


@registry.register_hparams
def mqp_ende_mq8_ff5440():
  # params=211M
  hparams = mqp_ende_mq8()
  hparams.d_ff = 5440
  return hparams


@registry.register_hparams
def mqp_ende_mq8_ff5440_local():
  hparams = mqp_ende_mq8_ff5440()
  hparams.decoder_local_attention_radius = 32
  return hparams


@registry.register_hparams
def mqp_ende_h4_kv256():
  hparams = mqp_ende_base()
  hparams.decoder_num_heads = 4
  hparams.encoder_num_heads = 4
  hparams.d_kv = 256
  return hparams


@registry.register_hparams
def mqp_ende_h2_kv512():
  hparams = mqp_ende_base()
  hparams.decoder_num_heads = 2
  hparams.encoder_num_heads = 2
  hparams.d_kv = 512
  return hparams


@registry.register_hparams
def mqp_ende_h1_kv1024():
  hparams = mqp_ende_base()
  hparams.decoder_num_heads = 1
  hparams.encoder_num_heads = 1
  hparams.d_kv = 1024
  return hparams


@registry.register_hparams
def mqp_ende_h4_ff5632():
  hparams = mqp_ende_base()
  hparams.decoder_num_heads = 4
  hparams.encoder_num_heads = 4
  hparams.d_ff = 5632
  return hparams


@registry.register_hparams
def mqp_ende_h2_ff6400():
  hparams = mqp_ende_base()
  hparams.decoder_num_heads = 2
  hparams.encoder_num_heads = 2
  hparams.d_ff = 6400
  return hparams


@registry.register_hparams
def mqp_ende_h1_ff6784():
  hparams = mqp_ende_base()
  hparams.decoder_num_heads = 1
  hparams.encoder_num_heads = 1
  hparams.d_ff = 6784
  return hparams


@registry.register_hparams
def mqp_ende_h2_kv64_ff6784():
  hparams = mqp_ende_base()
  hparams.decoder_num_heads = 2
  hparams.encoder_num_heads = 2
  hparams.d_kv = 64
  hparams.d_ff = 6784
  return hparams


@registry.register_hparams
def mqp_ende_h4_kv32_ff6784():
  hparams = mqp_ende_base()
  hparams.decoder_num_heads = 4
  hparams.encoder_num_heads = 4
  hparams.d_kv = 32
  hparams.d_ff = 6784
  return hparams


@registry.register_hparams
def mqp_ende_h8_kv16_ff6784():
  hparams = mqp_ende_base()
  hparams.decoder_num_heads = 8
  hparams.encoder_num_heads = 8
  hparams.d_kv = 16
  return hparams


@registry.register_hparams
def mqp_lm1b_base():
  """Series of architectures for language modeling."""
  hparams = mtf_transformer2.mtf_unitransformer_base()
  hparams.d_model = 1024
  hparams.max_length = 256
  hparams.batch_size = 256
  # Parameters for my_layer_stack()
  hparams.num_hidden_layers = 6
  hparams.d_ff = 8192
  hparams.d_kv = 128
  hparams.num_heads = 8
  hparams.learning_rate_decay_steps = 13600
  hparams.layout = "batch:batch;vocab:model;d_ff:model;heads:model"
  hparams.mesh_shape = "batch:32"
  return hparams


@registry.register_hparams
def mqp_lm1b_mq8():
  hparams = mqp_lm1b_base()
  hparams.num_heads = 8
  hparams.num_memory_heads = 1
  return hparams


@registry.register_hparams
def mqp_lm1b_mq8_ff9088():
  hparams = mqp_lm1b_mq8()
  hparams.d_ff = 9088
  return hparams


@registry.register_hparams
def mqp_lm1b_h1_ff9984():
  hparams = mqp_lm1b_base()
  hparams.num_heads = 1
  hparams.d_ff = 9984
  return hparams


@registry.register_hparams
def mqp_lm1b_h2_kv64_ff9984():
  hparams = mqp_lm1b_base()
  hparams.num_heads = 2
  hparams.d_kv = 64
  hparams.d_ff = 9984
  return hparams


@registry.register_hparams
def mqp_lm1b_h4_kv32_ff9984():
  hparams = mqp_lm1b_base()
  hparams.num_heads = 4
  hparams.d_kv = 32
  hparams.d_ff = 9984
  return hparams


@registry.register_hparams
def mqp_lm1b_h8_kv16_ff9984():
  hparams = mqp_lm1b_base()
  hparams.num_heads = 8
  hparams.d_kv = 16
  hparams.d_ff = 9984
  return hparams
