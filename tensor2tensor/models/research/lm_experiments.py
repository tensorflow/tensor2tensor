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
"""Experiments with Language Models.

Train languagemodel_lm1b32k_packed and measure log-ppl/token (dev).
These numbers need to be multiplied by 1.107893 to get log-ppl/word
 for comparison with published results.

Basic training regimen is 300k steps * 8 cores * batch_size=4096
   = about 10 epochs

Make sure to eval on CPU or GPU using a large number of steps (1000), since the
TPU eval code doesn't know how to stop at the end of the dev data.  Also need
to set activation_type=float32 for eval, since there is currently a conflict
between daisy_chain_getter and activation_type=bfloat16.

RESULTS:
  lmx_base:      log-ppl/tok=3.40   PPL/word=43.2   (10 hours*8 cores)
  lmx_h1k_f4k:
  lmx_h2k_f8k:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.models import transformer
from tensor2tensor.utils import registry


@registry.register_hparams
def lmx_base():
  """Transformer on languagemodel_lm1b32k_packed.  50M Params."""
  hparams = transformer.transformer_tpu()
  # sharing is counterproductive when underparameterized
  hparams.shared_embedding_and_softmax_weights = False
  # we judge by log-ppl, so label smoothing hurts.
  hparams.label_smoothing = 0.0
  # This makes the batch size on GPU the same as on TPU for a packed problem
  # with sequence length 256.
  # TODO(noam): fix the mess that is the data reading pipeline.
  hparams.max_length = 256
  # larger batch since we only have a decoder
  hparams.batch_size = 4096
  # save some memory so we can have a larger model
  hparams.activation_dtype = "bfloat16"
  return hparams


@registry.register_hparams
def lmx_h1k_f4k():
  """Transformer on languagemodel_lm1b32k_packed.  140M Params."""
  hparams = lmx_base()
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  return hparams


@registry.register_hparams
def lmx_h2k_f8k():
  """HParams for training languagemodel_lm1b32k_packed.  430M Params."""
  hparams = lmx_base()
  hparams.hidden_size = 2048
  hparams.filter_size = 8192
  return hparams


@registry.register_hparams
def lmx_h3k_f12k():
  """HParams for training languagemodel_lm1b32k_packed.  880M Params."""
  hparams = lmx_base()
  hparams.hidden_size = 3072
  hparams.filter_size = 12288
  hparams.batch_size = 2048
  hparams.weight_dtype = "bfloat16"
  return hparams


@registry.register_hparams
def lmx_h4k_f16k():
  """HParams for training languagemodel_lm1b32k_packed.  1470M Params."""
  hparams = lmx_base()
  hparams.hidden_size = 4096
  hparams.filter_size = 16384
  hparams.batch_size = 1024
  hparams.weight_dtype = "bfloat16"
  return hparams


@registry.register_hparams
def lmx_relative():
  """Language model using relative attention."""
  hparams = lmx_base()
  hparams.self_attention_type = "dot_product_relative_v2"
  hparams.activation_dtype = "float32"
  hparams.weight_dtype = "float32"
  return hparams


@registry.register_hparams
def lmx_relative_nopos():
  """Language model using relative attention and no positional encoding."""
  hparams = lmx_relative()
  hparams.pos = "none"
  return hparams


@registry.register_hparams
def lmx_moe():
  """Transformer with mixture of experts.  140M Params."""
  hparams = lmx_base()
  hparams.ffn_layer = "local_moe_tpu"
  return hparams


@registry.register_hparams
def lmx_moe_h1k_f4k_x32():
  """Transformer with mixture of experts.  890M Params."""
  hparams = lmx_h1k_f4k()
  hparams.ffn_layer = "local_moe_tpu"
  hparams.moe_num_experts = 32
  hparams.weight_dtype = "bfloat16"
  hparams.batch_size = 8192
  return hparams


@registry.register_hparams
def lmx_moe_h1k_f8k_x16():
  """Transformer with mixture of experts.  890M Params."""
  hparams = lmx_h1k_f4k()
  hparams.filter_size = 8192
  hparams.ffn_layer = "local_moe_tpu"
  hparams.moe_num_experts = 16
  hparams.weight_dtype = "bfloat16"
  hparams.batch_size = 8192
  return hparams


@registry.register_hparams
def lmx_h1k_f64k():
  """HParams for training languagemodel_lm1b32k_packed.  880M Params."""
  hparams = lmx_base()
  hparams.hidden_size = 1024
  hparams.filter_size = 65536
  hparams.batch_size = 2048
  return hparams
