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

"""Languaeg modeling experiments in mtf."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.models import mtf_transformer
from tensor2tensor.models.research import moe
from tensor2tensor.utils import registry


@registry.register_hparams
def xmoe_dense_4k():
  """Series of architectural experiments on cheap language models.

  For all of these architectures, we run on languagemodel_lm1b8k_packed
  for 32000 steps.

  All log-perplexities are per-token - multiply by 1.298 for per-word

  Results:
  model             params(M)  einsum  alltoall  mxu-util  log-ppl
  xmoe_dense_4k     30         3.0e12  0         45%        3.31
  xmoe_dense_8k     46         4.7e12  0         49%        3.24
  xmoe_dense_64k    282        2.8e13  0                    3.06
  xmoe_top_2        282        4.0e12  3.4e8     36%        3.07
  xmoe_top_2_c15    282        4.5e12  4.0e8     38%        3.07
  xmoe_2d           282        5.3e12  7.6e8     34%        3.06

  Trained at 4x the batch size:
  xmoe_2d_88        1090       2.1e13  3.0e9     24%        3.07

  Note: configurations and code are likely to change without notice.

  Returns:
    a hparams
  """
  hparams = mtf_transformer.mtf_transformer_base_lm()
  hparams.attention_dropout = 0.0
  hparams.relu_dropout = 0.0
  hparams.layer_prepostprocess_dropout = 0.0

  # The following hparams are constant across all these experiments.
  hparams.batch_size = 128
  hparams.d_model = 512
  hparams.d_kv = 128
  hparams.num_heads = 4
  hparams.decoder_layers = ["att", "drd"] * 4
  hparams.shared_embedding_and_softmax_weights = False
  hparams.learning_rate_schedule = "rsqrt_decay"

  # We will vary the following parameters related to the ffn/moe layers.
  hparams.d_ff = 4096
  hparams.layout = "batch:batch;vocab:model;d_ff:model;heads:model"
  hparams.mesh_shape = "batch:8"
  return hparams


@registry.register_hparams
def xmoe_dense_8k():
  hparams = xmoe_dense_4k()
  hparams.d_ff = 8192
  return hparams


@registry.register_hparams
def xmoe_dense_64k():
  """Very wide layer- run on 4x4."""
  hparams = xmoe_dense_4k()
  hparams.d_ff = 65536
  hparams.mesh_shape = "model:4,batch:8"
  return hparams


@registry.register_hparams
def xmoe_top_2():
  """Mixture of experts (16 experts)."""
  hparams = xmoe_dense_4k()
  moe.set_default_moe_hparams(hparams)
  hparams.mesh_shape = "all:8"
  hparams.layout = "batch:all;experts:all"
  return hparams


@registry.register_hparams
def xmoe_top_2_c15():
  """Mixture of experts."""
  hparams = xmoe_top_2()
  hparams.moe_capacity_factor_train = 1.5
  return hparams


@registry.register_hparams
def xmoe_2d():
  """Two-dimensional hierarchical mixture of 16 experts."""
  hparams = xmoe_top_2()
  hparams.decoder_layers = ["att", "hmoe"] * 4
  hparams.mesh_shape = "b0:2;b1:4"
  hparams.outer_batch_size = 4
  hparams.layout = "outer_batch:b0;inner_batch:b1,expert_x:b1,expert_y:b0"
  hparams.moe_num_experts = [4, 4]
  return hparams


@registry.register_hparams
def xmoe_2d_debug():
  """For debugging.

  Running this model on TPU without the hack of casting to bfloat16 for
  alltoall results in nan on the first step.
  TODO(noam): debug

  Returns:
    a hparams
  """
  hparams = xmoe_2d()
  hparams.decoder_layers = ["hmoe"] * 1
  hparams.activation_dtype = "float32"
  return hparams


@registry.register_hparams
def xmoe_2d_c15():
  """Mixture of experts."""
  hparams = xmoe_2d()
  hparams.moe_capacity_factor_train = 1.5
  return hparams


@registry.register_hparams
def xmoe_2d_x64():
  """Two-dimensional hierarchical mixture of 64 experts."""
  hparams = xmoe_2d()
  # hparams.mesh_shape = "b0:4;b1:8"
  hparams.outer_batch_size = 4
  hparams.moe_num_experts = [8, 8]
  return hparams


@registry.register_hparams
def xmoe2_dense(sz):
  """Series of architectural experiments on language modeling.

  Larger models than the ones above.

  All models are trained on sequences of 1024 tokens.

  We assume infinite training data, so no dropout necessary.
  We process 2^36 tokens in training = 524288 steps at batch size 128

  TODO(noam): find a large enough dataset for these experiments.

  You can use languagemodel_wiki_noref_v8k_l1k, but this is too small,
  so training will cover about 9 epochs.

  Note: configurations and code are likely to change without notice.

  Run on TPU 4x4 for 524288 steps unless otherwise indicated.

  Args:
    sz: an integer

  Returns:
    a hparams
  """
  hparams = mtf_transformer.mtf_transformer_paper_lm(sz)
  hparams.attention_dropout = 0.0
  hparams.relu_dropout = 0.0
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.max_length = 1024
  hparams.batch_size = 128
  hparams.learning_rate_schedule = "rsqrt_decay*linear_decay"
  hparams.learning_rate_decay_steps = 65536
  hparams.layout = "batch:batch;vocab:model;d_ff:model;heads:model"
  hparams.mesh_shape = "batch:32"
  return hparams


@registry.register_hparams
def xmoe2_dense_0():
  return xmoe2_dense(0)


@registry.register_hparams
def xmoe2_dense_1():
  return xmoe2_dense(1)


@registry.register_hparams
def xmoe2_dense_2():
  return xmoe2_dense(2)


@registry.register_hparams
def xmoe2_dense_3():
  return xmoe2_dense(3)


@registry.register_hparams
def xmoe2_v1():
  """Model incorporating mixture-of-experts and local-attention.

  ~6B parameters

  32 experts in 3 hierarchichal moe layers.

  Returns:
    a hparams
  """
  hparams = xmoe2_dense(0)
  moe.set_default_moe_hparams(hparams)
  hparams.decoder_layers = (
      ["local_att", "local_att", "drd",
       "att", "drd", "local_att", "local_att", "hmoe"] * 4)[:-1]
  hparams.d_ff = 2048
  hparams.d_kv = 128
  hparams.moe_hidden_size = 32768
  hparams.mesh_shape = "b0:4;b1:8"
  hparams.layout = "outer_batch:b0;inner_batch:b1,expert_x:b1,expert_y:b0"
  hparams.outer_batch_size = 4
  hparams.moe_num_experts = [8, 4]
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def xmoe2_v1_x128():
  """128 experts, ~25B params - Train for 131072 steps on 8x8."""
  hparams = xmoe2_v1()
  hparams.moe_num_experts = [16, 8]
  hparams.outer_batch_size = 8
  hparams.mesh_shape = "b0:8;b1:16"
  hparams.batch_size = 512
  hparams.learning_rate_decay_steps = 16384
  return hparams


@registry.register_hparams
def xmoe2_tiny():
  """Test on local cpu."""
  hparams = xmoe2_v1()
  hparams.decoder_layers = ["local_att", "att", "drd", "hmoe"]
  hparams.d_model = 128
  hparams.moe_hidden_size = 512
  hparams.outer_batch_size = 0
  hparams.batch_size = 2
  hparams.mesh_shape = ""
  hparams.activation_dtype = "float32"
  return hparams
