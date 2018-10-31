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

"""Experiments with mixture-of-experts architectures."""


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
  for 32k-96 steps (1-3 epochs) on one TPU (8 cores).

  All log-perplexities are per-token - multiply by 1.298 for per-word

  Results:
  model             params(M)  einsum  alltoall  mxu-util  log-ppl(1ep) (3ep)
  xmoe_dense_4k     30         3.0e12  0         45%        3.31
  xmoe_dense_8k     46         4.7e12  0         49%        3.24
  xmoe_dense_64k    282        2.8e13  0                    3.06
  xmoe_top_2        282        4.0e12  3.4e8     36%        3.07
  xmoe_top_2_c15    282        4.5e12  4.0e8     38%        3.07
  xmoe_2d           282        5.3e12  7.6e8     34%        3.06

  Trained at 4x the batch size:
  xmoe_2d_88        1090       2.1e13  3.0e9     24%

  Note: configurations and code are likely to change without notice.

  Returns:
    a hparams
  """
  hparams = mtf_transformer.mtf_transformer_base_lm()

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
  hparams = xmoe_dense_4k()
  hparams.d_ff = 65536
  hparams.mesh_shape = "model:4,batch:8"
  return hparams


@registry.register_hparams
def xmoe_top_2():
  """Mixture of experts."""
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
def mtf_transformer_lm_moe():
  """Mixture of experts language model.

  Compare to mtf_transformer.mtf_transformer_lm_baseline()

  Run this on 2x2 on languagemodel_lm1b32k_packed for 272000 steps (10 epochs)
  900M params.

  Results on LM1B:
         params/10^9  log-ppl(per-token)
         0.90         TODO(noam): rerun experiment

  Returns:
    a hparams
  """
  hparams = mtf_transformer.mtf_transformer_lm_baseline()
  hparams.decoder_layers = ["att", "moe"] * 4
  moe.set_default_moe_hparams(hparams)
  hparams.mesh_shape = "all:8"
  hparams.layout = "batch:all;experts:all"
  return hparams


@registry.register_hparams
def xmoe_2d():
  """Two-dimensional hierarchical mixture of experts."""
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
def xmoe_2d_88():
  """Two-dimensional hierarchical mixture of experts."""
  hparams = xmoe_2d()
  hparams.mesh_shape = "b0:4;b1:8"
  hparams.batch_size = 512
  hparams.outer_batch_size = 4
  hparams.moe_num_experts = [8, 8]
  return hparams


@registry.register_hparams
def xmoe_wiki_base(sz):
  """Series of architectural experiments on wikipedia text.

  For all of these architectures, we run on languagemodel_wiki_noref_v8k_l1k
  for 3 epochs.  (training set has ~7390100 sequences each of length 1024)
  1 epoch = 57500 steps at batch_size=128

  Results:
  model             params(M)  einsum  alltoall  mxu-util  log-ppl(1ep) (3ep)

  Note: configurations and code are likely to change without notice.

  Args:
    sz: an integer

  Returns:
    a hparams
  """
  hparams = mtf_transformer.mtf_transformer_paper_lm(sz)

  hparams.max_length = 1024
  hparams.batch_size = 128
  hparams.learning_rate_decay_steps = 57500
  hparams.layout = "batch:batch;vocab:model;d_ff:model;heads:model"
  hparams.mesh_shape = "batch:32"
  return hparams


@registry.register_hparams
def xmoe_wiki_base_0():
  return xmoe_wiki_base(0)


@registry.register_hparams
def xmoe_wiki_base_1():
  return xmoe_wiki_base(1)


@registry.register_hparams
def xmoe_wiki_base_2():
  return xmoe_wiki_base(2)


@registry.register_hparams
def xmoe_wiki_base_3():
  return xmoe_wiki_base(3)


@registry.register_hparams
def xmoe_wiki_x():
  """Baseline set of parameters for mixture-of-experts.

  ~6B parameters

  Returns:
    a hparams
  """
  hparams = xmoe_wiki_base(0)
  moe.set_default_moe_hparams(hparams)
  hparams.decoder_layers = (
      ["att", "drd", "att", "drd", "att", "hmoe"] * 3 +
      ["att", "drd", "att", "drd"])
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
def xmoe_wiki_x_a32():
  """Test 32-bit activations."""
  hparams = xmoe_wiki_x()
  hparams.activation_dtype = "float32"
  return hparams


@registry.register_hparams
def xmoe_wiki_x128():
  """128 experts, ~25B params on 8x8."""
  hparams = xmoe_wiki_x()
  hparams.moe_num_experts = [16, 8]
  hparams.outer_batch_size = 8
  hparams.mesh_shape = "b0:8;b1:16"
  hparams.batch_size = 512
  hparams.learning_rate_decay_steps = 14375
  return hparams


@registry.register_hparams
def xmoe_wiki_x_tiny():
  """Test on local cpu."""
  hparams = xmoe_wiki_x()
  hparams.decoder_layers = (["att", "drd", "hmoe"] * 2 + ["att", "drd"])
  hparams.moe_hidden_size = 512
  hparams.batch_size = 16
  hparams.mesh_shape = ""
  hparams.activation_dtype = "float32"
  return hparams
