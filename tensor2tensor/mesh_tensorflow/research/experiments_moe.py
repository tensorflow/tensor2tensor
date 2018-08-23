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

from tensor2tensor.mesh_tensorflow import mtf_transformer
from tensor2tensor.mesh_tensorflow.research import moe
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
  xmoe_dense_64k                       0                    3.06
  xmoe_top_2        282        4.0e12  3.4e8     36%
  xmoe_top_2_c15    282        4.5e12  4.0e8     38%

  Note: configurations and code are likely to change without notice.

  Returns:
    a hparams
  """
  hparams = mtf_transformer.mtf_transformer_base()

  # The following hparams are constant across all these experiments.
  hparams.label_smoothing = 0.0
  hparams.batch_size = 128
  hparams.d_model = 512
  hparams.d_kv = 128
  hparams.num_heads = 4
  hparams.num_decoder_layers = 4
  hparams.shared_embedding_and_softmax_weights = False
  hparams.learning_rate_schedule = "rsqrt_decay"

  # We will vary the following parameters related to the ffn/moe layers.
  hparams.feedforward_layer = "dense_relu_dense"
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
  moe.set_default_moe_hparams(hparams)
  hparams.mesh_shape = "all:8"
  hparams.layout = "batch:all;experts:all"
  hparams.feedforward_layer = "moe"
  return hparams


