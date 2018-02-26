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

"""Experiments with Adafactor.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.models import transformer
from tensor2tensor.utils import registry


def mimic_adam_with_adafactor(hparams):
  """Switch from Adam to Adafactor, approximating the behavior of Adam.

  Some minor things may be different, like epsilon and beta1 correction.

  Args:
    hparams: model hyperparameters where "Adam" in hparams.optimizer
  """
  assert "Adam" in hparams.optimizer
  hparams.optimizer = "Adafactor"
  hparams.optimizer_adafactor_beta1 = hparams.optimizer_adam_beta1
  hparams.optimizer_adafactor_beta2 = hparams.optimizer_adam_beta2
  hparams.optimizer_adafactor_multiply_by_parameter_scale = False
  hparams.optimizer_adafactor_factored = False
  hparams.optimizer_adafactor_clipping_threshold = None
  hparams.optimizer_adafactor_decay_type = "Adam"


@registry.register_hparams
def afx_adam():
  """Old version - Adam."""
  hparams = transformer.transformer_base_v2()
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.999
  hparams.symbol_modality_num_shards = 1
  hparams.batch_size = 2048
  hparams.optimizer = "Adam"
  hparams.learning_rate_schedule = (
      "constant*rsqrt_decay*linear_warmup*rsqrt_hidden_size")
  hparams.learning_rate_constant = 2.0
  return hparams


@registry.register_hparams
def afx_mimic_adam():
  """Emulating Adam - should be very similar to afx_adam."""
  hparams = afx_adam()
  mimic_adam_with_adafactor(hparams)
  return hparams


@registry.register_hparams
def afx_base():
  """Baseline - no momentum, beta=0.999."""
  hparams = afx_mimic_adam()
  hparams.optimizer_adafactor_beta1 = 0.0
  return hparams


@registry.register_hparams
def afx_factored():
  hparams = afx_base()
  hparams.optimizer_adafactor_factored = True
  return hparams


@registry.register_hparams
def afx_fast():
  hparams = afx_base()
  hparams.optimizer_adafactor_beta2 = 0.9
  return hparams


@registry.register_hparams
def afx_clip():
  hparams = afx_base()
  hparams.optimizer_adafactor_clipping_threshold = 1.0
  return hparams


@registry.register_hparams
def afx_clip2():
  hparams = afx_base()
  hparams.optimizer_adafactor_clipping_threshold = 2.0
  return hparams


@registry.register_hparams
def afx_clip_factored():
  hparams = afx_clip()
  hparams.optimizer_adafactor_factored = True
  return hparams


@registry.register_hparams
def afx_pow05():
  hparams = afx_base()
  hparams.optimizer_adafactor_decay_type = "pow"
  hparams.optimizer_adafactor_memory_exponent = 0.5
  return hparams


@registry.register_hparams
def afx_pow08():
  hparams = afx_pow05()
  hparams.optimizer_adafactor_memory_exponent = 0.8
  return hparams


@registry.register_hparams
def afx_pow10():
  hparams = afx_pow05()
  hparams.optimizer_adafactor_memory_exponent = 1.0
  return hparams


@registry.register_hparams
def afx_pow08_clip():
  hparams = afx_pow08()
  hparams.optimizer_adafactor_clipping_threshold = 1.0
  return hparams


@registry.register_hparams
def afx_relative():
  hparams = afx_base()
  hparams.optimizer_adafactor_multiply_by_parameter_scale = True
  hparams.learning_rate_schedule = "rsqrt_decay"
  hparams.learning_rate_warmup_steps = 10000
  return hparams


@registry.register_hparams
def afx_unscale():
  hparams = afx_base()
  hparams.shared_embedding_and_softmax_weights = False
  hparams.multiply_embedding_mode = "none"
  return hparams


@registry.register_hparams
def afx_unscale_relative():
  hparams = afx_unscale()
  hparams.optimizer_adafactor_multiply_by_parameter_scale = True
  hparams.learning_rate_schedule = "rsqrt_decay"
  hparams.learning_rate_warmup_steps = 10000
  return hparams


@registry.register_hparams
def afx_adafactor():
  """Adafactor with recommended learning rate schedule."""
  hparams = afx_adam()
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_schedule = "rsqrt_decay"
  hparams.learning_rate_warmup_steps = 10000
  return hparams
