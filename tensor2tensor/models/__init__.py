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

"""Models defined in T2T. Imports here force registration."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

# pylint: disable=unused-import

from tensor2tensor.layers import modalities  # pylint: disable=g-import-not-at-top
from tensor2tensor.models import basic
from tensor2tensor.models import bytenet
from tensor2tensor.models import distillation
from tensor2tensor.models import evolved_transformer
from tensor2tensor.models import image_transformer
from tensor2tensor.models import image_transformer_2d
from tensor2tensor.models import lstm
from tensor2tensor.models import neural_assistant
from tensor2tensor.models import neural_gpu
from tensor2tensor.models import resnet
from tensor2tensor.models import revnet
from tensor2tensor.models import shake_shake
from tensor2tensor.models import slicenet
from tensor2tensor.models import text_cnn
from tensor2tensor.models import transformer
from tensor2tensor.models import vanilla_gan
from tensor2tensor.models import xception
from tensor2tensor.models.neural_architecture_search import nas_model
from tensor2tensor.models.research import adafactor_experiments
from tensor2tensor.models.research import aligned
from tensor2tensor.models.research import autoencoders
from tensor2tensor.models.research import cycle_gan
from tensor2tensor.models.research import gene_expression
from tensor2tensor.models.research import neural_stack
from tensor2tensor.models.research import rl
from tensor2tensor.models.research import shuffle_network
from tensor2tensor.models.research import similarity_transformer
from tensor2tensor.models.research import super_lm
from tensor2tensor.models.research import transformer_moe
from tensor2tensor.models.research import transformer_nat
from tensor2tensor.models.research import transformer_parallel
from tensor2tensor.models.research import transformer_revnet
from tensor2tensor.models.research import transformer_sketch
from tensor2tensor.models.research import transformer_symshard
from tensor2tensor.models.research import transformer_vae
from tensor2tensor.models.research import universal_transformer
from tensor2tensor.models.video import basic_deterministic
from tensor2tensor.models.video import basic_recurrent
from tensor2tensor.models.video import basic_stochastic
from tensor2tensor.models.video import emily
from tensor2tensor.models.video import savp
from tensor2tensor.models.video import sv2p
from tensor2tensor.utils import contrib
from tensor2tensor.utils import registry

# The following models can't be imported under TF2
if not contrib.is_tf2:
  # pylint: disable=g-import-not-at-top
  from tensor2tensor.models.research import attention_lm
  from tensor2tensor.models.research import attention_lm_moe
  from tensor2tensor.models.research import glow
  from tensor2tensor.models.research import lm_experiments
  from tensor2tensor.models.research import moe_experiments
  from tensor2tensor.models.research import multiquery_paper
  from tensor2tensor.models import mtf_image_transformer
  from tensor2tensor.models import mtf_resnet
  from tensor2tensor.models import mtf_transformer
  from tensor2tensor.models import mtf_transformer2
  from tensor2tensor.models.research import vqa_attention
  from tensor2tensor.models.research import vqa_recurrent_self_attention
  from tensor2tensor.models.research import vqa_self_attention
  from tensor2tensor.models.video import epva
  from tensor2tensor.models.video import next_frame_glow
  # pylint: enable=g-import-not-at-top

# pylint: disable=unused-import

# pylint: enable=unused-import


def model(name):
  return registry.model(name)
