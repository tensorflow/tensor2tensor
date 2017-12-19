# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

# Dependency imports

# pylint: disable=unused-import

from tensor2tensor.layers import modalities
from tensor2tensor.models import aligned
from tensor2tensor.models import attention_lm
from tensor2tensor.models import attention_lm_moe
from tensor2tensor.models import bluenet
from tensor2tensor.models import bytenet
from tensor2tensor.models import cycle_gan
from tensor2tensor.models import gene_expression
from tensor2tensor.models import lstm
from tensor2tensor.models import multimodel
from tensor2tensor.models import neural_gpu
from tensor2tensor.models import resnet
from tensor2tensor.models import revnet
from tensor2tensor.models import shake_shake
from tensor2tensor.models import slicenet
from tensor2tensor.models import super_lm
from tensor2tensor.models import transformer
from tensor2tensor.models import transformer_moe
from tensor2tensor.models import transformer_revnet
from tensor2tensor.models import transformer_sketch
from tensor2tensor.models import transformer_vae
from tensor2tensor.models import vanilla_gan
from tensor2tensor.models import xception
# pylint: enable=unused-import
