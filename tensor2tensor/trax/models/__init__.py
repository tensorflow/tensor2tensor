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

"""Models defined in trax."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin

from tensor2tensor.trax.models import mlp
from tensor2tensor.trax.models import resnet
from tensor2tensor.trax.models import transformer


# Ginify
def model_configure(*args, **kwargs):
  kwargs["module"] = "trax.models"
  return gin.external_configurable(*args, **kwargs)


# pylint: disable=invalid-name
MLP = model_configure(mlp.MLP)
Resnet50 = model_configure(resnet.Resnet50)
WideResnet = model_configure(resnet.WideResnet)
TransformerLM = model_configure(transformer.TransformerLM)
