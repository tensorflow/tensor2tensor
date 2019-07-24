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

"""Optimizers defined in trax."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin

from tensor2tensor.trax.optimizers import base


def opt_configure(*args, **kwargs):
  kwargs["module"] = "trax.optimizers"
  return gin.external_configurable(*args, **kwargs)

# Optimizers (using upper-case names).
# pylint: disable=invalid-name
SGD = opt_configure(base.SGD)
Momentum = opt_configure(base.Momentum)
RMSProp = opt_configure(base.RMSProp)
Adam = opt_configure(base.Adam)
Adafactor = opt_configure(base.Adafactor)
SM3 = opt_configure(base.SM3)
