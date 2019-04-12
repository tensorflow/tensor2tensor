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

"""Loss functions and layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
from tensor2tensor.trax.backend import numpy as np
from tensor2tensor.trax.stax import slax


@gin.configurable(blacklist=['logpred', 'target'])
def kl_div(logpred, target, eps=np.finfo(np.float32).eps):
  """Calculate KL-divergence."""
  return np.sum(target * (np.log(target + eps) - logpred))


def crossentropy_loss(logpred, target):
  """Calculate crossentropy loss."""
  return - np.mean(
      np.sum(logpred * slax.one_hot(target, logpred.shape[-1]), axis=-1))


@gin.configurable(blacklist=['logpred', 'target', 'size'])
def label_smoothed_loss(logpred, target, size, padding_idx=0, smoothing=0.0):
  """Returns a label-smoothing loss-criterion function."""
  confidence = 1.0 - smoothing
  zerosmoothed = smoothing / (size - 2)
  delta = confidence - zerosmoothed
  assert logpred.shape[1] == size
  truedist = (np.full_like(logpred, zerosmoothed) +
              delta * slax.one_hot(target, size))
  truedist *= (1 - (np.arange(size) == padding_idx))
  truedist *= (1 - (target == padding_idx))[:, np.newaxis]
  return kl_div(logpred, truedist, eps=1e-6)
