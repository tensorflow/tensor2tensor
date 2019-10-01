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

"""Trax metrics layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.trax.backend import numpy as np
from tensor2tensor.trax.layers import base
from tensor2tensor.trax.layers import combinators as cb
from tensor2tensor.trax.layers import core


@base.layer(n_inputs=2, n_outputs=1)
def CrossEntropy(x, axis=-1, **kw):
  del kw
  prediction, target = x
  return np.sum(prediction * core.one_hot(target, prediction.shape[-1]),
                axis=axis)


@base.layer(n_inputs=2, n_outputs=1)
def L2(x, axis=-1, **kw):
  del kw
  prediction, target = x
  return np.sum((prediction - target)**2, axis=axis)


@base.layer(n_inputs=2, n_outputs=1)
def Accuracy(x, axis=-1, **kw):
  del kw
  prediction, target = x
  predicted_class = np.argmax(prediction, axis=axis)
  return np.equal(predicted_class, target)


@base.layer()
def WeightMask(target, mask_id=0, **kw):
  del kw
  if mask_id is None:
    return np.ones_like(target)
  return 1.0 - np.equal(target, mask_id).astype(np.float32)


@base.layer(n_inputs=2, n_outputs=1)
def WeightedMean(x, **kw):
  del kw
  metric, weights = x
  weights_sum = np.sum(weights)
  return np.sum(metric * weights) / weights_sum


def MaskedScalar(metric_layer, mask_id=None, has_weights=False):
  """Metric as scalar compatible with Trax masking."""
  # Stack of (inputs, targets) --> (metric, weight-mask).
  metric_and_mask = [
      cb.Parallel(
          [],
          cb.Dup()  # Duplicate targets
      ),
      cb.Parallel(
          metric_layer,  # Metric: (inputs, targets) --> metric
          WeightMask(mask_id=mask_id)  # pylint: disable=no-value-for-parameter
      )
  ]
  if not has_weights:
    # Take (metric, weight-mask) and return the weighted mean.
    return cb.Serial([metric_and_mask, WeightedMean()])  # pylint: disable=no-value-for-parameter
  return cb.Serial([
      metric_and_mask,
      cb.Parallel(
          [],
          cb.Multiply()  # Multiply given weights by mask_id weights
      ),
      WeightedMean()  # pylint: disable=no-value-for-parameter
  ])


def CrossEntropyScalar(mask_id=None, has_weights=False):
  """Cross-entropy as scalar compatible with Trax masking."""
  return MaskedScalar(CrossEntropy(), mask_id=mask_id, has_weights=has_weights)  # pylint: disable=no-value-for-parameter


NegLogPerplexityScalar = CrossEntropyScalar


def CrossEntropyLossScalar(mask_id=None, has_weights=False):
  """Cross-entropy loss as scalar compatible with Trax masking."""
  return cb.Serial(
      CrossEntropyScalar(mask_id=mask_id, has_weights=has_weights),
      core.MulConstant(constant=-1.0)
  )


def L2Scalar(mask_id=None, has_weights=False):
  """L2 as scalar compatible with Trax masking."""
  return MaskedScalar(L2(), mask_id=mask_id, has_weights=has_weights)  # pylint: disable=no-value-for-parameter


def L2LossScalar(mask_id=None, has_weights=False):
  """L2 loss as scalar compatible with Trax masking."""
  return cb.Serial(
      L2Scalar(mask_id=mask_id, has_weights=has_weights),
      core.MulConstant(constant=-1.0)
  )


def AccuracyScalar(mask_id=None, has_weights=False):
  """Accuracy as scalar compatible with Trax masking."""
  return MaskedScalar(Accuracy(), mask_id=mask_id, has_weights=has_weights)  # pylint: disable=no-value-for-parameter
