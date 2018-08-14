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
"""Utilities to assist in pruning models."""

import numpy as np

from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_pruning_strategy
def weight(w, sparsity):
  """Weight-level magnitude pruning."""
  w_shape = common_layers.shape_list(w)
  k = int(np.prod(w_shape[:-1]))
  count = tf.to_int32(k * sparsity)
  mask = common_layers.weight_targeting(w, count)
  return (1 - mask) * w


@registry.register_pruning_strategy
def unit(w, sparsity):
  """Unit-level magnitude pruning."""
  w_shape = common_layers.shape_list(w)
  count = tf.to_int32(w_shape[-1] * sparsity)
  mask = common_layers.unit_targeting(w, count)
  return (1 - mask) * w


def sparsify(sess, eval_model, pruning_strategy, pruning_params):
  """Prune the weights of a model and evaluate."""
  weights = tf.trainable_variables()

  def should_prune(name):
    """Whether to prune a weight or not."""
    in_whitelist = not pruning_params.white_list or any(
        e in name for e in pruning_params.white_list)
    in_blacklist = any(e in name for e in pruning_params.black_list)

    if pruning_params.white_list and not in_whitelist:
      return False
    elif in_blacklist:
      return False

    return True

  weights = [w for w in weights if should_prune(w.name)]
  tf.logging.info("Pruning weights: %s" % weights)
  unpruned_weights = sess.run(weights)

  reset_op = tf.no_op()
  for w, ow in zip(weights, unpruned_weights):
    op = tf.assign(w, ow)
    reset_op = tf.group(reset_op, op)

  for sparsity in pruning_params.sparsities:
    set_weights_op = tf.no_op()
    for w in weights:
      op = tf.assign(w, pruning_strategy(w, sparsity))
      set_weights_op = tf.group(set_weights_op, op)
    sess.run(set_weights_op)

    acc = eval_model()
    tf.logging.info("\tPruning to sparsity = %f: acc = %f" % (sparsity, acc))
    sess.run(reset_op)
