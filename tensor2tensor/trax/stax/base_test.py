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

"""Tests for Stax base layer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl.testing import absltest
from jax import random
import numpy as onp
import tensor2tensor.trax.stax as stax


def random_inputs(rng, input_shape):
  if isinstance(input_shape, tuple):
    return rng.randn(*input_shape).astype(onp.float32)
  elif isinstance(input_shape, list):
    return [random_inputs(rng, shape) for shape in input_shape]
  else:
    raise TypeError(type(input_shape))


def check_shape_agreement(test_case, init_fun, apply_fun, input_shape):
  rng_key1, rng_key2 = random.split(random.PRNGKey(0))
  result_shape, params = init_fun(rng_key1, input_shape)
  inputs = random_inputs(onp.random.RandomState(0), input_shape)
  result = apply_fun(params, inputs, rng=rng_key2)
  test_case.assertEqual(result.shape, result_shape)
  return result_shape


def check_staxlayer(test_case, staxlayer, input_shape):
  init_fun, apply_fun = staxlayer
  return check_shape_agreement(test_case, init_fun, apply_fun, input_shape)


# Helper functions for testing Lambda wrapper against functions involving
# complicated input trees:
def _enumerate_trees_w_leaves(n_leaves):
  """Construct all rooted trees with n leaves."""
  def enumtree(*args):
    n_args = len(args)
    # trivial cases:
    if n_args == 0:
      return []
    if n_args == 1:
      return args
    # general case of 2 or more args:
    # build index array
    idxs = range(0, n_args)
    trees = []
    # we consider all possible subsets of size n_set to gather
    for n_set in range(2, n_args+1):
      idxsets = list(itertools.combinations(idxs, n_set))
      for idxset in idxsets:
        # recurse by joining all subtrees with
        # n_set leaves and (n_args - n_set) leaves
        arg_set = tuple(args[i] for i in idxs if i in idxset)
        arg_coset = tuple(args[i] for i in idxs if i not in idxset)
        if arg_coset:
          trees.extend(tuple(itertools.product(enumtree(*arg_set),
                                               enumtree(*arg_coset))))
        else:
          # trivial case where arg_set is entire set
          trees.append(arg_set)
    return trees
  # return enumerated trees with integers as leaves
  return enumtree(*range(n_leaves))


def _build_combinator_tree(input_treespec, in_vars):
  """Build a trivial Staxlayer that takes a complicated tree of inputs."""
  parallel_args = []
  for e in input_treespec:
    if isinstance(e, int):
      parallel_args.append(in_vars[e])
    elif isinstance(e, tuple):
      parallel_args.append(_build_combinator_tree(e, in_vars))
  return stax.serial(stax.parallel(*parallel_args), stax.FanInSum)


class SlaxTest(absltest.TestCase):

  def test_flatten_n(self):
    input_shape = (29, 87, 10, 20, 30)

    actual_shape = check_staxlayer(self, stax.Flatten(1), input_shape)
    self.assertEqual(actual_shape, (29, 87 * 10 * 20 * 30))

    actual_shape = check_staxlayer(self, stax.Flatten(2), input_shape)
    self.assertEqual(actual_shape, (29, 87, 10 * 20 * 30))

    actual_shape = check_staxlayer(self, stax.Flatten(3), input_shape)
    self.assertEqual(actual_shape, (29, 87, 10, 20 * 30))

    actual_shape = check_staxlayer(self, stax.Flatten(4), input_shape)
    self.assertEqual(actual_shape, (29, 87, 10, 20, 30))

    # Not enough dimensions.
    with self.assertRaises(ValueError):
      check_staxlayer(self, stax.Flatten(5), input_shape)

    with self.assertRaises(ValueError):
      check_staxlayer(self, stax.Flatten(6), input_shape)

  def test_div(self):
    init_fun, apply_fun = stax.Div(2)
    input_np = onp.array([[1, 2, 3], [4, 5, 6]], dtype=onp.float32)
    input_shape = input_np.shape
    _, _ = init_fun(None, input_shape)
    output_np = apply_fun(None, input_np)
    # absltest doesn't have ndarray equalities.
    expected_output_np = input_np / 2.0
    self.assertAlmostEqual(
        0.0,
        onp.sum((output_np - expected_output_np) ** 2),
        delta=1e-6)


if __name__ == "__main__":
  absltest.main()
