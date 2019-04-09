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

"""Tests for Stax Extensions."""
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


def check_staxlayer(test_case, staxlayer, input_shape):
  init_fun, apply_fun = staxlayer
  check_shape_agreement(test_case, init_fun, apply_fun, input_shape)


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

  # Lambdas replace the staxlayer input stream with a placeholder that
  # _should_ break any use of unbound variables in the input stream.
  def testLambda_forbidden_access(self):
    with self.assertRaises(ValueError):
      for tree_spec in _enumerate_trees_w_leaves(2):
        @stax.Lambda
        def lambda_fun(x, y):  # pylint: disable=unused-argument
          return _build_combinator_tree(tree_spec,  # pylint: disable=cell-var-from-loop
                                        # try to read from input stream
                                        # rather than bound vars
                                        (x, stax.Identity))
        check_staxlayer(self, lambda_fun, [(1, 5, 7, 11),]*2)

  # Exhaustively test the tricky part of Lambda - input combinator
  # "initialization" for all 2412 trees of stax serial and parallel
  # combinators of up to six variables.  This probably covers most
  # practical use patterns!

  # The variables in for loops below are used immediately, disable lint warning
  # for this section:
  # pylint: disable=cell-var-from-loop
  def testLambda_1_arg(self):
    @stax.Lambda
    def lambda_fun(x):
      return _build_combinator_tree((0,), (x,))
    check_staxlayer(self, lambda_fun, (1, 5, 7, 11))

  def testLambda_2_args(self):
    for tree_spec in _enumerate_trees_w_leaves(2):
      @stax.Lambda
      def lambda_fun(x, y):
        return _build_combinator_tree(tree_spec, (x, y))
      check_staxlayer(self, lambda_fun, [(1, 5, 7, 11),]*2)

  def testLambda_3_args(self):
    for tree_spec in _enumerate_trees_w_leaves(3):
      @stax.Lambda
      def lambda_fun(x, y, z):
        return _build_combinator_tree(tree_spec, (x, y, z))
      check_staxlayer(self, lambda_fun, [(1, 5, 7, 11),]*3)

  def testLambda_4_args(self):
    for tree_spec in _enumerate_trees_w_leaves(4):
      @stax.Lambda
      def lambda_fun(x, y, z, w):
        return _build_combinator_tree(tree_spec, (x, y, z, w))
      check_staxlayer(self, lambda_fun, [(1, 5, 7, 11),]*4)

  def testLambda_5_args(self):
    for tree_spec in _enumerate_trees_w_leaves(5):
      @stax.Lambda
      def lambda_fun(x, y, z, w, v):
        return _build_combinator_tree(tree_spec, (x, y, z, w, v))
      check_staxlayer(self, lambda_fun, [(1, 5, 7, 11),]*5)

  # TODO(mattjj,levskaya): timing out, re-enable with longer timeout?
  def DISABLED_testLambda_6_args(self):  # pylint: disable=invalid-name
    for tree_spec in _enumerate_trees_w_leaves(6):
      @stax.Lambda
      def lambda_fun(x, y, z, w, v, u):
        return _build_combinator_tree(tree_spec, (x, y, z, w, v, u))
      check_staxlayer(self, lambda_fun, [(1, 5, 7, 11),]*6)

  # Test a few other cases, unused variables, non-input-tree use of
  # bound Lambda input variables.
  def testLambda_4_args_only_3_used(self):
    for tree_spec in _enumerate_trees_w_leaves(3):
      @stax.Lambda
      def lambda_fun(x, y, z, w):  # pylint: disable=unused-argument
        return _build_combinator_tree(tree_spec, (x, y, z))
      check_staxlayer(self, lambda_fun, [(1, 5, 7, 11),]*4)

  def testLambda_4_args_only_2_used(self):
    for tree_spec in _enumerate_trees_w_leaves(2):
      @stax.Lambda
      def lambda_fun(x, y, z, w):  # pylint: disable=unused-argument
        return _build_combinator_tree(tree_spec, (x, y))
      check_staxlayer(self, lambda_fun, [(1, 5, 7, 11),]*4)

  def testLambda_4_args_only_1_used(self):
    @stax.Lambda
    def lambda_fun(x, y, z, w):  # pylint: disable=unused-argument
      return _build_combinator_tree((0,), (x,))
    check_staxlayer(self, lambda_fun, [(1, 5, 7, 11),]*4)

  def testLambda_5_args_2_post_input_tree(self):
    for tree_spec in _enumerate_trees_w_leaves(3):
      @stax.Lambda
      def lambda_fun1(x, y, z, w, v):
        input_tree = _build_combinator_tree(tree_spec, (x, y, z))
        return stax.serial(input_tree,
                           stax.FanOut(3),
                           stax.parallel(stax.Identity, w, v),
                           stax.FanInSum)
      check_staxlayer(self, lambda_fun1, [(1, 5, 7, 11),]*5)

      @stax.Lambda
      def lambda_fun2(x, y, z, w, v):
        input_tree = _build_combinator_tree(tree_spec, (x, y, z))
        return stax.serial(input_tree,
                           stax.FanOut(3),
                           stax.parallel(w, stax.Identity, v),
                           stax.FanInSum)
      check_staxlayer(self, lambda_fun2, [(1, 5, 7, 11),]*5)

      @stax.Lambda
      def lambda_fun3(x, y, z, w, v):
        input_tree = _build_combinator_tree(tree_spec, (x, y, z))
        return stax.serial(input_tree,
                           stax.FanOut(3),
                           stax.parallel(w, v, stax.Identity),
                           stax.FanInSum)
      check_staxlayer(self, lambda_fun3, [(1, 5, 7, 11),]*5)
  # pylint: enable=cell-var-from-loop


if __name__ == "__main__":
  absltest.main()
