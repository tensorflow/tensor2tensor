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

"""Trax base optimizer class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.trax.backend import numpy as np
from tensor2tensor.trax.layers import base as layers


def tree_flatten(tree):
  """Flatten a tree into a list."""
  if isinstance(tree, (list, tuple)):
    # In python, sum of lists starting from [] is the concatenation.
    return sum([tree_flatten(t) for t in tree], [])
  if isinstance(tree, dict):
    # Only use the values in case of a dictionary node.
    return sum([tree_flatten(v) for v in tree.values()], [])
  return [tree]


def tree_unflatten(flat, tree):
  """Unflatten a list into a tree given the tree shape as second argument.

  Args:
    flat: a flat list of elements to be assembled into a tree.
    tree: a tree with the structure we want to have in the new tree.

  Returns:
    A pair (new_tree, rest_of_flat) where the new tree that has the structure
    of tree but with leaves from flat, and the remaining elements of flat if
    more were provided than the number of leaves of tree (useful for recursion).
  """
  if isinstance(tree, (list, tuple)):
    new_tree, rest = [], flat
    for t in tree:
      new_t, rest = tree_unflatten(rest, t)
      new_tree.append(new_t)
    new_tree = tuple(new_tree) if isinstance(tree, tuple) else new_tree
    return new_tree, rest
  if isinstance(tree, dict):
    new_tree, rest = {}, flat
    for k in tree:
      new_v, rest = tree_unflatten(rest, tree[k])
      new_tree[k] = new_v
    return new_tree, rest
  return flat[0], flat[1:]


class Optimizer(object):
  """Optimizer object, base class. Maps per-parameter functions to trees."""

  def __init__(self, step_size):
    """Optimizers take the step size function (learning rate) as argument."""
    if callable(step_size):
      self._step_size = step_size
    else:
      self._step_size = lambda _: step_size

  def init(self, x):
    """Create optimizer slots for the parameter x."""
    raise NotImplementedError

  def update(self, i, g, x, s):
    """Update the parameter x at step i with gradient g using state s."""
    raise NotImplementedError

  # End subclass interface.

  def step_size(self, i):
    return self._step_size(i)

  def tree_init(self, x_tree):
    return [self.init(x) for x in tree_flatten(x_tree)]

  def _update_and_check(self, i, g, x, s):
    new_x, new_s = self.update(i, g, x, s)
    if isinstance(x, np.ndarray):
      assert isinstance(new_x, np.ndarray), ("The type of the new parameter "
                                             "values should be np.ndarray; "
                                             "got %s" % type(new_x))
      assert new_x.dtype == x.dtype, ("The dtype of the new parameter values "
                                      "(%s) is not the same as the old one (%s)"
                                      % (new_x.dtype, x.dtype))
    return new_x, new_s

  def tree_update(self, i, grad_tree, x_tree, opt_state):
    grad_flat = tree_flatten(grad_tree)
    x_flat = tree_flatten(x_tree)
    updated_pairs = [self._update_and_check(i, g, x, s)
                     for (g, x, s) in zip(grad_flat, x_flat, opt_state)]
    new_x_flat, new_opt_state = zip(*updated_pairs)
    new_x, _ = tree_unflatten(new_x_flat, x_tree)
    return new_x, new_opt_state


# Utilities.


def l2_norm(tree):
  """Compute the l2 norm of a pytree of arrays. Useful for weight decay."""
  leaves = tree_flatten(tree)
  return np.sqrt(sum(np.vdot(x, x) for x in leaves))


def clip_grads(grad_tree, max_norm):
  """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
  norm = l2_norm(grad_tree)
  normalize = lambda g: np.where(norm < max_norm, g, g * (max_norm / norm))
  return layers.nested_map(normalize, grad_tree)


# Optimizers.


class SGD(Optimizer):
  """Plain SGD optimizer."""

  def init(self, x):
    return None

  def update(self, i, g, x, state):
    del state
    return x - (self.step_size(i) * g).astype(x.dtype), None


class Momentum(Optimizer):
  """Nestrov momentum optimizer."""

  def __init__(self, step_size, mass=0.9):
    """Initializer with a step size function and mass."""
    super(Momentum, self).__init__(step_size)
    self._mass = mass

  def init(self, x):
    return np.zeros_like(x)

  def update(self, i, g, x, velocity):
    new_velocity = self._mass * velocity - (1. - self._mass) * g
    return x + (self.step_size(i) * new_velocity).astype(x.dtype), new_velocity


class RMSProp(Optimizer):
  """RMSProp optimizer."""

  def __init__(self, step_size, gamma=0.9, eps=1e-8):
    """Initializer with a step size function, gamma and epsilon."""
    super(RMSProp, self).__init__(step_size)
    self._gamma = gamma
    self._epsilon = eps

  def init(self, x):
    return np.ones_like(x)

  def update(self, i, g, x, avg_sq_grad):
    avg_sq_grad = avg_sq_grad * self._gamma + g**2 * (1. - self._gamma)
    x = x - (self.step_size(i) * g /
             (np.sqrt(avg_sq_grad) + self._epsilon)).astype(x.dtype)
    return x, avg_sq_grad


class Adam(Optimizer):
  """Adam optimizer."""

  def __init__(self, step_size, b1=0.9, b2=0.999, eps=1e-8):
    """Create the Adam optimizer.

    Args:
      step_size: a callable representing a step size schedule
        that maps the iteration index to positive scalar.
      b1: optional, a positive scalar value for beta_1, the exponential decay
        rate for the first moment estimates (default 0.9).
      b2: optional, a positive scalar value for beta_2, the exponential decay
         rate for the second moment estimates (default 0.999).
      eps: optional, a positive scalar value for epsilon, a small constant for
        numerical stability (default 1e-8).
    """
    super(Adam, self).__init__(step_size)
    self._b1 = b1
    self._b2 = b2
    self._eps = eps

  def init(self, x):
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    return m, v

  def update(self, i, g, x, state):
    m, v = state
    b1, b2, eps = self._b1, self._b2, self._eps
    m = (1 - b1) * g + b1 * m  # First  moment estimate.
    v = (1 - b2) * (g ** 2) + b2 * v  # Second moment estimate.
    mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
    vhat = v / (1 - b2 ** (i + 1))
    x = x - (self.step_size(i) * mhat / (np.sqrt(vhat) + eps)).astype(x.dtype)
    return x, (m, v)


class SM3(Optimizer):
  """SM3 optimizer."""

  def __init__(self, step_size, momentum=0.9):
    """Create the SM3 optimizer.

    Memory-Efficient Adaptive Optimization for Large-Scale Learning.
    https://arxiv.org/abs/1901.11150

    Args:
      step_size: a callable representing a step size schedule
        that maps the iteration index to positive scalar.
      momentum: optional, a positive scalar value for momentum
    """
    super(SM3, self).__init__(step_size)
    self._momentum = momentum

  def init(self, x):
    vs = [np.zeros(sz, dtype=x.dtype) for sz in x.shape]
    return (np.zeros_like(x), vs)

  def _update_diagonal(self, step, g, x, m, v):
    v[0] += g * g
    preconditioner = np.where(v[0] > 0, 1.0 / np.sqrt(v[0]),
                              np.zeros_like(v[0]))
    preconditioned_g = preconditioner * g
    m = (1 - self._momentum) * preconditioned_g + self._momentum * m
    x = x - (self.step_size(step) * m).astype(x.dtype)
    return x, (m, v)

  def _expanded_shape(self, shape, axis):
    # Replaces a `shape` of [M, N, K] with 1 in all dimensions except for i.
    # For eg: i = 1 returns [1, N, 1].
    rank = len(shape)
    return [1] * axis + [shape[axis]] + [1] * (rank - axis - 1)

  def _minimum(self, tensor_list):
    minimum = tensor_list[0]
    for i in range(1, len(tensor_list)):
      minimum = np.minimum(minimum, tensor_list[i])
    return minimum

  def _update_sketched(self, step, g, x, m, v):
    """Update for higher-rank parameters."""
    shape = x.shape
    rank = len(shape)
    reshaped_accumulators = [np.reshape(v[i], self._expanded_shape(shape, i))
                             for i in range(rank)]
    current_accumulator = self._minimum(reshaped_accumulators)
    current_accumulator += g * g
    accumulator_inv_sqrt = np.where(current_accumulator > 0.0,
                                    1.0 / np.sqrt(current_accumulator),
                                    np.zeros_like(current_accumulator))
    preconditioned_gradient = g * accumulator_inv_sqrt
    m = (1.0 - self._momentum) * preconditioned_gradient + self._momentum * m
    x = x - (self.step_size(step) * m).astype(x.dtype)
    for i in range(len(v)):
      axes = list(range(int(i))) + list(range(int(i) + 1, rank))
      dim_accumulator = np.amax(current_accumulator, axis=axes)
      v[i] = dim_accumulator
    return x, (m, v)

  def update(self, i, g, x, state):
    m, v = state
    shape = x.shape
    rank = len(shape)
    if rank > 1:
      return self._update_sketched(i, g, x, m, v)
    else:
      return self._update_diagonal(i, g, x, m, v)
