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

  def __init__(self, learning_rate, *init_opt_params):
    """Initialize the optimizer.

    Takes the initial optimizer parameters as positional arguments. They are fed
    back to the optimizer in tree_update, in the same order. They can be changed
    between updates, e.g. for learning rate schedules.

    The constructor should be overridden in derived classes to give names to the
    optimizer parameters, so the gin configuration can set them.

    Args:
      learning_rate: The initial learning rate.
      *init_opt_params: Initial values of any additional optimizer parameters.
    """
    self._init_opt_params = tuple(
        map(np.array, (learning_rate,) + init_opt_params))

  def init(self, params):
    """Create optimizer slots for the given parameters."""
    raise NotImplementedError

  def update(self, step, grads, params, slots, opt_params):
    """Update a single parameter array.

    Args:
      step: Current step.
      grads: Gradients.
      params: Parameters.
      slots: Optimizer slots (e.g. gradient moments).
      opt_params: Optimizer (hyper)parameters (e.g. learning rate, momentum).

    Returns:
      (new_params, new_slots)
    """
    raise NotImplementedError

  # End subclass interface.

  def tree_init(self, param_tree):
    return (
        [self.init(param) for param in tree_flatten(param_tree)],
        self._init_opt_params,
    )

  def _update_and_check(self, step, grads, params, slots, opt_params):
    """Update a single parameter array and check types."""
    new_params, new_slots = self.update(
        step, grads, params, slots, opt_params)
    if isinstance(params, np.ndarray):
      assert isinstance(new_params, np.ndarray), (
          "The type of the new parameter values should be np.ndarray; got %s" %
          type(new_params))
      assert new_params.dtype == params.dtype, (
          "The dtype of the new parameter values (%s) is not the same as the "
          "old one (%s)" % (new_params.dtype, params.dtype))
    return new_params, new_slots

  def tree_update(self, step, grad_tree, param_tree, slots, opt_params):
    grads_flat = tree_flatten(grad_tree)
    params_flat = tree_flatten(param_tree)
    updated_pairs = [
        self._update_and_check(step, grad, param, slot, opt_params)
        for (grad, param, slot) in zip(grads_flat, params_flat, slots)
    ]
    new_params_flat, new_slots = zip(*updated_pairs)
    new_params, _ = tree_unflatten(new_params_flat, param_tree)
    return new_params, new_slots


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

  def init(self, params):
    return None

  def update(self, step, grads, params, slots, opt_params):
    del step
    del slots
    (learning_rate,) = opt_params
    return params - (learning_rate * grads).astype(params.dtype), None


class Momentum(Optimizer):
  """Nesterov momentum optimizer."""

  def __init__(self, learning_rate, mass=0.9):  # pylint: disable=useless-super-delegation
    super(Momentum, self).__init__(learning_rate, mass)

  def init(self, params):
    return np.zeros_like(params)

  def update(self, step, grads, params, velocity, opt_params):
    del step
    (learning_rate, mass) = opt_params
    new_velocity = mass * velocity - (1. - mass) * grads
    new_params = params + (learning_rate * new_velocity).astype(params.dtype)
    return (new_params, new_velocity)


class RMSProp(Optimizer):
  """RMSProp optimizer."""

  def __init__(self, learning_rate, gamma=0.9, eps=1e-8):  # pylint: disable=useless-super-delegation
    super(RMSProp, self).__init__(learning_rate, gamma, eps)

  def init(self, params):
    return np.ones_like(params)

  def update(self, step, grads, params, avg_sq_grad, opt_params):
    del step
    (learning_rate, gamma, eps) = opt_params
    avg_sq_grad = avg_sq_grad * gamma + grads**2 * (1. - gamma)
    params = params - (learning_rate * grads /
                       (np.sqrt(avg_sq_grad) + eps)).astype(params.dtype)
    return params, avg_sq_grad


class Adam(Optimizer):
  """Adam optimizer."""

  def __init__(self, learning_rate, weight_decay_rate=1e-5,  # pylint: disable=useless-super-delegation
               b1=0.9, b2=0.999, eps=1e-5):
    """Create the Adam optimizer.

    Args:
      learning_rate: a postitive scalar value for the initial learning rate.
      weight_decay_rate: rate at which to decay weights.
      b1: optional, a positive scalar value for beta_1, the exponential decay
        rate for the first moment estimates (default 0.9).
      b2: optional, a positive scalar value for beta_2, the exponential decay
         rate for the second moment estimates (default 0.999).
      eps: optional, a positive scalar value for epsilon, a small constant for
        numerical stability (default 1e-8).
    """
    super(Adam, self).__init__(learning_rate, weight_decay_rate, b1, b2, eps)

  def init(self, params):
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    return m, v

  def update(self, step, grads, params, slots, opt_params):
    m, v = slots
    learning_rate, weight_decay_rate, b1, b2, eps = opt_params
    m = (1 - b1) * grads + b1 * m  # First  moment estimate.
    v = (1 - b2) * (grads ** 2) + b2 * v  # Second moment estimate.
    mhat = m / (1 - b1 ** (step + 1))  # Bias correction.
    vhat = v / (1 - b2 ** (step + 1))
    params = (1 - weight_decay_rate) * params - (
        learning_rate * mhat / (np.sqrt(vhat) + eps)).astype(params.dtype)
    return params, (m, v)


class Adafactor(Optimizer):
  """Adafactor optimizer."""

  def __init__(self,
               learning_rate,
               factored=True,
               multiply_by_parameter_scale=True,
               do_clipping=True,
               do_momentum=False,
               beta1=0.0,
               decay_rate=0.8,
               clipping_threshold=1.0,
               weight_decay_rate=1e-5,
               epsilon1=1e-30,
               epsilon2=1e-3):
    """Create the Adafactor optimizer.

    Adafactor is described in https://arxiv.org/abs/1804.04235.

    Args:
      learning_rate: float: trax-provided learning rate.
      factored: boolean: whether to use factored second-moment estimator for 2d
        variables.
      multiply_by_parameter_scale: boolean: if True, then scale provided
        learning_rate by parameter norm. if False, provided learning_rate is
        absolute step size.
      do_clipping: whether to clip gradients; if True, set clipping_theshold.
      do_momentum: whether to use momentum; if True, set beta1.
      beta1: a float value between 0 and 1, enables momentum and uses extra
        memory if nonzero!  Off by default.
      decay_rate: float: controls second-moment exponential decay schedule.
      clipping_threshold: an optional float >= 1, if None no update clipping.
      weight_decay_rate: rate at which to decay weights.
      epsilon1: Regularization constant for squared gradient.
      epsilon2: Regularization constant for parameter scale.
    """
    # These 4 parameters are not configurable once the class is created.
    self._factored = factored
    self._multiply_by_parameter_scale = multiply_by_parameter_scale
    self._do_clipping = do_clipping
    self._do_momentum = do_momentum
    # Dynamically configurable parameters will be passed to the update function.
    super(Adafactor, self).__init__(
        learning_rate, beta1, decay_rate, clipping_threshold,
        weight_decay_rate, epsilon1, epsilon2)

  @staticmethod
  def _decay_rate_pow(i, exponent=0.8):
    """Default Adafactor second-moment decay schedule."""
    t = np.array(i, np.float32) + 1.0
    return 1.0 - t**(-exponent)

  def init(self, params):
    shape = params.shape
    slots = []
    if self._factored and len(shape) >= 2:
      v_row = np.zeros(shape[:-1], dtype=np.float32)
      v_col = np.zeros(shape[:-2] + shape[-1:], dtype=np.float32)
      slots.extend([v_row, v_col])
    else:
      v = np.zeros_like(params)
      slots.append(v)
    if self._do_momentum:
      m = np.zeros_like(params)
      slots.append(m)
    return slots

  def update(self, step, grads, params, slots, opt_params):
    updates = []
    (learning_rate, beta1, decay_rate, clipping_threshold,
     weight_decay_rate, epsilon1, epsilon2) = opt_params
    decay_rate = self._decay_rate_pow(step, exponent=decay_rate)
    update_scale = learning_rate
    if self._multiply_by_parameter_scale:
      update_scale *= np.maximum(
          np.sqrt(np.mean(params * params)), epsilon2)
    mixing_rate = 1.0 - decay_rate

    grads_sqr = grads * grads + epsilon1
    if self._factored and len(params.shape) >= 2:
      v_row = slots.pop(0)
      v_col = slots.pop(0)
      new_v_row = decay_rate * v_row + mixing_rate * np.mean(grads_sqr, axis=-1)
      new_v_col = decay_rate * v_col + mixing_rate * np.mean(grads_sqr, axis=-2)
      updates.extend([new_v_row, new_v_col])
      row_col_mean = np.mean(new_v_row, axis=-1, keepdims=True)
      row_factor = (new_v_row / row_col_mean)**-0.5
      col_factor = (new_v_col)**-0.5
      y = (
          grads * np.expand_dims(row_factor, axis=-1) *
          np.expand_dims(col_factor, axis=-2))
    else:
      v = slots.pop(0)
      new_v = decay_rate * v + mixing_rate * grads_sqr
      updates.append(new_v)
      y = grads * (new_v)**-0.5

    if self._do_clipping:
      clipping_denom = (
          np.maximum(1.0, np.sqrt(np.mean(y * y)) / clipping_threshold))
      y /= clipping_denom

    subtrahend = update_scale * y
    if self._do_momentum:
      m = slots.pop(0)
      new_m = beta1 * m + (1.0 - beta1) * subtrahend
      subtrahend = new_m
      updates.append(new_m)

    new_params = (1 - weight_decay_rate) * params - subtrahend
    return new_params, updates


class SM3(Optimizer):
  """SM3 optimizer."""

  def __init__(self, learning_rate, momentum=0.9):  # pylint: disable=useless-super-delegation
    """Create the SM3 optimizer.

    Memory-Efficient Adaptive Optimization for Large-Scale Learning.
    https://arxiv.org/abs/1901.11150

    Args:
      learning_rate: a postitive scalar value for the initial learning rate.
      momentum: optional, a positive scalar value for momentum
    """
    super(SM3, self).__init__(learning_rate, momentum)

  def init(self, params):
    vs = [np.zeros(sz, dtype=params.dtype) for sz in params.shape]
    return (np.zeros_like(params), vs)

  def _update_diagonal(self, grads, params, m, v, opt_params):
    (learning_rate, momentum) = opt_params
    v[0] += grads * grads
    preconditioner = np.where(v[0] > 0, 1.0 / np.sqrt(v[0]),
                              np.zeros_like(v[0]))
    preconditioned_grads = preconditioner * grads
    m = (1 - momentum) * preconditioned_grads + momentum * m
    params = params - (learning_rate * m).astype(params.dtype)
    return params, (m, v)

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

  def _update_sketched(self, grads, params, m, v, opt_params):
    """Update for higher-rank parameters."""
    (learning_rate, momentum) = opt_params
    shape = params.shape
    rank = len(shape)
    reshaped_accumulators = [np.reshape(v[i], self._expanded_shape(shape, i))
                             for i in range(rank)]
    current_accumulator = self._minimum(reshaped_accumulators)
    current_accumulator += grads * grads
    accumulator_inv_sqrt = np.where(current_accumulator > 0.0,
                                    1.0 / np.sqrt(current_accumulator),
                                    np.zeros_like(current_accumulator))
    preconditioned_gradient = grads * accumulator_inv_sqrt
    m = (1.0 - momentum) * preconditioned_gradient + momentum * m
    params = params - (learning_rate * m).astype(params.dtype)
    for i in range(len(v)):
      axes = list(range(int(i))) + list(range(int(i) + 1, rank))
      dim_accumulator = np.amax(current_accumulator, axis=axes)
      v[i] = dim_accumulator
    return params, (m, v)

  def update(self, step, grads, params, slots, opt_params):
    del step
    m, v = slots
    shape = params.shape
    rank = len(shape)
    if rank > 1:
      return self._update_sketched(grads, params, m, v, opt_params)
    else:
      return self._update_diagonal(grads, params, m, v, opt_params)
