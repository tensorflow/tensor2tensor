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

"""Trax backend: all the primitive functions needed."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import gin

import jax
from jax import lax
from jax import random as jax_random
import jax.numpy as jnp
import jax.scipy.special as jax_special
import numpy as onp
import tensorflow_datasets as tfds



def jax_conv(inp, fltr, window_strides, padding, dimension_numbers,
             filter_dilation=None):
  """A wrapper around `lax.conv_general_dilated`.

  It requires `dimension_numbers` and disallows `inp_dilation`.

  Args:
    inp: an (N+2)-D array. The input of the convolution.
    fltr: an (N+2)-D array. The filter (i.e. kernel) of the convolution.
    window_strides: the strides for moving the convolution window.
    padding: a string, either "VALID" or "SAME". The padding algorithm.
    dimension_numbers: a tuple of three strings encoding the data format of
      input, filter and output. "I" means input; "O" means output; "C" means
      channel; other characters such as "W", "H" and "D" means spatial
      dimensions.
    filter_dilation: the dilation rates for the filter. Dilating the filter
      means adding "holes" to the filter.

  Returns:
    An (N+2)-D array. The convolution result.
  """
  return lax.conv_general_dilated(inp, fltr, window_strides, padding,
                                  lhs_dilation=None,
                                  rhs_dilation=filter_dilation,
                                  dimension_numbers=dimension_numbers)


def _pooling_general(inputs, reducer, init_val, rescaler=None,
                     pool_size=(2, 2), strides=None, padding="VALID"):
  """Helper: general pooling computation used in pooling layers later."""
  spatial_strides = strides or (1,) * len(pool_size)
  rescale = rescaler(pool_size, spatial_strides, padding) if rescaler else None
  dims = (1,) + pool_size + (1,)  # NHWC
  strides = (1,) + spatial_strides + (1,)
  out = lax.reduce_window(inputs, init_val, reducer, dims, strides, padding)
  return rescale(out, inputs) if rescale else out


def jax_max_pool(x, pool_size, strides, padding):
  return _pooling_general(x, lax.max, -jnp.inf, pool_size=pool_size,
                          strides=strides, padding=padding)


def jax_sum_pool(x, pool_size, strides, padding):
  return _pooling_general(x, lax.add, 0., pool_size=pool_size,
                          strides=strides, padding=padding)


def _normalize_by_window_size(dims, spatial_strides, padding):  # pylint: disable=invalid-name
  def rescale(outputs, inputs):
    one = jnp.ones(inputs.shape[1:-1], dtype=inputs.dtype)
    window_sizes = lax.reduce_window(
        one, 0., lax.add, dims, spatial_strides, padding)
    return outputs / window_sizes[..., jnp.newaxis]
  return rescale


def jax_avg_pool(x, pool_size, strides, padding):
  return _pooling_general(x, lax.add, 0., _normalize_by_window_size,
                          pool_size, strides=strides, padding=padding)


def nested_map(x, f):
  """Map the function f to the nested structure x (dicts, tuples, lists)."""
  if isinstance(x, list):
    return [nested_map(y, f) for y in x]
  if isinstance(x, tuple):
    return tuple([nested_map(y, f) for y in x])
  return f(x)


class ShapeType(object):
  """Store shape and type."""

  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = dtype

  def __repr__(self):
    return "[shape:" + str(self.shape) + ", dtype:" + str(self.dtype) + "]"


def jax_eval_on_shapes(f):
  """Returns a function that evaluates `f` given input shapes and dtypes.

  It transforms function `f` to a function that performs the same computation as
  `f` but only on shapes and dtypes (a.k.a. shape inference).

  Args:
    f: the function to be transformed.

  Returns:
    A function whose input arguments can be either the same as `f`'s or only
    their shapes/dtypes represented by `ShapeType`, and whose return values are
    `ShapeType`s with the same nested structure as `f`'s return values.
  """
  def shape_fun(*args, **kwargs):
    jax_shapes = jax.eval_shape(f, *args, **kwargs)
    return nested_map(jax_shapes, lambda x: ShapeType(x.shape, x.dtype))
  return shape_fun


# The default value of dtype is different from jax_random.randint
def jax_randint(key, shape, minval, maxval, dtype=onp.int32):
  """Sample uniform random values in [minval, maxval) with given shape/dtype.

  Args:
    key: a PRNGKey used as the random key.
    shape: a tuple of nonnegative integers representing the shape.
    minval: int or array of ints broadcast-compatible with ``shape``, a minimum
      (inclusive) value for the range.
    maxval: int or array of ints broadcast-compatible with  ``shape``, a maximum
      (exclusive) value for the range.
    dtype: optional, an int dtype for the returned values (default int32).

  Returns:
    A random array with the specified shape and dtype.
  """
  return jax_random.randint(key, shape, minval=minval, maxval=maxval,
                            dtype=dtype)


_JAX_BACKEND = {
    "name": "jax",
    "np": jnp,
    "logsumexp": jax_special.logsumexp,
    "conv": jax_conv,
    "avg_pool": jax_avg_pool,
    "max_pool": jax_max_pool,
    "sum_pool": jax_sum_pool,
    "jit": jax.jit,
    "grad": jax.grad,
    "pmap": jax.pmap,
    "eval_on_shapes": jax_eval_on_shapes,
    "random_uniform": jax_random.uniform,
    "random_randint": jax_randint,
    "random_normal": jax_random.normal,
    "random_bernoulli": jax_random.bernoulli,
    "random_get_prng": jax.jit(jax_random.PRNGKey),
    "random_split": jax_random.split,
    "dataset_as_numpy": tfds.as_numpy,
}


_NUMPY_BACKEND = {
    "name": "numpy",
    "np": onp,
    "jit": (lambda f: f),
    "random_get_prng": lambda seed: None,
    "random_split": lambda prng, num=2: (None,) * num,
}


def get_name():
  return backend()["name"]


def logsumexp(*args, **kwargs):
  return backend()["logsumexp"](*args, **kwargs)


def conv(*args, **kwargs):
  return backend()["conv"](*args, **kwargs)


def avg_pool(*args, **kwargs):
  return backend()["avg_pool"](*args, **kwargs)


def max_pool(*args, **kwargs):
  return backend()["max_pool"](*args, **kwargs)


def sum_pool(*args, **kwargs):
  return backend()["sum_pool"](*args, **kwargs)


def jit(*args, **kwargs):
  return backend()["jit"](*args, **kwargs)


def grad(*args, **kwargs):
  return backend()["grad"](*args, **kwargs)


def pmap(*args, **kwargs):
  return backend()["pmap"](*args, **kwargs)


def eval_on_shapes(*args, **kwargs):
  return backend()["eval_on_shapes"](*args, **kwargs)


def dataset_as_numpy(*args, **kwargs):
  return backend()["dataset_as_numpy"](*args, **kwargs)


# For numpy and random modules, we need to call "backend()" lazily, only when
# the function is called -- so that it can be set by gin configs.
# (Otherwise, backend() is called on import before gin-config is parsed.)
# To do that, we make objects to encapsulated these modules.


class RandomBackend(object):
  """Backend providing random functions."""

  def get_prng(self, seed):
    return backend()["random_get_prng"](seed)

  def split(self, prng, num=2):
    return backend()["random_split"](prng, num)

  def uniform(self, *args, **kwargs):
    return backend()["random_uniform"](*args, **kwargs)

  def randint(self, *args, **kwargs):
    return backend()["random_randint"](*args, **kwargs)

  def normal(self, *args, **kwargs):
    return backend()["random_normal"](*args, **kwargs)

  def bernoulli(self, *args, **kwargs):
    return backend()["random_bernoulli"](*args, **kwargs)


random = RandomBackend()


# A class that just forwards attribute accesses to backend's numpy object.
class NumpyBackend(object):

  def __getattr__(self, attr):
    return getattr(backend()["np"], attr)


numpy = NumpyBackend()




override_backend_name = None


@gin.configurable()
def backend(name="jax"):
  name = name if not override_backend_name else override_backend_name
  if name == "numpy":
    return _NUMPY_BACKEND
  return _JAX_BACKEND


@contextlib.contextmanager
def use_backend(name):
  global override_backend_name
  prev_name = override_backend_name
  override_backend_name = name
  # Run the decorated function in try-finally in case it throws, e.g. for tests.
  try:
    yield
  finally:
    override_backend_name = prev_name
