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
from jax import random as jax_random
import jax.numpy as jnp
import jax.scipy.special as jax_special
import numpy as onp



_JAX_BACKEND = {
    "name": "jax",
    "np": jnp,
    "logsumexp": jax_special.logsumexp,
    "jit": jax.jit,
    "grad": jax.grad,
    "pmap": jax.pmap,
    "random_uniform": jax_random.uniform,
    "random_normal": jax_random.normal,
    "random_bernoulli": jax_random.bernoulli,
    "random_get_prng": jax.jit(jax_random.PRNGKey),
    "random_split": jax_random.split,
}


_NUMPY_BACKEND = {
    "name": "numpy",
    "np": onp,
    "jit": (lambda f: f),
}


def get_name():
  return backend()["name"]


def logsumexp(*args, **kwargs):
  return backend()["logsumexp"](*args, **kwargs)


def jit(*args, **kwargs):
  return backend()["jit"](*args, **kwargs)


def grad(*args, **kwargs):
  return backend()["grad"](*args, **kwargs)


def pmap(*args, **kwargs):
  return backend()["pmap"](*args, **kwargs)


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
  yield
  override_backend_name = prev_name
