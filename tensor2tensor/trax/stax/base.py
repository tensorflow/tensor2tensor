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

"""Base layer class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from jax.tree_util import register_pytree_node as _register_pytree_node


# Staxlayer binding to python variables
# ------------------------------------------------------------------------------
# Stax params-tree leaf type to mark bound subtrees references.
class _TreeMarker(dict):
  pass
# Add this leaf-type to JAX's tree-walker.
_register_pytree_node(_TreeMarker,
                      lambda xs: (tuple(), None),
                      lambda _, xs: _TreeMarker())


# TODO(lukaszkaiser): make this the base layer class (share by object).
class Share(tuple):
  """Layer parameter caching function to allow weight sharing.

  Args:
    A staxlayer: an (init_fun, apply_fun) pair.

  Returns:
    A 'parameter-bound' staxlayer that can be assigned to a python variable.
  Wherever this value is needed elsewhere in the stax tree, call this bound
  variable and all occurrences will share parameters that will automatically
  be updated by Stax optimizers.
  """

  def __init__(self, staxlayer):  # pylint: disable=super-init-not-called
    self._orig_init_fun, self._orig_apply_fun = staxlayer
    self._first_init = True
    self.params = None  # cached staxlayer params

  def _init_fun(self, rng_key, input_shape):  # pylint: disable=missing-docstring
    if self._first_init:
      # point of first subgraph initialization call: sets params, output_shape
      self._first_init = False
      out_shape, self.params = self._orig_init_fun(rng_key, input_shape)
      return out_shape, self.params
    else:
      # point of subgraph reuse:
      # params are just a marker to apply_funs signalling subgraph params reuse
      out_shape, _ = self._orig_init_fun(rng_key, input_shape)
      return out_shape, _TreeMarker()

  def _apply_fun(self, params, inputs, **kwargs):
    if isinstance(params, _TreeMarker):
      # point of subgraph reuse: calculate new value with cached params
      return self._orig_apply_fun(self.params, inputs, **kwargs)
    else:
      # point of first subgraph application to params: cache params
      self.params = params
      return self._orig_apply_fun(params, inputs, **kwargs)

  # when unpacking this (init, apply) pair we return the wrapped funs
  def __iter__(self):
    return iter((self._init_fun, self._apply_fun))
