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

"""Tests for tensor2tensor.trax.backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import jax.numpy as jnp
import numpy as onp
from tensor2tensor.trax import backend as backend_lib
from tensorflow import test


class BackendTest(test.TestCase):

  def setUp(self):
    gin.clear_config()

  def override_gin(self, bindings):
    gin.parse_config_files_and_bindings(None, bindings)

  def test_backend_imports_correctly(self):
    backend = backend_lib.backend()
    self.assertEqual(jnp, backend["np"])
    self.assertNotEqual(onp, backend["np"])

    self.override_gin("backend.name = 'numpy'")

    backend = backend_lib.backend()
    self.assertNotEqual(jnp, backend["np"])
    self.assertEqual(onp, backend["np"])

  def test_numpy_backend_delegation(self):
    # Assert that we are getting JAX's numpy backend.
    backend = backend_lib.backend()
    numpy = backend_lib.numpy
    self.assertEqual(jnp, backend["np"])

    # Assert that `numpy` calls the appropriate gin configured functions and
    # properties.
    self.assertTrue(numpy.isinf(numpy.inf))
    self.assertEqual(jnp.isinf, numpy.isinf)
    self.assertEqual(jnp.inf, numpy.inf)

    # Assert that we will now get the pure numpy backend.

    self.override_gin("backend.name = 'numpy'")

    backend = backend_lib.backend()
    numpy = backend_lib.numpy
    self.assertEqual(onp, backend["np"])

    # Assert that `numpy` calls the appropriate gin configured functions and
    # properties.
    self.assertTrue(numpy.isinf(numpy.inf))
    self.assertEqual(onp.isinf, numpy.isinf)
    self.assertEqual(onp.inf, numpy.inf)

if __name__ == "__main__":
  test.main()
