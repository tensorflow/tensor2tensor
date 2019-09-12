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

"""Tests for base layer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from tensor2tensor.trax import backend
from tensor2tensor.trax.layers import base


class BaseLayerTest(absltest.TestCase):

  def test_layer_decorator_and_shape_agreement(self):
    @base.layer()
    def add_one(x, **unused_kwargs):
      return x + 1

    output_shape = base.check_shape_agreement(
        add_one(), (12, 17))  # pylint: disable=no-value-for-parameter
    self.assertEqual(output_shape, (12, 17))

  def test_custom_zero_grad(self):

    class IdWithZeroGrad(base.Layer):

      def call(self, x, params, **kwargs):
        del params, kwargs
        return x, ()

      def new_parameters(self, input_shapes, input_dtype, rng):
        del input_shapes, input_dtype, rng
        return (), ()

      @property
      def has_custom_grad(self):
        return True

      def custom_grad(self, inputs, output, ct, params, state, **kwargs):
        return (backend.numpy.zeros_like(ct), ())

    layer = IdWithZeroGrad()
    rng = backend.random.get_prng(0)
    params = ()
    input_shape = (9, 17)
    random_input = backend.random.uniform(rng, input_shape, minval=-1.0,
                                          maxval=1.0)
    f = lambda x: backend.numpy.mean(layer(x, params, rng=rng)[0])
    grad = backend.grad(f)(random_input)
    self.assertEqual(grad.shape, input_shape)  # Gradient for each input.
    self.assertEqual(sum(sum(grad * grad)), 0.0)  # Each one is 0.

  def test_custom_id_grad(self):

    class IdWithIdGrad(base.Layer):

      def call(self, x, params, **kwargs):
        del params, kwargs
        return x, ()

      def new_parameters(self, input_shapes, input_dtype, rng):
        del input_shapes, input_dtype, rng
        return (), ()

      @property
      def has_custom_grad(self):
        return True

      def custom_grad(self, inputs, output, ct, params, state, **kwargs):
        return (inputs, ())

    layer = IdWithIdGrad()
    rng = backend.random.get_prng(0)
    params = ()
    input_shape = (9, 17)
    random_input = backend.random.uniform(rng, input_shape, minval=-1.0,
                                          maxval=1.0)
    f = lambda x: backend.numpy.mean(layer(x, params, rng=rng)[0])
    grad = backend.grad(f)(random_input)
    self.assertEqual(grad.shape, input_shape)  # Gradient for each input.
    self.assertEqual(sum(sum(grad)), sum(sum(random_input)))  # Same as input.

if __name__ == "__main__":
  absltest.main()
