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
from tensor2tensor.trax.layers import base


class BaseLayerTest(absltest.TestCase):

  def test_layer_decorator_and_shape_agreement(self):
    @base.layer()
    def add_one(x, **unused_kwargs):
      return x + 1

    output_shape = base.check_shape_agreement(
        add_one(), (12, 17))  # pylint: disable=no-value-for-parameter
    self.assertEqual(output_shape, (12, 17))


if __name__ == "__main__":
  absltest.main()
