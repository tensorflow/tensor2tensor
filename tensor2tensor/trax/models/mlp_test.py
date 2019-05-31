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

"""Tests for MLP."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from tensor2tensor.trax import backend
from tensor2tensor.trax import layers as tl
from tensor2tensor.trax.models import mlp


class MLPTest(absltest.TestCase):

  def test_mlp_forward_shape(self):
    """Run the MLP model forward and check output shape."""
    input_shape = (3, 28, 28, 1)
    model = mlp.MLP(d_hidden=32, n_output_classes=10)
    final_shape = tl.check_shape_agreement(model, input_shape)
    self.assertEqual((3, 10), final_shape)


if __name__ == '__main__':
  absltest.main()
