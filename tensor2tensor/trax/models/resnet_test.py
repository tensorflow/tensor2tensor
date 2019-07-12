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

"""Tests for Resnet models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from tensor2tensor.trax import backend
from tensor2tensor.trax import layers as tl
from tensor2tensor.trax.models import resnet


class ResnetTest(absltest.TestCase):

  def test_resnet(self):
    input_shape = (3, 256, 256, 3)
    model = resnet.Resnet50(d_hidden=8, n_output_classes=10)
    final_shape = tl.check_shape_agreement(model, input_shape)
    self.assertEqual((3, 10), final_shape)

  def test_wide_resnet(self):
    input_shape = (3, 32, 32, 3)
    model = resnet.WideResnet(n_blocks=1, n_output_classes=10)
    final_shape = tl.check_shape_agreement(model, input_shape)
    self.assertEqual((3, 10), final_shape)



if __name__ == '__main__':
  absltest.main()
