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

"""Tests for Transformer models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from tensor2tensor.trax import backend
from tensor2tensor.trax import layers as tl
from tensor2tensor.trax.models import transformer


class TransformerTest(absltest.TestCase):

  def test_transformer_lm_forward_shape(self):
    """Run the Transformer LM forward and check output shape."""
    vocab_size = 16
    input_shape = [3, 5]
    model = transformer.TransformerLM(
        vocab_size, d_feature=32, d_feedforward=64, n_layers=2, n_heads=2)
    final_shape = tl.check_shape_agreement(
        model, tuple(input_shape), integer_inputs=True)
    self.assertEqual(tuple(input_shape + [vocab_size]), final_shape)

  def test_transformer_forward_shape(self):
    """Run the Transformer forward and check output shape."""
    vocab_size = 16
    single_input_shape = [3, 5]
    input_shape = (tuple(single_input_shape), tuple(single_input_shape))
    model = transformer.Transformer(
        vocab_size, d_feature=32, d_feedforward=64, n_layers=2, n_heads=2)
    final_shape = tl.check_shape_agreement(
        model, input_shape, integer_inputs=True)
    self.assertEqual(tuple(single_input_shape + [vocab_size]), final_shape)


if __name__ == '__main__':
  absltest.main()
