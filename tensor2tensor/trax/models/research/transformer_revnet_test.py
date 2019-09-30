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

"""Tests for Transformer-Revnet models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
from tensor2tensor.trax import layers as tl
from tensor2tensor.trax.models.research import transformer_revnet


class TransformerRevnetTest(parameterized.TestCase):

  def test_transformer_lm_forward_shape(self):
    """Run the TransformerRevnet LM forward and check output shape."""
    vocab_size = 16
    input_shape = ((1, 8), (1, 8))
    model = transformer_revnet.TransformerRevnetLM(
        vocab_size, d_model=32, d_ff=64,
        d_attention_key=16, d_attention_value=16, n_layers=1, n_heads=2,
        max_len=16, n_chunks=2, n_attention_chunks=1)
    final_shape = tl.check_shape_agreement(
        model, tuple(input_shape), integer_inputs=True)
    self.assertEqual(((1, 8, 16), (1, 8, 16)), final_shape)


if __name__ == '__main__':
  absltest.main()
