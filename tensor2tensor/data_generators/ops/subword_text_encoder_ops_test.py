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

"""Tests for subword_text_encoder_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators.ops import subword_text_encoder_ops
import tensorflow as tf

vocab_file = (
    "third_party/py/tensor2tensor/data_generators/ops/testdata/subwords")


class SubwordTextEncoderOpsTest(tf.test.TestCase):

  def test_subword_text_encoder_encode(self):
    s = "the quick brown fox jumps over the lazy dog"
    encoded = subword_text_encoder_ops.subword_text_encoder_encode(
        s, vocab_file)
    self.assertAllEqual(encoded, [2, 3, 4, 5, 6, 7, 8, 9, 2, 11, 12, 1])


if __name__ == "__main__":
  tf.enable_eager_execution()
  tf.test.main()
