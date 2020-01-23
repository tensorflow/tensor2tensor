# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""Tests for tensor2tensor.data_generators.translate_ende."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import translate_ende

import tensorflow.compat.v1 as tf


class TranslateEndeTest(tf.test.TestCase):
  """Tests that some TranslateEnde subclasses inherit information correctly."""

  def test_vocab_size(self):
    wmt_8k = translate_ende.TranslateEndeWmt8k()
    wmt_32k = translate_ende.TranslateEndeWmt32k()
    self.assertEqual(wmt_8k.approx_vocab_size, 8192)
    self.assertEqual(wmt_32k.approx_vocab_size, 32768)

  def test_additional_datasets(self):
    wmt_8k = translate_ende.TranslateEndeWmt8k()
    wmt_32k = translate_ende.TranslateEndeWmt32k()
    self.assertListEqual(wmt_8k.additional_training_datasets, [])
    self.assertListEqual(wmt_32k.additional_training_datasets, [])

  def test_source_data_files(self):
    wmt_8k = translate_ende.TranslateEndeWmt8k()
    wmt_32k = translate_ende.TranslateEndeWmt32k()
    eval_split = problem.DatasetSplit.EVAL
    train_split = problem.DatasetSplit.TRAIN

    wmt_8k_eval_files = wmt_8k.source_data_files(eval_split)
    wmt_32k_eval_files = wmt_32k.source_data_files(eval_split)
    self.assertListEqual(wmt_8k_eval_files, wmt_32k_eval_files)
    self.assertGreater(len(wmt_8k_eval_files), 0)

    wmt_8k_train_files = wmt_8k.source_data_files(train_split)
    wmt_32k_train_files = wmt_32k.source_data_files(train_split)
    self.assertListEqual(wmt_8k_train_files, wmt_32k_train_files)
    self.assertGreater(len(wmt_8k_train_files), 0)


if __name__ == '__main__':
  tf.test.main()
