# Copyright 2017 Google Inc. All Rights Reserved.
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
# ==============================================================================

"""Tests for iterator_utils.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import lookup_ops

from ..utils import iterator_utils


class IteratorUtilsTest(tf.test.TestCase):

  def testGetIterator(self):
    tgt_vocab_table = src_vocab_table = lookup_ops.index_table_from_tensor(
        tf.constant(["a", "b", "c", "eos", "sos"]))
    src_dataset = tf.contrib.data.Dataset.from_tensor_slices(
        tf.constant(["c c a", "c a", "d", "f e a g"]))
    tgt_dataset = tf.contrib.data.Dataset.from_tensor_slices(
        tf.constant(["a b", "b c", "", "c c"]))
    hparams = tf.contrib.training.HParams(
        random_seed=3,
        num_buckets=5,
        source_reverse=False,
        eos="eos",
        sos="sos")
    batch_size = 2
    src_max_len = 3
    iterator = iterator_utils.get_iterator(
        src_dataset=src_dataset,
        tgt_dataset=tgt_dataset,
        src_vocab_table=src_vocab_table,
        tgt_vocab_table=tgt_vocab_table,
        batch_size=batch_size,
        sos=hparams.sos,
        eos=hparams.eos,
        source_reverse=hparams.source_reverse,
        random_seed=hparams.random_seed,
        num_buckets=hparams.num_buckets,
        src_max_len=src_max_len)
    table_initializer = tf.tables_initializer()
    source = iterator.source
    target_input = iterator.target_input
    target_output = iterator.target_output
    src_seq_len = iterator.source_sequence_length
    tgt_seq_len = iterator.target_sequence_length
    self.assertEqual([None, None], source.shape.as_list())
    self.assertEqual([None, None], target_input.shape.as_list())
    self.assertEqual([None, None], target_output.shape.as_list())
    self.assertEqual([None], src_seq_len.shape.as_list())
    self.assertEqual([None], tgt_seq_len.shape.as_list())
    with self.test_session() as sess:
      sess.run(table_initializer)
      sess.run(iterator.initializer)

      (source_v, src_len_v, target_input_v, target_output_v, tgt_len_v) = (
          sess.run((source, src_seq_len, target_input, target_output,
                    tgt_seq_len)))
      self.assertAllEqual(
          [[-1, -1, 0], # "f" == unknown, "e" == unknown, a
           [2, 0, 3]],  # c a eos -- eos is padding
          source_v)
      self.assertAllEqual([3, 2], src_len_v)
      self.assertAllEqual(
          [[4, 2, 2],   # sos c c
           [4, 1, 2]],  # sos b c
          target_input_v)
      self.assertAllEqual(
          [[2, 2, 3],   # c c eos
           [1, 2, 3]],  # b c eos
          target_output_v)
      self.assertAllEqual([3, 3], tgt_len_v)

      (source_v, src_len_v, target_input_v, target_output_v, tgt_len_v) = (
          sess.run((source, src_seq_len, target_input, target_output,
                    tgt_seq_len)))
      self.assertAllEqual(
          [[2, 2, 0]],  # c c a
          source_v)
      self.assertAllEqual([3], src_len_v)
      self.assertAllEqual(
          [[4, 0, 1]],  # sos a b
          target_input_v)
      self.assertAllEqual(
          [[0, 1, 3]],  # a b eos
          target_output_v)
      self.assertAllEqual([3], tgt_len_v)

      with self.assertRaisesOpError("End of sequence"):
        sess.run(source)


  def testGetIteratorWithSkipCount(self):
    tgt_vocab_table = src_vocab_table = lookup_ops.index_table_from_tensor(
        tf.constant(["a", "b", "c", "eos", "sos"]))
    src_dataset = tf.contrib.data.Dataset.from_tensor_slices(
        tf.constant(["c c a", "c a", "d", "f e a g"]))
    tgt_dataset = tf.contrib.data.Dataset.from_tensor_slices(
        tf.constant(["a b", "b c", "", "c c"]))
    hparams = tf.contrib.training.HParams(
        random_seed=3,
        num_buckets=5,
        source_reverse=False,
        eos="eos",
        sos="sos")
    batch_size = 2
    src_max_len = 3
    skip_count = tf.placeholder(shape=(), dtype=tf.int64)
    iterator = iterator_utils.get_iterator(
        src_dataset=src_dataset,
        tgt_dataset=tgt_dataset,
        src_vocab_table=src_vocab_table,
        tgt_vocab_table=tgt_vocab_table,
        batch_size=batch_size,
        sos=hparams.sos,
        eos=hparams.eos,
        source_reverse=hparams.source_reverse,
        random_seed=hparams.random_seed,
        num_buckets=hparams.num_buckets,
        src_max_len=src_max_len,
        skip_count=skip_count)
    table_initializer = tf.tables_initializer()
    source = iterator.source
    target_input = iterator.target_input
    target_output = iterator.target_output
    src_seq_len = iterator.source_sequence_length
    tgt_seq_len = iterator.target_sequence_length
    self.assertEqual([None, None], source.shape.as_list())
    self.assertEqual([None, None], target_input.shape.as_list())
    self.assertEqual([None, None], target_output.shape.as_list())
    self.assertEqual([None], src_seq_len.shape.as_list())
    self.assertEqual([None], tgt_seq_len.shape.as_list())
    with self.test_session() as sess:
      sess.run(table_initializer)
      sess.run(iterator.initializer, feed_dict={skip_count: 3})

      (source_v, src_len_v, target_input_v, target_output_v, tgt_len_v) = (
          sess.run((source, src_seq_len, target_input, target_output,
                    tgt_seq_len)))
      self.assertAllEqual(
          [[-1, -1, 0]], # "f" == unknown, "e" == unknown, a
          source_v)
      self.assertAllEqual([3], src_len_v)
      self.assertAllEqual(
          [[4, 2, 2]],   # sos c c
          target_input_v)
      self.assertAllEqual(
          [[2, 2, 3]],   # c c eos
          target_output_v)
      self.assertAllEqual([3], tgt_len_v)

      with self.assertRaisesOpError("End of sequence"):
        sess.run(source)

      # Re-init iterator with skip_count=0.
      sess.run(iterator.initializer, feed_dict={skip_count: 0})

      (source_v, src_len_v, target_input_v, target_output_v, tgt_len_v) = (
          sess.run((source, src_seq_len, target_input, target_output,
                    tgt_seq_len)))
      self.assertAllEqual(
          [[-1, -1, 0], # "f" == unknown, "e" == unknown, a
           [2, 0, 3]],  # c a eos -- eos is padding
          source_v)
      self.assertAllEqual([3, 2], src_len_v)
      self.assertAllEqual(
          [[4, 2, 2],   # sos c c
           [4, 1, 2]],  # sos b c
          target_input_v)
      self.assertAllEqual(
          [[2, 2, 3],   # c c eos
           [1, 2, 3]],  # b c eos
          target_output_v)
      self.assertAllEqual([3, 3], tgt_len_v)

      (source_v, src_len_v, target_input_v, target_output_v, tgt_len_v) = (
          sess.run((source, src_seq_len, target_input, target_output,
                    tgt_seq_len)))
      self.assertAllEqual(
          [[2, 2, 0]],  # c c a
          source_v)
      self.assertAllEqual([3], src_len_v)
      self.assertAllEqual(
          [[4, 0, 1]],  # sos a b
          target_input_v)
      self.assertAllEqual(
          [[0, 1, 3]],  # a b eos
          target_output_v)
      self.assertAllEqual([3], tgt_len_v)

      with self.assertRaisesOpError("End of sequence"):
        sess.run(source)


  def testGetInferIterator(self):
    src_vocab_table = lookup_ops.index_table_from_tensor(
        tf.constant(["a", "b", "c", "eos", "sos"]))
    src_dataset = tf.contrib.data.Dataset.from_tensor_slices(
        tf.constant(["c c a", "c a", "d", "f e a g"]))
    hparams = tf.contrib.training.HParams(
        random_seed=3,
        source_reverse=False,
        eos="eos",
        sos="sos")
    batch_size = 2
    src_max_len = 3
    iterator = iterator_utils.get_infer_iterator(
        src_dataset=src_dataset,
        src_vocab_table=src_vocab_table,
        batch_size=batch_size,
        eos=hparams.eos,
        source_reverse=hparams.source_reverse,
        src_max_len=src_max_len)
    table_initializer = tf.tables_initializer()
    source = iterator.source
    seq_len = iterator.source_sequence_length
    self.assertEqual([None, None], source.shape.as_list())
    self.assertEqual([None], seq_len.shape.as_list())
    with self.test_session() as sess:
      sess.run(table_initializer)
      sess.run(iterator.initializer)

      (source_v, seq_len_v) = sess.run((source, seq_len))
      self.assertAllEqual(
          [[2, 2, 0],   # c c a
           [2, 0, 3]],  # c a eos
          source_v)
      self.assertAllEqual([3, 2], seq_len_v)

      (source_v, seq_len_v) = sess.run((source, seq_len))
      self.assertAllEqual(
          [[-1, 3, 3],    # "d" == unknown, eos eos
           [-1, -1, 0]],  # "f" == unknown, "e" == unknown, a
          source_v)
      self.assertAllEqual([1, 3], seq_len_v)

      with self.assertRaisesOpError("End of sequence"):
        sess.run((source, seq_len))


if __name__ == "__main__":
  tf.test.main()
