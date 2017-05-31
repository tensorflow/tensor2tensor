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

"""For loading data into NMT models."""
from __future__ import print_function

import collections

import tensorflow as tf


__all__ = ["BatchedInput", "get_iterator", "get_infer_iterator"]


# NOTE(ebrevdo): When we subclass this, instances' __dict__ becomes empty.
class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("initializer",
                                           "source",
                                           "target_input",
                                           "target_output",
                                           "source_sequence_length",
                                           "target_sequence_length"))):
  pass


def get_infer_iterator(
    src_dataset, hparams, src_vocab_table, batch_size, src_max_len=None):
  # TODO(ebrevdo): Add shuffling as an option.
  # TODO(ebrevdo): make lookup default value the "unk" symbol?
  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(hparams.eos)),
                       tf.int32)
  source_reverse = hparams.source_reverse
  src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)
  if source_reverse:
    src_dataset = src_dataset.map(lambda src: tf.reverse(src, axis=[0]))
  if src_max_len:
    src_dataset = src_dataset.map(lambda src: src[:src_max_len])
  # Convert the word strings to ids
  src_dataset = src_dataset.map(
      lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))
  # Add in the word counts.
  src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The entry is the source line rows;
        # this has unknown-length vectors.  The last entry is
        # the source row size; this is a scalar.
        padded_shapes=(tf.TensorShape([None]),  # src
                       tf.TensorShape([])),     # src_len
        # Pad the source sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(src_eos_id,  # src
                        0))          # src_len -- unused

  batched_dataset = batching_func(src_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, src_seq_len) = batched_iter.get_next()
  return BatchedInput(
      initializer=batched_iter.initializer,
      source=src_ids,
      target_input=None,
      target_output=None,
      source_sequence_length=src_seq_len,
      target_sequence_length=None)


def get_iterator(src_dataset, tgt_dataset, hparams,
                 src_vocab_table, tgt_vocab_table, batch_size,
                 src_max_len=None, tgt_max_len=None, bucket=True):
  # TODO(ebrevdo): Add shuffling as an option.
  # TODO(ebrevdo): make lookup default value the "unk" symbol?
  src_eos_id = tf.cast(
      src_vocab_table.lookup(tf.constant(hparams.eos)),
      tf.int32)
  tgt_sos_id = tf.cast(
      tgt_vocab_table.lookup(tf.constant(hparams.sos)),
      tf.int32)
  tgt_eos_id = tf.cast(
      tgt_vocab_table.lookup(tf.constant(hparams.eos)),
      tf.int32)
  source_reverse = hparams.source_reverse
  src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)
  tgt_dataset = tgt_dataset.map(lambda tgt: tf.string_split([tgt]).values)
  src_tgt_dataset = tf.contrib.data.Dataset.zip((src_dataset, tgt_dataset))

  # Filter zero length input sequences.
  src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

  if src_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src[:src_max_len], tgt))
  if tgt_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src, tgt[:tgt_max_len]))
  if source_reverse:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.reverse(src, axis=[0]), tgt))
  # Convert the word strings to ids.  Word strings that are not in the
  # vocab get the lookup table's default_value integer.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                   tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)))
  # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (src,
                   tf.concat(([tgt_sos_id], tgt), 0),
                   tf.concat((tgt, [tgt_eos_id]), 0)))
  # Add in the word counts.  Subtract one from the target to avoid counting
  # the target_input <eos> tag (resp. target_output <sos> tag).
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt_in, tgt_out: (
          src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)))
  # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.
        padded_shapes=(tf.TensorShape([None]),  # src
                       tf.TensorShape([None]),  # tgt_input
                       tf.TensorShape([None]),  # tgt_output
                       tf.TensorShape([]),      # src_len
                       tf.TensorShape([])),     # tgt_len
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(src_eos_id,  # src
                        tgt_eos_id,  # tgt_input
                        tgt_eos_id,  # tgt_output
                        0,           # src_len -- unused
                        0))          # tgt_len -- unused
  if bucket:
    def key_func(unused_1, unused_2, unused_3, src_len, unused_4):
      # Bucket sentence pairs by the length of their source sentence.
      # Pairs with source length [0, 10) go to bucket 0, source
      # length [10, 20) go to bucket 1, etc.
      # Pairs with source length over 100 words all go into an
      # "overflow" bucket 10.
      return tf.to_int64(tf.minimum(10, src_len // 10))
    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)
    batched_dataset = src_tgt_dataset.group_by_window(
        key_func=key_func, reduce_func=reduce_func, window_size=batch_size)
  else:
    batched_dataset = batching_func(src_tgt_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len, tgt_seq_len) = (
      batched_iter.get_next())
  return BatchedInput(
      initializer=batched_iter.initializer,
      source=src_ids,
      target_input=tgt_input_ids,
      target_output=tgt_output_ids,
      source_sequence_length=src_seq_len,
      target_sequence_length=tgt_seq_len)
