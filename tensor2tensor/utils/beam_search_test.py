# Copyright 2017 Google Inc.
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

"""Tests for tensor2tensor.beam_search."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
from tensor2tensor.utils import beam_search

import tensorflow as tf


class BeamSearchTest(tf.test.TestCase):

  def testShapes(self):
    batch_size = 2
    beam_size = 3
    vocab_size = 4
    decode_length = 10

    initial_ids = tf.constant([0, 0])  # GO

    def symbols_to_logits(_):
      # Just return random logits
      return tf.random_uniform((batch_size * beam_size, vocab_size))

    final_ids, final_probs = beam_search.beam_search(
        symbols_to_logits, initial_ids, beam_size, decode_length, vocab_size,
        0.)

    self.assertEqual(final_ids.get_shape().as_list(), [None, beam_size, None])

    self.assertEqual(final_probs.get_shape().as_list(), [None, beam_size])

  def testComputeTopkScoresAndSeq(self):
    batch_size = 2
    beam_size = 3

    sequences = tf.constant([[[2, 3], [4, 5], [6, 7], [19, 20]],
                             [[8, 9], [10, 11], [12, 13], [80, 17]]])

    scores = tf.constant([[-0.1, -2.5, 0., -1.5],
                          [-100., -5., -0.00789, -1.34]])
    flags = tf.constant([[True, False, False, True],
                         [False, False, False, True]])

    topk_seq, topk_scores, topk_flags = beam_search.compute_topk_scores_and_seq(
        sequences, scores, scores, flags, beam_size, batch_size)

    with self.test_session():
      topk_seq = topk_seq.eval()
      topk_scores = topk_scores.eval()
      topk_flags = topk_flags.eval()

    exp_seq = [[[6, 7], [2, 3], [19, 20]], [[12, 13], [80, 17], [10, 11]]]
    exp_scores = [[0., -0.1, -1.5], [-0.00789, -1.34, -5.]]

    exp_flags = [[False, True, True], [False, True, False]]
    self.assertAllEqual(exp_seq, topk_seq)
    self.assertAllClose(exp_scores, topk_scores)
    self.assertAllEqual(exp_flags, topk_flags)

  def testGreedyBatchOne(self):
    batch_size = 1
    beam_size = 1
    vocab_size = 2
    decode_length = 3

    initial_ids = tf.constant([0] * batch_size)  # GO

    # Test that beam search finds the most probable sequence.
    # These probabilities represent the following search
    #
    #               G0 (0)
    #                  / \
    #                /     \
    #              /         \
    #            /             \
    #         0(0.7)          1(0.3)
    #           / \
    #          /   \
    #         /     \
    #     0(0.4) 1(0.6)
    #        /\
    #       /  \
    #      /    \
    #    0(0.5) 1(0.5)
    # and the following decoding probabilities
    # 0000 - 0.7 * 0.4  * 0.1
    # 0001 - 0.7 * 0.4  * 0.9
    # 001 - 0.7 * 0.6 (Best)
    # 01 = 0.3
    #
    # 001 is the most likely sequence under these probabilities.
    probabilities = tf.constant([[[0.7, 0.3]], [[0.4, 0.6]], [[0.5, 0.5]]])

    def symbols_to_logits(ids):
      pos = tf.shape(ids)[1]
      logits = tf.to_float(tf.log(probabilities[pos - 1, :]))
      return logits

    final_ids, final_probs = beam_search.beam_search(
        symbols_to_logits,
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        0.0,
        eos_id=1)

    with self.test_session():
      ids = final_ids.eval()
      probs = final_probs.eval()
    self.assertAllEqual([[[0, 0, 1]]], ids)
    self.assertAllClose([[0.7 * 0.6]], np.exp(probs))

  def testNotGreedyBeamTwo(self):
    batch_size = 1
    beam_size = 2
    vocab_size = 3
    decode_length = 3

    initial_ids = tf.constant([0] * batch_size)  # GO
    probabilities = tf.constant([[[0.1, 0.1, 0.8], [0.1, 0.1, 0.8]],
                                 [[0.4, 0.5, 0.1], [0.2, 0.4, 0.4]],
                                 [[0.05, 0.9, 0.05], [0.4, 0.4, 0.2]]])

    def symbols_to_logits(ids):
      pos = tf.shape(ids)[1]
      logits = tf.to_float(tf.log(probabilities[pos - 1, :]))
      return logits

    final_ids, final_probs = beam_search.beam_search(
        symbols_to_logits,
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        0.0,
        eos_id=1)

    with self.test_session():
      ids = final_ids.eval()
      probs = final_probs.eval()
    self.assertAllEqual([[[0, 2, 1, 0], [0, 2, 0, 1]]], ids)
    self.assertAllClose([[0.8 * 0.5, 0.8 * 0.4 * 0.9]], np.exp(probs))

  def testGreedyWithCornerCase(self):
    batch_size = 1
    beam_size = 1
    vocab_size = 3
    decode_length = 2

    initial_ids = tf.constant([0] * batch_size)  # GO
    probabilities = tf.constant([[0.2, 0.1, 0.7], [0.4, 0.1, 0.5]])

    def symbols_to_logits(ids):
      pos = tf.shape(ids)[1]
      logits = tf.to_float(tf.log(probabilities[pos - 1, :]))
      return logits

    final_ids, final_probs = beam_search.beam_search(
        symbols_to_logits,
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        0.0,
        eos_id=1)

    with self.test_session():
      ids = final_ids.eval()
      probs = final_probs.eval()
    self.assertAllEqual([[[0, 2, 2]]], ids)
    self.assertAllClose([[0.7 * 0.5]], np.exp(probs))

  def testNotGreedyBatchTwoBeamTwoWithAlpha(self):
    batch_size = 2
    beam_size = 2
    vocab_size = 3
    decode_length = 3

    initial_ids = tf.constant([0] * batch_size)  # GO
    # Probabilities for position * batch * beam * vocab
    # Probabilities have been set such that with alpha = 3.5, the less probable
    # but longer sequence will have a better score than the shorter sequence
    # with higher log prob in batch 1, and the order will be reverse in batch
    # 2. That is, the shorter sequence will still have a higher score in spite
    # of the length penalty
    probabilities = tf.constant([[[[0.1, 0.1, 0.8], [0.1, 0.1, 0.8]],
                                  [[0.1, 0.1, 0.8], [0.1, 0.1, 0.8]]],
                                 [[[0.4, 0.5, 0.1], [0.2, 0.4, 0.4]],
                                  [[0.3, 0.6, 0.1], [0.2, 0.4, 0.4]]],
                                 [[[0.05, 0.9, 0.05], [0.4, 0.4, 0.2]],
                                  [[0.05, 0.9, 0.05], [0.4, 0.4, 0.2]]]])

    def symbols_to_logits(ids):
      pos = tf.shape(ids)[1]
      logits = tf.to_float(tf.log(probabilities[pos - 1, :]))
      return logits

    final_ids, final_scores = beam_search.beam_search(
        symbols_to_logits,
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        3.5,
        eos_id=1)

    with self.test_session():
      ids = final_ids.eval()
      scores = final_scores.eval()
    self.assertAllEqual([[[0, 2, 0, 1], [0, 2, 1, 0]], [[0, 2, 1, 0],
                                                        [0, 2, 0, 1]]], ids)
    self.assertAllClose([[
        np.log(0.8 * 0.4 * 0.9) / (8. / 6.)**3.5,
        np.log(0.8 * 0.5) / (7. / 6.)**3.5
    ], [
        np.log(0.8 * 0.6) / (7. / 6.)**3.5,
        np.log(0.8 * 0.3 * 0.9) / (8. / 6.)**3.5
    ]], scores)

  def testNotGreedyBeamTwoWithAlpha(self):
    batch_size = 1
    beam_size = 2
    vocab_size = 3
    decode_length = 3

    initial_ids = tf.constant([0] * batch_size)  # GO
    # Probabilities for position * batch * beam * vocab
    # Probabilities have been set such that with alpha = 3.5, the less probable
    # but longer sequence will have a better score that the shorter sequence
    # with higher log prob.
    probabilities = tf.constant([[[0.1, 0.1, 0.8], [0.1, 0.1, 0.8]],
                                 [[0.4, 0.5, 0.1], [0.2, 0.4, 0.4]],
                                 [[0.05, 0.9, 0.05], [0.4, 0.4, 0.2]]])

    def symbols_to_logits(ids):
      pos = tf.shape(ids)[1]
      logits = tf.to_float(tf.log(probabilities[pos - 1, :]))
      return logits

    # Disable early stopping
    final_ids, final_scores = beam_search.beam_search(
        symbols_to_logits,
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        3.5,
        eos_id=1)

    with self.test_session():
      ids = final_ids.eval()
      scores = final_scores.eval()
    self.assertAllClose([[
        np.log(0.8 * 0.4 * 0.9) / (8. / 6.)**3.5,
        np.log(0.8 * 0.5) / (7. / 6.)**3.5
    ]], scores)
    self.assertAllEqual([[[0, 2, 0, 1], [0, 2, 1, 0]]], ids)


if __name__ == "__main__":
  tf.test.main()
