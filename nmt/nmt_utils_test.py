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

"""Tests for nmt_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

import utils.misc_utils as misc_utils
import utils.nmt_utils as nmt_utils


class NmtUtilsTest(tf.test.TestCase):

  def assertAllClose(self, *args, **kwargs):
    kwargs["atol"] = 1e-4  # For GPU tests
    kwargs["rtol"] = 1e-4  # For GPU tests
    return super(NmtUtilsTest, self).assertAllClose(*args, **kwargs)

  def _to_text(self, sents, vocab):
    return [misc_utils.int2text(sent, vocab) for sent in sents]

  # def _prepare_seq2seq_data(self):
  #   # src
  #   src_vocab = ["<unk>", "<s>", "</s>", "anh", "yeu", "em", "tinh",
  #                "co", "ay", "hon"]
  #   src_vocab_hash = {word: id for id, word in enumerate(src_vocab)}
  #   sents = [
  #       "anh yeu em !",
  #       "tinh yeu",
  #       "co ay hon",
  #   ]
  #   src_examples = [misc_utils.text2int(sent, src_vocab_hash)
  #                   for sent in sents]
  #   src_max_len = 4
  #   src_eos_id = src_vocab_hash["</s>"]

  #   # tgt
  #   tgt_vocab = ["<unk>", "<s>", "</s>", "i", "love", "you",
  #                "she", "kisses"]
  #   tgt_vocab_hash = {word: id for id, word in enumerate(tgt_vocab)}
  #   sents = [
  #       "i love you !",
  #       "love",
  #       "she kisses",
  #   ]
  #   tgt_examples = [misc_utils.text2int(sent, tgt_vocab_hash)
  #                   for sent in sents]
  #   tgt_max_len = 4
  #   tgt_sos_id = tgt_vocab_hash["<s>"]
  #   tgt_eos_id = tgt_vocab_hash["</s>"]
  #   return (src_examples, tgt_examples, src_max_len, tgt_max_len,
  #           src_vocab, tgt_vocab,
  #           src_eos_id, tgt_sos_id, tgt_eos_id)

  # def testPrepareBatchSeq2Seq(self):
  #   (src_examples, tgt_examples, src_max_len, tgt_max_len,
  #    src_vocab, tgt_vocab,
  #    src_eos_id, tgt_sos_id, tgt_eos_id) = self._prepare_seq2seq_data()
  #   source_reverse = False
  #   task = "seq2seq"
  #   batch = nmt_utils. _prepare_batch(
  #       src_examples, tgt_examples, src_max_len, tgt_max_len,
  #       src_eos_id, tgt_sos_id, tgt_eos_id, source_reverse, task)

  #   # Note the expect results here is time-major,
  #   #   so read by column for individual sent
  #   expected_encoder_inputs = ["anh tinh co",
  #                              "yeu yeu ay",
  #                              "em </s> hon",
  #                              "<unk> </s> </s>"]
  #   #  fix the problem of dropping the last words
  #   #   those longest sentences in a batch
  #   expected_decoder_inputs = ["<s> <s> <s>",
  #                              "i love she",
  #                              "love </s> kisses",
  #                              "you </s> </s>"]
  #   expected_decoder_outputs = ["i love she",
  #                               "love </s> kisses",
  #                               "you </s> </s>",
  #                               "</s> </s> </s>"]
  #   expected_decoder_weights = np.array([[1., 1., 1.],
  #                                        [1., 1., 1.],
  #                                        [1., 0., 1.],
  #                                        [1., 0., 0.]],
  #                                       dtype=np.float32)

  #   print(self._to_text(batch["encoder_inputs"], src_vocab))
  #   print(self._to_text(batch["decoder_inputs"], tgt_vocab))
  #   print(self._to_text(batch["decoder_outputs"], tgt_vocab))
  #   print(batch["decoder_weights"])

  #   self.assertEqual(expected_encoder_inputs,
  #                    self._to_text(batch["encoder_inputs"], src_vocab))
  #   self.assertEqual(expected_decoder_inputs,
  #                    self._to_text(batch["decoder_inputs"], tgt_vocab))
  #   self.assertEqual(expected_decoder_outputs,
  #                    self._to_text(batch["decoder_outputs"], tgt_vocab))
  #   self.assertAllClose(expected_decoder_weights, batch["decoder_weights"])
  #   self.assertEqual(len(src_examples), batch["size"])
  #   self.assertEqual([4, 2, 3], batch["encoder_lengths"])
  #   self.assertEqual([4, 2, 3], batch["decoder_lengths"])  # add </s>

  # def testPrepareBatchSeq2SeqSourceReverse(self):
  #   (src_examples, tgt_examples, src_max_len, tgt_max_len,
  #    src_vocab, tgt_vocab,
  #    src_eos_id, tgt_sos_id, tgt_eos_id) = self._prepare_seq2seq_data()
  #   source_reverse = True
  #   task = "seq2seq"
  #   batch = nmt_utils._prepare_batch(
  #       src_examples, tgt_examples, src_max_len, tgt_max_len,
  #       src_eos_id, tgt_sos_id, tgt_eos_id, source_reverse, task)

  #   # Note the expect results here is time-major,
  #   #   so read by column for individual sent
  #   expected_encoder_inputs = ["<unk> yeu hon",
  #                              "em tinh ay",
  #                              "yeu </s> co",
  #                              "anh </s> </s>"]
  #   #  fix the problem of dropping the last words
  #   #   those longest sentences in a batch
  #   expected_decoder_inputs = ["<s> <s> <s>",
  #                              "i love she",
  #                              "love </s> kisses",
  #                              "you </s> </s>"]
  #   expected_decoder_outputs = ["i love she",
  #                               "love </s> kisses",
  #                               "you </s> </s>",
  #                               "</s> </s> </s>"]
  #   expected_decoder_weights = np.array([[1., 1., 1.],
  #                                        [1., 1., 1.],
  #                                        [1., 0., 1.],
  #                                        [1., 0., 0.]],
  #                                       dtype=np.float32)

  #   print(self._to_text(batch["encoder_inputs"], src_vocab))
  #   print(self._to_text(batch["decoder_inputs"], tgt_vocab))
  #   print(self._to_text(batch["decoder_outputs"], tgt_vocab))
  #   print(batch["decoder_weights"])

  #   self.assertEqual(expected_encoder_inputs,
  #                    self._to_text(batch["encoder_inputs"], src_vocab))
  #   self.assertEqual(expected_decoder_inputs,
  #                    self._to_text(batch["decoder_inputs"], tgt_vocab))
  #   self.assertEqual(expected_decoder_outputs,
  #                    self._to_text(batch["decoder_outputs"], tgt_vocab))
  #   self.assertAllClose(expected_decoder_weights, batch["decoder_weights"])
  #   self.assertEqual(len(src_examples), batch["size"])
  #   self.assertEqual([4, 2, 3], batch["encoder_lengths"])
  #   self.assertEqual([4, 2, 3], batch["decoder_lengths"])  # add </s>

  # def testPrepareBatchInferenceSeq2Seq(self):
  #   (src_examples, _, src_max_len, _, src_vocab, _,
  #    src_eos_id, _, _) = self._prepare_seq2seq_data()
  #   source_reverse = False
  #   task = "seq2seq"
  #   batch = nmt_utils. _prepare_batch_inference(
  #       src_examples, src_max_len, src_eos_id, source_reverse, task)

  #   # Note the expect results here is time-major,
  #   #   so read by column for individual sent
  #   expected_encoder_inputs = ["anh tinh co",
  #                              "yeu yeu ay",
  #                              "em </s> hon",
  #                              "<unk> </s> </s>"]
  #   print(self._to_text(batch["encoder_inputs"], src_vocab))

  #   self.assertEqual(expected_encoder_inputs,
  #                    self._to_text(batch["encoder_inputs"], src_vocab))
  #   self.assertEqual(len(src_examples), batch["size"])
  #   self.assertEqual([4, 2, 3], batch["encoder_lengths"])

  #   self.assertFalse("decoder_inputs" in batch)
  #   self.assertFalse("decoder_outputs" in batch)
  #   self.assertFalse("decoder_weights" in batch)
  #   self.assertFalse("decoder_lengths" in batch)

  # def _prepare_seq2label_data(self):
  #   # src
  #   src_vocab = ["<unk>", "<s>", "</s>", "anh", "yeu", "em", "tinh",
  #                "co", "ay", "hon"]
  #   src_vocab_hash = {word: id for id, word in enumerate(src_vocab)}
  #   sents = [
  #       "anh yeu em !",
  #       "tinh yeu",
  #       "co ay hon",
  #   ]
  #   src_examples = [misc_utils.text2int(sent, src_vocab_hash)
  #                   for sent in sents]
  #   src_max_len = 4
  #   src_eos_id = src_vocab_hash["</s>"]

  #   # tgt
  #   tgt_vocab = ["yes", "no"]
  #   tgt_vocab_hash = {word: id for id, word in enumerate(tgt_vocab)}
  #   sents = [
  #       "yes",
  #       "yes",
  #       "no",
  #   ]
  #   tgt_examples = [misc_utils.text2int(sent, tgt_vocab_hash)
  #                   for sent in sents]
  #   tgt_max_len = 1
  #   tgt_sos_id = None
  #   tgt_eos_id = None
  #   return (src_examples, tgt_examples, src_max_len, tgt_max_len,
  #           src_vocab, tgt_vocab,
  #           src_eos_id, tgt_sos_id, tgt_eos_id)

  # def testPrepareBatchSeq2Label(self):
  #   (src_examples, tgt_examples, src_max_len, tgt_max_len,
  #    src_vocab, tgt_vocab,
  #    src_eos_id, tgt_sos_id, tgt_eos_id) = self._prepare_seq2label_data()
  #   source_reverse = False
  #   task = "seq2label"
  #   batch = nmt_utils. _prepare_batch(
  #       src_examples, tgt_examples, src_max_len, tgt_max_len,
  #       src_eos_id, tgt_sos_id, tgt_eos_id, source_reverse, task)

  #   # Note the expect results here is time-major
  #   #   so read by column for individual sent
  #   # For seq2label, we add an extract </s>
  #   expected_encoder_inputs = ["anh tinh co",
  #                              "yeu yeu ay",
  #                              "em </s> hon",
  #                              "<unk> </s> </s>",
  #                              "</s> </s> </s>"]
  #   expected_decoder_outputs = ["yes", "yes", "no"]

  #   print(self._to_text(batch["encoder_inputs"], src_vocab))
  #   print(self._to_text(batch["decoder_outputs"], tgt_vocab))

  #   self.assertEqual(expected_encoder_inputs,
  #                    self._to_text(batch["encoder_inputs"], src_vocab))
  #   self.assertEqual(expected_decoder_outputs,
  #                    self._to_text(batch["decoder_outputs"], tgt_vocab))
  #   self.assertEqual(len(src_examples), batch["size"])
  #   self.assertEqual([5, 3, 4], batch["encoder_lengths"])  # add </s>

if __name__ == "__main__":
  tf.test.main()
