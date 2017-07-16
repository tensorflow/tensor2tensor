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

"""Tests for vocab_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import tensorflow as tf

from ..utils import vocab_utils


class VocabUtilsTest(tf.test.TestCase):

  def testCheckVocab(self):
    # Create a vocab file
    vocab_dir = os.path.join(tf.test.get_temp_dir(), "vocab_dir")
    os.makedirs(vocab_dir)
    vocab_file = os.path.join(vocab_dir, "vocab_file")
    vocab = ["a", "b", "c"]
    with codecs.getwriter("utf-8")(tf.gfile.GFile(vocab_file, "wb")) as f:
      for word in vocab:
        f.write("%s\n" % word)

    # Call vocab_utils
    out_dir = os.path.join(tf.test.get_temp_dir(), "out_dir")
    os.makedirs(out_dir)
    vocab_size, new_vocab_file = vocab_utils.check_vocab(
        vocab_file, out_dir)

    # Assert: we expect the code to add  <unk>, <s>, </s> and
    # create a new vocab file
    self.assertEqual(len(vocab) + 3, vocab_size)
    self.assertEqual(os.path.join(out_dir, "vocab_file"), new_vocab_file)
    new_vocab = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(new_vocab_file, "rb")) as f:
      for line in f:
        new_vocab.append(line.strip())
    self.assertEqual(
        [vocab_utils.UNK, vocab_utils.SOS, vocab_utils.EOS] + vocab, new_vocab)


if __name__ == "__main__":
  tf.test.main()
