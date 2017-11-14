# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

"""Translate generators test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import tempfile

# Dependency imports

import six
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate

import tensorflow as tf


class TranslateTest(tf.test.TestCase):

  def testCharacterGenerator(self):
    # Generate a trivial source and target file.
    tmp_dir = self.get_temp_dir()
    (_, tmp_file_path) = tempfile.mkstemp(dir=tmp_dir)
    if six.PY2:
      enc_f = lambda s: s
    else:
      enc_f = lambda s: s.encode("utf-8")
    with io.open(tmp_file_path + ".src", "wb") as src_file:
      src_file.write(enc_f("source1\n"))
      src_file.write(enc_f("source2\n"))
    with io.open(tmp_file_path + ".tgt", "wb") as tgt_file:
      tgt_file.write(enc_f("target1\n"))
      tgt_file.write(enc_f("target2\n"))

    # Call character generator on the generated files.
    results_src, results_tgt = [], []
    character_vocab = text_encoder.ByteTextEncoder()
    for dictionary in translate.character_generator(
        tmp_file_path + ".src", tmp_file_path + ".tgt", character_vocab):
      self.assertEqual(sorted(list(dictionary)), ["inputs", "targets"])
      results_src.append(dictionary["inputs"])
      results_tgt.append(dictionary["targets"])

    # Check that the results match the files.
    # First check that the results match the encoded original strings;
    # this is a comparison of integer arrays.
    self.assertEqual(len(results_src), 2)
    self.assertEqual(results_src[0], character_vocab.encode("source1"))
    self.assertEqual(results_src[1], character_vocab.encode("source2"))
    self.assertEqual(results_tgt[0], character_vocab.encode("target1"))
    self.assertEqual(results_tgt[1], character_vocab.encode("target2"))
    # Then decode the results and compare with the original strings;
    # this is a comparison of strings
    self.assertEqual(character_vocab.decode(results_src[0]), "source1")
    self.assertEqual(character_vocab.decode(results_src[1]), "source2")
    self.assertEqual(character_vocab.decode(results_tgt[0]), "target1")
    self.assertEqual(character_vocab.decode(results_tgt[1]), "target2")

    # Clean up.
    os.remove(tmp_file_path + ".src")
    os.remove(tmp_file_path + ".tgt")
    os.remove(tmp_file_path)


if __name__ == "__main__":
  tf.test.main()
