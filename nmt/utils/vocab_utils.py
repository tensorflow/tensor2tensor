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

"""Utility to handle vocabularies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import tensorflow as tf

from ..utils import misc_utils as utils


UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0


def check_vocab(vocab_file, out_dir, sos=None, eos=None, unk=None):
  """Check if vocab_file doesn't exist, create from corpus_file."""
  if tf.gfile.Exists(vocab_file):
    utils.print_out("# Vocab file %s exists" % vocab_file)
    vocab = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
      vocab_size = 0
      for word in f:
        vocab_size += 1
        vocab.append(word.strip())

    # Verify if the vocab starts with unk, sos, eos
    # If not, prepend those tokens & generate a new vocab file
    if not unk: unk = UNK
    if not sos: sos = SOS
    if not eos: eos = EOS
    assert len(vocab) >= 3
    if vocab[0] != unk or vocab[1] != sos or vocab[2] != eos:
      utils.print_out("The first 3 vocab words [%s, %s, %s]"
                      " are not [%s, %s, %s]" %
                      (vocab[0], vocab[1], vocab[2], unk, sos, eos))
      vocab = [unk, sos, eos] + vocab
      vocab_size += 3
      new_vocab_file = os.path.join(out_dir, os.path.basename(vocab_file))
      with codecs.getwriter("utf-8")(tf.gfile.GFile(new_vocab_file, "wb")) as f:
        for word in vocab:
          f.write("%s\n" % word)
      vocab_file = new_vocab_file
  else:
    raise ValueError("vocab_file does not exist.")

  vocab_size = len(vocab)
  return vocab_size, vocab_file
