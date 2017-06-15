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

r"""Data preprocessor for lm1b benchmark.

Process the raw text file to replace out-of-vocab words with "<UNK>".

The input consists of a tokenized text file, where tokens are separated with
whitespace.

Outputs a similar text file where the OOV words have been repalced with UNK.
The whitespace in the output may be different.

This maintains compatibility with the benchmark, which does the same thing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

tf.app.flags.DEFINE_string("vocab_file", "",
                           "text file containing one word per line")

tf.app.flags.DEFINE_string("in_filepattern", "", "input filename")

tf.app.flags.DEFINE_string(
    "out_prefix", "", "The output filename is equal to out_prefix plus "
    "the last 15 characters of in_file. (e.g. -00001-of-00100)")

FLAGS = tf.app.flags.FLAGS


def replace_oov(vocab, in_file):
  """Replace out-of-vocab words with <UNK>."""
  out_file = FLAGS.out_prefix + in_file[-15:]
  print ("in_file", in_file, "out_file", out_file)
  with tf.gfile.Open(out_file, "w") as out:
    for line in tf.gfile.Open(in_file):
      words = line.split()
      for i in xrange(len(words)):
        if not vocab.get(words[i]):
          words[i] = "UNK"
      out_line = " ".join(words) + "\n"
      out.write(out_line)


def main(_):
  vocab = {}
  with tf.gfile.Open(FLAGS.vocab_file) as vocab_file:
    for line in vocab_file:
      vocab[line.strip()] = True

  in_files = tf.gfile.Glob(FLAGS.in_filepattern)
  assert in_files, "No matching input files"
  for in_file in in_files:
    replace_oov(vocab, in_file)

if __name__ == "__main__":
  tf.app.run()
