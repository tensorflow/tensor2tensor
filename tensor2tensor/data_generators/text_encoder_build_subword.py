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

r"""Program to build a SubwordTextEncoder.

The flags --min_count and --corpus_max_lines will affect the size of the
vocabulary.  Try changing these flags until you get a vocabulary
of the size you want.

Example usage:

python data_generators/text_encoder_build_subword.py \
    --corpus_filepattern=$LM1B_DIR/train-unk-* \
    --corpus_max_lines=17500 \
    --output_fn=$DATA_DIR/lm1b16k.subword_text_encoder \
    --logtostderr

python data_generators/text_encoder_build_subword.py \
    --corpus_filepattern=$LM1B_DIR/train-unk-* \
    --corpus_max_lines=270000 \
    --output_fn=$DATA_DIR/lm1b64k.subword_text_encoder \
    --logtostderr
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.data_generators import text_encoder

import tensorflow as tf

tf.app.flags.DEFINE_string('output_fn', '/tmp/my.subword_text_encoder',
                           'where to store the SubwordTextEncoder')
tf.app.flags.DEFINE_string('corpus_filepattern', '',
                           'Corpus of one or more text files')
tf.app.flags.DEFINE_integer('min_count', 5, 'Minimum subtoken count in corpus')
tf.app.flags.DEFINE_integer('corpus_max_lines', 10000,
                            'How many lines of corpus to read')
tf.app.flags.DEFINE_integer('num_iterations', 4, 'Number of iterations')
FLAGS = tf.app.flags.FLAGS


def main(unused_argv):
  gs = text_encoder.SubwordTextEncoder()
  if not FLAGS.corpus_filepattern:
    raise ValueError('Must provide --corpus_filepattern')
  token_counts = text_encoder.SubwordTextEncoder.get_token_counts(
      FLAGS.corpus_filepattern, FLAGS.corpus_max_lines)
  gs.build_from_token_counts(token_counts, FLAGS.output_fn, FLAGS.min_count,
                             FLAGS.num_iterations)


if __name__ == '__main__':
  tf.app.run()
