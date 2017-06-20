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

r"""Convert language modeling data to tf.Example format.

Uses SubwordTextEncoder.

For each line, we generate a tf.Example, with "targets" equal to a sequence
of subwords (integers), ending in subword id 1 for end-of-sequence.  We add
a dummy feature "inputs"=[0] for compatability with seq-to-seq models.

If FLAGS.combine_to_length is nonzero, then we combine multiple sequences into
examples of a constant length, possibly with some padding at the end.


How to preprocess lm1b - billion word benchmark
TODO(noam): should these instructions be made into a script and moved elsewhere?


# Download data into $DATADIR/
http://www.statmt.org/lm-benchmark/\
1-billion-word-language-modeling-benchmark-r13output.tar.gz
http://download.tensorflow.org/models/LM_LSTM_CNN/vocab-2016-09-10.txt

# unpack data
cd $DATADIR
tar xvf 1-billion-word-language-modeling-benchmark-r13output.tar.gz

# replace oov words with UNK
./blaze-bin/third_party/py/tensor2tensor/data_generators/replace_oov \
--vocab_file=$DATADIR/vocab-2016-09-10.txt \
--in_filepattern=\
$DATADIR/1-billion-word-language-modeling-benchmark-r13output/\
heldout-monolingual.tokenized.shuffled/news.en.heldout-00000-of-00050 \
--out_prefix=$DATADIR/dev-unk \
--logtostderr

wc $DATADIR/dev-unk-00000-of-00050
#  -> 6075 153583 826189
# dev set tokens including EOS = 6075 + 153583 = 159658

$BINARYDIR/replace_oov \
--vocab_file=$DATADIR/vocab-2016-09-10.txt \
--in_filepattern=\
$DATADIR/1-billion-word-language-modeling-benchmark-r13output/\
training-monolingual.tokenized.shuffled/news.en-?????-of-00100 \
--out_prefix=$DATADIR/train-unk \
--logtostderr

# build vocabularies
$BINARYDIR/\
text_encoder_build_subword \
 --corpus_filepattern=$DATADIR/train-unk-* \
 --corpus_max_lines=17500 \
 --output_fn=$DATADIR/lm1b_16k.subword_text_encoder \
 --logtostderr

$BINARYDIR/\
text_encoder_build_subword \
 --corpus_filepattern=$DATADIR/train-unk-* \
 --corpus_max_lines=270000 \
 --output_fn=$DATADIR/lm1b_64k.subword_text_encoder \
 --logtostderr

# generate training and dev data

# 16k vocab

$BINARYDIR/lm_example \
--logtostderr \
--vocab_file=$DATADIR/lm1b_16k.subword_text_encoder \
--in_filepattern=$DATADIR/dev-unk* \
--out_prefix=$DATADIR/lm1b_16k-dev

#  -> total subwords: 189068
#  perplexity exponent = 189068 / 159658 = 1.184206

mv $DATADIR/lm1b_16k-dev-00000-of-00050 $DATADIR/lm1b_16k-dev-00000-of-00001

$BINARYDIR/\
text_encoder_inspect_subword \
--logtostderr \
--vocab_file=$DATADIR/lm1b_16k.subword_text_encoder \
--in_file=$DATADIR/lm1b_16k-dev-00000-of-00001 | more

$BINARYDIR/lm_example \
--logtostderr \
--vocab_file=$DATADIR/lm1b_16k.subword_text_encoder \
--in_filepattern=$DATADIR/train-unk* \
--out_prefix=$DATADIR/lm1b_16k-train

# 64k vocab

$BINARYDIR/lm_example \
--logtostderr \
--vocab_file=$DATADIR/lm1b_64k.subword_text_encoder \
--in_filepattern=$DATADIR/dev-unk* \
--out_prefix=$DATADIR/lm1b_64k-dev

#  -> total subwords: 170366
#  perplexity exponent = 170366 / 159658 = 1.067068

mv $DATADIR/lm1b_64k-dev-00000-of-00050 $DATADIR/lm1b_64k-dev-00000-of-00001

$BINARYDIR/\
text_encoder_inspect_subword \
--logtostderr \
--vocab_file=$DATADIR/lm1b_64k.subword_text_encoder \
--in_file=$DATADIR/lm1b_64k-dev-00000-of-00001 | more

$BINARYDIR/lm_example \
--logtostderr \
--vocab_file=$DATADIR/lm1b_64k.subword_text_encoder \
--in_filepattern=$DATADIR/train-unk* \
--out_prefix=$DATADIR/lm1b_64k-train

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import text_encoder

import tensorflow as tf

tf.app.flags.DEFINE_string(
    "vocab_file", "", "SubwordTextEncoder vocabulary file")

tf.app.flags.DEFINE_integer(
    "combine_to_length", 0,
    "If positive, concatenate documents to form examples with length exactly"
    " equal to this value.  Documents are still suffixed with subword id=1. "
    " Examples are padded with subword id=0.")

tf.app.flags.DEFINE_string("in_filepattern", "", "input filename")

tf.app.flags.DEFINE_string(
    "out_prefix", "", "The output filename is equal to out_prefix plus "
    "the last 15 characters of in_file. (e.g. -00001-of-00100)")

FLAGS = tf.app.flags.FLAGS


def _make_example(ids, raw_num_bytes):
  if FLAGS.combine_to_length > 0:
    ids += [0] * (FLAGS.combine_to_length - len(ids))
  return generator_utils.to_example({
      "targets": ids,
      "inputs": [0],
      "raw_num_bytes": [raw_num_bytes]
  }).SerializeToString()


def convert_file(in_file, encoder):
  """Convert a file to examples."""
  total_bytes = 0
  total_subwords = 0
  total_documents = 0
  dropped_documents = 0

  combined_subwords = []
  combined_num_bytes = 0

  out_file = FLAGS.out_prefix + in_file[-15:]
  writer = tf.python_io.TFRecordWriter(out_file)
  out_file = FLAGS.out_prefix + in_file[-15:]
  print ("in_file", in_file, "out_file", out_file)
  for line in tf.gfile.Open(in_file):
    total_documents += 1
    assert line[-1] == "\n"
    num_bytes = len(line)
    total_bytes += num_bytes
    line = line[:-1]
    subwords = encoder.encode(line) + [1]
    total_subwords += len(subwords)
    if FLAGS.combine_to_length:
      if len(combined_subwords) + len(subwords) > FLAGS.combine_to_length:
        writer.write(_make_example(combined_subwords, combined_num_bytes))
        combined_subwords = []
        combined_num_bytes = 0
      if len(subwords) <= FLAGS.combine_to_length:
        combined_subwords.extend(subwords)
        combined_num_bytes += num_bytes
      else:
        dropped_documents += 1
    else:
      writer.write(_make_example(subwords, num_bytes))
  if combined_subwords:
    writer.write(_make_example(combined_subwords, combined_num_bytes))
  writer.close()

  tf.logging.info("total bytes: %d", total_bytes)
  tf.logging.info("total subwords: %d", total_subwords)
  tf.logging.info("bytes per subword: %f", total_bytes / total_subwords)
  tf.logging.info("total documents: %d", total_documents)
  tf.logging.info("dropped documents: %d", dropped_documents)


def main(_):
  """Convert a file to examples."""
  encoder = text_encoder.SubwordTextEncoder(FLAGS.vocab_file)

  in_files = tf.gfile.Glob(FLAGS.in_filepattern)
  assert in_files, "No matching input files"
  for in_file in in_files:
    convert_file(in_file, encoder)


if __name__ == "__main__":
  tf.app.run()
