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

r"""Inspect a TFRecord file of tensorflow.Example and show tokenizations.

python data_generators/text_encoder_inspect_subword.py \
    --logtostderr \
    --vocab_file=$DATA_DIR/tokens.vocab.8192 \
    --in_file=$DATA_DIR/wmt_ende_tokens_8k-train-00000-of-00100
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.data_generators import text_encoder

import tensorflow as tf

tf.app.flags.DEFINE_string("vocab_file", "",
                           "SubwordTextEncoder vocabulary file")

tf.app.flags.DEFINE_string("in_file", "", "input filename")

FLAGS = tf.app.flags.FLAGS


def ShowSequence(subtokenizer, subtokens, label):
  print("%s decoded = %s" % (label, subtokenizer.decode(subtokens)))
  print("%s subtoken ids = %s" % (label, subtokens))
  print("%s subtoken strings = %s" %
        (label,
         [subtokenizer.subtoken_to_subtoken_string(s) for s in subtokens]))
  print("")


def main(_):
  """Convert a file to examples."""
  subtokenizer = text_encoder.SubwordTextEncoder(FLAGS.vocab_file)
  reader = tf.python_io.tf_record_iterator(FLAGS.in_file)
  for record in reader:
    x = tf.train.Example()
    x.ParseFromString(record)
    inputs = [int(i) for i in x.features.feature["inputs"].int64_list.value]
    targets = [int(i) for i in x.features.feature["targets"].int64_list.value]
    ShowSequence(subtokenizer, inputs, "inputs")
    ShowSequence(subtokenizer, targets, "targets")


if __name__ == "__main__":
  tf.app.run()
