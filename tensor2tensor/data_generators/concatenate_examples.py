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

r"""Convert seq-seq examples to "concatenated" examples.

The concatenated example has no "inputs".
Instead the source is at the beginning of the target.

We can now use a simple language model.

Example:
seq-seq mode:
{
  "inputs": subtokenizer.encode("I love you.") + [1]
  "targets": subtokenizer.encode("Je t'aime.") + [1]
}
->
concatenated mode:
{
  "inputs": [0]
  "targets": (subtokenizer.encode("source English I love you.") + [1]
              + subtokenizer.encode("target French Je t'aime.") + [1])
}

We add a dummy feature "inputs"=[0] for compatability with seq-to-seq models.

If FLAGS.combine_to_length is nonzero, then we combine multiple examples into
examples of a constant length, possibly with some padding at the end.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import text_encoder
import tensorflow as tf

tf.app.flags.DEFINE_string("vocab_file", "",
                           "SubwordTextEncoder vocabulary file")

tf.app.flags.DEFINE_boolean(
    "random_reverse", False,
    "If true, write half of the example with source/target reversed")

tf.app.flags.DEFINE_boolean(
    "count_everything", False,
    "If true, assign positive weights to designators, source and target. "
    "If false, assign positive weights only to target.")

tf.app.flags.DEFINE_string("source_domain_string", "English", "")
tf.app.flags.DEFINE_string("target_domain_string", "French", "")

tf.app.flags.DEFINE_integer(
    "combine_to_length", 0,
    "If positive, concatenate examples to form examples with target length "
    " equal to this value. Targets are padded with subtoken id=0.")

tf.app.flags.DEFINE_string("in_file", "", "input filename")

tf.app.flags.DEFINE_string(
    "out_prefix", "/usr/local/google/tmp/concat",
    "The output filename is equal to out_prefix plus "
    "the last 15 characters of in_file. (e.g. -00001-of-00100)")

FLAGS = tf.app.flags.FLAGS


def _make_example(ids, weights, raw_num_bytes):
  if FLAGS.combine_to_length > 0:
    ids += [0] * (FLAGS.combine_to_length - len(ids))
  return generator_utils.to_example({
      "targets": ids,
      "target_weights": weights,
      "inputs": [0],
      "raw_num_bytes": [raw_num_bytes]
  }).SerializeToString()


def main(_):
  """Convert a file to examples."""
  subtokenizer = text_encoder.SubwordTextEncoder(FLAGS.vocab_file)
  total_bytes = 0
  total_subtokens = 0
  total_examples = 0
  dropped_examples = 0

  combined_subtokens = []
  combined_num_bytes = 0
  combined_weights = []

  source_specifier = subtokenizer.encode("source " + FLAGS.source_domain_string)
  target_specifier = subtokenizer.encode("target " + FLAGS.target_domain_string)
  if FLAGS.random_reverse:
    r_source_specifier = subtokenizer.encode("source " +
                                             FLAGS.target_domain_string)
    r_target_specifier = subtokenizer.encode("target " +
                                             FLAGS.source_domain_string)

  reader = tf.python_io.tf_record_iterator(FLAGS.in_file)

  out_file = FLAGS.out_prefix + FLAGS.in_file[-15:]
  writer = tf.python_io.TFRecordWriter(out_file)

  for record in reader:
    total_examples += 1
    if total_examples % 1000 == 0:
      tf.logging.info("total_examples: %d", total_examples)
    x = tf.train.Example()
    x.ParseFromString(record)
    inputs = [i for i in x.features.feature["inputs"].int64_list.value]
    targets = [i for i in x.features.feature["targets"].int64_list.value]
    should_reverse = FLAGS.random_reverse and random.random() < 0.5
    source_bytes = len(subtokenizer.decode(inputs[:-1])) + 1
    target_bytes = len(subtokenizer.decode(targets[:-1])) + 1
    if not should_reverse:
      subtokens = source_specifier + inputs + target_specifier + targets
      weights = ([0.0] *
                 (len(source_specifier) + len(inputs) + len(target_specifier)) +
                 [1.0] * len(targets))
      num_bytes = target_bytes
    else:
      subtokens = r_source_specifier + targets + r_target_specifier + inputs
      weights = (
          [0.0] *
          (len(r_source_specifier) + len(targets) + len(r_target_specifier)) +
          [1.0] * len(inputs))
      num_bytes = source_bytes
    if FLAGS.count_everything:
      weights = [1.0] * len(subtokens)
      num_bytes = source_bytes + target_bytes
    total_bytes += num_bytes
    total_subtokens += sum(weights)
    if FLAGS.combine_to_length:
      if combined_subtokens and (len(combined_subtokens) + len(subtokens) >
                                 FLAGS.combine_to_length):
        writer.write(
            _make_example(combined_subtokens, combined_weights,
                          combined_num_bytes))
        combined_subtokens = []
        combined_weights = []
        combined_num_bytes = 0
      if len(subtokens) <= FLAGS.combine_to_length:
        combined_subtokens.extend(subtokens)
        combined_weights.extend(weights)
        combined_num_bytes += num_bytes
      else:
        dropped_examples += 1
    else:
      writer.write(_make_example(subtokens, weights, num_bytes))
  if combined_subtokens:
    writer.write(
        _make_example(combined_subtokens, combined_weights, combined_num_bytes))
  writer.close()

  tf.logging.info("total bytes: %d", total_bytes)
  tf.logging.info("total subtokens: %d", total_subtokens)
  tf.logging.info("bytes per subtoken: %f", total_bytes / total_subtokens)
  tf.logging.info("total documents: %d", total_examples)
  tf.logging.info("dropped documents: %d", dropped_examples)


if __name__ == "__main__":
  tf.app.run()
