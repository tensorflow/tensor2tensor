# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

python data_generators/inspect_tfrecord.py \
    --logtostderr \
    --print_targets \
    --subword_text_encoder_filename=$DATA_DIR/vocab.endefr.8192 \
    --input_filename=$DATA_DIR/wmt_ende_tokens_8k-train-00000-of-00100
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

from tensor2tensor.data_generators import text_encoder

import tensorflow.compat.v1 as tf


tf.flags.DEFINE_string("subword_text_encoder_filename", "",
                       "SubwordTextEncoder vocabulary file")
tf.flags.DEFINE_string("token_text_encoder_filename", "",
                       "TokenTextEncoder vocabulary file")
tf.flags.DEFINE_bool("byte_text_encoder", False, "use a ByteTextEncoder")
tf.flags.DEFINE_string("input_filename", "", "input filename")
tf.flags.DEFINE_bool("print_inputs", False, "Print decoded inputs to stdout")
tf.flags.DEFINE_bool("print_targets", False, "Print decoded targets to stdout")
tf.flags.DEFINE_bool("print_all", False, "Print all fields")

FLAGS = tf.flags.FLAGS


def main(_):
  """Convert a file to examples."""
  if FLAGS.subword_text_encoder_filename:
    encoder = text_encoder.SubwordTextEncoder(
        FLAGS.subword_text_encoder_filename)
  elif FLAGS.token_text_encoder_filename:
    encoder = text_encoder.TokenTextEncoder(FLAGS.token_text_encoder_filename)
  elif FLAGS.byte_text_encoder:
    encoder = text_encoder.ByteTextEncoder()
  else:
    encoder = None
  reader = tf.python_io.tf_record_iterator(FLAGS.input_filename)
  total_sequences = 0
  total_input_tokens = 0
  total_target_tokens = 0
  nonpadding_input_tokens = 0
  nonpadding_target_tokens = 0
  max_input_length = 0
  max_target_length = 0
  for record in reader:
    x = tf.train.Example()
    x.ParseFromString(record)
    inputs = [int(i) for i in x.features.feature["inputs"].int64_list.value]
    targets = [int(i) for i in x.features.feature["targets"].int64_list.value]
    if FLAGS.print_inputs:
      print("INPUTS:\n" + encoder.decode(inputs) if encoder else inputs)
    if FLAGS.print_targets:
      print("TARGETS:\n" + encoder.decode(targets) if encoder else targets)
    nonpadding_input_tokens += len(inputs) - inputs.count(0)
    nonpadding_target_tokens += len(targets) - targets.count(0)
    total_input_tokens += len(inputs)
    total_target_tokens += len(targets)
    total_sequences += 1
    max_input_length = max(max_input_length, len(inputs))
    max_target_length = max(max_target_length, len(targets))
    if FLAGS.print_all:
      for k, v in six.iteritems(x.features.feature):
        print("%s: %s" % (k, v.int64_list.value))

  print("total_sequences: %d" % total_sequences)
  print("total_input_tokens: %d" % total_input_tokens)
  print("total_target_tokens: %d" % total_target_tokens)
  print("nonpadding_input_tokens: %d" % nonpadding_input_tokens)
  print("nonpadding_target_tokens: %d" % nonpadding_target_tokens)
  print("max_input_length: %d" % max_input_length)
  print("max_target_length: %d" % max_target_length)


if __name__ == "__main__":
  tf.app.run()
