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

"""Generally useful utility functions."""
from __future__ import print_function

import codecs
import collections
import json
import math
import os
import sys
import time

import numpy as np
import tensorflow as tf

sys.stdout = codecs.getwriter("utf-8")(sys.stdout)


def safe_exp(value):
  """Exponentiation with catching of overflow error."""
  try:
    ans = math.exp(value)
  except OverflowError:
    ans = float("inf")
  return ans


def print_time(s, start_time):
  """Take a start time, print elapsed duration, and return a new time."""
  print("%s, time %ds, %s." % (s, (time.time() - start_time), time.ctime()))
  sys.stdout.flush()
  return time.time()


def print_out(s, f=None, new_line=True):
  """Similar to print but with support to flush and output to a file."""
  if f:
    if new_line:
      f.write("%s\n" % s)
    else:
      f.write("%s" % s)

  # stdout
  if new_line:
    sys.stdout.write("%s\n" % s)
  else:
    sys.stdout.write("%s" % s)
  sys.stdout.flush()


def print_hparams(hparams, skip_patterns=None):
  """Print hparams, can skip keys based on pattern."""
  values = hparams.values()
  for key in sorted(values.iterkeys()):
    if not skip_patterns or all(
        [skip_pattern not in key for skip_pattern in skip_patterns]):
      print_out("  %s=%s" % (key, str(values[key])))


def load_hparams(model_dir):
  """Load hparams from an existing model directory."""
  hparams_file = os.path.join(model_dir, "hparams")
  if tf.gfile.Exists(hparams_file):
    print_out("# Loading hparams from %s" % hparams_file)
    with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "r")) as f:
      hparams_values = json.load(f)
      hparams = tf.contrib.training.HParams(**hparams_values)
    return hparams
  else:
    return None


def save_hparams(out_dir, hparams):
  """Save hparams."""
  hparams_file = os.path.join(out_dir, "hparams")
  print_out("  saving hparams to %s" % hparams_file)
  with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "w")) as f:
    f.write(hparams.to_json())


def debug_tensor(s, msg=None, summarize=10):
  """Print the shape and value of a tensor at test time. Return a new tensor."""
  if not msg:
    msg = s.name
  return tf.Print(s, [tf.shape(s), s], msg + " ", summarize=summarize)


def add_summary(summary_writer, global_step, tag, value):
  """Add a new summary to the current summary_writer.
  Useful to log things that are not part of the training graph, e.g., tag=BLEU.
  """
  summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
  summary_writer.add_summary(summary, global_step)


def get_config_proto(log_device_placement=False, allow_soft_placement=True):
  # GPU options:
  # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
  config_proto = tf.ConfigProto(
      log_device_placement=log_device_placement,
      allow_soft_placement=allow_soft_placement)
  config_proto.gpu_options.allow_growth = True
  return config_proto


def load_list_map(path):
  """Read a list of strings into file, one item per line.

  Args:
    path: path to the file containing the list.

  Returns:
    values: a list of strings.

  Raises:
    ValueError: if the provided path does not exist or is empty.
  """

  if not path or not tf.gfile.Exists(path):
    raise ValueError("File %s not found.", path)
  else:
    print_out("# Loading string list from %s" % path)
    value_map = {}
    with codecs.getreader("utf-8")(tf.gfile.GFile(path, "r")) as f:
      for line in f:
        value_map[line.strip()] = 1
      return value_map


def int2text(indices, vocab, ignore_map=None):
  """Convert a sequence of integers into words using a vocab."""
  if (not hasattr(indices, "__len__") and  # for numpy array
      not isinstance(indices, collections.Iterable)):
    indices = [indices]

  if not ignore_map:
    words = [vocab[index] for index in indices]
  else:
    for index in indices:
      if vocab[index] in ignore_map:
        continue
      words.append(vocab[index])
  return " ".join(words)


def bpe_int2text(indices, vocab, ignore_map=None, delimiter="@@"):
  """Convert a sequence of integers into words using a vocab."""
  words = []
  word = ""
  delimiter_len = len(delimiter)
  for index in indices:
    symbol = vocab[index]
    if len(symbol) >= delimiter_len and symbol[-delimiter_len:] == delimiter:
      word += symbol[:-delimiter_len]
    else:  # end of a word
      word += symbol
      if not ignore_map or word not in ignore_map:
        words.append(word)
      word = ""
  return " ".join(words)


def build_buckets(max_seq_len, num_buckets):
  """Build buckets. For example, [10, 20, 30, 40, 50]."""
  buckets = []
  bucket_width = int((max_seq_len + num_buckets - 1) / num_buckets)
  for i in range(num_buckets):
    value = bucket_width * (i + 1)
    # the last value is max_seq_len-1
    if value > max_seq_len:
      value = max_seq_len - 1
    buckets.append(value)
  print_out("# buckets: %s" % str(buckets))
  return buckets


def select_bucket(buckets_scale):
  """Choose a bucket according to data distribution.

  We pick a random number in [0, 1] and
  use the corresponding interval in buckets_scale.
  """
  random_number_01 = np.random.random_sample()
  for i in range(len(buckets_scale)):
    if buckets_scale[i] > random_number_01:
      return i


def text2int(line, vocab_hash, unk="<unk>"):
  """Turn a line of text into a sequence of indices."""
  words = line.split()
  ids = []
  for word in words:
    ids.append(vocab_hash[word] if word in vocab_hash else vocab_hash[unk])
  return ids


def read_text_to_int(path, vocab_hash, unk="<unk>", max_train=0):
  """Read text data from a file and convert to integers.

  Args:
   path: path to a text file.
   vocab_hash: to map strings to integers
   unk: to look up vocab_hash if we do not know a word.
   max_train: maximum number of lines to read, all other will be ignored;
     if <=0, data files will be read completely (no limit).
  Returns:
   data_set: a list of sequences of integers
   max_len: maximum sequence length
  Raises:
    ValueError: if unk is not in the vocab.
  """
  if unk not in vocab_hash:
    raise ValueError("No unk word %s in the vocab hash" % unk)

  data_set = []
  max_len = 0
  sent_id_max_len = -1
  min_len = 1e10
  sent_id_min_len = -1
  total_len = 0
  with codecs.getreader("utf-8")(tf.gfile.GFile(path, "r")) as in_file:
    counter = 0
    print_out("  reading file %s" % path, new_line=False)
    for line in in_file:
      counter += 1

      # Process line
      ids = text2int(line.strip(), vocab_hash, unk)
      data_set.append(ids)

      # Check len
      num_words = len(ids)
      if num_words > max_len:
        max_len = num_words
        sent_id_max_len = counter
      if num_words < min_len:
        min_len = num_words
        sent_id_min_len = counter
      total_len += num_words

      # Max train
      if counter == max_train:
        break

      # Log
      if counter % 1000000 == 0:
        print_out(" (%dM) " % (counter / 1000000), new_line=False)
    print_out("    num lines = %d, max len %d (sent %d), min len %d (sent %d),"
              " avg len %.2f" % (counter, max_len, sent_id_max_len, min_len,
                                 sent_id_min_len, float(total_len) / counter))

  return data_set, max_len
