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

"""Utilities for data generators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import io
import os
import tarfile
import urllib

# Dependency imports

import six
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder
from tensor2tensor.data_generators.tokenizer import Tokenizer

import tensorflow as tf


def to_example(dictionary):
  """Helper: build tf.Example from (string -> int/float/str list) dictionary."""
  features = {}
  for (k, v) in six.iteritems(dictionary):
    if not v:
      raise ValueError("Empty generated field: %s", str((k, v)))
    if isinstance(v[0], six.integer_types):
      features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    elif isinstance(v[0], float):
      features[k] = tf.train.Feature(float_list=tf.train.FloatList(value=v))
    elif isinstance(v[0], six.string_types):
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    else:
      raise ValueError("Value is neither an int nor a float; v: %s type: %s" %
                       (str(v[0]), str(type(v[0]))))
  return tf.train.Example(features=tf.train.Features(feature=features))


def generate_files_distributed(generator,
                               output_name,
                               output_dir,
                               num_shards=1,
                               max_cases=None,
                               task_id=0):
  """generate_files but with a single writer writing to shard task_id."""
  assert task_id < num_shards
  output_filename = "%s-%.5d-of-%.5d" % (output_name, task_id, num_shards)
  output_file = os.path.join(output_dir, output_filename)
  tf.logging.info("Writing to file %s", output_file)
  writer = tf.python_io.TFRecordWriter(output_file)

  counter = 0
  for case in generator:
    if counter % 100000 == 0:
      tf.logging.info("Generating case %d for %s." % (counter, output_name))
    counter += 1
    if max_cases and counter > max_cases:
      break
    sequence_example = to_example(case)
    writer.write(sequence_example.SerializeToString())

  writer.close()
  return output_file


def generate_files(generator,
                   output_name,
                   output_dir,
                   num_shards=1,
                   max_cases=None):
  """Generate cases from a generator and save as TFRecord files.

  Generated cases are transformed to tf.Example protos and saved as TFRecords
  in sharded files named output_dir/output_name-00..N-of-00..M=num_shards.

  Args:
    generator: a generator yielding (string -> int/float/str list) dictionaries.
    output_name: the file name prefix under which output will be saved.
    output_dir: directory to save the output to.
    num_shards: how many shards to use (defaults to 1).
    max_cases: maximum number of cases to get from the generator;
      if None (default), we use the generator until StopIteration is raised.

  Returns:
    List of output file paths.
  """
  writers = []
  output_files = []
  for shard in xrange(num_shards):
    output_filename = "%s-%.5d-of-%.5d" % (output_name, shard, num_shards)
    output_file = os.path.join(output_dir, output_filename)
    output_files.append(output_file)
    writers.append(tf.python_io.TFRecordWriter(output_file))

  counter, shard = 0, 0
  for case in generator:
    if counter % 100000 == 0:
      tf.logging.info("Generating case %d for %s." % (counter, output_name))
    counter += 1
    if max_cases and counter > max_cases:
      break
    sequence_example = to_example(case)
    writers[shard].write(sequence_example.SerializeToString())
    shard = (shard + 1) % num_shards

  for writer in writers:
    writer.close()

  return output_files


def maybe_download(directory, filename, url):
  """Download filename from url unless it's already in directory.

  Args:
    directory: path to the directory that will be used.
    filename: name of the file to download to (do nothing if it already exists).
    url: URL to download from.

  Returns:
    The path to the downloaded file.
  """
  if not tf.gfile.Exists(directory):
    tf.logging.info("Creating directory %s" % directory)
    os.mkdir(directory)
  filepath = os.path.join(directory, filename)
  if not tf.gfile.Exists(filepath):
    tf.logging.info("Downloading %s to %s" % (url, filepath))
    filepath, _ = urllib.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    tf.logging.info("Succesfully downloaded %s, %s bytes." % (filename,
                                                              statinfo.st_size))
  else:
    tf.logging.info("Not downloading, file already found: %s" % filepath)
  return filepath


def gunzip_file(gz_path, new_path):
  """Unzips from gz_path into new_path.

  Args:
    gz_path: path to the zipped file.
    new_path: path to where the file will be unzipped.
  """
  tf.logging.info("Unpacking %s to %s" % (gz_path, new_path))
  with gzip.open(gz_path, "rb") as gz_file:
    with io.open(new_path, "wb") as new_file:
      for line in gz_file:
        new_file.write(line)


# TODO(aidangomez): en-fr tasks are significantly over-represented below
_DATA_FILE_URLS = [
    # German-English
    [
        "http://data.statmt.org/wmt16/translation-task/training-parallel-nc-v11.tgz",  # pylint: disable=line-too-long
        [
            "training-parallel-nc-v11/news-commentary-v11.de-en.en",
            "training-parallel-nc-v11/news-commentary-v11.de-en.de"
        ]
    ],
    # German-English & French-English
    [
        "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz", [
            "commoncrawl.de-en.en", "commoncrawl.de-en.de",
            "commoncrawl.fr-en.en", "commoncrawl.fr-en.fr"
        ]
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz", [
            "training/europarl-v7.de-en.en", "training/europarl-v7.de-en.de",
            "training/europarl-v7.fr-en.en", "training/europarl-v7.fr-en.fr"
        ]
    ],
    # French-English
    [
        "http://www.statmt.org/wmt10/training-giga-fren.tar",
        ["giga-fren.release2.fixed.en.gz", "giga-fren.release2.fixed.fr.gz"]
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-un.tgz",
        ["un/undoc.2000.fr-en.en", "un/undoc.2000.fr-en.fr"]
    ],
]


def get_or_generate_vocab(tmp_dir, vocab_filename, vocab_size):
  """Generate a vocabulary from the datasets listed in _DATA_FILE_URLS."""
  vocab_filepath = os.path.join(tmp_dir, vocab_filename)
  if os.path.exists(vocab_filepath):
    vocab = SubwordTextEncoder(vocab_filepath)
    return vocab

  tokenizer = Tokenizer()
  for source in _DATA_FILE_URLS:
    url = source[0]
    filename = os.path.basename(url)
    read_type = "r:gz" if "tgz" in filename else "r"

    compressed_file = maybe_download(tmp_dir, filename, url)

    with tarfile.open(compressed_file, read_type) as corpus_tar:
      corpus_tar.extractall(tmp_dir)

    for lang_file in source[1]:
      tf.logging.info("Reading file: %s" % lang_file)
      filepath = os.path.join(tmp_dir, lang_file)

      # For some datasets a second extraction is necessary.
      if ".gz" in lang_file:
        tf.logging.info("Unpacking subdirectory %s" % filepath)
        new_filepath = os.path.join(tmp_dir, lang_file[:-3])
        gunzip_file(filepath, new_filepath)
        filepath = new_filepath

      # Use Tokenizer to count the word occurrences.
      with tf.gfile.GFile(filepath, mode="r") as source_file:
        file_byte_budget = 3.5e5 if "en" in filepath else 7e5
        for line in source_file:
          if file_byte_budget <= 0:
            break
          line = line.strip()
          file_byte_budget -= len(line)
          _ = tokenizer.encode(line)

  vocab = SubwordTextEncoder.build_to_target_size(
      vocab_size, tokenizer.token_counts, vocab_filepath, 1, 1e3)
  return vocab


def read_records(filename):
  reader = tf.python_io.tf_record_iterator(filename)
  records = []
  for record in reader:
    records.append(record)
    if len(records) % 10000 == 0:
      tf.logging.info("read: %d", len(records))
  return records


def write_records(records, out_filename):
  writer = tf.python_io.TFRecordWriter(out_filename)
  for count, record in enumerate(records):
    writer.write(record)
    if count % 10000 == 0:
      tf.logging.info("write: %d", count)
  writer.close()
