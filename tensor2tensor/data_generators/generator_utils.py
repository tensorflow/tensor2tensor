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

"""Utilities for data generators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import gzip
import os
import random
import stat
import tarfile

# Dependency imports

import requests
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import six.moves.urllib_request as urllib  # Imports urllib on Python2, urllib.request on Python3

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import tokenizer

import tensorflow as tf

UNSHUFFLED_SUFFIX = "-unshuffled"


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
      if not six.PY2:  # Convert in python 3.
        v = [bytes(x, "utf-8") for x in v]
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    elif isinstance(v[0], bytes):
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    else:
      raise ValueError("Value for %s is not a recognized type; v: %s type: %s" %
                       (k, str(v[0]), str(type(v[0]))))
  return tf.train.Example(features=tf.train.Features(feature=features))


def generate_files_distributed(generator,
                               output_name,
                               output_dir,
                               num_shards=1,
                               max_cases=None,
                               task_id=0):
  """generate_files but with a single writer writing to shard task_id."""
  assert task_id < num_shards
  output_filename = sharded_name(output_name, task_id, num_shards)
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


def _data_filenames(output_name, output_dir, num_shards):
  return [
      os.path.join(output_dir, fname)
      for fname in shard_filepath(output_name, num_shards)
  ]


def train_data_filenames(problem, output_dir, num_shards):
  return _data_filenames(problem + "-train", output_dir, num_shards)


def dev_data_filenames(problem, output_dir, num_shards):
  return _data_filenames(problem + "-dev", output_dir, num_shards)


def test_data_filenames(problem, output_dir, num_shards):
  return _data_filenames(problem + "-test", output_dir, num_shards)


def combined_data_filenames(problem, output_dir, num_training_shards):
  return (train_data_filenames(problem, output_dir, num_training_shards) +
          dev_data_filenames(problem, output_dir, 1) + test_data_filenames(
              problem, output_dir, 1))


def sharded_name(base_name, shard, total_shards):
  return "%s-%.5d-of-%.5d" % (base_name, shard, total_shards)


def shard_filepath(fname, num_shards):
  return [
      sharded_name(fname, shard, num_shards) for shard in xrange(num_shards)
  ]


def generate_files(generator, output_filenames, max_cases=None):
  """Generate cases from a generator and save as TFRecord files.

  Generated cases are transformed to tf.Example protos and saved as TFRecords
  in sharded files named output_dir/output_name-00..N-of-00..M=num_shards.

  Args:
    generator: a generator yielding (string -> int/float/str list) dictionaries.
    output_filenames: List of output file paths.
    max_cases: maximum number of cases to get from the generator;
      if None (default), we use the generator until StopIteration is raised.
  """
  num_shards = len(output_filenames)
  writers = [tf.python_io.TFRecordWriter(fname) for fname in output_filenames]
  counter, shard = 0, 0
  for case in generator:
    if counter > 0 and counter % 100000 == 0:
      tf.logging.info("Generating case %d." % counter)
    counter += 1
    if max_cases and counter > max_cases:
      break
    sequence_example = to_example(case)
    writers[shard].write(sequence_example.SerializeToString())
    shard = (shard + 1) % num_shards

  for writer in writers:
    writer.close()


def download_report_hook(count, block_size, total_size):
  """Report hook for download progress.

  Args:
    count: current block number
    block_size: block size
    total_size: total size
  """
  percent = int(count * block_size * 100 / total_size)
  print("\r%d%%" % percent + " completed", end="\r")


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
    inprogress_filepath = filepath + ".incomplete"
    inprogress_filepath, _ = urllib.urlretrieve(
        url, inprogress_filepath, reporthook=download_report_hook)
    # Print newline to clear the carriage return from the download progress
    print()
    tf.gfile.Rename(inprogress_filepath, filepath)
    statinfo = os.stat(filepath)
    tf.logging.info("Successfully downloaded %s, %s bytes." %
                    (filename, statinfo.st_size))
  else:
    tf.logging.info("Not downloading, file already found: %s" % filepath)
  return filepath


def maybe_download_from_drive(directory, filename, url):
  """Download filename from google drive unless it's already in directory.

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
  confirm_token = None
  if tf.gfile.Exists(filepath):
    tf.logging.info("Not downloading, file already found: %s" % filepath)
    return filepath

  # Since the file is big, drive will scan it for virus and take it to a
  # warning page. We find the confirm token on this page and append it to the
  # URL to start the download process.
  confirm_token = None
  session = requests.Session()
  response = session.get(url, stream=True)
  for k, v in response.cookies.items():
    if k.startswith("download_warning"):
      confirm_token = v

  if confirm_token:
    url = url + "&confirm=" + confirm_token
  tf.logging.info("Downloading %s to %s" % (url, filepath))

  response = session.get(url, stream=True)
  # Now begin the download.
  chunk_size = 16 * 1024
  with open(filepath, "wb") as f:
    for chunk in response.iter_content(chunk_size):
      if chunk:
        f.write(chunk)

  # Print newline to clear the carriage return from the download progress
  print()
  statinfo = os.stat(filepath)
  tf.logging.info("Successfully downloaded %s, %s bytes." % (filename,
                                                             statinfo.st_size))
  return filepath


def gunzip_file(gz_path, new_path):
  """Unzips from gz_path into new_path.

  Args:
    gz_path: path to the zipped file.
    new_path: path to where the file will be unzipped.
  """
  if tf.gfile.Exists(new_path):
    tf.logging.info("File %s already exists, skipping unpacking" % new_path)
    return
  tf.logging.info("Unpacking %s to %s" % (gz_path, new_path))
  # We may be unpacking into a newly created directory, add write mode.
  mode = stat.S_IRWXU or stat.S_IXGRP or stat.S_IRGRP or stat.S_IROTH
  os.chmod(os.path.dirname(new_path), mode)
  with gzip.open(gz_path, "rb") as gz_file:
    with tf.gfile.GFile(new_path, mode="wb") as new_file:
      for line in gz_file:
        new_file.write(line)


def get_or_generate_vocab_inner(data_dir, vocab_filename, vocab_size,
                                generator):
  """Inner implementation for vocab generators.

  Args:
    data_dir: The base directory where data and vocab files are stored. If None,
        then do not save the vocab even if it doesn't exist.
    vocab_filename: relative filename where vocab file is stored
    vocab_size: target size of the vocabulary constructed by SubwordTextEncoder
    generator: a generator that produces tokens from the vocabulary

  Returns:
    A SubwordTextEncoder vocabulary object.
  """
  if data_dir is None:
    vocab_filepath = None
  else:
    vocab_filepath = os.path.join(data_dir, vocab_filename)

  if vocab_filepath is not None and tf.gfile.Exists(vocab_filepath):
    tf.logging.info("Found vocab file: %s", vocab_filepath)
    vocab = text_encoder.SubwordTextEncoder(vocab_filepath)
    return vocab

  tf.logging.info("Generating vocab file: %s", vocab_filepath)
  token_counts = defaultdict(int)
  for item in generator:
    for tok in tokenizer.encode(text_encoder.native_to_unicode(item)):
      token_counts[tok] += 1

  vocab = text_encoder.SubwordTextEncoder.build_to_target_size(
      vocab_size, token_counts, 1, 1e3)

  if vocab_filepath is not None:
    vocab.store_to_file(vocab_filepath)
  return vocab


def get_or_generate_vocab(data_dir, tmp_dir, vocab_filename, vocab_size,
                          sources):
  """Generate a vocabulary from the datasets in sources."""

  def generate():
    tf.logging.info("Generating vocab from: %s", str(sources))
    for source in sources:
      url = source[0]
      filename = os.path.basename(url)
      compressed_file = maybe_download(tmp_dir, filename, url)

      for lang_file in source[1]:
        tf.logging.info("Reading file: %s" % lang_file)
        filepath = os.path.join(tmp_dir, lang_file)

        # Extract from tar if needed.
        if not tf.gfile.Exists(filepath):
          read_type = "r:gz" if filename.endswith("tgz") else "r"
          with tarfile.open(compressed_file, read_type) as corpus_tar:
            corpus_tar.extractall(tmp_dir)

        # For some datasets a second extraction is necessary.
        if lang_file.endswith(".gz"):
          new_filepath = os.path.join(tmp_dir, lang_file[:-3])
          if tf.gfile.Exists(new_filepath):
            tf.logging.info(
                "Subdirectory %s already exists, skipping unpacking" % filepath)
          else:
            tf.logging.info("Unpacking subdirectory %s" % filepath)
            gunzip_file(filepath, new_filepath)
          filepath = new_filepath

        # Use Tokenizer to count the word occurrences.
        with tf.gfile.GFile(filepath, mode="r") as source_file:
          file_byte_budget = 1e6
          counter = 0
          countermax = int(source_file.size() / file_byte_budget / 2)
          for line in source_file:
            if counter < countermax:
              counter += 1
            else:
              if file_byte_budget <= 0:
                break
              line = line.strip()
              file_byte_budget -= len(line)
              counter = 0
              yield line

  return get_or_generate_vocab_inner(data_dir, vocab_filename, vocab_size,
                                     generate())


def get_or_generate_tabbed_vocab(data_dir, tmp_dir, source_filename,
                                 index, vocab_filename, vocab_size):
  r"""Generate a vocabulary from a tabbed source file.

  The source is a file of source, target pairs, where each line contains
  a source string and a target string, separated by a tab ('\t') character.
  The index parameter specifies 0 for the source or 1 for the target.

  Args:
    data_dir: path to the data directory.
    tmp_dir: path to the temporary directory.
    source_filename: the name of the tab-separated source file.
    index: index.
    vocab_filename: the name of the vocabulary file.
    vocab_size: vocabulary size.

  Returns:
    The vocabulary.
  """
  def generate():
    filepath = os.path.join(tmp_dir, source_filename)
    tf.logging.info("Generating vocab from %s", filepath)
    with tf.gfile.GFile(filepath, mode="r") as source_file:
      for line in source_file:
        line = line.strip()
        if line and "\t" in line:
          parts = line.split("\t", 1)
          part = parts[index].strip()
          yield part

  return get_or_generate_vocab_inner(data_dir, vocab_filename, vocab_size,
                                     generate())


def get_or_generate_txt_vocab(data_dir, vocab_filename, vocab_size,
                              filepatterns):
  """Generate a vocabulary from txt files with example-per-line."""
  if isinstance(filepatterns, str):
    filepatterns = [filepatterns]

  def generate():
    tf.logging.info("Generating vocab from %s", filepatterns)
    for filepattern in filepatterns:
      for filename in tf.gfile.Glob(filepattern):
        with tf.gfile.GFile(filename, mode="r") as source_file:
          for line in source_file:
            yield line.strip()

  return get_or_generate_vocab_inner(data_dir, vocab_filename, vocab_size,
                                     generate())


def read_records(filename):
  reader = tf.python_io.tf_record_iterator(filename)
  records = []
  for record in reader:
    records.append(record)
    if len(records) % 100000 == 0:
      tf.logging.info("read: %d", len(records))
  return records


def write_records(records, out_filename):
  writer = tf.python_io.TFRecordWriter(out_filename)
  for count, record in enumerate(records):
    writer.write(record)
    if count > 0 and count % 100000 == 0:
      tf.logging.info("write: %d", count)
  writer.close()


def generate_dataset_and_shuffle(train_gen,
                                 train_paths,
                                 dev_gen,
                                 dev_paths,
                                 shuffle=True):
  generate_files(train_gen, train_paths)
  generate_files(dev_gen, dev_paths)
  if shuffle:
    shuffle_dataset(train_paths + dev_paths)


def shuffle_dataset(filenames):
  tf.logging.info("Shuffling data...")
  for fname in filenames:
    records = read_records(fname)
    random.shuffle(records)
    out_fname = fname.replace(UNSHUFFLED_SUFFIX, "")
    write_records(records, out_fname)
    tf.gfile.Remove(fname)
