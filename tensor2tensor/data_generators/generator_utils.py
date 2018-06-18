# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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
import multiprocessing as mp
import os
import random
import stat
import tarfile
import tempfile
import requests
import six
from six.moves import range  # pylint: disable=redefined-builtin
# Imports urllib on Python2, urllib.request on Python3
import six.moves.urllib_request as urllib

from tensor2tensor.data_generators import text_encoder

import tensorflow as tf

UNSHUFFLED_SUFFIX = "-unshuffled"


def to_example(dictionary):
  """Helper: build tf.Example from (string -> int/float/str list) dictionary."""
  features = {}
  for (k, v) in six.iteritems(dictionary):
    if not v:
      raise ValueError("Empty generated field: %s" % str((k, v)))
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
    example = to_example(case)
    writer.write(example.SerializeToString())

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
      sharded_name(fname, shard, num_shards) for shard in range(num_shards)
  ]


def outputs_exist(filenames):
  for out_fname in filenames:
    out_fname = out_fname.replace(UNSHUFFLED_SUFFIX, "")
    if tf.gfile.Exists(out_fname):
      return out_fname


def generate_files(generator, output_filenames,
                   max_cases=None, cycle_every_n=1):
  """Generate cases from a generator and save as TFRecord files.

  Generated cases are transformed to tf.Example protos and saved as TFRecords
  in sharded files named output_dir/output_name-00..N-of-00..M=num_shards.

  Args:
    generator: a generator yielding (string -> int/float/str list) dictionaries.
    output_filenames: List of output file paths.
    max_cases: maximum number of cases to get from the generator;
      if None (default), we use the generator until StopIteration is raised.
    cycle_every_n: how many cases from the generator to take before
      switching to the next shard; by default set to 1, switch every case.
  """
  if outputs_exist(output_filenames):
    tf.logging.info("Skipping generator because outputs files exist")
    return
  tmp_filenames = [fname + ".incomplete" for fname in output_filenames]
  num_shards = len(output_filenames)
  writers = [tf.python_io.TFRecordWriter(fname) for fname in tmp_filenames]
  counter, shard = 0, 0
  for case in generator:
    if case is None:
      continue
    if counter % 100000 == 0:
      tf.logging.info("Generating case %d." % counter)
    counter += 1
    if max_cases and counter > max_cases:
      break
    example = to_example(case)
    writers[shard].write(example.SerializeToString())
    if counter % cycle_every_n == 0:
      shard = (shard + 1) % num_shards

  for writer in writers:
    writer.close()

  for tmp_name, final_name in zip(tmp_filenames, output_filenames):
    tf.gfile.Rename(tmp_name, final_name)

  tf.logging.info("Generated %s Examples", counter)


def download_report_hook(count, block_size, total_size):
  """Report hook for download progress.

  Args:
    count: current block number
    block_size: block size
    total_size: total size
  """
  percent = int(count * block_size * 100 / total_size)
  print("\r%d%%" % percent + " completed", end="\r")


def maybe_download(directory, filename, uri):
  """Download filename from uri unless it's already in directory.

  Copies a remote file to local if that local file does not already exist.  If
  the local file pre-exists this function call, it does not check that the local
  file is a copy of the remote.

  Remote filenames can be filepaths, any URI readable by tensorflow.gfile, or a
  URL.

  Args:
    directory: path to the directory that will be used.
    filename: name of the file to download to (do nothing if it already exists).
    uri: URI to copy (or download) from.

  Returns:
    The path to the downloaded file.
  """
  tf.gfile.MakeDirs(directory)
  filepath = os.path.join(directory, filename)
  if tf.gfile.Exists(filepath):
    tf.logging.info("Not downloading, file already found: %s" % filepath)
    return filepath

  tf.logging.info("Downloading %s to %s" % (uri, filepath))
  try:
    tf.gfile.Copy(uri, filepath)
  except tf.errors.UnimplementedError:
    if uri.startswith("http"):
      inprogress_filepath = filepath + ".incomplete"
      inprogress_filepath, _ = urllib.urlretrieve(
          uri, inprogress_filepath, reporthook=download_report_hook)
      # Print newline to clear the carriage return from the download progress
      print()
      tf.gfile.Rename(inprogress_filepath, filepath)
    else:
      raise ValueError("Unrecognized URI: " + filepath)
  statinfo = os.stat(filepath)
  tf.logging.info("Successfully downloaded %s, %s bytes." %
                  (filename, statinfo.st_size))
  return filepath


def maybe_download_from_drive(directory, filename, url):
  """Download filename from Google drive unless it's already in directory.

  Args:
    directory: path to the directory that will be used.
    filename: name of the file to download to (do nothing if it already exists).
    url: URL to download from.

  Returns:
    The path to the downloaded file.
  """
  if not tf.gfile.Exists(directory):
    tf.logging.info("Creating directory %s" % directory)
    tf.gfile.MakeDirs(directory)
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
                                generator, max_subtoken_length=None,
                                reserved_tokens=None):
  """Inner implementation for vocab generators.

  Args:
    data_dir: The base directory where data and vocab files are stored. If None,
      then do not save the vocab even if it doesn't exist.
    vocab_filename: relative filename where vocab file is stored
    vocab_size: target size of the vocabulary constructed by SubwordTextEncoder
    generator: a generator that produces tokens from the vocabulary
    max_subtoken_length: an optional integer.  Set this to a finite value to
      avoid quadratic costs during vocab building.
    reserved_tokens: List of reserved tokens. `text_encoder.RESERVED_TOKENS`
      should be a prefix of `reserved_tokens`. If `None`, defaults to
      `RESERVED_TOKENS`.

  Returns:
    A SubwordTextEncoder vocabulary object.
  """
  if data_dir and vocab_filename:
    vocab_filepath = os.path.join(data_dir, vocab_filename)
    if tf.gfile.Exists(vocab_filepath):
      tf.logging.info("Found vocab file: %s", vocab_filepath)
      return text_encoder.SubwordTextEncoder(vocab_filepath)
  else:
    vocab_filepath = None

  tf.logging.info("Generating vocab file: %s", vocab_filepath)
  vocab = text_encoder.SubwordTextEncoder.build_from_generator(
      generator, vocab_size, max_subtoken_length=max_subtoken_length,
      reserved_tokens=reserved_tokens)

  if vocab_filepath:
    tf.gfile.MakeDirs(data_dir)
    vocab.store_to_file(vocab_filepath)

  return vocab


def get_or_generate_vocab(data_dir, tmp_dir, vocab_filename, vocab_size,
                          sources, file_byte_budget=1e6):
  """Generate a vocabulary from the datasets in sources."""

  def generate():
    """Generate lines for vocabulary generation."""
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

        with tf.gfile.GFile(filepath, mode="r") as source_file:
          file_byte_budget_ = file_byte_budget
          counter = 0
          countermax = int(source_file.size() / file_byte_budget_ / 2)
          for line in source_file:
            if counter < countermax:
              counter += 1
            else:
              if file_byte_budget_ <= 0:
                break
              line = line.strip()
              file_byte_budget_ -= len(line)
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


def _shuffle_single(fname):
  records = read_records(fname)
  random.shuffle(records)
  out_fname = fname.replace(UNSHUFFLED_SUFFIX, "")
  write_records(records, out_fname)
  tf.gfile.Remove(fname)


def shuffle_dataset(filenames):
  """Shuffles the dataset."""
  if outputs_exist(filenames):
    tf.logging.info("Skipping shuffle because output files exist")
    return
  tf.logging.info("Shuffling data...")
  if len(filenames) > 1:
    pool = mp.Pool(min(len(filenames), 20))
    pool.map(_shuffle_single, filenames)
  else:
    _shuffle_single(filenames[0])
  tf.logging.info("Data shuffled.")


class SequencePacker(object):
  """Helper for constructing a packed example of sequence examples.

  See comments to pack_examples()
  """

  def __init__(self, first_sequence, spacing=2):
    self._spacing = spacing
    self._ids = first_sequence[:]
    self._segmentation = [1] * len(first_sequence)
    self._position = list(range(len(first_sequence)))

  def add(self, ids):
    padding = [0] * self._spacing
    self._ids.extend(padding + ids)
    next_segment_num = self._segmentation[-1] + 1 if self._segmentation else 1
    self._segmentation.extend(padding + [next_segment_num] * len(ids))
    self._position.extend(padding + list(range(len(ids))))

  def can_fit(self, ids, packed_length):
    return len(self._ids) + self._spacing + len(ids) <= packed_length

  def to_dict(self):
    return {"inputs": [0],
            "targets": self._ids,
            "targets_segmentation": self._segmentation,
            "targets_position": self._position}


class SequencePairPacker(object):
  """Helper for packing sequence-to-sequence examples into bigger examples.

  See comments to pack_examples()
  """

  def __init__(self, first_sequence_pair, spacing=2):
    self._inputs = SequencePacker(first_sequence_pair[0], spacing)
    self._targets = SequencePacker(first_sequence_pair[1], spacing)

  def add(self, pair):
    self._inputs.add(pair[0])
    self._targets.add(pair[1])

  def can_fit(self, pair, packed_length):
    return (self._inputs.can_fit(pair[0], packed_length) and
            self._targets.can_fit(pair[1], packed_length))

  def to_dict(self):
    ret = self._targets.to_dict()
    inputs_dict = self._inputs.to_dict()
    ret["inputs"] = inputs_dict["targets"]
    ret["inputs_segmentation"] = inputs_dict["targets_segmentation"]
    ret["inputs_position"] = inputs_dict["targets_position"]
    return ret


def pack_examples(examples,
                  has_inputs,
                  packed_length=256,
                  spacing=2,
                  queue_size=10,
                  chop_long_sequences=False):
  """Pack examples into longer examples.

  If has_inputs=False, we are packing single-sequence examples with
  targets only and no inputs.

  In this case, we concatenate the targets from several examples to form
  each new example.  We insert a number of zeros for spacing between the
  original sequences.  This is to help the sequences stay separate
  under convolutions.  If chop_long_sequences is set, then any input sequence
  longer than packed_length gets chopped up into multiple examples.  Otherwise,
  long sequences are emitted as singletons.

  If has_inputs=True, then we are packing sequence-to-sequence
  examples.  We combine several examples by concatenating the inputs
  (as above) and concatenating the targets (as above).  Chopping of
  long sequences is not supported.

  The packed examples are represented as dictionaries containing:
    "inputs", "targets": the packed sequences described above
    "inputs_segmentation", "targets_segmentation":
       Sequences aligned with "inputs", "targets" specifying to which original
       sequence each position belongs.  Numbering starts from 1, and 0 is used
       for spacing.  This information is useful for preventing attention across
       segments.
       e.g. [1 1 1 1 1 1 0 0 2 2 2 0 0 3 3 3 3 3 0 0 4 4 4]
     "inputs_position", "targets_position":
       Sequences aligned with "inputs", "targets" specifying position within
       the original sequence.  This is useful for positional encodings.
       e.g. [0 1 2 3 4 5 0 0 0 1 2 0 0 0 1 2 3 4 0 0 0 1 2]

  Args:
    examples: a generator returning feature dictionaries.
    has_inputs: a boolean
    packed_length: an integer
    spacing: an integer
    queue_size: an integer
    chop_long_sequences: a boolean

  Yields:
    feature dictionaries.
  """
  packer = SequencePairPacker if has_inputs else SequencePacker
  combined = []
  for example in examples:
    x = ((example["inputs"], example["targets"])
         if has_inputs else example["targets"])
    if chop_long_sequences and len(x) > packed_length:
      assert not has_inputs
      num_fragments = len(x) // packed_length
      for i in range(num_fragments):
        yield packer(
            x[packed_length * i:packed_length * (i + 1)], spacing).to_dict()
      x = x[packed_length * num_fragments:]
    added = False
    for c in combined:
      if c.can_fit(x, packed_length):
        c.add(x)
        added = True
        break
    if not added:
      if len(combined) == queue_size:
        yield combined[0].to_dict()
        combined = combined[1:]
      combined.append(packer(x, spacing))
  for c in combined:
    yield c.to_dict()


def make_tmp_dir(suffix="", prefix="tmp", dir=None):  # pylint: disable=redefined-builtin
  """Make a temporary directory."""
  if dir is None:
    return tempfile.mkdtemp(suffix, prefix, dir)
  else:
    while True:
      rand_term = random.randint(1, 9999)
      tmp_dir = os.path.join(dir, "%s%d%s" % (prefix, rand_term, suffix))
      if tf.gfile.Exists(tmp_dir):
        continue
      tf.gfile.MakeDirs(tmp_dir)
      break
    return tmp_dir


def tfrecord_iterator_for_problem(problem, data_dir,
                                  dataset_split=tf.estimator.ModeKeys.TRAIN):
  """Iterate over the records on disk for the Problem."""
  filenames = tf.gfile.Glob(problem.filepattern(data_dir, mode=dataset_split))
  example_spec = problem.example_reading_spec()[0]
  return tfrecord_iterator(filenames, example_spec=example_spec)


def tfrecord_iterator(filenames, gzipped=False, example_spec=None):
  """Yields records from TFRecord files.

  Args:
    filenames: list<str>, list of TFRecord filenames to read from.
    gzipped: bool, whether the TFRecord files are gzip-encoded.
    example_spec: dict<str feature name, tf.VarLenFeature/tf.FixedLenFeature>,
      if provided, will parse each record as a tensorflow.Example proto.

  Yields:
    Records (or parsed Examples, if example_spec is provided) from files.
  """
  with tf.Graph().as_default():
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    def _load_records(filename):
      return tf.data.TFRecordDataset(
          filename,
          compression_type=tf.constant("GZIP") if gzipped else None,
          buffer_size=16 * 1000 * 1000)

    dataset = dataset.flat_map(_load_records)

    def _parse_example(ex_ser):
      return tf.parse_single_example(ex_ser, example_spec)

    if example_spec:
      dataset = dataset.map(_parse_example, num_parallel_calls=32)
    dataset = dataset.prefetch(100)
    record_it = dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
      while True:
        try:
          ex = sess.run(record_it)
          yield ex
        except tf.errors.OutOfRangeError:
          break
