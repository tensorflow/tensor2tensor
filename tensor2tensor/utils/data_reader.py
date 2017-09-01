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

"""Data reader module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

# Dependency imports

import numpy as np

import six
from six.moves import zip  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.data_generators.problem import preprocess_examples_common
from tensor2tensor.utils import registry

import tensorflow as tf


def examples_reader(data_sources,
                    data_fields_to_features,
                    training,
                    capacity=32,
                    data_items_to_decoders=None,
                    data_items_to_decode=None):
  """Reads Examples from data_sources and decodes to Tensors.

  The dictionary data_fields_to_features for an image dataset can be:

  data_fields_to_features = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
    'image/class/label': tf.FixedLenFeature(
        [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
  }

  and for a simple algorithmic dataset with variable-length data it is:

  data_fields_to_features = {
    'inputs': tf.VarLenFeature(tf.int64),
    'targets': tf.VarLenFeature(tf.int64),
  }

  The data_items_to_decoders dictionary argument can be left as None if there
  is no decoding to be performed. But, e.g. for images, it should be set so that
  the images are decoded from the features, e.g., for MNIST:

  data_items_to_decoders = {
    'image': tfexample_decoder.Image(
      image_key = 'image/encoded',
      format_key = 'image/format',
      shape=[28, 28],
      channels=1),
    'label': tfexample_decoder.Tensor('image/class/label'),
  }

  These arguments are compatible with the use of tf.contrib.slim.data module,
  see there for more documentation.

  Args:
    data_sources: a list or tuple of sources from which the data will be read,
      for example [/path/to/train@128, /path/to/train2*, /tmp/.../train3*]
    data_fields_to_features: a dictionary from data fields in the data sources
      to features, such as tf.VarLenFeature(tf.int64), see above for examples.
    training: a Boolean, whether to read for training or evaluation.
    capacity: integer, buffer capacity; set to 2 * max_batch_size or more.
    data_items_to_decoders: a dictionary mapping data items (that will be
      in the returned result) to decoders that will decode them using features
      defined in data_fields_to_features; see above for examples. By default
      (if this is None), we grab the tensor from every feature.
    data_items_to_decode: a subset of data items that will be decoded;
      by default (if this is None), we decode all items.

  Returns:
    A tf.contrib.data.Dataset of dict<feature name, Tensor>
  """

  def decode_record(record):
    """Serialized Example to dict of <feature name, Tensor>."""
    example_serialized = record
    item_decoders = data_items_to_decoders
    if item_decoders is None:
      item_decoders = {
          field: tf.contrib.slim.tfexample_decoder.Tensor(field)
          for field in data_fields_to_features
      }

    decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
        data_fields_to_features, item_decoders)

    decode_items = data_items_to_decode
    if decode_items is None:
      decode_items = list(item_decoders)

    decoded = decoder.decode(example_serialized, items=decode_items)
    return dict(zip(decode_items, decoded))

  with tf.name_scope("examples_in"):
    data_files = tf.contrib.slim.parallel_reader.get_data_files(data_sources)
    if training:
      random.shuffle(data_files)
    dataset = tf.contrib.data.TFRecordDataset(data_files)
    num_threads = min(4 if training else 1, len(data_files))
    dataset = dataset.map(decode_record, num_threads=num_threads)
    if training:
      dataset = dataset.shuffle(capacity)
    # Loop inifinitely if training, just once otherwise
    dataset = dataset.repeat(None if training else 1)
    return dataset


def preprocessing(examples, data_file_pattern):
  """Preprocessing of examples."""
  # This function is for obsolete problems only, as we're porting them
  # all to the Problem class and its preprocess_examples method. Don't add.
  if "image" in data_file_pattern:

    def resize(img, size):
      return tf.to_int64(
          tf.image.resize_images(img, [size, size], tf.image.ResizeMethod.AREA))

    if "img2img" in data_file_pattern:
      inputs = examples["inputs"]
      examples["inputs"] = resize(inputs, 16)
      examples["targets"] = resize(inputs, 64)
  elif "audio" in data_file_pattern:
    # Reshape audio to proper shape
    sample_count = tf.to_int32(examples.pop("audio/sample_count"))
    sample_width = tf.to_int32(examples.pop("audio/sample_width"))
    channel_count = 1
    examples["inputs"] = tf.reshape(examples["inputs"],
                                    [sample_count, sample_width, channel_count])
    if "wsj" in data_file_pattern:
      examples["inputs"] = tf.bitcast(examples["inputs"], tf.int32)
  elif "a2q_20161229" in data_file_pattern:
    # we forgot the EOS when we preprocessed this data.
    examples["targets"] = tf.concat([examples["targets"], [1]], 0)
  return examples


def cast_int64_to_int32(features):
  f = {}
  for k, v in six.iteritems(features):
    if v.dtype == tf.int64:
      v = tf.to_int32(v)
    f[k] = v
  return f


def feature_placeholders(data_fields):
  feature_map = {}
  for (field, tp) in data_fields:
    if not field.startswith("targets"):
      feature_map[field] = tf.placeholder(
          dtype=tp, shape=[None] * 4, name=field)
  return feature_map


def default_example_reading_spec(data_file_pattern):
  """Example reading spec for problem_hparams problems."""
  # This function is for problems that have yet to be ported to the new Problem
  # API. Do not add here.
  data_items_to_decoders = None
  # Read from image TFRecords if the file has "image" in its name.
  if data_file_pattern and "image" in data_file_pattern:
    label_key = "image/class/label"
    data_fields = {
        "image/encoded": tf.FixedLenFeature((), tf.string),
        "image/format": tf.FixedLenFeature((), tf.string),
        label_key: tf.VarLenFeature(tf.int64)
    }
    data_items_to_decoders = {
        "inputs":
            tf.contrib.slim.tfexample_decoder.Image(
                image_key="image/encoded",
                format_key="image/format",
                channels=1 if "mnist" in data_file_pattern else 3),
        "targets":
            tf.contrib.slim.tfexample_decoder.Tensor(label_key),
    }
  elif data_file_pattern and "audio" in data_file_pattern:
    data_type = tf.int64 if "timit" in data_file_pattern else tf.float32
    data_fields = {
        "inputs": tf.VarLenFeature(data_type),
        "audio/sample_count": tf.FixedLenFeature((), tf.int64),
        "audio/sample_width": tf.FixedLenFeature((), tf.int64),
        "targets": tf.VarLenFeature(tf.int64),
    }
  else:
    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets": tf.VarLenFeature(tf.int64)
    }
  return data_fields, data_items_to_decoders


def read_examples(problem,
                  data_file_pattern,
                  capacity,
                  mode=tf.contrib.learn.ModeKeys.TRAIN):
  """Create Dataset of Example for problem and data_file_pattern."""
  if problem is None:
    data_fields, data_items_to_decoders = default_example_reading_spec(
        data_file_pattern)
  else:
    data_fields, data_items_to_decoders = problem.example_reading_spec()

  if data_file_pattern is None:
    # Create placeholders for input, rather than reading data from disk.
    return feature_placeholders(data_fields)

  is_training = mode == tf.contrib.learn.ModeKeys.TRAIN
  dataset = examples_reader(
      [data_file_pattern],
      data_fields,
      training=is_training,
      capacity=capacity,
      data_items_to_decoders=data_items_to_decoders)
  return dataset


def input_pipeline(problem, data_file_pattern, capacity, mode, hparams,
                   batching_scheme):
  """Input pipeline, returns a dictionary of batched and padded tensors.

  Args:
    problem: Problem instance for which to build the input pipeline.
    data_file_pattern: file pattern for input files.
    capacity: int, data pipeline buffer capacity.
    mode: tf.contrib.learn.ModeKeys entry.
    hparams: an HParams object.
    batching_scheme: a dictionary containing
      "boundaries": a list of integers for the boundaries that will be
        used for bucketing; see bucket_by_sequence_length for more details.
      "batch_sizes": a list of batch sizes corresponding to the buckets
      "max_length": an integer.  We drop sequences which are longer.

  Returns:
    dict <feature name, batched and padded Tensor>
  """
  is_training = mode == tf.contrib.learn.ModeKeys.TRAIN
  num_threads = 4 if is_training else 1

  with tf.name_scope("input_pipeline"):
    dataset = read_examples(problem, data_file_pattern, capacity, mode=mode)
    dataset = dataset.map(
        lambda ex: _preprocess(ex, problem, data_file_pattern, hparams, mode),
        num_threads=num_threads)
    dataset = dataset.filter(
        lambda ex: _example_too_big(ex, batching_scheme["max_length"]))

    dataset = bucket_by_sequence_length(
        dataset, _example_length, batching_scheme["boundaries"],
        batching_scheme["batch_sizes"], batching_scheme["window_size"])
    # We reshuffle the batches to prevent many long-sequence batches at once.
    # TODO(rsepassi): Rm hasattr call once new dynamic window size functionality
    # is in a stable TF release.
    if (batching_scheme["shuffle_queue_size"] is not None and
        not hasattr(dataset, "apply")):
      dataset = dataset.shuffle(batching_scheme["shuffle_queue_size"])
    batched_examples = dataset.make_one_shot_iterator().get_next()
    return batched_examples


def _preprocess(example, problem, data_file_pattern, hparams, mode):
  """Preprocessing for example."""
  if problem is None:
    example = preprocess_examples_common(example, hparams)
    example = preprocessing(example, data_file_pattern)
  else:
    example = problem.preprocess_examples(example, mode, hparams)

  # We do not want int64s as they are not supported on GPUs.
  example = cast_int64_to_int32(example)

  return example


def _example_length(example):
  length = 0
  # Length of the example is the maximum length of the feature lengths
  for v in example.values():
    # For images the sequence length is the size of the spatial dimensions.
    feature_length = (tf.shape(v)[0] if len(v.get_shape()) < 3 else
                      tf.shape(v)[0] * tf.shape(v)[1])
    length = tf.maximum(length, feature_length)
  return length


def _example_too_big(example, max_length):
  return tf.less_equal(_example_length(example), max_length)


def bucket_by_sequence_length(dataset, example_length_fn, bucket_boundaries,
                              bucket_batch_sizes, window_size):
  """Bucket entries in dataset by length.

  Args:
    dataset: Dataset of dict<feature name, Tensor>.
    example_length_fn: function from example to int, determines the length of
      the example, which will determine the bucket it goes into.
    bucket_boundaries: list<int>, boundaries of the buckets.
    bucket_batch_sizes: list<int>, batch size per bucket.
    window_size: an integer divisible by all elements of bucket_batch_sizes

  Returns:
    Dataset of padded and batched examples.
  """
  with tf.name_scope("bucket_by_seq_length"):

    def example_to_bucket_id(example):
      """Return int64 id of the length bucket for this example."""
      seq_length = example_length_fn(example)

      boundaries = list(bucket_boundaries)
      buckets_min = [np.iinfo(np.int32).min] + boundaries
      buckets_max = boundaries + [np.iinfo(np.int32).max]
      conditions_c = tf.logical_and(
          tf.less_equal(buckets_min, seq_length),
          tf.less(seq_length, buckets_max))
      bucket_id = tf.reduce_min(tf.where(conditions_c))

      return bucket_id

    def window_size_fn(bucket_id):
      # window size = batch size
      batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)
      window_size = batch_sizes[bucket_id]
      return window_size

    def batching_fn(bucket_id, grouped_dataset):
      batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)
      batch_size = batch_sizes[bucket_id]

      # Pad each dimension of each feature so that they match.
      padded_shapes = dict(
          [(name, [None] * len(shape))
           for name, shape in grouped_dataset.output_shapes.items()])
      return grouped_dataset.padded_batch(batch_size, padded_shapes)

    # TODO(rsepassi): Rm branch once the new group_by_window functionality is in
    # a stable TF release.
    if hasattr(dataset, "apply"):
      # If the Dataset supports dynamic window size, use it.
      dataset = dataset.apply(
          tf.contrib.data.group_by_window,
          args=(example_to_bucket_id, batching_fn, None, window_size_fn))
    else:
      dataset = dataset.group_by_window(example_to_bucket_id, batching_fn,
                                        window_size)
    return dataset


def _bucket_boundaries(max_length, min_length=8, length_bucket_step=1.1):
  """A default set of length-bucket boundaries."""
  assert min_length <= max_length
  assert length_bucket_step > 1.0
  x = min_length
  boundaries = []
  while x < max_length:
    boundaries.append(x)
    x = max(x + 1, int(x * length_bucket_step))
  return boundaries


def _batching_scheme(batch_size,
                     max_length,
                     min_length_bucket,
                     length_bucket_step,
                     drop_long_sequences=False,
                     shard_multiplier=1,
                     length_multiplier=1):
  """A batching scheme based on model hyperparameters.

  Every batch containins a number of sequences divisible by `shard_multiplier`.

  Args:
    batch_size: int, total number of tokens in a batch.
    max_length: int, sequences longer than this will be skipped. Defaults to
      batch_size.
    min_length_bucket: int
    length_bucket_step: float greater than 1.0
    drop_long_sequences: bool, if True, then sequences longer than
      `max_length` are dropped.  This prevents generating batches with
      more than the usual number of tokens, which can cause out-of-memory
      errors.
    shard_multiplier: an integer increasing the batch_size to suit splitting
      across datashards.
    length_multiplier: an integer multiplier that is used to increase the
      batch sizes and sequence length tolerance.

  Returns:
     A dictionary with parameters that can be passed to input_pipeline:
       * boundaries: list of bucket boundaries
       * batch_sizes: list of batch sizes for each length bucket
       * max_length: int, maximum length of an example
  """
  max_length = max_length or batch_size
  boundaries = _bucket_boundaries(max_length, min_length_bucket,
                                  length_bucket_step)
  boundaries = [boundary * length_multiplier for boundary in boundaries]
  max_length *= length_multiplier
  batch_sizes = [
      max(1, batch_size // length) for length in boundaries + [max_length]
  ]
  max_batch_size = max(batch_sizes)
  # Since the Datasets API only allows a single constant for window_size,
  # and it needs divide all bucket_batch_sizes, we pick a highly-compoisite
  # window size and then round down all batch sizes to divisors of that window
  # size, so that a window can always be divided evenly into batches.
  # TODO(noam): remove this when Dataset API improves.
  highly_composite_numbers = [
      1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680,
      2520, 5040, 7560, 10080, 15120, 20160, 25200, 27720, 45360, 50400, 55440,
      83160, 110880, 166320, 221760, 277200, 332640, 498960, 554400, 665280,
      720720, 1081080, 1441440, 2162160, 2882880, 3603600, 4324320, 6486480,
      7207200, 8648640, 10810800, 14414400, 17297280, 21621600, 32432400,
      36756720, 43243200, 61261200, 73513440, 110270160
  ]
  window_size = max(
      [i for i in highly_composite_numbers if i <= 3 * max_batch_size])
  divisors = [i for i in xrange(1, window_size + 1) if window_size % i == 0]
  batch_sizes = [max([d for d in divisors if d <= bs]) for bs in batch_sizes]
  window_size *= shard_multiplier
  batch_sizes = [bs * shard_multiplier for bs in batch_sizes]
  # The Datasets API splits one window into multiple batches, which
  # produces runs of many consecutive batches of the same size.  This
  # is bad for training.  To solve this, we will shuffle the batches
  # using a queue which must be several times as large as the maximum
  # number of batches per window.
  max_batches_per_window = window_size // min(batch_sizes)
  shuffle_queue_size = max_batches_per_window * 3
  ret = {
      "boundaries": boundaries,
      "batch_sizes": batch_sizes,
      "max_length": (max_length if drop_long_sequences else 10**9),
      "shuffle_queue_size": shuffle_queue_size,
      "window_size": window_size,
  }
  tf.logging.info("batching_scheme = %s" % ret)
  return ret


def hparams_to_batching_scheme(hparams,
                               drop_long_sequences=False,
                               shard_multiplier=1,
                               length_multiplier=1):
  """Wrapper around _batching_scheme with hparams."""
  return _batching_scheme(
      batch_size=hparams.batch_size,
      max_length=hparams.max_length,
      min_length_bucket=hparams.min_length_bucket,
      length_bucket_step=hparams.length_bucket_step,
      drop_long_sequences=drop_long_sequences,
      shard_multiplier=shard_multiplier,
      length_multiplier=length_multiplier)


def constant_batching_scheme(constant_batch_size_in_sequences):
  """A batching scheme with constant batch size.

  Args:
    constant_batch_size_in_sequences: an integer

  Returns:
     a dictionary
  """
  boundaries = _bucket_boundaries(1024)
  batch_sizes = [constant_batch_size_in_sequences] * (1 + len(boundaries))
  return {
      "boundaries": boundaries,
      "batch_sizes": batch_sizes,
      "max_length": 10**9,
      "shuffle_queue_size": None,
      "window_size": constant_batch_size_in_sequences,
  }


def get_data_filepatterns(problems, data_dir, mode):
  """Return the location of a dataset for a given mode."""
  datasets = []
  for problem in problems.split("-"):
    try:
      problem = registry.problem(problem).dataset_filename()
    except ValueError:
      problem, _, _ = problem_hparams.parse_problem_name(problem)
    path = os.path.join(data_dir, problem)
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      datasets.append("%s-train*" % path)
    else:
      datasets.append("%s-dev*" % path)
  return datasets
