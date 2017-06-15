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

"""Data reader module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

# Dependency imports

import six
from six.moves import zip  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.models import common_layers

import tensorflow as tf


def examples_queue(data_sources,
                   data_fields_to_features,
                   training,
                   capacity=32,
                   data_items_to_decoders=None,
                   data_items_to_decode=None):
  """Contruct a queue of training or evaluation examples.

  This function will create a reader from files given by data_sources,
  then enqueue the tf.Examples from these files, shuffling if training
  is true, and finally parse these tf.Examples to tensors.

  The dictionary data_fields_to_features for an image dataset can be this:

  data_fields_to_features = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
    'image/class/label': tf.FixedLenFeature(
        [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
  }

  and for a simple algorithmic dataset with variable-length data it is this:

  data_fields_to_features = {
    'inputs': tf.VarLenFeature(tf.int64),
    'targets': tf.VarLenFeature(tf.int64),
  }

  The data_items_to_decoders dictionary argument can be left as None if there
  is no decoding to be performed. But, e.g. for images, it should be set so that
  the images are decoded from the features, e.g., like this for MNIST:

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
    capacity: integer, queue capacity; set to 2 * max_batch_size or more.
    data_items_to_decoders: a dictionary mapping data items (that will be
      in the returned result) to decoders that will decode them using features
      defined in data_fields_to_features; see above for examples. By default
      (if this is None), we grab the tensor from every feature.
    data_items_to_decode: a subset of data items that will be decoded;
      by default (if this is None), we decode all items.

  Returns:
    A dictionary mapping each data_field to a corresponding 1D int64 tensor
    read from the created queue.

  Raises:
    ValueError: if no files are found with the provided data_prefix or no data
      fields were provided.
  """
  with tf.name_scope("examples_queue"):
    # Read serialized examples using slim parallel_reader.
    num_epochs = None if training else 1
    _, example_serialized = tf.contrib.slim.parallel_reader.parallel_read(
        data_sources,
        tf.TFRecordReader,
        num_epochs=num_epochs,
        shuffle=training,
        capacity=2 * capacity,
        min_after_dequeue=capacity,
        num_readers=4 if training else 1)

    if data_items_to_decoders is None:
      data_items_to_decoders = {
          field: tf.contrib.slim.tfexample_decoder.Tensor(field)
          for field in data_fields_to_features
      }

    decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
        data_fields_to_features, data_items_to_decoders)

    if data_items_to_decode is None:
      data_items_to_decode = list(data_items_to_decoders)

    decoded = decoder.decode(example_serialized, items=data_items_to_decode)
    return {
        field: tensor
        for (field, tensor) in zip(data_items_to_decode, decoded)
    }


def input_pipeline(data_file_pattern, capacity, mode):
  """Input pipeline, returns a dictionary of tensors from queues."""
  # Read from image TFRecords if the file has "image" in its name.
  if data_file_pattern and "image" in data_file_pattern:
    data_fields = {
        "image/encoded": tf.FixedLenFeature((), tf.string),
        "image/format": tf.FixedLenFeature((), tf.string),
        "image/class/label": tf.VarLenFeature(tf.int64)
    }
    data_items_to_decoders = {
        "inputs":
            tf.contrib.slim.tfexample_decoder.Image(
                image_key="image/encoded",
                format_key="image/format",
                channels=1 if "mnist" in data_file_pattern else 3),
        "targets":
            tf.contrib.slim.tfexample_decoder.Tensor("image/class/label"),
    }
  elif data_file_pattern and "audio" in data_file_pattern:
    data_type = tf.int64 if "timit" in data_file_pattern else tf.float32
    data_fields = {
        "inputs": tf.VarLenFeature(data_type),
        "audio/sample_count": tf.FixedLenFeature((), tf.int64),
        "audio/sample_width": tf.FixedLenFeature((), tf.int64),
        "targets": tf.VarLenFeature(tf.int64),
    }
    data_items_to_decoders = None
  else:
    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets": tf.VarLenFeature(tf.int64)
    }
    data_items_to_decoders = None

  # Create placeholders for input, rather than reading data from disk.
  if data_file_pattern is None:
    feature_map = {}
    for (field, tp) in data_fields:
      if field != "targets":
        feature_map[field] = tf.placeholder(
            dtype=tp, shape=[None] * 4, name=field)
    return feature_map

  # Now the non-trivial case construction.
  examples = examples_queue(
      [data_file_pattern],
      data_fields,
      training=(mode == tf.contrib.learn.ModeKeys.TRAIN),
      capacity=capacity,
      data_items_to_decoders=data_items_to_decoders)

  if "image" in data_file_pattern:
    # Small single-example pre-processing for images.
    examples["inputs"] = tf.cast(examples["inputs"], tf.int64)
    if ("image_imagenet" in data_file_pattern or
        "image_mscoco" in data_file_pattern):
      # For imagnet/coco, resize images to 299x299 as is standard.
      def resize(img):
        return tf.to_int64(tf.image.resize_images(img, [299, 299]))

      def preprocess(img):
        img = tf.image.resize_images(img, [360, 360])
        img = common_layers.image_augmentation(tf.to_float(img) / 255.)
        return tf.to_int64(img * 255.)

      inputs = examples["inputs"]
      if mode == tf.contrib.learn.ModeKeys.TRAIN:
        examples["inputs"] = tf.cond(  # Preprocess 80% of the time.
            tf.less(tf.random_uniform([]), 0.8),
            lambda img=inputs: preprocess(img),
            lambda img=inputs: resize(img))
      else:
        examples["inputs"] = tf.to_int64(resize(inputs))
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

  # We do not want int64s as they do are not supported on GPUs.
  return {k: tf.to_int32(v) for (k, v) in six.iteritems(examples)}


def batch_examples(examples, batching_scheme):
  """Given a queue of examples, create batches of examples with similar lengths.

  We assume that examples is a dictionary with string keys and tensor values,
  possibly coming from a queue, e.g., constructed by examples_queue above.
  Each tensor in examples is assumed to be 1D. We will put tensors of similar
  length into batches togeter. We return a dictionary with the same keys as
  examples, and with values being batches of size batch_size. If elements have
  different lengths, they are padded with 0s. This function is based on
  tf.contrib.training.bucket_by_sequence_length so see there for details.

  For example, if examples is a queue containing [1, 2, 3] and [4], then
  this function with batch_size=2 will return a batch [[1, 2, 3], [4, 0, 0]].

  Args:
    examples: a dictionary with string keys and 1D tensor values.
    batching_scheme: a dictionary containing
      "boundaries": a list of integers for the boundaries that will be
        used for bucketing; see tf.contrib.training.bucket_by_sequence_length
        for more details.
      "batch_sizes": a list of batch sizes corresponding to the buckets
      "max_length": an integer.  We drop sequences which are longer.

  Returns:
    A dictionary with the same keys as examples and with values being batches
    of examples padded with 0s, i.e., [batch_size x length] tensors.
  """
  with tf.name_scope("batch_examples"):
    # The queue to bucket on will be chosen based on maximum length.
    max_length = 0
    for v in examples.values():
      # For images the sequence length is the size of the spatial dimensions.
      sequence_length = (tf.shape(v)[0] if len(v.get_shape()) < 3 else
                         tf.shape(v)[0] * tf.shape(v)[1])
      max_length = tf.maximum(max_length, sequence_length)
    (_, outputs) = tf.contrib.training.bucket_by_sequence_length(
        max_length,
        examples,
        batching_scheme["batch_sizes"],
        [b + 1 for b in batching_scheme["boundaries"]],
        capacity=2,  # Number of full batches to store, we don't need many.
        bucket_capacities=[2 * b for b in batching_scheme["batch_sizes"]],
        dynamic_pad=True,
        keep_input=(max_length <= batching_scheme["max_length"]))
    return outputs


def bucket_boundaries(max_length, min_length=8, mantissa_bits=2):
  """A default set of length-bucket boundaries."""
  x = min_length
  boundaries = []
  while x < max_length:
    boundaries.append(x)
    x += 2**max(0, int(math.log(x, 2)) - mantissa_bits)
  return boundaries


def hparams_to_batching_scheme(hparams,
                               drop_long_sequences=False,
                               shard_multiplier=1,
                               length_multiplier=1):
  """A batching scheme based on model hyperparameters.

  Every batch containins a number of sequences divisible by `shard_multiplier`.

  If `drop_long_sequences` is True, then sequences longer than
  `hparams.batch_size` are dropped.  This prevents generating batches with
  more than the usual number of tokens, which can cause out-of-memory errors.

  Args:
    hparams: a hyperparameters.
    drop_long_sequences: a boolean.
    shard_multiplier: an integer increasing the batch_size to suit splitting
      across datashards.
    length_multiplier: an integer multiplier that is used to increase the
      batch sizes and sequence length tolerance.

  Returns:
     a dictionary
  """
  max_length = hparams.max_length or hparams.batch_size
  boundaries = bucket_boundaries(
      max_length, mantissa_bits=hparams.batching_mantissa_bits)
  batch_sizes = [
      max(1, hparams.batch_size // length)
      for length in boundaries + [max_length]
  ]
  batch_sizes = [b * shard_multiplier for b in batch_sizes]
  max_length *= length_multiplier
  boundaries = [boundary * length_multiplier for boundary in boundaries]
  return {
      "boundaries": boundaries,
      "batch_sizes": batch_sizes,
      "max_length": (max_length if drop_long_sequences else 10**9)
  }


def constant_batching_scheme(constant_batch_size_in_sequences):
  """A batching scheme with constant batch size.

  Args:
    constant_batch_size_in_sequences: an integer

  Returns:
     a dictionary
  """
  boundaries = bucket_boundaries(1024)
  batch_sizes = [constant_batch_size_in_sequences] * (1 + len(boundaries))
  return {
      "boundaries": boundaries,
      "batch_sizes": batch_sizes,
      "max_length": 10**9
  }


def get_datasets(problems, data_dir, mode):
  """Return the location of a dataset for a given mode."""
  datasets = []
  for problem in problems.split("-"):
    problem, _, _ = problem_hparams.parse_problem_name(problem)
    path = os.path.join(data_dir, problem)
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      datasets.append("%s-train*" % path)
    else:
      datasets.append("%s-dev*" % path)
  return datasets
