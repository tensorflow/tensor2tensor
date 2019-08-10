# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

import functools
import multiprocessing
import random

import six
from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.utils import mlperf_log

import tensorflow as tf


def cast_ints_to_int32(features):
  f = {}
  for k, v in sorted(six.iteritems(features)):
    if v.dtype in [tf.int64, tf.uint8]:
      v = tf.to_int32(v)
    f[k] = v
  return f


def example_length(example):
  length = 0
  # Length of the example is the maximum length of the feature lengths
  for _, v in sorted(six.iteritems(example)):
    # For images the sequence length is the size of the spatial dimensions.
    feature_length = tf.shape(v)[0]
    if len(v.get_shape()) > 2:
      feature_length = tf.shape(v)[0] * tf.shape(v)[1]
    length = tf.maximum(length, feature_length)
  return length


def example_valid_size(example, min_length, max_length):
  length = example_length(example)
  return tf.logical_and(
      length >= min_length,
      length <= max_length,
  )


def padded_batch(dataset, batch_size, padded_shapes=None):
  padded_shapes = padded_shapes or dict(
      [(name, [None] * len(shape))
       for name, shape in dataset.output_shapes.items()])
  return dataset.padded_batch(batch_size, padded_shapes)


def _bucket_boundaries(max_length, min_length=8, length_bucket_step=1.1):
  """A default set of length-bucket boundaries."""
  assert length_bucket_step > 1.0
  x = min_length
  boundaries = []
  while x < max_length:
    boundaries.append(x)
    x = max(x + 1, int(x * length_bucket_step))
  return boundaries


def batching_scheme(batch_size,
                    max_length,
                    min_length_bucket,
                    length_bucket_step,
                    drop_long_sequences=False,
                    shard_multiplier=1,
                    length_multiplier=1,
                    min_length=0):
  """A batching scheme based on model hyperparameters.

  Every batch contains a number of sequences divisible by `shard_multiplier`.

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
    min_length: int, sequences shorter than this will be skipped.

  Returns:
     A dictionary with parameters that can be passed to input_pipeline:
       * boundaries: list of bucket boundaries
       * batch_sizes: list of batch sizes for each length bucket
       * max_length: int, maximum length of an example

  Raises:
    ValueError: If min_length > max_length
  """
  max_length = max_length or batch_size
  if max_length < min_length:
    raise ValueError("max_length must be greater or equal to min_length")

  boundaries = _bucket_boundaries(max_length, min_length_bucket,
                                  length_bucket_step)
  boundaries = [boundary * length_multiplier for boundary in boundaries]
  max_length *= length_multiplier

  batch_sizes = [
      max(1, batch_size // length) for length in boundaries + [max_length]
  ]
  max_batch_size = max(batch_sizes)
  # Since the Datasets API only allows a single constant for window_size,
  # and it needs divide all bucket_batch_sizes, we pick a highly-composite
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
  divisors = [i for i in range(1, window_size + 1) if window_size % i == 0]
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
      "min_length": min_length,
      "max_length": (max_length if drop_long_sequences else 10**9),
      "shuffle_queue_size": shuffle_queue_size,
  }
  return ret


def hparams_to_batching_scheme(hparams,
                               drop_long_sequences=False,
                               shard_multiplier=1,
                               length_multiplier=1):
  """Wrapper around _batching_scheme with hparams."""
  return batching_scheme(
      batch_size=hparams.batch_size,
      min_length=hparams.min_length,
      max_length=hparams.max_length,
      min_length_bucket=hparams.min_length_bucket,
      length_bucket_step=hparams.length_bucket_step,
      drop_long_sequences=drop_long_sequences,
      shard_multiplier=shard_multiplier,
      length_multiplier=length_multiplier)


class DummyQueueRunner(object):
  """Can stand-in for a QueueRunner but does nothing."""

  def __init__(self):
    pass

  def create_threads(self, sess, coord=None, daemon=False, start=False):
    del sess, coord, daemon, start
    return []


def pad_for_tpu(shapes_dict, hparams, max_length):
  """Pads unknown features' dimensions for TPU."""
  padded_shapes = {}

  def get_filler(specified_max_length):
    if not specified_max_length:
      return max_length
    return min(specified_max_length, max_length)

  inputs_none_filler = get_filler(hparams.max_input_seq_length)
  targets_none_filler = get_filler(hparams.max_target_seq_length)

  def pad_one_shape(shape, none_filler):
    return [
        (dim if dim is not None else none_filler) for dim in shape.as_list()
    ]

  for key, shape in six.iteritems(shapes_dict):
    if key == "inputs":
      padded_shapes[key] = pad_one_shape(shape, inputs_none_filler)
    elif key == "targets":
      padded_shapes[key] = pad_one_shape(shape, targets_none_filler)
    else:
      padded_shapes[key] = pad_one_shape(shape, max_length)
  return padded_shapes


def cpu_count():
  """Return the number of available cores."""
  num_available_cores = multiprocessing.cpu_count()
  return num_available_cores


def _summarize_features(features, num_shards=1):
  with tf.name_scope("input_stats"):
    for (k, v) in six.iteritems(features):
      if isinstance(v, tf.Tensor) and v.get_shape().ndims > 1:
        tf.summary.scalar("%s_batch" % k, tf.shape(v)[0] // num_shards)
        tf.summary.scalar("%s_length" % k, tf.shape(v)[1])
        nonpadding = tf.to_float(tf.not_equal(v, 0))
        nonpadding_tokens = tf.reduce_sum(nonpadding)
        tf.summary.scalar("%s_nonpadding_tokens" % k, nonpadding_tokens)
        tf.summary.scalar("%s_nonpadding_fraction" % k,
                          tf.reduce_mean(nonpadding))


def standardize_shapes(features, batch_size=None):
  """Set the right shapes for the features."""
  for fname in ["inputs", "targets"]:
    if fname not in features:
      continue
    f = features[fname]
    while len(f.get_shape()) < 4:
      f = tf.expand_dims(f, axis=-1)
    features[fname] = f

  if batch_size:
    # Ensure batch size is set on all features
    for _, t in six.iteritems(features):
      shape = t.get_shape().as_list()
      shape[0] = batch_size
      t.set_shape(t.get_shape().merge_with(shape))
      # Assert shapes are fully known
      t.get_shape().assert_is_fully_defined()

  return features


def _are_shapes_fully_defined(shapes_dict):
  for shape in shapes_dict.values():
    if not shape.is_fully_defined():
      return False
  return True


def _file_num_records_cached(filename):
  """Return the number of TFRecords in a file."""
  # Cache the result, as this is expensive to compute
  if filename in _file_num_records_cache:
    return _file_num_records_cache[filename]
  ret = 0
  for _ in tf.python_io.tf_record_iterator(filename):
    ret += 1
  _file_num_records_cache[filename] = ret
  return ret


_file_num_records_cache = {}


def skip_random_fraction(dataset, data_file):
  # Skip a random fraction at the beginning of the stream.  The skip is
  # essential for synchronous highly-parallel training to avoid multiple
  # replicas reading the same data in lock-step.
  num_skip = random.randint(0, _file_num_records_cached(data_file))
  return dataset.skip(num_skip)


def pad_batch(features, batch_multiple):
  """Pad batch dim of features to nearest multiple of batch_multiple."""
  feature = list(features.items())[0][1]
  batch_size = tf.shape(feature)[0]
  mod = batch_size % batch_multiple
  has_mod = tf.cast(tf.cast(mod, tf.bool), tf.int32)
  batch_padding = batch_multiple * has_mod - mod

  padded_features = {}
  for k, feature in features.items():
    rank = len(feature.shape)
    paddings = [[0, 0] for _ in range(rank)]
    paddings[0][1] = batch_padding
    padded_feature = tf.pad(feature, paddings)
    padded_features[k] = padded_feature
  return padded_features


# TODO(lukaszkaiser): refactor the API to not be just a list of self params
#   but make sense for other uses too.
def input_fn(dataset,
             filepattern,
             skip_random_fraction_when_training,
             batch_size_means_tokens_param,
             batch_size_multiplier,
             max_length,
             mode,
             hparams,
             data_dir=None,
             params=None,
             config=None,
             force_repeat=False,
             prevent_repeat=False):
  """Builds input pipeline for problem.

  Args:
    dataset: the dataset to make input function from.
    filepattern: the pattern of files to read from.
    skip_random_fraction_when_training: whether to skip randomly when training.
    batch_size_means_tokens_param: whether batch size should mean tokens.
    batch_size_multiplier: how to multiply batch size when bucketing.
    max_length: maximum length,
    mode: tf.estimator.ModeKeys
    hparams: HParams, model hparams
    data_dir: str, data directory; if None, will use hparams.data_dir
    params: dict, may include "batch_size"
    config: RunConfig; should have the data_parallelism attribute if not using
      TPU
    force_repeat: bool, whether to repeat the data even if not training
    prevent_repeat: bool, whether to not repeat when in training mode.
      Overrides force_repeat.

  Returns:
    (features_dict<str name, Tensor feature>, Tensor targets)
  """
  is_training = mode == tf.estimator.ModeKeys.TRAIN
  if config and config.use_tpu:
    num_threads = 64
  else:
    num_threads = cpu_count() if is_training else 1

  if config and hasattr(config,
                        "data_parallelism") and config.data_parallelism:
    num_shards = config.data_parallelism.n
  else:
    num_shards = 1

  mlperf_log.transformer_print(
      key=mlperf_log.INPUT_MAX_LENGTH, value=max_length)

  def tpu_valid_size(example):
    return example_valid_size(example, hparams.min_length, max_length)

  def gpu_valid_size(example):
    drop_long_sequences = is_training or hparams.eval_drop_long_sequences
    max_validate_length = max_length if drop_long_sequences else 10**9
    return example_valid_size(example, hparams.min_length, max_validate_length)

  def define_shapes(example):
    batch_size = config and config.use_tpu and params["batch_size"]
    return standardize_shapes(example, batch_size=batch_size)

  # Read and preprocess
  data_dir = data_dir or (hasattr(hparams, "data_dir") and hparams.data_dir)

  if (force_repeat or is_training) and not prevent_repeat:
    # Repeat and skip a random number of records
    dataset = dataset.repeat()

  if is_training and skip_random_fraction_when_training:
    data_files = tf.contrib.slim.parallel_reader.get_data_files(filepattern)
    #  In continuous_train_and_eval when switching between train and
    #  eval, this input_fn method gets called multiple times and it
    #  would give you the exact same samples from the last call
    #  (because the Graph seed is set). So this skip gives you some
    #  shuffling.
    dataset = skip_random_fraction(dataset, data_files[0])

  dataset = dataset.map(cast_ints_to_int32, num_parallel_calls=num_threads)

  if batch_size_means_tokens_param:
    batch_size_means_tokens = True
  else:
    if _are_shapes_fully_defined(dataset.output_shapes):
      batch_size_means_tokens = False
    else:
      tf.logging.warning(
          "Shapes are not fully defined. Assuming batch_size means tokens.")
      batch_size_means_tokens = True

  # Batching
  if not batch_size_means_tokens:
    # Batch size means examples per datashard.
    if config and config.use_tpu:
      # on TPU, we use params["batch_size"], which specifies the number of
      # examples across all datashards
      batch_size = params["batch_size"]
      dataset = dataset.batch(batch_size, drop_remainder=True)
    else:
      batch_size = hparams.batch_size * num_shards
      dataset = dataset.batch(batch_size)
  else:
    # batch_size means tokens per datashard
    if config and config.use_tpu:
      dataset = dataset.filter(tpu_valid_size)
      padded_shapes = pad_for_tpu(dataset.output_shapes, hparams, max_length)
      # on TPU, we use params["batch_size"], which specifies the number of
      # examples across all datashards
      batch_size = params["batch_size"]
      if hparams.pad_batch:
        tf.logging.warn(
            "Padding the batch to ensure that remainder eval batches are "
            "processed. This may lead to incorrect metrics for "
            "non-zero-padded features, e.g. images. Use a smaller batch "
            "size that has no remainder in that case.")
        dataset = dataset.padded_batch(
            batch_size, padded_shapes, drop_remainder=False)
        dataset = dataset.map(
            functools.partial(pad_batch, batch_multiple=batch_size),
            num_parallel_calls=num_threads)
      else:
        dataset = dataset.padded_batch(
            batch_size, padded_shapes, drop_remainder=True)
    else:
      # On GPU, bucket by length
      dataset = dataset.filter(gpu_valid_size)
      cur_batching_scheme = hparams_to_batching_scheme(
          hparams,
          shard_multiplier=num_shards,
          length_multiplier=batch_size_multiplier)
      if hparams.use_fixed_batch_size:
        # Here  batch_size really means examples per datashard.
        cur_batching_scheme["batch_sizes"] = [hparams.batch_size]
        cur_batching_scheme["boundaries"] = []
      dataset = dataset.apply(
          tf.data.experimental.bucket_by_sequence_length(
              example_length, cur_batching_scheme["boundaries"],
              cur_batching_scheme["batch_sizes"]))

      if not is_training:
        batch_multiple = num_shards
        if hparams.use_fixed_batch_size:
          # Make sure the last batch has the same fixed size as the rest.
          batch_multiple *= hparams.batch_size
        if batch_multiple > 1:
          tf.logging.warn(
              "Padding the batch to ensure that remainder eval batches have "
              "a batch size divisible by the number of data shards. This may "
              "lead to incorrect metrics for non-zero-padded features, e.g. "
              "images. Use a single datashard (i.e. 1 GPU) in that case.")
          dataset = dataset.map(
              functools.partial(pad_batch, batch_multiple=batch_multiple),
              num_parallel_calls=num_threads)

  dataset = dataset.map(define_shapes, num_parallel_calls=num_threads)

  # Add shuffling for training batches. This is necessary along with record
  # level shuffling in the dataset generation. Record shuffling will shuffle
  # the examples. However, in some cases, it's possible that the shuffle
  # buffer size for record shuffling is smaller than the batch size. In such
  # cases, adding batch shuffling ensures that the data is in random order
  # during training
  if (is_training and hasattr(hparams, "batch_shuffle_size") and
      hparams.batch_shuffle_size):
    dataset = dataset.shuffle(hparams.batch_shuffle_size)

  # Split batches into chunks if targets are too long.
  # The new "chunk_number" feature is 0 for the first chunk and goes up then.
  # Chunks are reversed so the 0th chunk comes first, then the 1st and so on,
  # so models can attend to them in the order they arrive. The last chunk is
  # usually the one containing the end of the target sentence (EOS).
  chunk_length = hparams.get("split_targets_chunk_length", 0)
  max_chunks = hparams.get("split_targets_max_chunks", 100)
  if chunk_length > 0:
    def is_nonzero_chunk(example):
      """A chunk is zero if all targets are 0s."""
      return tf.less(0, tf.reduce_sum(tf.abs(example["targets"])))

    def split_on_length(example):
      """Split a batch of ditcs on length."""
      x = example["targets"]
      # TODO(kitaev): This code breaks if chunk_length * max_chunks < batch_size
      length_diff = chunk_length * max_chunks - tf.shape(x)[1]
      padded_x = tf.pad(x, [(0, 0), (0, length_diff), (0, 0), (0, 0)])
      chunks = [padded_x[:, i*chunk_length:(i+1)*chunk_length, :, :]
                for i in range(max_chunks - 1)]
      chunks.append(padded_x[:, (max_chunks - 1)*chunk_length:, :, :])
      new_example = {}
      # Setting chunk_number to be tf.range(max_chunks) is incompatible with TPU
      new_example["chunk_number"] = tf.concat([
          tf.expand_dims(tf.ones_like(c) * n, axis=0)
          for n, c in enumerate(chunks)
      ],
                                              axis=0)
      new_example["targets"] = tf.concat(
          [tf.expand_dims(c, axis=0) for c in chunks], axis=0)
      for k in example:
        if k != "targets":
          assert k != "chunk_number", (
              "Chunking code expects the chunk_number feature name to be "
              "available"
          )
          new_example[k] = tf.concat(
              [tf.expand_dims(example[k], axis=0) for _ in range(max_chunks)],
              axis=0)
      return tf.data.Dataset.from_tensor_slices(new_example)

    dataset = dataset.flat_map(split_on_length)
    dataset = dataset.filter(is_nonzero_chunk)

    # The chunking data pipeline thus far creates batches of examples where all
    # of the examples have the same chunk number. This can lead to periodic
    # fluctuations in the loss; for example, when all examples in the batch have
    # chunk number 0 the loss may be higher than midway through a sequence.
    # Enabling split_targets_strided_training adjusts the data so that each
    # batch includes examples at various points within a sequence.
    if is_training and hparams.split_targets_strided_training:
      # TODO(kitaev): make sure that shape inference works on GPU, not just TPU.
      inferred_batch_size = dataset.output_shapes["targets"].as_list()[0]
      if inferred_batch_size is None:
        raise ValueError(
            "Strided training is only implemented when the batch size can be "
            "inferred statically, for example when training on TPU."
        )
      chunk_stride = inferred_batch_size * max(
          1, max_chunks // inferred_batch_size) + 1

      def collapse_nested_datasets(example):
        """Converts a dataset of datasets to a dataset of tensor features."""
        new_example = {}
        for k, v in example.items():
          v = tf.data.experimental.get_single_element(
              v.batch(inferred_batch_size, drop_remainder=True))
          new_example[k] = v
        return tf.data.Dataset.from_tensor_slices(new_example)

      dataset = dataset.apply(tf.data.experimental.unbatch())
      dataset = dataset.window(inferred_batch_size, inferred_batch_size,
                               chunk_stride)
      dataset = dataset.flat_map(collapse_nested_datasets)
      dataset = dataset.batch(inferred_batch_size, drop_remainder=True)

  def prepare_for_output(example):
    if not config or not config.use_tpu:
      _summarize_features(example, num_shards)
    if mode == tf.estimator.ModeKeys.PREDICT:
      example["infer_targets"] = example.pop("targets")
      return example
    else:
      return example, example[hparams.get(
          key="labels_feature_name", default="targets")]

  dataset = dataset.map(prepare_for_output, num_parallel_calls=num_threads)
  dataset = dataset.prefetch(2)

  if mode == tf.estimator.ModeKeys.PREDICT:
    # This is because of a bug in the Estimator that short-circuits prediction
    # if it doesn't see a QueueRunner. DummyQueueRunner implements the
    # minimal expected interface but does nothing.
    tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, DummyQueueRunner())

  return dataset
