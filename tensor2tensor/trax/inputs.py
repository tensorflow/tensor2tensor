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

"""trax input pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import random

import gin

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

# Inputs is the trax tuple defining the input streams and shapes.
# * train_stream: training data that will be used for training
#     may include all the augmentation or selection the training wants
#     the shape of examples is [batch_fun.batch_size, ...]
# * train_eval_stream: training data used for evaluation
#     examples from training data but usually without augmentation
#     the shape of examples is [batch_fun.eval_batch_size, ...]
# * eval_stream: evaluation data stream
#     examples from evaluation data, usually without augmentation
#     the shape of examples is [batch_fun.eval_batch_size, ...]
# * input_shape: the shape of inputs
#     the [...] above, without batch size

Inputs = collections.namedtuple(
    "_Inputs",
    ["train_stream", "train_eval_stream", "eval_stream", "input_shape"])

# How many examples from the stream to skip at random during training.
# For now, we skip at most 100K examples for efficiency.
# TODO(lukaszkaiser): can we improve efficiency, should that be changed?
_MAX_SKIP_EXAMPLES = 1e5


@gin.configurable(blacklist=["num_devices"])
def inputs(num_devices, dataset_name, data_dir=None, input_name=None):
  """Make Inputs for built-in datasets.

  Args:
    num_devices: how many devices to build the inputs for.
    dataset_name: a TFDS or T2T dataset name. If it's a T2T dataset name, prefix
      with "t2t_".
    data_dir: data directory.
    input_name: optional, name of the inputs from the dictionary.

  Returns:
    trax.inputs.Inputs
  """
  assert data_dir, "Must provide a data directory"
  data_dir = os.path.expanduser(data_dir)

  (train_batches, train_eval_batches, eval_batches,
   input_name, input_shape) = _train_and_eval_batches(
       dataset_name, data_dir, input_name, num_devices)

  def train_input_fun():
    return dataset_to_stream(train_batches, input_name)

  def train_eval_input_fun():
    return dataset_to_stream(train_eval_batches, input_name)

  def eval_input_fun():
    return dataset_to_stream(eval_batches, input_name)

  return Inputs(train_stream=train_input_fun,
                train_eval_stream=train_eval_input_fun,
                eval_stream=eval_input_fun,
                input_shape=input_shape)


@gin.configurable(blacklist=["num_devices"])
def random_inputs(
    num_devices,
    input_shape=gin.REQUIRED, input_dtype=np.int32, input_range=(0, 255),
    output_shape=gin.REQUIRED, output_dtype=np.int32, output_range=(0, 9)):
  """Make random Inputs for debugging.

  Args:
    num_devices: how many devices to build the inputs for.
    input_shape: the shape of inputs (including batch dimension).
    input_dtype: the type of the inputs (int32 by default).
    input_range: the range of inputs (defaults to (0, 255)).
    output_shape: the shape of outputs (including batch dimension).
    output_dtype: the type of the outputs (int32 by default).
    output_range: the range of outputs (defaults to (0, 9)).

  Returns:
    trax.inputs.Inputs
  """
  if input_shape[0] % num_devices != 0:
    tf.logging.fatal(
        "num_devices[%d] should divide the first dimension of input_shape[%s]",
        num_devices, input_shape)
  if output_shape[0] % num_devices != 0:
    tf.logging.fatal(
        "num_devices[%d] should divide the first dimension of output_shape[%s]",
        num_devices, output_shape)

  def random_minibatches():
    """Generate a stream of random mini-batches."""
    if input_dtype in [np.float16, np.float32, np.float64]:
      rand = np.random.uniform
    else:
      rand = np.random.random_integers
    while True:
      inp = rand(input_range[0], input_range[1], input_shape)
      inp = inp.astype(input_dtype)
      out = rand(output_range[0], output_range[1], output_shape)
      out = out.astype(output_dtype)
      yield inp, out

  input_shape_without_batch = list(input_shape)[1:]
  return Inputs(train_stream=random_minibatches,
                train_eval_stream=random_minibatches,
                eval_stream=random_minibatches,
                input_shape=input_shape_without_batch)


def dataset_to_stream(dataset, input_name):
  """Takes a tf.Dataset and creates a numpy stream of ready batches."""
  for example in tfds.as_numpy(dataset):
    inp, out = example[0][input_name], example[1]
    if len(out.shape) > 1 and out.shape[-1] == 1:
      out = np.squeeze(out, axis=-1)
    yield inp, out


@gin.configurable(whitelist=["train_shuffle_files", "test_shuffle_files"])
def train_and_eval_dataset(dataset_name, data_dir, train_shuffle_files=True,
                           test_shuffle_files=False):
  """Return train and evaluation datasets, feature info and supervised keys.

  Args:
    dataset_name: a string, the name of the dataset; if it starts with "t2t_"
      then we'll search T2T Problem registry for it, otherwise we assume it
      is a dataset from TFDS and load it from there.
    data_dir: directory where the data is located.
    train_shuffle_files: Boolean determining whether or not to shuffle the train
      files at startup. Set to False if you want data determinism.
    test_shuffle_files: Boolean determining whether or not to shuffle the test
      files at startup. Set to False if you want data determinism.

  Returns:
    a 4-tuple consisting of:
     * the train tf.Daataset
     * the eval tf.Daataset
     * information about features: a python dictionary with feature names
         as keys and an object as value that provides .shape and .num_classes.
     * supervised_keys: information what's the input and what's the target,
         ie., a pair of lists with input and target feature names.
  """
  if dataset_name.startswith("t2t_"):
    return _train_and_eval_dataset_v1(dataset_name[4:], data_dir)
  dataset_builder = tfds.builder(dataset_name, data_dir=data_dir)
  info = dataset_builder.info
  splits = dataset_builder.info.splits
  if tfds.Split.TRAIN not in splits:
    raise ValueError("To train we require a train split in the dataset.")
  if tfds.Split.VALIDATION not in splits and "test" not in splits:
    raise ValueError("We require a validation or test split in the dataset.")
  eval_split = tfds.Split.VALIDATION
  if tfds.Split.VALIDATION not in splits:
    eval_split = tfds.Split.TEST
  train = tfds.load(
      name=dataset_name, split=tfds.Split.TRAIN,
      as_dataset_kwargs={"shuffle_files": train_shuffle_files})
  valid = tfds.load(
      name=dataset_name, split=eval_split,
      as_dataset_kwargs={"shuffle_files": test_shuffle_files})
  keys = None
  if info.supervised_keys:
    keys = ([info.supervised_keys[0]], [info.supervised_keys[1]])
  return train, valid, info.features, keys


def _make_info(shape_list, num_classes):
  """Create an info-like tuple for feature given some shapes and vocab size."""
  feature_info = collections.namedtuple("FeatureInfo", ["shape", "num_classes"])
  cur_shape = list(shape_list[0])
  # We need to merge the provided shapes, put None where they disagree.
  for shape in shape_list:
    if len(shape) != len(cur_shape):
      raise ValueError("Shapes need to have the same number of dimensions.")
    for i in range(len(shape)):
      if cur_shape[i] is not None:
        if shape[i] != cur_shape[i]:
          cur_shape[i] = None
  return feature_info(cur_shape, num_classes)


def _select_features(example, feature_list=None):
  """Select a subset of features from the example dict."""
  feature_list = feature_list or ["inputs", "targets"]
  return {f: example[f] for f in feature_list if f in example}


def _train_and_eval_dataset_v1(problem_name, data_dir):
  """Return train and evaluation datasets, feature info and supervised keys."""
  from tensor2tensor import problems  # pylint: disable=g-import-not-at-top
  assert not tf.executing_eagerly(), "tf.eager mode must be turned off."
  problem = problems.problem(problem_name)
  train_dataset = problem.dataset(tf.estimator.ModeKeys.TRAIN, data_dir)
  train_dataset = train_dataset.map(_select_features)
  eval_dataset = problem.dataset(tf.estimator.ModeKeys.EVAL, data_dir)
  eval_dataset = eval_dataset.map(_select_features)
  hparams = problem.get_hparams()
  # We take a few training examples to guess the shapes.
  input_shapes, target_shapes = [], []
  example_tensor = train_dataset.make_one_shot_iterator().get_next()
  sess = tf.Session()
  example1 = sess.run(example_tensor)
  example2 = sess.run(example_tensor)
  example3 = sess.run(example_tensor)
  # We use "inputs" as input except for purely auto-regressive tasks like
  # language models where "targets" are used as input_key.
  input_key = "inputs" if "inputs" in example1 else "targets"
  supervised_keys = ([input_key], ["targets"])
  for example in [example1, example2, example3]:
    input_shapes.append(list(example[input_key].shape))
    target_shapes.append(list(example["targets"].shape))
  input_vocab_size = hparams.vocab_size[input_key]
  target_vocab_size = hparams.vocab_size["targets"]
  input_info = _make_info(input_shapes, input_vocab_size)
  target_info = _make_info(target_shapes, target_vocab_size)
  info = {input_key: input_info, "targets": target_info}
  return train_dataset, eval_dataset, info, supervised_keys


@gin.configurable(blacklist=["dataset", "training", "shapes",
                             "target_names", "num_devices"])
def batch_fun(dataset, training, shapes, target_names, num_devices,
              batch_size_per_device=32, batch_size=None, eval_batch_size=32,
              bucket_length=32, buckets=None,
              batch_shuffle_size=128, max_eval_length=None):
  """Batching function."""
  del target_names
  # Batch size is batch_size_per_device * num_devices unless given directly.
  batch_size = batch_size or batch_size_per_device * num_devices
  # If bucketing is not specified, check if target shapes are variable.
  cur_batch_size = batch_size if training else eval_batch_size
  # Make cur_batch_size divisible by num_devices.
  cur_batch_size = max(cur_batch_size // num_devices, 1) * num_devices
  # Create heuristic buckets is none are specified.
  if buckets is None:
    variable_target_shapes = False
    target_shape = shapes[1]
    for dim in target_shape:
      if dim is None:
        variable_target_shapes = True
    tf.logging.info("Heuristically setting bucketing to %s based on shapes "
                    "of target tensors." % variable_target_shapes)
    if variable_target_shapes:
      bucket_boundaries = [bucket_length // 4, bucket_length // 2,
                           bucket_length, bucket_length * 2,
                           bucket_length * 4, bucket_length * 8,
                           bucket_length * 16]
      # We will pad to boundaries which pads to bucket_boundary - 1: add 1 here.
      bucket_boundaries = [b + 1 for b in bucket_boundaries]
      if not training:
        max_eval_length = max_eval_length or bucket_length * 32
        bucket_boundaries[-1] = max_eval_length
      bucket_batch_sizes = [cur_batch_size * 4, cur_batch_size * 2,
                            cur_batch_size, cur_batch_size // 2,
                            cur_batch_size // 4, cur_batch_size // 8,
                            cur_batch_size // 16, 1]
      if not training:
        bucket_batch_sizes[-2] = cur_batch_size // max_eval_length
      # Make batch sizes divisible by num_devices.
      bucket_batch_sizes = [max(b // num_devices, 1) * num_devices
                            for b in bucket_batch_sizes]
      buckets = (bucket_boundaries, bucket_batch_sizes)

  if buckets:
    tf.logging.info("Bucketing with buckets %s." % str(buckets))
    def example_length(_, target):
      return tf.shape(target)[0]
    boundaries, batch_sizes = buckets
    dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
        example_length, boundaries, batch_sizes,
        pad_to_bucket_boundary=True))
  else:
    dataset = dataset.padded_batch(cur_batch_size, shapes)
  if training:
    return dataset.shuffle(batch_shuffle_size)
  return dataset


# pylint: disable=unused-argument
@gin.configurable(blacklist=["dataset", "training"])
def cifar10_no_augmentation_preprocess(dataset, training):

  def cast_image(features, targets):
    features["image"] = tf.cast(features["image"], tf.float32) / 255.0
    return features, targets

  dataset = dataset.map(cast_image)
  return dataset


# pylint: disable=unused-argument
def no_preprocess(dataset, training):
  return dataset


@gin.configurable(blacklist=["dataset", "training"])
def lm1b_preprocess(dataset, training,
                    max_target_length=-1, max_eval_target_length=-1):
  """Preprocessing for LM1B: filter out targets exceeding maximum length."""

  def target_right_length(_, target):
    return tf.less(tf.shape(target)[0], max_target_length + 1)

  def eval_target_right_length(_, target):
    return tf.less(tf.shape(target)[0], max_eval_target_length + 1)

  if max_target_length > 0 and training:
    dataset = dataset.filter(target_right_length)

  if max_eval_target_length > 0 and not training:
    dataset = dataset.filter(eval_target_right_length)

  return dataset


@gin.configurable(whitelist=["preprocess_fun", "shuffle_buffer_size"])
def shuffle_and_batch_data(dataset,
                           target_names,
                           features_info,
                           training,
                           num_devices,
                           shuffle_buffer_size=1024,
                           preprocess_fun=no_preprocess):
  """Shuffle and batch the given dataset."""
  def append_targets(example):
    """Append targets to the example dictionary. Needed for Keras."""
    if len(target_names) == 1:
      return (example, example[target_names[0]])
    targets = {}
    for name in target_names:
      targets[name] = example[name]
    return (example, targets)
  dataset = dataset.map(append_targets)
  if training:
    dataset = dataset.repeat()
    # Skip a random fraction at the beginning of the stream.  The skip is
    # essential for synchronous highly-parallel training to avoid multiple
    # replicas reading the same data in lock-step.
    dataset = dataset.skip(random.randint(0, _MAX_SKIP_EXAMPLES))
  dataset = preprocess_fun(dataset, training)
  shapes = {k: features_info[k].shape for k in features_info}
  shapes = (shapes, shapes[target_names[0]])
  dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = batch_fun(dataset, training, shapes, target_names, num_devices)
  return dataset.prefetch(2)


def _train_and_eval_batches(dataset, data_dir, input_name, num_devices):
  """Return train and eval batches with input name and shape."""
  (train_data, eval_data, features_info, keys) = train_and_eval_dataset(
      dataset, data_dir)
  input_names, target_names = keys[0], keys[1]
  train_batches = shuffle_and_batch_data(
      train_data, target_names, features_info, training=True,
      num_devices=num_devices)
  train_eval_batches = shuffle_and_batch_data(  # Data for eval-on-train.
      train_data, target_names, features_info, training=False,
      num_devices=num_devices)
  eval_batches = shuffle_and_batch_data(
      eval_data, target_names, features_info, training=False,
      num_devices=num_devices)
  input_name = input_name or input_names[0]
  input_shape = features_info[input_name].shape
  return (train_batches, train_eval_batches, eval_batches,
          input_name, list(input_shape))
