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

import gin

import jax.numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds


Inputs = collections.namedtuple(
    "_Inputs", ["train_stream", "eval_stream", "input_shape"])


@gin.configurable()
def inputs(dataset_name, data_dir):
  """Make Inputs for built-in datasets.

  Args:
    dataset_name: a TFDS or T2T dataset name. If it's a T2T dataset name, prefix
      with "t2t_".
    data_dir: data directory.

  Returns:
    trax.inputs.Inputs
  """
  assert data_dir, "Must provide a data directory"
  data_dir = os.path.expanduser(data_dir)

  (train_batches, eval_batches,
   input_name, input_shape) = train_and_eval_batches(
       dataset_name, data_dir)

  def train_input_fun():
    return dataset_to_stream(train_batches, input_name)

  def eval_input_fun():
    return dataset_to_stream(eval_batches, input_name)

  return Inputs(train_stream=train_input_fun,
                eval_stream=eval_input_fun,
                input_shape=input_shape)


def dataset_to_stream(dataset, input_name):
  """Takes a tf.Dataset and creates a numpy stream of ready batches."""
  for example in tfds.as_numpy(dataset):
    inp, out = example[0][input_name], example[1]
    if len(out.shape) > 1 and out.shape[-1] == 1:
      out = np.squeeze(out, axis=-1)
    yield inp, out


def train_and_eval_dataset(dataset_name, data_dir):
  """Return train and evaluation datasets, feature info and supervised keys.

  Args:
    dataset_name: a string, the name of the dataset; if it starts with "t2t_"
      then we'll search T2T Problem registry for it, otherwise we assume it
      is a dataset from TFDS and load it from there.
    data_dir: directory where the data is located.

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
  train, valid = tfds.load(
      name=dataset_name, split=[tfds.Split.TRAIN, eval_split])
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
  return {f: example[f] for f in feature_list}


def _train_and_eval_dataset_v1(problem_name, data_dir):
  """Return train and evaluation datasets, feature info and supervised keys."""
  from tensor2tensor import problems  # pylint: disable=g-import-not-at-top
  problem = problems.problem(problem_name)
  train_dataset = problem.dataset(tf.estimator.ModeKeys.TRAIN, data_dir)
  train_dataset = train_dataset.map(_select_features)
  eval_dataset = problem.dataset(tf.estimator.ModeKeys.EVAL, data_dir)
  eval_dataset = eval_dataset.map(_select_features)
  supervised_keys = (["inputs"], ["targets"])
  hparams = problem.get_hparams()
  # We take a few training examples to guess the shapes.
  input_shapes, target_shapes = [], []
  example_tensor = train_dataset.make_one_shot_iterator().get_next()
  sess = tf.Session()
  example1 = sess.run(example_tensor)
  example2 = sess.run(example_tensor)
  example3 = sess.run(example_tensor)
  for example in [example1, example2, example3]:
    input_shapes.append(list(example["inputs"].shape))
    target_shapes.append(list(example["targets"].shape))
  input_vocab_size = hparams.vocab_size["inputs"]
  target_vocab_size = hparams.vocab_size["targets"]
  input_info = _make_info(input_shapes, input_vocab_size)
  target_info = _make_info(target_shapes, target_vocab_size)
  info = {"inputs": input_info, "targets": target_info}
  return train_dataset, eval_dataset, info, supervised_keys


@gin.configurable(blacklist=["dataset", "training"])
def preprocess_fun(dataset, training, max_target_length=-1):
  def target_right_length(_, target):
    return tf.less(tf.shape(target)[0], max_target_length + 1)
  if max_target_length > 0 and training:
    dataset = dataset.filter(target_right_length)
  return dataset


@gin.configurable(blacklist=["dataset", "training", "shapes", "target_names"])
def batch_fun(dataset, training, shapes, target_names,
              batch_size=32, eval_batch_size=32,
              bucket_length=32, buckets=None):
  """Batching function."""
  del target_names
  # If bucketing is not specified, check if target shapes are variable.
  cur_batch_size = batch_size if training else eval_batch_size
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
                           bucket_length * 4, bucket_length * 8]
      bucket_batch_sizes = [cur_batch_size * 4, cur_batch_size * 2,
                            cur_batch_size, cur_batch_size // 2,
                            cur_batch_size // 4, cur_batch_size // 8]
      buckets = (bucket_boundaries, bucket_batch_sizes)

  if buckets:
    tf.logging.info("Bucketing with buckets %s." % str(buckets))
    def example_length(_, target):
      return tf.shape(target)[0]
    boundaries, batch_sizes = buckets
    dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
        example_length, boundaries, batch_sizes, pad_to_bucket_boundary=True))
  else:
    dataset = dataset.padded_batch(cur_batch_size, shapes)
  return dataset


def shuffle_and_batch_data(dataset, target_names, features_info, training):
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
  shapes = {k: features_info[k].shape for k in features_info}
  shapes = (shapes, shapes[target_names[0]])
  dataset = dataset.shuffle(1024)
  dataset = preprocess_fun(dataset, training)
  dataset = batch_fun(dataset, training, shapes, target_names)
  return dataset.prefetch(32)


def train_and_eval_batches(dataset, data_dir):
  """Return train and eval batches with input name and shape."""
  (train_data, eval_data, features_info, keys) = train_and_eval_dataset(
      dataset, data_dir)
  input_names, target_names = keys[0], keys[1]
  train_batches = shuffle_and_batch_data(
      train_data, target_names, features_info, training=True)
  eval_batches = shuffle_and_batch_data(
      eval_data, target_names, features_info, training=False)
  input_shape = features_info[input_names[0]].shape
  return train_batches, eval_batches, input_names[0], list(input_shape)
