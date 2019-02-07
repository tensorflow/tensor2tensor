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

"""T2T models, configs and main training functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from tensor2tensor import problems
from tensor2tensor.v2.models import basic
from tensor2tensor.v2.models import resnet

import tensorflow as tf
import tensorflow_datasets as tfds

import gin.tf


# Since there are few models and configs for now, we use this simple registry.
# TODO(lukaszkaiser): find a better way to do this or remove altogether.
_MODEL_REGISTRY = {
    "basic_fc_relu": lambda: basic.BasicFcRelu,
    "basic_fc_large": basic.basic_fc_large,
    "basic_fc_relu_v2": lambda: basic.BasicFcReluV2,
    "resnet": lambda: resnet.Resnet,
}


def train_and_eval_dataset(dataset_name, data_dir):
  """Return train and evaluation datasets, feature info and supervised keys.

  Args:
    dataset_name: a string, the name of the dataset; if it starts with "v1_"
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
  if dataset_name.startswith("v1_"):
    return _train_and_eval_dataset_v1(dataset_name[3:], data_dir)
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


def _train_and_eval_dataset_v1(problem_name, data_dir):
  """Return train and evaluation datasets, feature info and supervised keys."""
  problem = problems.problem(problem_name)
  train_dataset = problem.dataset(tf.estimator.ModeKeys.TRAIN, data_dir)
  eval_dataset = problem.dataset(tf.estimator.ModeKeys.EVAL, data_dir)
  supervised_keys = (["inputs"], ["targets"])
  hparams = problem.get_hparams()
  # We take a few training examples to guess the shapes.
  input_shapes, target_shapes = [], []
  for example in train_dataset.take(3):
    input_shapes.append(example["inputs"].shape.as_list())
    target_shapes.append(example["targets"].shape.as_list())
  input_info = _make_info(
      input_shapes, hparams.modality["inputs"].top_dimensionality)
  target_info = _make_info(
      target_shapes, hparams.modality["targets"].top_dimensionality)
  info = {"inputs": input_info, "targets": target_info}
  return train_dataset, eval_dataset, info, supervised_keys


def shuffle_and_batch_data(dataset, batch_size, target_names, repeat=False):
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
  if repeat:
    dataset = dataset.repeat()
  shuffled = dataset.shuffle(128).batch(batch_size).prefetch(8)
  return shuffled


@gin.configurable(blacklist=["model"])
def model_compile(model,
                  optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=None):
  """Compile the model in Keras."""
  metrics = ["accuracy"] if metrics is None else metrics
  return model.compile(optimizer=optimizer,
                       loss=loss,
                       metrics=metrics)


# We include in gin config everything that could be useful to share between
# users, so when it gets saved in a .gin file it can be re-ran with few flags.
@gin.configurable(blacklist=["data_dir", "output_dir"])
def train_fn(data_dir=None, output_dir=None,
             model_class=gin.REQUIRED, dataset=gin.REQUIRED,
             input_names=None, target_names=None,
             batch_size=32, train_steps=1000, eval_steps=1, eval_frequency=100):
  """Train the given model on the given dataset.

  Args:
    data_dir: Directory where the data is located.
    output_dir: Directory where to put the logs and checkpoints.
    model_class: The model class to train.
    dataset: The name of the dataset to train on.
    input_names: List of strings with the names of the features on input.
    target_names: List of strings with the names of the target features.
    batch_size: integer, how many examples per batch.
    train_steps: for how many steps to train.
    eval_steps: for how many steps to do evaluation.
    eval_frequency: how often (every this many steps) to run evaluation.
  """
  train_data, eval_data, features_info, keys = train_and_eval_dataset(
      dataset, data_dir)
  if input_names is None:
    input_names = keys[0]
  if target_names is None:
    target_names = keys[1]
  # TODO(lukaszkaiser): The use of distribution strategy below fails like this:
  #   .../keras/models.py", line 93, in _clone_functional_model
  #      for layer in model._input_layers:
  #   AttributeError: 'BasicFcRelu' object has no attribute '_input_layers'
  # strategy = tf.distribute.MirroredStrategy()
  # with strategy.scope():
  model = model_class(features_info=features_info,
                      input_names=input_names, target_names=target_names)
  model_compile(model)
  train_batches = shuffle_and_batch_data(
      train_data, batch_size, target_names, repeat=True)
  eval_batches = shuffle_and_batch_data(eval_data, batch_size, target_names)

  # Training loop.
  callbacks = []
  callbacks.append(tf.keras.callbacks.History())
  callbacks.append(tf.keras.callbacks.BaseLogger())
  if output_dir is not None:
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=output_dir))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=output_dir, save_weights_only=True))
  model.fit(train_batches,
            epochs=train_steps // eval_frequency,
            steps_per_epoch=eval_frequency,
            validation_data=eval_batches,
            validation_steps=eval_steps,
            callbacks=callbacks)


def t2t_train(model_name, dataset_name,
              data_dir=None, output_dir=None, config_file=None, config=None):
  """Main function to train the given model on the given dataset.

  Args:
    model_name: The name of the model to train.
    dataset_name: The name of the dataset to train on.
    data_dir: Directory where the data is located.
    output_dir: Directory where to put the logs and checkpoints.
    config_file: the gin configuration file to use.
    config: string (in gin format) to override gin parameters.
  """
  if model_name not in _MODEL_REGISTRY:
    raise ValueError("Model %s not in registry. Available models:\n * %s." %
                     (model_name, "\n * ".join(_MODEL_REGISTRY.keys())))
  model_class = _MODEL_REGISTRY[model_name]()
  gin.bind_parameter("train_fn.model_class", model_class)
  gin.bind_parameter("train_fn.dataset", dataset_name)
  gin.parse_config_files_and_bindings(config_file, config)
  # TODO(lukaszkaiser): save gin config in output_dir if provided?
  train_fn(data_dir, output_dir=output_dir)
