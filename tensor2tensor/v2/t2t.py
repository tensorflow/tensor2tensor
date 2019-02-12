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
import os

from tensor2tensor import problems
from tensor2tensor.utils import data_reader
from tensor2tensor.v2.models import basic
from tensor2tensor.v2.models import resnet
from tensor2tensor.v2.models import transformer

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
    "transformer": transformer.transformer_base_single_gpu,
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


def _select_features(example, feature_list=None):
  """Select a subset of features from the example dict."""
  feature_list = feature_list or ["inputs", "targets"]
  return {f: example[f] for f in feature_list}


def _train_and_eval_dataset_v1(problem_name, data_dir):
  """Return train and evaluation datasets, feature info and supervised keys."""
  problem = problems.problem(problem_name)
  train_dataset = problem.dataset(tf.estimator.ModeKeys.TRAIN, data_dir)
  train_dataset = train_dataset.map(_select_features)
  eval_dataset = problem.dataset(tf.estimator.ModeKeys.EVAL, data_dir)
  eval_dataset = eval_dataset.map(_select_features)
  supervised_keys = (["inputs"], ["targets"])
  hparams = problem.get_hparams()
  # We take a few training examples to guess the shapes.
  input_shapes, target_shapes = [], []
  for example in train_dataset.take(3):
    input_shapes.append(example["inputs"].shape.as_list())
    target_shapes.append(example["targets"].shape.as_list())
  input_vocab_size = hparams.vocab_size["inputs"]
  target_vocab_size = hparams.vocab_size["targets"]
  input_info = _make_info(input_shapes, input_vocab_size)
  target_info = _make_info(target_shapes, target_vocab_size)
  info = {"inputs": input_info, "targets": target_info}
  return train_dataset, eval_dataset, info, supervised_keys


@gin.configurable(blacklist=["dataset", "training"])
def preprocess_fn(dataset, training, max_target_length=-1):
  def target_right_length(_, target):
    if max_target_length < 1 or not training:
      return tf.constant(True)
    return tf.less(tf.shape(target)[0], max_target_length + 1)
  dataset = dataset.filter(target_right_length)
  return dataset


@gin.configurable(blacklist=["dataset", "training", "shapes", "target_names"])
def batch_fn(dataset, training, shapes, target_names,
             batch_size=32, eval_batch_size=32, bucket_batch_length=32,
             bucket_max_length=256, bucket_min_length=8,
             bucket_length_step=1.1, buckets=None):
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
      batch_size_per_token = cur_batch_size * bucket_batch_length
      scheme = data_reader.batching_scheme(batch_size_per_token,
                                           bucket_max_length,
                                           bucket_min_length,
                                           bucket_length_step,
                                           drop_long_sequences=training)
      buckets = (scheme["boundaries"], scheme["batch_sizes"])

  if buckets:
    tf.logging.info("Bucketing with buckets %s." % str(buckets))
    def example_length(_, target):
      return tf.shape(target)[0]
    boundaries, batch_sizes = buckets
    dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
        example_length, boundaries, batch_sizes))
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
  dataset = dataset.shuffle(128)
  dataset = preprocess_fn(dataset, training)
  dataset = batch_fn(dataset, training, shapes, target_names)
  return dataset.prefetch(8)


@gin.configurable()
class T2TLearningRateSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """A LearningRateSchedule that uses a T2T config."""

  def __init__(self, schedule=None, constant=0.1, warmup_steps=200):
    """Applies the give T2T schedule string with the given parameters."""
    super(T2TLearningRateSchedule, self).__init__()
    self.schedule = schedule or "constant * linear_warmup * rsqrt_decay"
    self.constant = constant
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    ret = tf.constant(1.0)
    for name in [n.strip() for n in self.schedule.split("*")]:
      if name == "constant":
        ret *= self.constant
      elif name == "linear_warmup":
        ret *= tf.minimum(1.0, step / self.warmup_steps)
      elif name == "rsqrt_decay":
        ret *= tf.rsqrt(tf.maximum(step, self.warmup_steps))
      else:
        raise ValueError("Unknown factor %s." % name)
    tf.contrib.summary.scalar("learning_rate", ret)
    return ret

  def get_config(self):
    return {
        "schedule": self.schedule,
        "constant": self.constant,
        "warmup_steps": self.warmup_steps,
    }


@gin.configurable(blacklist=["model"])
def optimize_fn(model,
                optimizer=None,
                learning_rate_schedule=None,
                loss=None,
                metrics=None):
  """Compile the model in Keras."""
  learning_rate_schedule = learning_rate_schedule or T2TLearningRateSchedule()
  if optimizer:
    optimizer = optimizer(learning_rate=learning_rate_schedule)
  else:  # We use Adam by default with adjusted parameters.
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_schedule,
        beta_1=0.9, beta_2=0.997, epsilon=1e-9)
  metrics = metrics or [tf.keras.metrics.sparse_categorical_accuracy]
  def xent_loss(y, x):
    return tf.keras.backend.sparse_categorical_crossentropy(
        y, x, from_logits=True)
  loss = loss or xent_loss
  return model.compile(optimizer=optimizer,
                       loss=loss,
                       metrics=metrics)


# We include in gin config everything that could be useful to share between
# users, so when it gets saved in a .gin file it can be re-ran with few flags.
@gin.configurable(blacklist=["data_dir", "output_dir"])
def train_fn(data_dir=None, output_dir=None,
             model_class=gin.REQUIRED, dataset=gin.REQUIRED,
             input_names=None, target_names=None,
             train_steps=1000, eval_steps=1, eval_frequency=100):
  """Train the given model on the given dataset.

  Args:
    data_dir: Directory where the data is located.
    output_dir: Directory where to put the logs and checkpoints.
    model_class: The model class to train.
    dataset: The name of the dataset to train on.
    input_names: List of strings with the names of the features on input.
    target_names: List of strings with the names of the target features.
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
  optimize_fn(model)
  train_batches = shuffle_and_batch_data(
      train_data, target_names, features_info, training=True)
  eval_batches = shuffle_and_batch_data(
      eval_data, target_names, features_info, training=False)
  # Need to run one training step just to get optimizer variables to load.
  model.fit(train_batches, epochs=1, steps_per_epoch=1)

  # Training loop.
  callbacks = []
  callbacks.append(tf.keras.callbacks.History())
  callbacks.append(tf.keras.callbacks.BaseLogger())
  last_epoch = 0
  if output_dir is not None:
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=output_dir))
    output_format = os.path.join(output_dir, "model-{epoch:05d}")
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=output_format, save_weights_only=True))
    checkpoints = tf.gfile.Glob(os.path.join(output_dir, "model-*"))
    # Take basenames and strip the "model-" prefix.
    checkpoints = [os.path.basename(ckpt)[6:] for ckpt in checkpoints]
    # Get epoch numbers from the filenames and sort to obtain last epoch.
    epoch_numbers = [int(ckpt[:5]) for ckpt in checkpoints if len(ckpt) > 4]
    epoch_numbers.sort()
    if epoch_numbers:
      last_epoch = epoch_numbers[-1]
      saved_path = os.path.join(output_dir, "model-%05d" % last_epoch)
      model.load_weights(saved_path)
  model.fit(train_batches,
            epochs=train_steps // eval_frequency,
            steps_per_epoch=eval_frequency,
            validation_data=eval_batches,
            validation_steps=eval_steps,
            initial_epoch=last_epoch,
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
