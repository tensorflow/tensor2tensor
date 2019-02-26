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

"""J2J main training functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time

from absl import logging
import gin

import jax
from jax.experimental import optimizers
import jax.numpy as np

from tensor2tensor.jax import input_pipeline
from tensor2tensor.jax import jaxboard
# Import for gin configurable models
from tensor2tensor.jax import models  # pylint: disable=unused-import

from tensorflow.io import gfile

import tensorflow_datasets as tfds


@gin.configurable(blacklist=["step"])
def learning_rate(step,
                  schedule="constant * linear_warmup * rsqrt_decay",
                  constant=0.001,
                  warmup_steps=100):
  """Learning rate."""
  ret = 1.0
  for name in [n.strip() for n in schedule.split("*")]:
    if name == "constant":
      ret *= constant
    elif name == "linear_warmup":
      ret *= np.minimum(1.0, step / warmup_steps)
    elif name == "rsqrt_decay":
      ret /= np.sqrt(np.maximum(step, warmup_steps))
    else:
      raise ValueError("Unknown factor %s." % name)
  return ret


@gin.configurable()
def optimizer(name="adam",
              momentum_mass=0.9, rmsprop_gamma=0.9, rmsprop_eps=1e-8,
              adam_b1=0.9, adam_b2=0.997, adam_eps=1e-8):
  """Return the optimizer, by name."""
  if name == "sgd":
    return optimizers.sgd(learning_rate)
  if name == "momentum":
    return optimizers.momentum(learning_rate, mass=momentum_mass)
  if name == "rmsprop":
    return optimizers.rmsprop(
        learning_rate, gamma=rmsprop_gamma, eps=rmsprop_eps)
  if name == "adam":
    return optimizers.adam(learning_rate, b1=adam_b1, b2=adam_b2, eps=adam_eps)
  raise ValueError("Unknown optimizer %s" % str(name))


def one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)


def accuracy(batch, model_predictions):
  """Calculate accuracy."""
  _, targets = batch
  predicted_class = np.argmax(model_predictions, axis=-1)
  return np.mean(predicted_class == targets)


def neg_log_perplexity(batch, model_predictions):
  """Calculate negative log perplexity."""
  _, targets = batch
  hot_targets = one_hot(targets, model_predictions.shape[-1])
  return np.mean(np.sum(model_predictions * hot_targets, axis=-1))


def loss(params, batch, model_predict):
  """Calculate loss."""
  inputs, targets = batch
  preds = model_predict(params, inputs)
  return - np.mean(preds * one_hot(targets, preds.shape[-1]))


def dataset_to_stream(batches, input_name):
  """Takes a tf.Dataset and creates a numpy stream of ready batches."""
  for example in tfds.as_numpy(batches):
    inp, out = example[0][input_name], example[1]
    if len(out.shape) > 1 and out.shape[-1] == 1:
      out = np.squeeze(out, axis=-1)
    yield inp, out


def log(s, stdout=True):
  logging.info(s)
  if stdout:
    print(s)


def _make_directory(path):
  """Helper function: create directory if it doesn't exist yet."""
  if not gfile.exists(path):
    log("Creating directory %s" % path)
    gfile.mkdir(path)


def save_params_and_step(params, step, output_dir):
  """Save params and step in output dir."""
  if output_dir is not None:
    _make_directory(output_dir)
    params_file = os.path.join(output_dir, "model.pkl")
    with gfile.GFile(params_file, "wb") as f:
      pickle.dump((params, step), f)
    log("Model saved to %s" % params_file, stdout=False)


def load_params_and_step(output_dir):
  """Save params and step in output dir."""
  if output_dir is None:
    return None, None
  if not gfile.exists(output_dir):
    return None, None
  params_file = os.path.join(output_dir, "model.pkl")
  if not gfile.exists(params_file):
    return None, None
  with gfile.GFile(params_file, "r") as f:
    (params, step) = pickle.load(f)
  log("Model loaded from %s" % params_file)
  return params, step


def _make_summary_writer(output_path):
  _make_directory(output_path)
  return jaxboard.SummaryWriter(output_path)


# Metrics to calculate and report.
_metrics = {
    "accuracy": accuracy,
    "neg_log_perplexity": neg_log_perplexity,
    "loss": lambda x, y: - neg_log_perplexity(x, y),
}


# We include in gin config everything that could be useful to share between
# users, so when it gets saved in a .gin file it can be re-ran with few flags.
@gin.configurable(blacklist=["data_dir", "output_dir"])
def train_fn(data_dir=None, output_dir=None,
             model=gin.REQUIRED,
             dataset=gin.REQUIRED,
             train_steps=1000, eval_steps=10, eval_frequency=100):
  """Train the given model on the given dataset.

  Args:
    data_dir: Directory where the data is located.
    output_dir: Directory where to put the logs and checkpoints.
    model: The model to train (a function).
    dataset: The name of the dataset to train on.
    train_steps: for how many steps to train.
    eval_steps: for how many steps to do evaluation.
    eval_frequency: how often (every this many steps) to run evaluation.
  """
  (train_batches, eval_batches,
   input_name, input_shape) = input_pipeline.train_and_eval_batches(
       dataset, data_dir)
  train_stream = dataset_to_stream(train_batches, input_name)

  # Training loop.
  opt_init, opt_update = optimizer()
  model_init, model_predict = model()

  @jax.jit
  def update(i, opt_state, batch):
    params = optimizers.get_params(opt_state)
    return opt_update(i, jax.grad(loss)(
        params, batch, model_predict), opt_state)

  _, init_params = model_init([-1] + input_shape)
  step, train_sw, eval_sw = 0, None, None
  if output_dir is not None:
    _make_directory(output_dir)
    # Load parameters.
    loaded_params, loaded_step = load_params_and_step(output_dir)
    if loaded_params is not None:
      init_params = loaded_params
    if loaded_step is not None:
      step = loaded_step

    # Create summary writers.
    eval_sw = _make_summary_writer(os.path.join(output_dir, "eval"))
    train_sw = _make_summary_writer(os.path.join(output_dir, "train"))

  log("Starting training.")
  opt_state = opt_init(init_params)
  gin_config_saved = False
  cur_eval_frequency = 1  # First evaluation after the first training step.
  while step < train_steps:
    # Training.
    start_time = time.time()
    for _ in range(cur_eval_frequency):
      opt_state = update(step, opt_state, next(train_stream))
      if train_sw and step % 10 == 0:  # Log learning rate curve each 10 steps.
        train_sw.scalar("training/learning rate",
                        learning_rate(step), step=step)
      step += 1
    epoch_time = time.time() - start_time
    log("Step {}, last {} steps in {:0.2f} sec".format(
        step, cur_eval_frequency, epoch_time))

    # Save the model.
    params = optimizers.get_params(opt_state)
    save_params_and_step(params, step, output_dir)

    # Save the config if not saved yet.
    # Gin file only includes used parameters, so we save it at this point.
    if output_dir and not gin_config_saved:
      gin_config_saved = True
      config_path = os.path.join(output_dir, "config.gin")
      with gfile.GFile(config_path, "w") as f:
        f.write(gin.operative_config_str())

    # Evaluation.
    eval_stream = dataset_to_stream(eval_batches, input_name)
    eval_train_stream = dataset_to_stream(train_batches, input_name)
    train_metrics = {key: 0.0 for key in _metrics}
    eval_metrics = {key: 0.0 for key in _metrics}
    for _ in range(eval_steps):
      train_batch = next(eval_train_stream)
      train_predictions = model_predict(params, train_batch[0])
      eval_batch = next(eval_stream)
      eval_predictions = model_predict(params, eval_batch[0])
      for m in _metrics:
        train_metrics[m] += _metrics[m](
            train_batch, train_predictions) / float(eval_steps)
        eval_metrics[m] += _metrics[m](
            eval_batch, eval_predictions) / float(eval_steps)

    for m in _metrics:
      log("Step %d train %s %.8f" % (step, m, train_metrics[m]))
      prefix = "metrics/"
      if train_sw:
        train_sw.scalar(prefix + m, train_metrics[m], step=step)
      log("Step %d eval  %s %.8f" % (step, m, eval_metrics[m]))
      if eval_sw:
        eval_sw.scalar(prefix + m, eval_metrics[m], step=step)

    # Log non-metric reports and flush.
    if train_sw:
      if step > 1:  # Don't log performance of the first step.
        train_sw.scalar("training/steps per second",
                        cur_eval_frequency / epoch_time, step=step)
      train_sw.writer.flush()
    if eval_sw:
      eval_sw.writer.flush()

    # After the first step, Evaluate every eval_frequency steps.
    if cur_eval_frequency == 1 and eval_frequency != 1:
      cur_eval_frequency = eval_frequency - 1
    else:
      cur_eval_frequency = eval_frequency
