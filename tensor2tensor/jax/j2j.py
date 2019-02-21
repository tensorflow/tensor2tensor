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
# Import for gin configurable models
from tensor2tensor.jax import models  # pylint: disable=unused-import

from tensorflow import gfile

import tensorflow_datasets as tfds


@gin.configurable(blacklist=["step"])
def learning_rate(step, schedule=None, constant=0.001, warmup_steps=200):
  """Learning rate."""
  schedule = schedule or "constant * linear_warmup * rsqrt_decay"
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


def accuracy(params, batch, model_predict):
  """Calculate accuracy."""
  inputs, targets = batch
  predicted_class = np.argmax(model_predict(params, inputs), axis=1)
  return np.mean(predicted_class == targets)


def loss(params, batch, model_predict):
  """Calculate loss."""
  inputs, targets = batch
  preds = model_predict(params, inputs)
  return -np.mean(preds * one_hot(targets, preds.shape[-1]))


def dataset_to_stream(batches, input_name):
  """Takes a tf.Dataset and creates a numpy stream of ready batches."""
  for example in tfds.as_numpy(batches):
    inp, out = example[0][input_name], example[1]
    yield inp, out


def log(s, stdout=True):
  logging.info(s)
  if stdout:
    print(s)


def save_params_and_step(params, step, output_dir):
  """Save params and step in output dir."""
  if output_dir is not None:
    if not gfile.Exists(output_dir):
      log("Creating directory %s" % output_dir)
      gfile.MkDir(output_dir)
    params_file = os.path.join(output_dir, "model.pkl")
    with gfile.Open(params_file, "wb") as f:
      pickle.dump((params, step), f)
    log("Model saved to %s" % params_file, stdout=False)


def load_params_and_step(output_dir):
  """Save params and step in output dir."""
  if output_dir is None:
    return None, None
  if not gfile.Exists(output_dir):
    return None, None
  params_file = os.path.join(output_dir, "model.pkl")
  if not gfile.Exists(params_file):
    return None, None
  with gfile.Open(params_file, "r") as f:
    (params, step) = pickle.load(f)
  log("Model loaded from %s" % params_file)
  return params, step


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
  step = 0
  if output_dir is not None:
    loaded_params, loaded_step = load_params_and_step(output_dir)
    if loaded_params is not None:
      init_params = loaded_params
    if loaded_step is not None:
      step = loaded_step
  opt_state = opt_init(init_params)

  log("Starting training.")
  while step < train_steps:
    # Training.
    start_time = time.time()
    for _ in range(eval_frequency):
      opt_state = update(step, opt_state, next(train_stream))
      step += 1
    epoch_time = time.time() - start_time
    log("Step {}, last {} steps in {:0.2f} sec".format(
        step, eval_frequency, epoch_time))

    # Save the model.
    params = optimizers.get_params(opt_state)
    save_params_and_step(params, step, output_dir)

    # Evaluation.
    eval_stream = dataset_to_stream(eval_batches, input_name)
    eval_train_stream = dataset_to_stream(train_batches, input_name)
    train_acc, eval_acc = 0, 0
    for _ in range(eval_steps):
      train_acc += accuracy(params, next(eval_train_stream), model_predict)
      eval_acc += accuracy(params, next(eval_stream), model_predict)
    log("Train set accuracy {:0.4f}".format(train_acc / eval_steps))
    log("Eval  set accuracy {:0.4f}".format(eval_acc / eval_steps))
