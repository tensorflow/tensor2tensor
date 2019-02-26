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

import collections
import functools
import os
import pickle
import time

from absl import logging
import gin

import jax
from jax.experimental import optimizers
import jax.numpy as np

from tensor2tensor.jax import inputs as inputs_lib
from tensor2tensor.jax import jaxboard
# Import for gin configurable models
from tensor2tensor.jax import models  # pylint: disable=unused-import

from tensorflow.io import gfile


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


def log(s, stdout=True):
  logging.info(s)
  if stdout:
    print(s)


def step_log(step, s):
  log("Step % 6d: %s" % (step, s))


State = collections.namedtuple("_State", ["step", "params"])


def restore_state(output_dir):
  """Restore State."""
  empty_state = State(step=None, params=None)
  if output_dir is None or not gfile.exists(output_dir):
    return empty_state
  params_file = os.path.join(output_dir, "model.pkl")
  if not gfile.exists(params_file):
    return empty_state

  with gfile.GFile(params_file, "rb") as f:
    (params, step) = pickle.load(f)
  log("Model loaded from %s" % params_file)
  return State(step=step, params=params)


def save_state(state, output_dir, save_gin=True):
  """Save State and optionally gin config."""
  if not output_dir:
    return

  gfile.makedirs(output_dir)
  params_file = os.path.join(output_dir, "model.pkl")
  with gfile.GFile(params_file, "wb") as f:
    pickle.dump((state.params, state.step), f)
  log("Model saved to %s" % params_file, stdout=False)

  # Gin file only includes used parameters, so we save it at this point.
  if save_gin:
    config_path = os.path.join(output_dir, "config.gin")
    with gfile.GFile(config_path, "w") as f:
      f.write(gin.operative_config_str())


# Metrics to calculate and report.
_METRICS = {
    "accuracy": accuracy,
    "neg_log_perplexity": neg_log_perplexity,
    "loss": lambda x, y: - neg_log_perplexity(x, y),
}


# We include in gin config everything that could be useful to share between
# users, so when it gets saved in a .gin file it can be re-run with minimal
# flags.
@gin.configurable(blacklist=["data_dir", "output_dir"])
def train_fn(data_dir=None,
             output_dir=None,
             model=gin.REQUIRED,
             dataset=gin.REQUIRED,
             train_steps=1000,
             eval_steps=10,
             eval_frequency=100):
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
  # Make Inputs
  inputs = inputs_lib.make_inputs(dataset, data_dir)

  # Setup optimizer and model
  opt_init, opt_update = optimizer()
  model_init, model_predict = model()
  jit_predict = jax.jit(model_predict)

  # Setup state
  state = restore_state(output_dir)
  step = state.step or 0
  _, init_params = model_init([-1] + inputs.input_shape)
  opt_state = opt_init(state.params or init_params)

  # Create summary writers.
  train_sw, eval_sw = None, None
  if output_dir:
    gfile.makedirs(output_dir)
    train_sw = jaxboard.SummaryWriter(os.path.join(output_dir, "train"))
    eval_sw = jaxboard.SummaryWriter(os.path.join(output_dir, "eval"))

  # Make fast update function
  @jax.jit
  def update(i, opt_state, batch):
    params = optimizers.get_params(opt_state)
    return opt_update(i, jax.grad(loss)(
        params, batch, model_predict), opt_state)

  step_log(step, "starting training")
  train_gen = inputs.train_fn()
  is_first_step = True
  epoch_steps = 1  # First evaluation after the first training step.
  while step < train_steps:
    # Train
    start_time = time.time()
    for _ in range(epoch_steps):
      opt_state = update(step, opt_state, next(train_gen))
      if train_sw and step % 10 == 0:  # Log learning rate curve each 10 steps.
        train_sw.scalar("training/learning rate",
                        learning_rate(step), step=step)
      step += 1
    epoch_time = time.time() - start_time
    print()
    step_log(step, "%d train steps in %0.2f secs" % (epoch_steps, epoch_time))

    # Save state
    params = optimizers.get_params(opt_state)
    save_state(State(params=params, step=step), output_dir,
               save_gin=is_first_step)

    # Evaluate
    print()
    step_log(step, "starting evaluation")
    train_metrics, eval_metrics = evaluate(
        inputs, functools.partial(jit_predict, params), eval_steps)

    for m in _METRICS:
      step_log(step, "train %s %.8f" % (m, train_metrics[m]))
      prefix = "metrics/"
      if train_sw:
        train_sw.scalar(prefix + m, train_metrics[m], step=step)
      step_log(step, "eval  %s %.8f" % (m, eval_metrics[m]))
      if eval_sw:
        eval_sw.scalar(prefix + m, eval_metrics[m], step=step)

    # Log non-metric reports and flush.
    if train_sw and not is_first_step:
      train_sw.scalar("training/steps per second",
                      epoch_steps / epoch_time, step=step)
      train_sw.writer.flush()
    if eval_sw:
      eval_sw.writer.flush()

    # After the first step, train for eval_frequency steps before evaluating
    epoch_steps = (eval_frequency - 1) if is_first_step else eval_frequency
    is_first_step = False

  step_log(step, "finished training.")


def evaluate(inputs, predict_fn, eval_steps):
  """Evaluate.

  Args:
    inputs: Inputs namedtuple.
    predict_fn: function from inputs to predictions. params should already be
      partially applied.
    eval_steps: int, number of evaluation steps.

  Returns:
    train_metrics: dict
    eval_metrics: dict
  """
  eval_stream = inputs.eval_fn()
  eval_train_stream = inputs.train_fn()
  train_metrics = {key: 0.0 for key in _METRICS}
  eval_metrics = {key: 0.0 for key in _METRICS}
  for _ in range(eval_steps):
    train_batch = next(eval_train_stream)
    train_predictions = predict_fn(train_batch[0])
    eval_batch = next(eval_stream)
    eval_predictions = predict_fn(eval_batch[0])
    for m in _METRICS:
      train_metrics[m] += _METRICS[m](
          train_batch, train_predictions) / float(eval_steps)
      eval_metrics[m] += _METRICS[m](
          eval_batch, eval_predictions) / float(eval_steps)

  return train_metrics, eval_metrics
