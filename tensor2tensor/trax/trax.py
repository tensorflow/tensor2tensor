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

"""trax main training functions."""

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
from jax.experimental import optimizers as jax_opt
import jax.numpy as np

import six

from tensor2tensor.trax import history as trax_history
from tensor2tensor.trax import jaxboard
from tensor2tensor.trax import learning_rate as lr
from tensor2tensor.trax import optimizers as trax_opt

from tensorflow.io import gfile

# Imports for gin configurables
# TODO(trax): Move to trainer.py. Only here because of t2t_trainer usage.
# pylint: disable=unused-import,g-bad-import-order,reimported
from tensor2tensor.trax import inputs as _trax_inputs
from tensor2tensor.trax import models as _trax_models
from tensor2tensor.trax import optimizers as _trax_opt
# pylint: disable=unused-import,g-bad-import-order,reimported


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
  return - np.mean(np.sum(preds * one_hot(targets, preds.shape[-1]), axis=-1))


def log(s, stdout=True):
  logging.info(s)
  if stdout:
    print(s)


def step_log(step, s):
  log("Step % 6d: %s" % (step, s))


State = collections.namedtuple("_State", ["step", "params", "history"])


def restore_state(output_dir):
  """Restore State."""
  params_file = os.path.join(output_dir, "model.pkl")
  if not gfile.exists(params_file):
    return State(step=None, params=None, history=trax_history.History())

  with gfile.GFile(params_file, "rb") as f:
    (params, step, history) = pickle.load(f)
  log("Model loaded from %s" % params_file)
  return State(step=step, params=params, history=history)


def save_gin(output_dir, sw=None):
  config_path = os.path.join(output_dir, "config.gin")
  config_str = gin.operative_config_str()
  with gfile.GFile(config_path, "w") as f:
    f.write(config_str)
  if sw:
    sw.text("gin_config",
            jaxboard.markdownify_operative_config_str(config_str))


def save_state(state, output_dir):
  """Save State and optionally gin config."""
  params_file = os.path.join(output_dir, "model.pkl")
  with gfile.GFile(params_file, "wb") as f:
    pickle.dump((state.params, state.step, state.history), f)
  log("Model saved to %s" % params_file, stdout=False)


# Metrics to calculate and report.
_METRICS = {
    "accuracy": accuracy,
    "neg_log_perplexity": neg_log_perplexity,
    "loss": lambda x, y: - neg_log_perplexity(x, y),
}


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
      train_metrics[m] += (_METRICS[m](train_batch, train_predictions)
                           / float(eval_steps))
      eval_metrics[m] += (_METRICS[m](eval_batch, eval_predictions)
                          / float(eval_steps))

  return train_metrics, eval_metrics


def log_metrics(metrics, summ_writer, log_prefix, step, history=None):
  """Log metrics to summary writer and history."""
  rjust_len = max([len(name) for name in metrics])
  for name, value in six.iteritems(metrics):
    step_log(step, "%s %s | % .8f" % (log_prefix, name.rjust(rjust_len), value))
    full_name = "metrics/" + name
    if history:
      history.append(full_name, value, step, log_prefix)
    if summ_writer:
      summ_writer.scalar(full_name, value, step)


# TODO(trax):
# * Make configurable:
#   * loss
#   * metrics
# * Save/restore: pickle unsafe. Use np.array.savez + MessagePack?
# * Move metrics to metrics.py


@gin.configurable(blacklist=["output_dir"])
def train(output_dir,
          model=gin.REQUIRED,
          inputs=gin.REQUIRED,
          optimizer=trax_opt.adam,
          learning_rate_fn=lr.make_default_schedule,
          train_steps=1000,
          eval_steps=10,
          eval_frequency=100):
  """Train the model on the inputs.

  Args:
    output_dir: Directory where to put the logs and checkpoints.
    model: The model to train as a callable returning 2 callables, an init_fun
      and apply_fun.
    inputs: callable returning trax.inputs.Inputs.
    optimizer: The optimizer as a callable taking a learning_rate callable and
      returning 2 callables, opt_init and opt_update.
    learning_rate_fn: The learning rate callable that takes history and returns
      a function from step to learning rate (a float).
    train_steps: int, total number of training steps.
    eval_steps: int, num of steps per evaluation. If None or 0, eval disabled.
    eval_frequency: int, how often to run evaluation (every eval_frequency
      steps). If None or 0, eval disabled.

  Returns:
    trax.State
  """
  gfile.makedirs(output_dir)
  # Create summary writers and history.
  train_sw = jaxboard.SummaryWriter(os.path.join(output_dir, "train"))
  eval_sw = jaxboard.SummaryWriter(os.path.join(output_dir, "eval"))

  inputs = inputs()

  # Setup optimizer and model
  state = restore_state(output_dir)
  history = state.history
  learning_rate = learning_rate_fn(history)
  opt_init, opt_update = optimizer(learning_rate)
  model_init, model_predict = model()

  # Setup state
  step = state.step or 0
  params_initializer = lambda: model_init([-1] + inputs.input_shape)[1]
  opt_state = opt_init(state.params or params_initializer())

  # jit model_predict and update so they're fast
  jit_predict = jax.jit(model_predict)  # for evaluation

  @jax.jit
  def update(i, opt_state, batch):
    params = jax_opt.get_params(opt_state)
    return opt_update(i, jax.grad(loss)(
        params, batch, model_predict), opt_state)

  print()
  step_log(step, "starting training")
  inputs_stream = inputs.train_fn()
  eval_enabled = eval_steps and eval_frequency
  is_first_step = True
  # Evaluate after the first training step, then reset to normal_epoch_steps
  normal_epoch_steps = (eval_enabled and eval_frequency) or train_steps
  epoch_steps = 1
  while step < train_steps:
    print()  # separate logging for each loop iteration

    # Train
    start_time = time.time()
    for _ in range(epoch_steps):
      opt_state = update(step, opt_state, next(inputs_stream))
      if step % 10 == 0:  # Log learning rate curve each 10 steps.
        train_sw.scalar("training/learning rate",
                        learning_rate(step), step=step)
      step += 1
    epoch_time = time.time() - start_time
    step_log(step, "ran %d train steps in %0.2f secs" %
             (epoch_steps, epoch_time))

    # Evaluate
    params = jax_opt.get_params(opt_state)
    if eval_enabled:
      step_log(step, "starting evaluation")
      train_metrics, eval_metrics = evaluate(
          inputs, functools.partial(jit_predict, params), eval_steps)
      log_metrics(train_metrics, train_sw, "train", step, history=history)
      log_metrics(eval_metrics, eval_sw, "eval ", step, history=history)
      eval_sw.writer.flush()

    # Save state
    save_state(State(params=params, step=step, history=history), output_dir)

    # Gin only tracks the used parameters, so we save it after the first step.
    if is_first_step:
      save_gin(output_dir, train_sw)

    # Log non-metric reports.
    if not is_first_step:
      train_sw.scalar("training/steps per second",
                      epoch_steps / epoch_time, step=step)
    train_sw.writer.flush()

    # Update learning rate with new history.
    learning_rate = learning_rate_fn(history)

    # After the first step, train for normal_epoch_steps steps before evaluating
    epoch_steps = (
        (normal_epoch_steps - 1) if is_first_step else normal_epoch_steps)
    is_first_step = False

  print()
  step_log(step, "finished training")
  return State(params=params, step=step, history=history)
