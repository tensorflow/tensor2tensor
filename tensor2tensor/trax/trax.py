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
import itertools
import os
import pickle
import time

from absl import logging

import gin

import jax
from jax.experimental import optimizers as jax_opt
import jax.numpy as np
import jax.random as random

import six

from tensor2tensor.trax import history as trax_history
from tensor2tensor.trax import jaxboard
from tensor2tensor.trax import learning_rate as lr
from tensor2tensor.trax import optimizers as trax_opt
import tensor2tensor.trax.stax as stax

from tensorflow.io import gfile


@gin.configurable(blacklist=["inputs", "targets"])
def masked_mean(inputs, targets, mask_id=None):
  """Mean of the inputs but counting only those where targets != mask_id."""
  x = inputs.astype(np.float32)
  if mask_id is None:
    return np.mean(x)
  unmask = 1.0 - np.equal(targets, mask_id).astype(np.float32)
  return np.sum(x * unmask) / np.sum(unmask)


def accuracy(batch, model_predictions):
  """Calculate accuracy."""
  _, targets = batch
  predicted_class = np.argmax(model_predictions, axis=-1)
  correct = np.equal(predicted_class, targets)
  return masked_mean(correct, targets)


def neg_log_perplexity(batch, model_predictions):
  """Calculate negative log perplexity."""
  _, targets = batch
  hot_targets = stax.one_hot(targets, model_predictions.shape[-1])
  xent = np.sum(model_predictions * hot_targets, axis=-1)
  return masked_mean(xent, targets)


def loss(params, batch, model_predict):
  """Calculate loss."""
  inputs, targets = batch
  preds = model_predict(params, inputs)
  xent = np.sum(preds * stax.one_hot(targets, preds.shape[-1]), axis=-1)
  return - masked_mean(xent, targets)


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
  log("Model loaded from %s at step %d" % (params_file, step))
  logging.debug("From loaded model : history = %s", history)
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


def evaluate_train_and_eval(step, inputs, predict_fun, eval_steps,
                            train_sw=None, eval_sw=None, history=None):
  """Evalaute on train and eval data, and log metrics."""
  step_log(step, "Evaluation")
  train_metrics, eval_metrics = [
      evaluate(  # pylint: disable=g-complex-comprehension
          itertools.islice(input_stream(), eval_steps),
          predict_fun,
          _METRICS)
      for input_stream in
      [inputs.train_stream, inputs.eval_stream]]
  if train_sw:
    log_metrics(train_metrics, train_sw, "train", step, history=history)
  if eval_sw:
    log_metrics(eval_metrics, eval_sw, "eval", step, history=history)
  return train_metrics, eval_metrics


def evaluate(inputs_stream, predict_fun, metric_funs):
  """Evaluate.

  Args:
    inputs_stream: iterable of inputs to evaluate on.
    predict_fun: function from inputs to predictions. params should already be
      partially applied.
    metric_funs: dict from metric name to metric function, which takes inputs
      and predictions and returns a scalar metric value.

  Returns:
    metrics: dict from metric name to metric value averaged over the number of
      inputs.
  """
  metrics = collections.defaultdict(float)
  count = 0
  for inp in inputs_stream:
    count += 1
    preds = predict_fun(inp[0])
    for m, f in six.iteritems(metric_funs):
      metrics[m] += f(inp, preds)
  return {m: v / count for (m, v) in six.iteritems(metrics)}


def log_metrics(metrics, summ_writer, log_prefix, step, history=None):
  """Log metrics to summary writer and history."""
  rjust_len = max([len(name) for name in metrics])
  for name, value in six.iteritems(metrics):
    step_log(step, "%s %s | % .8f" % (
        log_prefix.ljust(5), name.rjust(rjust_len), value))
    full_name = "metrics/" + name
    if history:
      history.append(log_prefix, full_name, step, value)
    if summ_writer:
      summ_writer.scalar(full_name, value, step)


# TODO(trax):
# * Make configurable:
#   * loss
#   * metrics
# * Training loop callbacks/hooks/...
# * Save/restore: pickle unsafe. Use np.array.savez + MessagePack?
# * Move metrics to metrics.py
# * Setup namedtuples for interfaces (e.g. lr fun constructors can take a
#   LearningRateInit, metric funs, etc.).
# * Allow disabling eval


def epochs(steps=None, epoch_steps=1):
  """Iterator over epochs until steps is reached. 1-indexed.

  Args:
    steps: int, total number of steps. Infinite if None.
    epoch_steps: int, number of steps per epoch. Can also be an iterable<int> to
      enable variable length epochs.

  Yields:
    (epoch: int, epoch id, epoch_steps: int, number of steps in this epoch)
  """
  try:
    iter(epoch_steps)
  except TypeError:
    epoch_steps = itertools.repeat(epoch_steps)

  step = 0
  for epoch, epoch_steps in enumerate(epoch_steps):
    epoch_steps = min(epoch_steps, steps - step)
    yield (epoch + 1, epoch_steps)
    step += epoch_steps
    if steps and step >= steps:
      break


def _jit_update_fun(predict_fun, loss_fun, optimizer, lr_fun):
  """Get jit-ed update function for loss, optimizer, learning rate function."""
  @jax.jit
  def update(i, opt_state, batch):
    _, opt_update = optimizer(lr_fun)
    params = jax_opt.get_params(opt_state)
    return opt_update(i, jax.grad(loss_fun)(
        params, batch, predict_fun), opt_state)
  return update


@gin.configurable(blacklist=["output_dir"])
def train(output_dir,
          model=gin.REQUIRED,
          inputs=gin.REQUIRED,
          optimizer=trax_opt.adam,
          lr_schedule=lr.MultifactorSchedule,
          train_steps=1000,
          eval_steps=10,
          eval_frequency=100,
          run_debug_step=False):
  """Train the model on the inputs.

  Args:
    output_dir: Directory where to put the logs and checkpoints.
    model: The model to train as a callable returning 2 callables, an init_fun
      and apply_fun.
    inputs: callable returning trax.inputs.Inputs.
    optimizer: The optimizer as a callable taking a learning_rate callable and
      returning 2 callables, opt_init and opt_update.
    lr_schedule: A learning rate schedule as a function that takes history and
      returns a function from step to learning rate (a float).
    train_steps: int, total number of training steps.
    eval_steps: int, num of steps per evaluation. If None or 0, eval disabled.
    eval_frequency: int, how often to run evaluation (every eval_frequency
      steps). If None or 0, eval disabled.
    run_debug_step: bool, if True, will run the model and loss without @jit for
      one step.

  Returns:
    trax.State
  """
  rng = random.PRNGKey(0)
  gfile.makedirs(output_dir)
  # Create summary writers and history.
  train_sw = jaxboard.SummaryWriter(os.path.join(output_dir, "train"))
  eval_sw = jaxboard.SummaryWriter(os.path.join(output_dir, "eval"))

  inputs = inputs()

  # Setup optimizer and model
  state = restore_state(output_dir)
  history = state.history
  lr_fun = lr_schedule(history)
  opt_init, _ = optimizer(lr_fun)
  model_init, model_predict_original = model()
  # We need a model_predict that fills in the random generator if needed.
  def model_predict(x, y, **kwargs):
    """Same as model_predict_original but fill in rng if it isn't passed."""
    if "rng" in kwargs:
      return model_predict_original(x, y, **kwargs)
    return model_predict_original(x, y, rng=rng, **kwargs)

  # Setup state
  step = state.step or 0
  params_initializer = lambda: model_init([-1] + list(inputs.input_shape))[1]
  params = state.params or params_initializer()
  opt_state = opt_init(params)

  # jit model_predict and update so they're fast
  jit_model_predict = jax.jit(model_predict)  # for evaluation
  jit_update_fun = _jit_update_fun(model_predict, loss, optimizer, lr_fun)

  print()
  train_stream = inputs.train_stream()
  epoch_steps = itertools.chain([1,  # first epoch only 1 step
                                 eval_frequency - 1],
                                itertools.repeat(eval_frequency))
  step_log(step, "Starting training")

  # Non-compiled debug step helps find problems in models easier.
  if run_debug_step:
    debug_loss = loss(params, next(train_stream), model_predict)
    step_log(step, "Debug step loss %.8f" % debug_loss)

  for epoch, epoch_steps in epochs(train_steps, epoch_steps):
    # Log separator
    print()

    # Timer
    start_time = time.time()

    for _ in range(epoch_steps):
      # Train
      opt_state = jit_update_fun(step, opt_state, next(train_stream))
      step += 1

      # LR log
      if step == 1 or step % 10 == 0:
        train_sw.scalar("training/learning rate",
                        lr_fun(step), step=step)

    # Timer
    epoch_time = time.time() - start_time
    step_log(step, "Ran %d train steps in %0.2f secs" %
             (epoch_steps, epoch_time))
    if epoch_steps > 1:
      train_sw.scalar("training/steps per second",
                      epoch_steps / epoch_time, step=step)

    # Evaluate
    params = jax_opt.get_params(opt_state)
    evaluate_train_and_eval(
        step=step,
        inputs=inputs,
        predict_fun=functools.partial(jit_model_predict, params),
        eval_steps=eval_steps,
        train_sw=train_sw,
        eval_sw=eval_sw,
        history=history)

    # Save state
    save_state(State(params=params, step=step, history=history), output_dir)

    # Save Gin config
    # Gin only tracks the used parameters, so we save it after the first epoch.
    if epoch == 1:
      save_gin(output_dir, train_sw)

    # Update learning rate with new history
    old_lr_fun = lr_fun
    lr_fun = lr_schedule(history)
    if lr_fun != old_lr_fun:  # For performance, only jit if there is a change.
      jit_update_fun = _jit_update_fun(model_predict, loss, optimizer, lr_fun)

    # Flush summary writers
    train_sw.writer.flush()
    eval_sw.writer.flush()

  step_log(step, "Training done")
  return State(params=params, step=step, history=history)
