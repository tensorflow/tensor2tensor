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
import random
import sys
import time

from absl import logging

import gin

import jax
from jax import lax
import numpy
import six

from tensor2tensor.trax import backend
from tensor2tensor.trax import history as trax_history
from tensor2tensor.trax import inputs as trax_inputs
from tensor2tensor.trax import jaxboard
from tensor2tensor.trax import layers
from tensor2tensor.trax import learning_rate as lr
from tensor2tensor.trax import optimizers as trax_opt
from tensor2tensor.trax.backend import numpy as np
from tensor2tensor.trax.backend import random as jax_random

import tensorflow as tf
from tensorflow.io import gfile


def _make_list(predictions, targets):
  """Helper: make predictions and targets lists, check they match on length."""
  #  Our models sometimes return predictions in lists, make it a list always.
  # TODO(lukaszkaiser): make abstractions for nested structures and refactor.
  if not isinstance(predictions, (list, tuple)):
    if isinstance(targets, (list, tuple)):
      raise ValueError("Targets are a list or tuple but predictions are not.")
    predictions, targets = [predictions], [targets]
  if len(predictions) != len(targets):
    raise ValueError("Predictions and targets have different lengths.")
  return list(predictions), list(targets)


@gin.configurable(blacklist=["inputs", "targets"])
def masked_mean(inputs, targets, mask_id=None):
  """Mean of the inputs but counting only those where targets != mask_id."""
  inputs = [x.astype(np.float32) for x in inputs]
  # We assume all elements in the list contribute equally.
  # TODO(lukaszkaiser): remove this assumption (e.g., when masks differ).
  length = len(inputs)
  if mask_id is None:
    # TODO(lukaszkaiser): can we just divide the sum by length? XLA optimizes?
    return sum([np.mean(x) / length for x in inputs])
  unmask = [1.0 - np.equal(t, mask_id).astype(np.float32) for t in targets]
  return sum([np.sum(x * m) / (length * np.sum(m))
              for x, m in zip(inputs, unmask)])


def accuracy(batch, model_predictions):
  """Calculate accuracy."""
  _, targets = batch
  model_predictions, targets = _make_list(model_predictions, targets)
  correct = []
  for (prediction, target) in zip(model_predictions, targets):
    predicted_class = np.argmax(prediction, axis=-1)
    correct.append(np.equal(predicted_class, target))
  return masked_mean(correct, targets)


def neg_log_perplexity(batch, model_predictions):
  """Calculate negative log perplexity."""
  _, targets = batch
  model_predictions, targets = _make_list(model_predictions, targets)
  xent = []
  for (prediction, target) in zip(model_predictions, targets):
    hot_target = layers.one_hot(target, prediction.shape[-1])
    xent.append(np.sum(prediction * hot_target, axis=-1))
  return masked_mean(xent, targets)


def loss(params, batch, model_predict, rng):
  """Calculate loss."""
  inputs, targets = batch
  predictions = model_predict(inputs, params, rng=rng)
  predictions, targets = _make_list(predictions, targets)
  xent = []
  for (pred, target) in zip(predictions, targets):
    xent.append(np.sum(pred * layers.one_hot(target, pred.shape[-1]), axis=-1))
  return - masked_mean(xent, targets)


def log(s, stdout=True):
  logging.info(s)
  if stdout:
    print(s)
    sys.stdout.flush()


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


def save_state(state, output_dir, keep=False):
  """Save State and optionally gin config."""
  params_file = os.path.join(output_dir, "model.pkl")
  with gfile.GFile(params_file, "wb") as f:
    pickle.dump((state.params, state.step, state.history), f)
  if keep:
    params_file = os.path.join(output_dir, "model_{}.pkl".format(state.step))
    with gfile.GFile(params_file, "wb") as f:
      pickle.dump((state.params, state.step, state.history), f)
  log("Model saved to %s" % params_file, stdout=False)


# Metrics to calculate and report.
_METRICS = {
    "accuracy": accuracy,
    "neg_log_perplexity": neg_log_perplexity,
    "loss": lambda x, y: - neg_log_perplexity(x, y),
}


def evaluate_train_and_eval(step, inputs, predict_fun, eval_steps, rng,
                            train_sw=None, eval_sw=None, history=None):
  """Evalaute on train and eval data, and log metrics."""
  step_log(step, "Evaluation")
  train_metrics, eval_metrics = [
      evaluate(  # pylint: disable=g-complex-comprehension
          itertools.islice(input_stream(), eval_steps),
          predict_fun,
          _METRICS,
          rng)
      for input_stream in
      [inputs.train_eval_stream, inputs.eval_stream]]
  if train_sw:
    log_metrics(train_metrics, train_sw, "train", step, history=history)
  if eval_sw:
    log_metrics(eval_metrics, eval_sw, "eval", step, history=history)
  step_log(step, "Finished evaluation")
  return train_metrics, eval_metrics


def evaluate(inputs_stream, predict_fun, metric_funs, rng):
  """Evaluate.

  Args:
    inputs_stream: iterable of inputs to evaluate on.
    predict_fun: function from inputs to predictions. params should already be
      partially applied.
    metric_funs: dict from metric name to metric function, which takes inputs
      and predictions and returns a scalar metric value.
    rng: random number generator.

  Returns:
    metrics: dict from metric name to metric value averaged over the number of
      inputs.
  """
  metrics = collections.defaultdict(float)
  count = 0
  for inp in inputs_stream:
    count += 1
    rng, subrng = jax_random.split(rng)
    preds = predict_fun(inp[0], rng=subrng)
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


def get_random_number_generator_and_set_seed(seed=None):
  """Get a JAX random number generator and set random seed everywhere."""
  random.seed(seed)
  # While python random accepts None as seed and uses time/os seed then,
  # some other functions expect integers so we create one here.
  if seed is None:
    seed = random.randint(0, 2**31 - 1)
  tf.set_random_seed(seed)
  numpy.random.seed(seed)
  return jax_random.get_prng(seed)


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


def _jit_predict_fun(model_predict, num_devices):
  """Use jit on model_predict if required."""
  def predict(x, params=(), rng=None):
    """Predict function jited and parallelized as requested."""
    # On one device, jit and run.
    if num_devices == 1:
      return backend.jit(model_predict)(x, params, rng=rng)

    # Multi-devices, pmap and run.
    @functools.partial(backend.pmap, axis_name="batch")
    def mapped_predict(x, params, rng):
      return model_predict(x, params, rng=rng)
    pred = mapped_predict(
        reshape_by_device(x, num_devices),
        params,
        jax_random.split(rng, num_devices))
    # Need to reduce the [device, per-device-batch, ...] tensors back to
    # a [batch, ...] tensor. The tensors may be nested.
    if not isinstance(x, (list, tuple)):  # Not nested.
      batch_size = x.shape[0]
      return np.reshape(pred, [batch_size] + list(pred.shape[2:]))
    batch_size = x[0].shape[0]
    return [np.reshape(p, [batch_size] + list(p.shape[2:])) for p in pred]

  return predict


def _jit_update_fun(predict_fun, loss_fun, optimizer, lr_fun, num_devices):
  """Get jit-ed update function for loss, optimizer, learning rate function."""
  if num_devices == 1:  # TODO(lukaszkaiser): remove branch when not needed.
    def single_update(i, opt_state, batch, rng):
      rng, subrng = jax_random.split(rng[0])
      _, opt_update = optimizer(lr_fun)
      params = trax_opt.get_params(opt_state)
      return opt_update(i, backend.grad(loss_fun)(
          params, batch, predict_fun, rng), opt_state), [subrng]
    return backend.jit(single_update)

  @functools.partial(backend.pmap, axis_name="batch")
  def mapped_update(i, opt_state, batch, rng):
    """This is a multi-device version of the update function above."""
    # We assume all tensors have the first dimension = num_devices.
    rng, subrng = jax_random.split(rng)
    _, opt_update = optimizer(lr_fun)
    params = trax_opt.get_params(opt_state)
    grads = backend.grad(loss_fun)(params, batch, predict_fun, rng)
    grads = jax.tree_util.tree_map(
        lambda g: lax.psum(g, "batch"), grads)
    return opt_update(i, grads, opt_state), subrng

  def update(i, opt_state, batch, rng):
    return mapped_update(jax.replicate(i), opt_state, batch, rng)

  return update


def _reshape_by_device_single(x, num_devices):
  """Reshape x into a shape [num_devices, ...]."""
  x_shape = list(x.shape)
  batch_size = x_shape[0]
  batch_size_per_device = batch_size // num_devices
  # We require that num_devices divides batch_size evenly.
  if batch_size_per_device * num_devices != batch_size:
    logging.fatal(
        "We require that num_devices[%d] divides batch_size[%d] evenly.",
        num_devices, batch_size)
  # New shape.
  new_shape_prefix = [num_devices, batch_size_per_device]
  return np.reshape(x, new_shape_prefix + x_shape[1:])


def reshape_by_device(x, num_devices):
  """Reshape possibly nested x into a shape [num_devices, ...]."""
  return layers.nested_map(
      x, lambda x: _reshape_by_device_single(x, num_devices))


@gin.configurable(blacklist=["output_dir"])
def train(output_dir,
          model=gin.REQUIRED,
          loss_fun=loss,
          inputs=trax_inputs.inputs,
          optimizer=trax_opt.adam,
          lr_schedule=lr.MultifactorSchedule,
          train_steps=1000,
          save_steps=None,
          eval_steps=10,
          eval_frequency=100,
          num_devices=None,
          random_seed=None,
          run_debug_step=False,
          save_forward_graph=False):
  """Train the model on the inputs.

  Args:
    output_dir: Directory where to put the logs and checkpoints.
    model: The model to train as a callable returning 2 callables, an init_fun
      and apply_fun.
    loss_fun: callable with signature: params, trax.inputs.Inputs, model, rng
      -> loss.
    inputs: callable returning trax.inputs.Inputs.
    optimizer: The optimizer as a callable taking a learning_rate callable and
      returning 2 callables, opt_init and opt_update.
    lr_schedule: A learning rate schedule as a function that takes history and
      returns a function from step to learning rate (a float).
    train_steps: int, total number of training steps.
    save_steps: list of integers. Keep a model file at each of the supplied save
      steps.
    eval_steps: int, num of steps per evaluation. If None or 0, eval disabled.
    eval_frequency: int, how often to run evaluation (every eval_frequency
      steps). If None or 0, eval disabled.
    num_devices: how many devices to use (if None, default, use all available)
    random_seed: the random seed to use; time/os dependent if None (default).
    run_debug_step: bool, if True, will run the model and loss without @jit for
      one step.
    save_forward_graph: bool, if True, save forward computation graph to file.

  Returns:
    trax.State
  """
  if save_steps is None:
    save_steps = []
  num_devices = num_devices or jax.lib.xla_bridge.device_count()
  rng = get_random_number_generator_and_set_seed(random_seed)
  gfile.makedirs(output_dir)
  # Create summary writers and history.
  train_sw = jaxboard.SummaryWriter(os.path.join(output_dir, "train"))
  eval_sw = jaxboard.SummaryWriter(os.path.join(output_dir, "eval"))

  inputs = inputs(num_devices)

  # Setup optimizer and model
  state = restore_state(output_dir)
  history = state.history
  lr_fun = lr_schedule(history)
  opt_init, _ = optimizer(lr_fun)
  model_train = model(mode="train")
  model_predict_eval = model(mode="eval")

  # Setup state
  step = state.step or 0
  rng, init_rng = jax_random.split(rng)
  rngs = jax_random.split(rng, num_devices)
  first_shape = inputs.input_shape[0]
  # If the inputs are a tuple/list, add [-1] (batch) to each element.
  if isinstance(first_shape, (list, tuple)):
    model_input_shape = tuple(
        [tuple([-1] + list(shape)) for shape in inputs.input_shape])
  else:  # Otherwise just add [-1] to the input shape.
    model_input_shape = tuple([-1] + list(inputs.input_shape))
  params = state.params or model_train.initialize(model_input_shape, init_rng)
  opt_state = opt_init(params)
  if num_devices > 1:  # TODO(lukaszkaiser): use everywhere when pmap is stable.
    opt_state = jax.replicate(opt_state)

  # jit model_predict and update so they're fast
  jit_model_predict_eval = _jit_predict_fun(model_predict_eval, num_devices)
  jit_update_fun = _jit_update_fun(
      model_train, loss_fun, optimizer, lr_fun, num_devices)

  train_stream = inputs.train_stream()
  epoch_steps = [train_steps]  # Only training if eval_frequency is 0 or None.
  if eval_frequency and eval_steps > 0:
    epoch_steps = itertools.chain([1,  # first epoch only 1 step
                                   eval_frequency - 1],
                                  itertools.repeat(eval_frequency))
  step_log(step, "Starting training using %d devices" % num_devices)

  # Non-compiled debug step helps find problems in models easier.
  if run_debug_step:
    debug_loss = loss_fun(params, next(train_stream), model_train, rng)
    step_log(step, "Debug step loss %.8f" % debug_loss)

  for epoch, epoch_steps in epochs(train_steps, epoch_steps):
    # Log separator
    print()

    # Timer
    start_time = time.time()

    for _ in range(epoch_steps):
      # Train
      next_train_batch = next(train_stream)
      if num_devices > 1:  # TODO(lukaszkaiser): use everywhere when possible.
        next_train_batch = reshape_by_device(next_train_batch, num_devices)
      opt_state, rngs = jit_update_fun(step, opt_state, next_train_batch, rngs)
      step += 1

      if step in save_steps:
        save_state(State(params=params, step=step, history=history),
                   output_dir,
                   keep=True)

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

    # Print number of parameters
    params = trax_opt.get_params(opt_state)
    if step == 1:
      sizes = layers.sizes(params)
      total_size = layers.nested_reduce(sizes, sum)
      step_log(step, "Total trainable parameters size: %d" % total_size)

    # Evaluate
    evaluate_train_and_eval(
        step=step,
        inputs=inputs,
        predict_fun=functools.partial(jit_model_predict_eval, params=params),
        eval_steps=eval_steps,
        rng=rng,
        train_sw=train_sw,
        eval_sw=eval_sw,
        history=history)

    # Save computation graph
    if save_forward_graph and step == 1:
      # Dump forward computation graph to file.
      computation = jax.xla_computation(model_predict_eval)(
          next_train_batch[0], params=params, rng=rng)
      with gfile.GFile(os.path.join(output_dir, "forward_graph.dot"), "w") as f:
        f.write(computation.GetHloDotGraph())

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
      jit_update_fun = _jit_update_fun(
          model_train, loss_fun, optimizer, lr_fun, num_devices)

    # Flush summary writers
    train_sw.flush()
    eval_sw.flush()

  step_log(step, "Training done")
  return State(params=params, step=step, history=history)
