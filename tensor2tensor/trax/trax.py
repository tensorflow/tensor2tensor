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

"""Trax main training functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import itertools
import os
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
from tensor2tensor.trax import utils
from tensor2tensor.trax.backend import numpy as np
from tensor2tensor.trax.backend import random as jax_random

import tensorflow as tf
from tensorflow.io import gfile


def unpack_batch(batch, has_weights):
  """Unpacks a training batch into inputs, targets and weights."""
  if has_weights:
    assert len(batch) == 3  # (inputs, targets, weights)
    return batch
  else:
    inputs, targets = batch
    if isinstance(targets, (list, tuple)):
      # If weights are not provided, use scalar 1s and rely on broadcasting.
      weights = [1.0] * len(targets)
    else:
      weights = 1.0
    return inputs, targets, weights


def _make_list(predictions, targets, weights):
  """Make predictions, targets and weights lists, check they match on length."""
  #  Our models sometimes return predictions in lists, make it a list always.
  # TODO(lukaszkaiser): make abstractions for nested structures and refactor.
  if not isinstance(predictions, (list, tuple)):
    if isinstance(targets, (list, tuple)):
      raise ValueError("Targets are a list or tuple but predictions are not.")
    if isinstance(weights, (list, tuple)):
      raise ValueError("Weights are a list or tuple but predictions are not.")
    predictions, targets, weights = [predictions], [targets], [weights]
  if len(predictions) != len(targets):
    raise ValueError("Predictions and targets have different lengths.")
  if len(predictions) != len(weights):
    raise ValueError("Predictions and weights have different lengths.")
  return list(predictions), list(targets), list(weights)


@gin.configurable(blacklist=["inputs", "targets", "weights"])
def masked_mean(inputs, targets, weights, mask_id=None):
  """Weighted mean of the inputs, excluding where targets == mask_id."""
  inputs = [x.astype(np.float32) for x in inputs]
  # We assume all elements in the list contribute equally.
  # TODO(lukaszkaiser): remove this assumption (e.g., when masks differ).
  length = len(inputs)
  if mask_id is not None:
    weights = [w * (1.0 - np.equal(t, mask_id).astype(np.float32))
               for t, w in zip(targets, weights)]
  weight_sums = [np.float32(t.size) if np.isscalar(w) else np.sum(w)
                 for w, t in zip(weights, targets)]
  return sum([np.sum(x * w) / (length * s)
              for x, w, s in zip(inputs, weights, weight_sums)])


def accuracy(batch, model_predictions, has_weights):
  """Calculate accuracy."""
  _, targets, weights = unpack_batch(batch, has_weights)
  model_predictions, targets, weights = _make_list(
      model_predictions, targets, weights)
  correct = []
  for (prediction, target) in zip(model_predictions, targets):
    predicted_class = np.argmax(prediction, axis=-1)
    correct.append(np.equal(predicted_class, target))
  return masked_mean(correct, targets, weights)


def neg_log_perplexity(batch, model_predictions, has_weights):
  """Calculate negative log perplexity."""
  _, targets, weights = unpack_batch(batch, has_weights)
  model_predictions, targets, weights = _make_list(
      model_predictions, targets, weights)
  xent = []
  for (prediction, target) in zip(model_predictions, targets):
    hot_target = layers.one_hot(target, prediction.shape[-1])
    xent.append(np.sum(prediction * hot_target, axis=-1))
  return masked_mean(xent, targets, weights)


def _stack_inputs_targets_and_get_predictions(inputs_and_targets):
  """Helper to stack inputs and targets and retrieve predictions from output."""
  # Inputs and targets can be lists - we build a flat one to input to the model.
  model_inp = []
  for x in inputs_and_targets:
    if not isinstance(x, (list, tuple)):
      model_inp.append(x)
    else:
      model_inp.extend(x)
  # We retrieve as many predictions from model output as many there were inputs.
  inp = inputs_and_targets[0]
  inp_len = len(inp) if isinstance(inp, (list, tuple)) else 1
  get_pred = lambda x: x[0] if inp_len == 1 else x[:inp_len]
  return tuple(model_inp), get_pred


def loss(params, batch, model_predict, state, rng, has_weights):
  """Calculate loss."""
  inputs, targets, weights = unpack_batch(batch, has_weights)
  model_input, get_preds = _stack_inputs_targets_and_get_predictions(
      [inputs, targets])
  # Call model, predictions will be the returned stack, usually consisting of
  # the prediction tensor and the targets.
  predictions, state = model_predict(model_input, params, state, rng=rng)
  predictions = get_preds(predictions)
  predictions, targets, weights = _make_list(predictions, targets, weights)
  xent = []
  for (pred, target) in zip(predictions, targets):
    xent.append(np.sum(pred * layers.one_hot(target, pred.shape[-1]), axis=-1))
  return - masked_mean(xent, targets, weights), state


def log(s, stdout=True):
  logging.info(s)
  if stdout:
    print(s)
    sys.stdout.flush()


def step_log(step, s):
  log("Step % 6d: %s" % (step, s))


State = collections.namedtuple("_State", [
    "step",       # Current training step number.
    "opt_state",  # OptState.
    "history",    # trax.history.History.
    "model_state",
])


OptState = collections.namedtuple("_OptState", [
    "params",      # Model parameters.
    "slots",       # Per-parameter optimizer state, e.g. gradient moments.
    "opt_params",  # Optimizer (hyper)parameters, e.g. learning rate, momentum.
])


def restore_state(output_dir):
  """Restore State."""
  params_file = os.path.join(output_dir, "model.pkl")
  if not gfile.exists(params_file):
    return State(step=None, opt_state=None, history=trax_history.History(),
                 model_state=None)

  pkl_module = utils.get_pickle_module()
  with gfile.GFile(params_file, "rb") as f:
    (opt_state, step, history, model_state) = pkl_module.load(f)
  log("Model loaded from %s at step %d" % (params_file, step))
  logging.debug("From loaded model : history = %s", history)
  return State(step=step, opt_state=OptState(*opt_state), history=history,
               model_state=model_state)


def _save_gin(output_dir, sw=None):
  config_path = os.path.join(output_dir, "config.gin")
  config_str = gin.operative_config_str()
  with gfile.GFile(config_path, "w") as f:
    f.write(config_str)
  if sw:
    sw.text("gin_config",
            jaxboard.markdownify_operative_config_str(config_str))


def save_state(state, output_dir, keep=False):
  """Save State and optionally gin config."""
  pkl_module = utils.get_pickle_module()
  params_file = os.path.join(output_dir, "model.pkl")
  with gfile.GFile(params_file, "wb") as f:
    pkl_module.dump((tuple(state.opt_state), state.step, state.history,
                     state.model_state), f)
  if keep:
    params_file = os.path.join(output_dir, "model_{}.pkl".format(state.step))
    with gfile.GFile(params_file, "wb") as f:
      pkl_module.dump((tuple(state.opt_state), state.step, state.history,
                       state.model_state), f)
  log("Model saved to %s" % params_file, stdout=False)


def _save_replicated(opt_state, step, history, model_state, n_devices,
                     output_dir, keep):
  """Save state but given a possibly replicated opt_state."""
  if n_devices > 1:
    first_replica = lambda x: x[0]
    opt_state = OptState(*layers.nested_map(opt_state, first_replica))
  # This line, while optional, allows JAX to transfer arrays from the device to
  # the host in parallel, which is particularly important for cloud TPU.
  if backend.get_name() == "jax":
    opt_state = jax.device_get(opt_state)
  save_state(State(opt_state=opt_state, step=step, history=history,
                   model_state=model_state), output_dir, keep=keep)


def _print_n_params(opt_state, n_devices, step):
  """Print out the number of parameters."""
  sizes = layers.sizes(opt_state.params)
  if n_devices > 1:
    unreplicate = lambda x: x[0]
    single_params = layers.nested_map(opt_state.params, unreplicate)
    sizes = layers.sizes(single_params)
  total_size = layers.nested_reduce(sizes, sum)
  step_log(step, "Total trainable parameters size: %d" % total_size)


# Metrics to calculate and report.
_METRICS = {
    "accuracy": accuracy,
    "neg_log_perplexity": neg_log_perplexity,
    "loss": lambda *args, **kwargs: - neg_log_perplexity(*args, **kwargs),
}


def evaluate_train_and_eval(step, eval_stream, train_eval_stream,
                            predict_fn, eval_steps, state, rng, has_weights,
                            train_sw=None, eval_sw=None, history=None):
  """Evalaute on train and eval data, and log metrics."""
  step_log(step, "Evaluation")
  metrics_list = []
  for input_stream in [train_eval_stream, eval_stream]:
    metrics, state = evaluate(  # pylint: disable=g-complex-comprehension
        itertools.islice(input_stream, eval_steps),
        predict_fn,
        _METRICS,
        state,
        rng,
        has_weights)
    metrics_list.append(metrics)
  # Unpack in the same order we've iterated over streams in the loop above.
  train_metrics, eval_metrics = metrics_list  # pylint: disable=unbalanced-tuple-unpacking
  if train_sw:
    log_metrics(train_metrics, train_sw, "train", step, history=history)
  if eval_sw:
    log_metrics(eval_metrics, eval_sw, "eval", step, history=history)
  step_log(step, "Finished evaluation")
  return train_metrics, eval_metrics, state


def evaluate(inputs_stream, predict_fn, metric_fns, state, rng, has_weights):
  """Evaluate.

  Args:
    inputs_stream: iterable of inputs to evaluate on.
    predict_fn: function from inputs to predictions. params should already be
      partially applied.
    metric_fns: dict from metric name to metric function, which takes inputs
      and predictions and returns a scalar metric value.
    state: start state for `predict_fn`.
    rng: random number generator.
    has_weights: bool, whether weights are included in the inputs.

  Returns:
    metrics: dict from metric name to metric value averaged over the number of
      inputs.
    state: end state for `predict_fn`.
  """
  metrics = collections.defaultdict(float)
  count = 0
  for inp in inputs_stream:
    count += 1
    rng, subrng = jax_random.split(rng)
    model_inp, get_preds = _stack_inputs_targets_and_get_predictions(inp)
    # Call model, preds will be the returned stack, usually (pred, targets).
    preds, state = predict_fn(model_inp, state=state, rng=subrng)
    pred = get_preds(preds)
    for m, f in six.iteritems(metric_fns):
      metrics[m] += f(inp, pred, has_weights=has_weights)
  return {m: v / count for (m, v) in six.iteritems(metrics)}, state


def evaluate_loss_train_and_eval(step, eval_stream, train_eval_stream,
                                 compute_loss_fn, eval_steps,
                                 state, rngs, has_weights,
                                 train_sw=None, eval_sw=None, history=None):
  """More efficient evaluation that logs only the loss on train & eval data."""
  assert not has_weights, (
      "MemoryEfficientTrainer doesn't support has_weights")
  step_log(step, "Evaluation")
  train_eval_metrics = []
  for input_stream in [train_eval_stream, eval_stream]:
    total = 0.0
    count = 0.0
    for inp in itertools.islice(input_stream, eval_steps):
      loss_values, state, rngs = compute_loss_fn(inp, state, rngs)
      total += float(numpy.mean(loss_values))
      count += 1.0
    metrics = {"loss": total / count}
    train_eval_metrics.append(metrics)
  # Unpack in the same order we've iterated over streams in the loop above.
  train_metrics, eval_metrics = train_eval_metrics  # pylint: disable=unbalanced-tuple-unpacking
  if train_sw:
    log_metrics(train_metrics, train_sw, "train", step, history=history)
  if eval_sw:
    log_metrics(eval_metrics, eval_sw, "eval", step, history=history)
  step_log(step, "Finished evaluation")
  return train_metrics, eval_metrics, state


def log_metrics(metrics, summ_writer, log_prefix, step, history=None):
  """Log metrics to summary writer and history."""
  rjust_len = max([0] + [len(name) for name in metrics])
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


def epochs(total_steps, steps_to_skip, epoch_steps):
  """Generates the number of steps in each epoch before reaching total_steps.

  Args:
    total_steps: int, total number of steps.
    steps_to_skip: int, number of steps to skip because of a restart.
    epoch_steps: iterable of int, numbers of steps in each epoch.

  Yields:
    epoch_steps: int, number of steps in this epoch
  """
  steps_to_go = total_steps - steps_to_skip
  epoch_steps = iter(epoch_steps)

  # Remove the desired number of steps from the stream.
  for steps_this_epoch in epoch_steps:
    if steps_this_epoch > steps_to_skip:
      # Put back the number of steps left in the unfinished epoch.
      epoch_steps = itertools.chain(
          [steps_this_epoch - steps_to_skip], epoch_steps)
    if steps_this_epoch >= steps_to_skip:
      break
    steps_to_skip -= steps_this_epoch

  # Yield the remaining steps per epoch up to total_steps.
  for steps_this_epoch in epoch_steps:
    steps_this_epoch = min(steps_this_epoch, steps_to_go)
    yield steps_this_epoch
    steps_to_go -= steps_this_epoch
    if steps_to_go == 0:
      break


@gin.configurable
def _jit_predict_fn(model_predict, n_devices, jit=True):
  """Returns a JIT-compiled predict function (unless jit=False)."""

  if n_devices == 1:
    return backend.jit(model_predict) if jit else model_predict

  # Multi-devices, pmap and run.
  @functools.partial(backend.pmap, axis_name="batch")
  def mapped_predict(x, params, state, rng):
    return model_predict(x, params, state, rng=rng)

  def predict(x, params=(), state=(), rng=None):
    """Predict function jited and parallelized as requested."""
    pred, state = mapped_predict(
        reshape_by_device(x, n_devices),
        params,
        state,
        jax_random.split(rng, n_devices))
    # Need to reduce the [device, per-device-batch, ...] tensors back to
    # a [batch, ...] tensor. The tensors may be nested.
    def combine(x):
      batch_size = x.shape[0] * x.shape[1]
      return np.reshape(x, [batch_size] + list(x.shape[2:]))
    return layers.nested_map(pred, combine), state

  return predict


@gin.configurable
def _jit_update_fn(predict_fn, loss_fn, optimizer, n_devices, jit=True):
  """Returns a (JIT-compiled) function that computes updates for one step."""
  if n_devices == 1:  # TODO(lukaszkaiser): remove branch when not needed.
    def single_update(i, opt_state, batch, state, rng):
      params, slots, opt_params = opt_state
      rng, subrng = jax_random.split(rng[0])
      grad_fn = backend.grad(loss_fn, has_aux=True)
      grads, state = grad_fn(params, batch, predict_fn, state, rng)
      return optimizer.tree_update(
          i, grads, params, slots, opt_params), state, [subrng]
    return backend.jit(single_update) if jit else single_update

  # Else, for n_devices > 1:
  @functools.partial(backend.pmap, axis_name="batch")
  def mapped_update(i, opt_state, batch, state, rng):
    """This is a multi-device version of the update function above."""
    # We assume all tensors have the first dimension = n_devices.
    params, slots, opt_params = opt_state
    rng, subrng = jax_random.split(rng)
    grad_fn = backend.grad(loss_fn, has_aux=True)
    grads, state = grad_fn(params, batch, predict_fn, state, rng)
    grads = jax.tree_util.tree_map(
        lambda g: lax.psum(g, "batch"), grads)
    return optimizer.tree_update(
        i, grads, params, slots, opt_params), state, subrng

  def update(i, opt_state, batch, state, rng):
    return mapped_update(numpy.repeat(i, n_devices), opt_state, batch, state,
                         rng)

  return update


@gin.configurable
def _jit_compute_loss_fn(predict_fn, loss_fn, n_devices, jit=True):
  """Returns a (JIT-compiled) function that computes the loss for one step."""
  if n_devices == 1:  # TODO(lukaszkaiser): remove branch when not needed.
    def single_compute_loss(opt_state, batch, state, rng):
      rng, subrng = jax_random.split(rng[0])
      loss_val, state = loss_fn(opt_state[0], batch, predict_fn, state, rng)
      return loss_val, state, [subrng]
    return backend.jit(single_compute_loss) if jit else single_compute_loss

  # Else, for n_devices > 1:
  @functools.partial(backend.pmap, axis_name="batch")
  def mapped_compute_loss(opt_state, batch, state, rng):
    """This is a multi-device version of the update function above."""
    # We assume all tensors have the first dimension = n_devices.
    rng, subrng = jax_random.split(rng)
    loss_val, state = loss_fn(opt_state[0], batch, predict_fn, state, rng)
    return loss_val, state, subrng

  def compute_loss(opt_state, batch, state, rng):
    return mapped_compute_loss(
        opt_state, reshape_by_device(batch, n_devices), state, rng)

  return compute_loss


@gin.configurable
def _is_jit_init(value=True):
  return value


def _reshape_by_device_single(x, n_devices):
  """Reshape x into a shape [n_devices, ...]."""
  x_shape = list(x.shape)
  batch_size = x_shape[0]
  batch_size_per_device = batch_size // n_devices
  # We require that n_devices divides batch_size evenly.
  if batch_size_per_device * n_devices != batch_size:
    logging.fatal(
        "We require that n_devices[%d] divides batch_size[%d] evenly.",
        n_devices, batch_size)
  # New shape.
  new_shape_prefix = [n_devices, batch_size_per_device]
  return np.reshape(x, new_shape_prefix + x_shape[1:])


def reshape_by_device(x, n_devices):
  """Reshape possibly nested x into a shape [n_devices, ...]."""
  return layers.nested_map(
      x, lambda x: _reshape_by_device_single(x, n_devices))


def multi_device_put(x, devices=None, reuse=True):
  """Memory efficient multi-device replication in JAX.

  Args:
    x: jax DeviceArray or numpy ndarray to be replicated.
    devices: a jax.devices() list or subset thereof of devices to
      replicate onto.  Should match the list passed to any pmaps
      ingesting the replicated array.
    reuse: bool. If x is a DeviceArray whether to reuse its backing
      device_buffer in the resulting ShardedDeviceArray.

  Returns:
    A ShardedDeviceArray with dtype = x.dtype and shape =
    (n_devices,) + x.shape that's backed by replica
    device_buffers on each device.
  """
  # Convert _FilledConstants that don't have device_buffer, etc.
  if type(x) != jax.xla.DeviceArray:  # pylint: disable=unidiomatic-typecheck
    x = np.array(x)
  if not devices:
    devices = jax.devices()
  n_devices = len(devices)
  x_aval = jax.xla.abstractify(x)
  broadcast_x_aval = jax.abstract_arrays.ShapedArray(
      (n_devices,) + x_aval.shape,
      x_aval.dtype)
  if reuse:
    other_device_ordinals = [dv.id for dv in jax.devices()
                             if dv != x.device_buffer.device()]
    broadcast_buffers = ([x.device_buffer,] +
                         [jax.xla.xc.Buffer.from_pyval(x, device=i)
                          for i in other_device_ordinals])
  else:
    broadcast_buffers = [jax.xla.xc.Buffer.from_pyval(x, device=i)
                         for i in range(n_devices)]
  return jax.pxla.ShardedDeviceArray(broadcast_x_aval, broadcast_buffers)


def _repeat_stream(stream):
  """Repeat a stream indefinitely."""
  while True:
    for example in stream():
      yield example


@gin.configurable(whitelist=[])
class Trainer(object):
  """Trax trainer.

  A trainer allows to make training steps, train for full epochs,
  save the training state and access evaluation data.
  """

  def __init__(self, model, loss_fn, optimizer, lr_schedule, inputs,
               output_dir=None, random_seed=None, n_devices=None,
               save_steps=None, should_save=True, has_weights=False,
               nontrainable_param_map=None):
    if save_steps is None:
      save_steps = []
    self._save_steps = save_steps
    self._should_save = should_save
    self._has_weights = has_weights
    loss_fn = functools.partial(loss_fn, has_weights=self._has_weights)
    device_count = jax.lib.xla_bridge.device_count()
    n_devices = n_devices or device_count
    # TODO(lukaszkaiser): remove this restriction when possible.
    if n_devices != device_count:
      raise ValueError("Jax cannot work yet with n_devices != all devices: "
                       "%d != %d" % (n_devices, device_count))
    self._n_devices = n_devices
    rng = get_random_number_generator_and_set_seed(random_seed)
    inputs = inputs(n_devices)
    self._inputs = inputs

    # Initialize the learning rate to a dummy value. It will be set in reset().
    opt = optimizer(learning_rate=0.0)

    # Setup the model.
    model_train = model(mode="train")
    model_predict_eval = model(mode="eval")

    # Setup state.
    rng, init_rng = jax_random.split(rng)
    self._rngs = jax_random.split(rng, n_devices)
    first_shape = inputs.input_shape[0]
    # If the inputs are a tuple/list, add [None] (batch) to each element.
    if isinstance(first_shape, (list, tuple)):
      model_input_shape = tuple(
          tuple([None] + list(shape)) for shape in inputs.input_shape)
      model_target_shape = tuple(
          tuple([None] + list(shape)) for shape in inputs.target_shape)
    else:  # Otherwise just add [None] to the input shape.
      model_input_shape = tuple([None] + list(inputs.input_shape))
      model_target_shape = tuple([None] + list(inputs.target_shape))
    # Change all None to 1 in input and target shape.
    model_input_shape = layers.nested_map(
        model_input_shape, lambda x: x if x else 1)
    model_target_shape = layers.nested_map(
        model_target_shape, lambda x: x if x else 1)
    def initialize(input_shape, input_dtype, target_shape, target_dtype, rng):
      """Helper to initialize the model."""
      # Combine inputs and targets on the stack.
      if not isinstance(input_dtype, (list, tuple)):
        input_dtype = [input_dtype]
        input_shape = [input_shape]
      if not isinstance(target_dtype, (list, tuple)):
        target_dtype = [target_dtype]
        target_shape = [target_shape]
      full_type = list(input_dtype) + list(target_dtype)
      full_shape = list(input_shape) + list(target_shape)
      # We need to create a new model instance and not reuse `model_train` here,
      # because `m.initialize` puts cached parameter values in `m` and hence the
      # next call of `m.initialize` will give wrong results.
      params, state = model(mode="train").initialize(full_shape, full_type, rng)
      (slots, opt_params) = opt.tree_init(params)
      return (OptState(params, slots, opt_params), state)
    if _is_jit_init():
      # JIT parameter initialization to avoid memory fragmentation
      initialize = backend.jit(initialize, static_argnums=(0, 1, 2, 3))
    self._initialize = lambda: initialize(  # pylint: disable=g-long-lambda
        model_input_shape, self._inputs.input_dtype,
        model_target_shape, self._inputs.target_dtype, init_rng)

    # jit model_predict and update so they're fast
    self._jit_model_predict_eval = _jit_predict_fn(
        model_predict_eval, n_devices)
    self._jit_update_fn = _jit_update_fn(model_train, loss_fn, opt, n_devices)

    self._model_train = model_train
    self._model_predict_eval = model_predict_eval
    self._loss_fn = loss_fn
    # TODO(pkozakowski): "Learning rate schedules" are currently able to control
    # control all optimizer parameters and model state, so let's rename them
    # accordingly.
    self._lr_schedule = lr_schedule

    if nontrainable_param_map is None:
      nontrainable_param_map = {}
    self._nontrainable_param_map = nontrainable_param_map

    # Those fields will be set in reset().
    self._output_dir = None
    self._train_sw = None
    self._eval_sw = None
    self._history = None
    self._lr_fn = None
    self._opt_state = None
    self._step = None
    self._model_state = None

    if output_dir is not None:
      self.reset(output_dir)

  def reset(self, output_dir):
    """Reset the model parameters.

    Restores the parameters from the given output_dir if a checkpoint exists,
    otherwise randomly initializes them.

    Does not re-jit the model.

    Args:
      output_dir: Output directory.
    """
    self._output_dir = output_dir
    gfile.makedirs(output_dir)
    # Create summary writers and history.
    self._train_sw = jaxboard.SummaryWriter(os.path.join(output_dir, "train"))
    self._eval_sw = jaxboard.SummaryWriter(os.path.join(output_dir, "eval"))

    # Reset the train and eval streams.
    self._train_stream = self._inputs.train_stream()
    # TODO(lukaszkaiser): add an option to evaluate exactly on the full eval
    #   set by adding a padding and stopping the stream when too large.
    self._eval_stream = _repeat_stream(self._inputs.eval_stream)
    self._train_eval_stream = _repeat_stream(self._inputs.train_eval_stream)

    # Restore the training state.
    state = restore_state(output_dir)
    self._step = state.step or 0
    history = state.history
    self._lr_fn = self._lr_schedule(history)
    self._history = history
    if state.opt_state:
      opt_state = state.opt_state
      model_state = state.model_state
    else:
      opt_state, model_state = self._initialize()
      model_state = layers.nested_map(
          model_state, self._maybe_replicate)
    self._opt_state = OptState(*layers.nested_map(
        opt_state, self._maybe_replicate))
    self._model_state = model_state
    if not state.opt_state:
      self._maybe_save_state(keep=False)

    self.update_nontrainable_params()

  @property
  def step(self):
    return self._step

  @property
  def n_devices(self):
    return self._n_devices

  @property
  def state(self):
    return State(
        opt_state=self._opt_state, step=self._step, history=self._history,
        model_state=self._model_state)

  @property
  def nontrainable_params(self):
    # TODO(lukaszkaiser): it makes no sense to use an accelerator (e.g. TPU)
    # in op-by-op mode just to compute the learning rate. However, there
    # should be a cleaner approach that forceably swapping out the backend.
    with backend.use_backend("numpy"):
      return self._lr_fn(self._step)

  def _maybe_replicate(self, x):
    if self._n_devices > 1:
      if backend.get_name() == "jax":
        return multi_device_put(x)
      else:
        return np.broadcast_to(x, (self._n_devices,) + x.shape)
    else:
      return x

  def _maybe_save_state(self, keep):
    if self._should_save:
      _save_replicated(self._opt_state, self._step, self._history,
                       self._model_state, self._n_devices, self._output_dir,
                       keep)

  def save_gin(self):
    _save_gin(self._output_dir, self._train_sw)

  def print_n_params(self):
    _print_n_params(self._opt_state, self._n_devices, self._step)

  def _map_to_state_dicts(self, f):
    """Map the function f to all dicts in model state."""
    def nested_map(x, f):
      if isinstance(x, list):
        return [nested_map(y, f) for y in x]
      if isinstance(x, tuple):
        return tuple([nested_map(y, f) for y in x])
      if isinstance(x, dict) and len(x) == 1:
        return f(x)
      return x
    return nested_map(self._model_state, f)

  def _state_dicts_update(self, state_dict):
    assert len(state_dict.keys()) == 1
    key = list(state_dict.keys())[0]
    value = np.array(state_dict[key])
    return {key: np.array(self.update_model_state(key, value))}

  def update_model_state(self, key, value):
    """Updates model state based on nontrainable_params."""
    # Translate model state keys to nontrainable param names.
    if key in self._nontrainable_param_map:
      param_name = self._nontrainable_param_map[key]
    else:
      # If a key is not in mapping, it stays the same.
      param_name = key
    if param_name in self.nontrainable_params:
      if self._step == 0:
        log("Mapping model state key {} to nontrainable param {}.".format(
            key, param_name
        ))
        return self._maybe_replicate(
            np.array(self.nontrainable_params[param_name])
        )
    return value

  def _train_step(self, next_train_batch):
    """Run one training step and update self._opt_state."""
    # Calculate the current optimizer parameters.
    # TODO(pkozakowski): Optimizer parameters get polluted with model state,
    # which doesn't break anything but is weird. Filter it out.
    opt_param_updates = layers.nested_map(
        self.nontrainable_params, lambda x: self._maybe_replicate(np.array(x))
    )
    opt_state = self._opt_state
    opt_state.opt_params.update(opt_param_updates)

    # Run the update.
    (params, slots), self._model_state, self._rngs = self._jit_update_fn(
        self._step, opt_state, next_train_batch, self._model_state, self._rngs)
    self._model_state = self._map_to_state_dicts(self._state_dicts_update)
    self._opt_state = opt_state._replace(params=params, slots=slots)
    self._step += 1

  def train_epoch(self, epoch_steps, eval_steps):
    """Train for one epoch."""
    # Log separator
    print()

    # Timer
    start_time = time.time()

    for _ in range(epoch_steps):
      # Train
      next_train_batch = next(self._train_stream)
      if self._n_devices > 1:  # TODO(lukaszkaiser): use everywhere if possible.
        next_train_batch = reshape_by_device(next_train_batch, self._n_devices)

      self._train_step(next_train_batch)

      if self._step in self._save_steps:
        self._maybe_save_state(keep=True)

      # Log nontrainable params (learning rate, dropout etc.)
      if self._step == 1 or self._step % 10 == 0:
        for (name, value) in self.nontrainable_params.items():
          self._train_sw.scalar("training/{}".format(name), value)

    # Timer
    epoch_time = time.time() - start_time
    step_log(self._step, "Ran %d train steps in %0.2f secs" %
             (epoch_steps, epoch_time))
    if epoch_steps > 1:
      self._train_sw.scalar("training/steps per second",
                            epoch_steps / epoch_time, step=self._step)

    # Evaluate in parallel
    self.evaluate(eval_steps)

    # Save state
    self._maybe_save_state(keep=False)

    # Flush summary writers
    self._train_sw.flush()
    self._eval_sw.flush()

  def evaluate(self, eval_steps):
    """Evaluate the model and log metrics."""
    _, rng = jax_random.split(self._rngs[0])
    _, _, self._model_state = evaluate_train_and_eval(
        step=self._step,
        eval_stream=self._eval_stream,
        train_eval_stream=self._train_eval_stream,
        predict_fn=functools.partial(self._jit_model_predict_eval,
                                     params=self._opt_state[0]),
        eval_steps=eval_steps,
        state=self._model_state,
        rng=rng,
        train_sw=self._train_sw,
        eval_sw=self._eval_sw,
        history=self._history,
        has_weights=self._has_weights)

    # Save the optimizer params in the history
    for (name, value) in self.nontrainable_params.items():
      self._history.append("train", "training/{}".format(name), self._step,
                           value)

  def update_nontrainable_params(self):
    self._lr_fn = self._lr_schedule(self._history)

  def save_computation_graphs(self, save_backward_graph):
    """Dump computation graphs to files."""
    if self._n_devices != 1:
      return  # TODO(lukaszkaiser): make this work with more devices.
    next_train_batch = next(self._train_stream)
    output_dir = self._output_dir
    if self._n_devices > 1:
      next_train_batch = reshape_by_device(next_train_batch, self._n_devices)
    params = self._opt_state[0]
    forward_computation = jax.xla_computation(self._model_predict_eval)(
        next_train_batch, params=params, state=self._model_state,
        rng=self._rngs[0])
    with gfile.GFile(os.path.join(output_dir, "forward.txt"), "w") as f:
      f.write(forward_computation.GetHloText())
    with gfile.GFile(os.path.join(output_dir, "forward.dot"), "w") as f:
      f.write(forward_computation.GetHloDotGraph())
    backward_computation = jax.xla_computation(self._jit_update_fn)(
        self._step, self._opt_state, next_train_batch, self._model_state,
        self._rngs)
    with gfile.GFile(os.path.join(output_dir, "backward.txt"), "w") as f:
      f.write(backward_computation.GetHloText())
    if save_backward_graph:  # Backward graphs can be large so we guard it.
      with gfile.GFile(os.path.join(output_dir, "backward.dot"), "w") as f:
        f.write(backward_computation.GetHloDotGraph())


@gin.configurable(whitelist=[])
class MemoryEfficientTrainer(Trainer):
  """Trax trainer that aims to minimize memory usage.
  """
  # TODO(kitaev): memory efficiency should be a feature of the main Trainer
  # class, but there's a separate class for now because this trainer only
  # supports evaluating the loss (and not any other metrics).

  def __init__(self, *args, **kwargs):
    super(MemoryEfficientTrainer, self).__init__(*args, **kwargs)
    # Model predictions can use large amounts of memory. The memory-efficient
    # approach is to compute metrics on each replica and then aggregate. For now
    # we only implement computing the loss, and not any other metrics.
    self._jit_compute_loss = _jit_compute_loss_fn(
        self._model_predict_eval, self._loss_fn, self._n_devices)
    assert not self._has_weights, (
        "MemoryEfficientTrainer doesn't support has_weights")

  def evaluate(self, eval_steps):
    # Evaluate only the loss function (a more efficient, jitted, implementation)
    _, _, self._model_state = evaluate_loss_train_and_eval(
        step=self._step,
        eval_stream=self._eval_stream,
        train_eval_stream=self._train_eval_stream,
        compute_loss_fn=functools.partial(self._jit_compute_loss,
                                          self._opt_state),
        eval_steps=eval_steps,
        state=self._model_state,
        rngs=self._rngs,
        has_weights=self._has_weights,
        train_sw=self._train_sw,
        eval_sw=self._eval_sw,
        history=self._history)

  def save_computation_graphs(self, save_backward_graph):
    # TODO(kitaev): implement saving graphs while making sure that no op-by-op
    # execution happens in the process.
    del save_backward_graph
    return


@gin.configurable(blacklist=["output_dir"])
def train(output_dir,
          model=gin.REQUIRED,
          loss_fn=loss,
          inputs=trax_inputs.inputs,
          optimizer=trax_opt.Adafactor,
          lr_schedule=lr.MultifactorSchedule,
          trainer_class=Trainer,
          train_steps=1000,
          save_steps=None,
          eval_steps=10,
          eval_frequency=100,
          n_devices=None,
          random_seed=None,
          save_graphs=True,
          save_backward_graph=False,
          has_weights=False):
  """Train the model on the inputs.

  Args:
    output_dir: Directory where to put the logs and checkpoints.
    model: The model to train as a callable returning 2 callables, an init_fn
      and apply_fn.
    loss_fn: callable with signature: params, trax.inputs.Inputs, model, state,
      rng -> loss.
    inputs: callable returning trax.inputs.Inputs.
    optimizer: The optimizer (see optimizers/base.py for signature).
    lr_schedule: A learning rate schedule as a function that takes history and
      returns a function from step to learning rate (a float).
    trainer_class: The trainer class to use.
    train_steps: int, total number of training steps.
    save_steps: list of integers. Keep a model file at each of the supplied save
      steps.
    eval_steps: int, num of steps per evaluation. If None or 0, eval disabled.
    eval_frequency: int, how often to run evaluation (every eval_frequency
      steps). If None or 0, eval disabled.
    n_devices: how many devices to use (if None, default, use all available)
    random_seed: the random seed to use; time/os dependent if None (default).
    save_graphs: bool, if True, save computation graph to file.
    save_backward_graph: bool, if True, save backward graph to file too.
    has_weights: bool, whether weights are included in the inputs.
  Returns:
    trax.State
  """
  trainer = trainer_class(model, loss_fn, optimizer, lr_schedule, inputs,
                          output_dir,
                          random_seed=random_seed, n_devices=n_devices,
                          save_steps=save_steps, has_weights=has_weights)

  epoch_steps = [train_steps]  # Only training if eval_frequency is 0 or None
  if eval_frequency and eval_steps > 0:
    epoch_steps = itertools.chain([1,  # first epoch only 1 step
                                   eval_frequency - 1],
                                  itertools.repeat(eval_frequency))
  step_log(trainer.step,
           "Starting training using %d devices" % trainer.n_devices)

  for epoch_steps in epochs(train_steps, trainer.step, epoch_steps):
    trainer.train_epoch(epoch_steps, eval_steps)

    # Update nontrainable parameters with new history
    trainer.update_nontrainable_params()

    # Bookkeeping we do at the first step
    if trainer.step == 1:
      # Print number of parameters
      trainer.print_n_params()

      # Save computation graph (single-device only for now)
      if (save_graphs and backend.get_name() == "jax"):
        trainer.save_computation_graphs(save_backward_graph)

      # Save Gin config
      trainer.save_gin()

  step_log(trainer.step, "Training done")
  return trainer.state
