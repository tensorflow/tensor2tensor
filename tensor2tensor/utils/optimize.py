# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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

"""Optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np

from tensor2tensor.utils import yellowfin

import tensorflow as tf

from tensorflow.python.framework import dtypes


def optimize(loss, learning_rate, hparams, use_tpu=False):
  """Minimize loss."""
  loss = weight_decay_and_noise(loss, hparams, learning_rate)
  loss = tf.identity(loss, name="total_loss")
  log_variable_sizes(verbose=hparams.summarize_vars)
  diet_vars = [
      v for v in tf.global_variables() if v.dtype == dtypes.float16_ref
  ]
  log_variable_sizes(
      diet_vars, "Diet Variables", verbose=hparams.summarize_vars)
  opt = ConditionalOptimizer(hparams.optimizer, learning_rate, hparams, use_tpu)
  if use_tpu:
    opt = tf.contrib.tpu.CrossShardOptimizer(opt)

  tf.summary.scalar("learning_rate", learning_rate)
  opt_summaries = ["loss", "global_gradient_norm"]
  if hparams.summarize_grads:
    tf.logging.info("Summarizing gradients")
    opt_summaries.extend(["gradients", "gradient_norm"])

  if hparams.clip_grad_norm:
    tf.logging.info("Clipping gradients, norm: %0.5f", hparams.clip_grad_norm)
  if hparams.grad_noise_scale:
    tf.logging.info("Adding noise to gradients, noise scale: %0.5f",
                    hparams.grad_noise_scale)

  train_op = tf.contrib.layers.optimize_loss(
      name="training",
      loss=loss,
      global_step=tf.train.get_or_create_global_step(),
      learning_rate=learning_rate,
      clip_gradients=hparams.clip_grad_norm or None,
      gradient_noise_scale=hparams.grad_noise_scale or None,
      optimizer=opt,
      summaries=opt_summaries,
      colocate_gradients_with_ops=True)
  return train_op


class ConditionalOptimizer(tf.train.Optimizer):
  """Conditional optimizer."""

  def __init__(self, optimizer_name, lr, hparams, use_tpu=False):
    if optimizer_name == "Adam" and use_tpu:
      # LazyAdamOptimizer does not work on TPU
      optimizer_name = "TrueAdam"

    tf.logging.info("Using optimizer %s", optimizer_name)

    if optimizer_name == "Adam":
      # We change the default epsilon for Adam and re-scale lr.
      # Using LazyAdam as it's much faster for large vocabulary embeddings.
      self._opt = tf.contrib.opt.LazyAdamOptimizer(
          lr / 500.0,
          beta1=hparams.optimizer_adam_beta1,
          beta2=hparams.optimizer_adam_beta2,
          epsilon=hparams.optimizer_adam_epsilon)
    elif optimizer_name == "Momentum":
      self._opt = tf.train.MomentumOptimizer(
          lr,
          momentum=hparams.optimizer_momentum_momentum,
          use_nesterov=hparams.optimizer_momentum_nesterov)
    elif optimizer_name == "YellowFin":
      self._opt = yellowfin.YellowFinOptimizer(
          learning_rate=lr, momentum=hparams.optimizer_momentum_momentum)
    elif optimizer_name == "TrueAdam":
      self._opt = tf.train.AdamOptimizer(
          lr / 500.0,
          beta1=hparams.optimizer_adam_beta1,
          beta2=hparams.optimizer_adam_beta2,
          epsilon=hparams.optimizer_adam_epsilon)
    elif optimizer_name == "Adafactor":
      self._opt = AdafactorOptimizer(lr / 500.0)
    else:
      self._opt = tf.contrib.layers.OPTIMIZER_CLS_NAMES[optimizer_name](lr)

  def compute_gradients(self, loss, var_list=None, **kwargs):
    return self._opt.compute_gradients(loss, var_list, **kwargs)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    return self._opt.apply_gradients(
        grads_and_vars, global_step=global_step, name=name)


def _sqrt_decay(step):
  """Decay like 1 / sqrt(step), multiplied by 500 to normalize."""
  return 500.0 / tf.sqrt(tf.maximum(step, 1.0))


def _exp_decay_after(step, rate, from_which_step):
  """Decay exponentially by rate (per step) starting at from_which_step."""
  return tf.cond(
      step < from_which_step,
      lambda: tf.constant(1.0),
      lambda: rate**(step - from_which_step),
      name="exponential_decay_step_cond")


def piecewise_learning_rate(step, boundaries, values):
  """Scale learning rate according to the given schedule.

  Multipliers are not cumulative.

  Args:
    step: global step
    boundaries: List of steps to transition on.
    values: Multiplier to apply at each boundary transition.

  Returns:
    Scaled value for the learning rate.
  """
  values = [1.0] + values
  return tf.train.piecewise_constant(
      step, boundaries, values, name="piecewise_lr")


def learning_rate_decay(hparams, warmup_steps=0):
  """Learning rate decay multiplier."""
  scheme = hparams.learning_rate_decay_scheme
  warmup_steps = tf.to_float(warmup_steps)
  global_step = tf.to_float(tf.train.get_or_create_global_step())

  if not scheme or scheme == "none":
    return tf.constant(1.)

  tf.logging.info("Applying learning rate decay: %s.", scheme)

  if scheme == "exp":
    decay_steps = hparams.learning_rate_decay_steps
    p = (global_step - warmup_steps) / decay_steps
    if hparams.learning_rate_decay_staircase:
      p = tf.floor(p)
    return tf.pow(hparams.learning_rate_decay_rate, p)

  if scheme == "piecewise":
    return piecewise_learning_rate(global_step,
                                   hparams.learning_rate_boundaries,
                                   hparams.learning_rate_multiples)

  if scheme == "cosine":
    cycle_steps = hparams.learning_rate_cosine_cycle_steps
    cycle_position = global_step % (2 * cycle_steps)
    cycle_position = cycle_steps - tf.abs(cycle_steps - cycle_position)
    return 0.5 * (1 + tf.cos(np.pi * cycle_position / cycle_steps))

  if scheme == "cyclelinear10x":
    # Cycle the rate linearly by 10x every warmup_steps, up and down.
    cycle_steps = warmup_steps
    cycle_position = global_step % (2 * cycle_steps)
    cycle_position = tf.to_float(  # Normalize to the interval [-1, 1].
        cycle_position - cycle_steps) / float(cycle_steps)
    cycle_position = 1.0 - tf.abs(cycle_position)  # 0 to 1 and back to 0.
    return (cycle_position + 0.1) * 3.0  # 10x difference each cycle (0.3-3).

  if scheme == "sqrt":
    return _sqrt_decay(global_step - warmup_steps)

  raise ValueError("Unrecognized learning rate decay scheme: %s" %
                   hparams.learning_rate_decay_scheme)


def learning_rate_warmup(warmup_steps, warmup_schedule="exp"):
  """Learning rate warmup multiplier."""
  if not warmup_steps:
    return tf.constant(1.)

  tf.logging.info("Applying %s learning rate warmup for %d steps",
                  warmup_schedule, warmup_steps)

  warmup_steps = tf.to_float(warmup_steps)
  global_step = tf.to_float(tf.train.get_or_create_global_step())

  if warmup_schedule == "exp":
    return tf.exp(tf.log(0.01) / warmup_steps)**(warmup_steps - global_step)
  else:
    assert warmup_schedule == "linear"
    start = tf.constant(0.35)
    return ((tf.constant(1.) - start) / warmup_steps) * global_step + start


def learning_rate_decay_with_warmup(hparams, num_worker_replicas=1):
  """Learning rate decay rate with warmup based on hparams."""
  warmup_steps = hparams.learning_rate_warmup_steps * num_worker_replicas
  warmup = learning_rate_warmup(warmup_steps)

  decay = learning_rate_decay(hparams, warmup_steps)

  global_step = tf.train.get_or_create_global_step()
  return tf.where(global_step < warmup_steps, warmup, decay)


def learning_rate_schedule(hparams, num_worker_replicas=1):
  """Learning rate schedule based on hparams."""
  schedule = hparams.learning_rate_schedule
  warmup_steps = tf.to_float(hparams.learning_rate_warmup_steps)
  global_step = tf.to_float(tf.train.get_or_create_global_step())
  if hparams.learning_rate_decay_scheme == "noam":
    # backwards compatiblity with previous behavior
    schedule = "linear_warmup_rsqrt_decay"
  if schedule == "warmup_and_decay":
    return learning_rate_decay_with_warmup(hparams, num_worker_replicas)
  elif schedule == "linear_warmup_rsqrt_decay":
    return 5000.0 * hparams.hidden_size**-0.5 * tf.minimum(
        (global_step + 1) * warmup_steps**-1.5, (global_step + 1)**-0.5)
  else:
    raise ValueError("Unrecognized learning rate schedule: %s" % schedule)


def weight_decay_and_noise(loss, hparams, learning_rate, var_list=None):
  """Apply weight decay and weight noise."""
  if var_list is None:
    var_list = tf.trainable_variables()

  decay_vars = [v for v in var_list]
  noise_vars = [v for v in var_list if "/body/" in v.name]

  weight_decay_loss = weight_decay(hparams.weight_decay, decay_vars)
  tf.summary.scalar("losses/weight_decay", weight_decay_loss)
  weight_noise_ops = weight_noise(hparams.weight_noise, learning_rate,
                                  noise_vars)

  with tf.control_dependencies(weight_noise_ops):
    loss = tf.identity(loss)

  loss += weight_decay_loss
  return loss


def weight_noise(noise_rate, learning_rate, var_list):
  """Apply weight noise to vars in var_list."""
  if not noise_rate:
    return [tf.no_op()]

  tf.logging.info("Applying weight noise scaled by learning rate, "
                  "noise_rate: %0.5f", noise_rate)

  noise_ops = []

  for v in var_list:
    with tf.device(v._ref().device):  # pylint: disable=protected-access
      scale = noise_rate * learning_rate * 0.001
      tf.summary.scalar("weight_noise_scale", scale)
      noise = tf.truncated_normal(v.shape) * scale
      noise_op = v.assign_add(noise)
      noise_ops.append(noise_op)

  return noise_ops


def weight_decay(decay_rate, var_list, skip_biases=True):
  """Apply weight decay to vars in var_list."""
  if not decay_rate:
    return 0.

  tf.logging.info("Applying weight decay, decay_rate: %0.5f", decay_rate)

  weight_decays = []
  for v in var_list:
    # Weight decay.
    # This is a heuristic way to detect biases that works for main tf.layers.
    is_bias = len(v.shape.as_list()) == 1 and v.name.endswith("bias:0")
    if not (skip_biases and is_bias):
      with tf.device(v.device):
        v_loss = tf.nn.l2_loss(v)
      weight_decays.append(v_loss)

  return tf.add_n(weight_decays) * decay_rate


def log_variable_sizes(var_list=None, tag=None, verbose=False):
  """Log the sizes and shapes of variables, and the total size.

  Args:
    var_list: a list of variables; defaults to trainable_variables
    tag: a string; defaults to "Trainable Variables"
    verbose: bool, if True, log every weight; otherwise, log total size only.
  """
  if var_list is None:
    var_list = tf.trainable_variables()
  if tag is None:
    tag = "Trainable Variables"

  if not var_list:
    return

  name_to_var = {v.name: v for v in var_list}
  total_size = 0
  for v_name in sorted(list(name_to_var)):
    v = name_to_var[v_name]
    v_size = int(np.prod(np.array(v.shape.as_list())))
    if verbose:
      tf.logging.info("Weight    %s\tshape    %s\tsize    %d",
                      v.name[:-2].ljust(80),
                      str(v.shape).ljust(20), v_size)
    total_size += v_size
  tf.logging.info("%s Total size: %d", tag, total_size)


def get_variable_initializer(hparams):
  """Get variable initializer from hparams."""
  if not hparams.initializer:
    return None

  tf.logging.info("Using variable initializer: %s", hparams.initializer)
  if hparams.initializer == "orthogonal":
    return tf.orthogonal_initializer(gain=hparams.initializer_gain)
  elif hparams.initializer == "uniform":
    max_val = 0.1 * hparams.initializer_gain
    return tf.random_uniform_initializer(-max_val, max_val)
  elif hparams.initializer == "normal_unit_scaling":
    return tf.variance_scaling_initializer(
        hparams.initializer_gain, mode="fan_avg", distribution="normal")
  elif hparams.initializer == "uniform_unit_scaling":
    return tf.variance_scaling_initializer(
        hparams.initializer_gain, mode="fan_avg", distribution="uniform")
  else:
    raise ValueError("Unrecognized initializer: %s" % hparams.initializer)


class AdafactorOptimizer(tf.train.Optimizer):
  """Optimizer that implements the Adafactor algorithm.

  Adafactor is similar to RMSProp (ADAM, etc.), but takes advantage of the
  structure of weight matrices to use less memory and to be more resilient to
  sudden large gradients.

  The RMSProp algorithm works on each component independently as follows:
    w -= grad * learning_rate / sqrt(estimated_mean_square_grad)

    learning_rate is the desired update magnitude, and
    estimated_mean_square_grad is computed by exponential smoothing of the
    square of the gradient.

  Adafactor addresses two shortcomings of RMSProp:

  1. In RMSProp (ADAM, etc), maintaining estimated_mean_square_grad requires
     memory equal to the number of parameters.  This can be an impediment to
     training large models on GPU/TPU systems with limited memory.

     Adafactor uses less memory.
     For an AxB weight matrix, instead of keeping a full AxB
     estimated_mean_square_grad matrix, Adafactor keeps only
     exponentially-smoothed row and column means, and bases its estimates on
     those means.   Thus the memory requirements drop from `2AB` to `A+B`.

  2. Depending on the decay rate of the exponential smoothing, we run into one
     of two problems.

     If the decay rate is high (short memory), we see the problem described
     here - worse final quality:
       On the Convergence of Adam and Beyond
       https://openreview.net/forum?id=ryQu7f-RZ

     If the decay rate is low (long memory), then the estimate does not adjust
     rapidly to suddenly large gradients, and the model diverges.
     Suddenly large gradients (which we will call anomalies), may happen either
     due to weird training data, or because the model has just learned something
     important and can now rush to exploit it.  Momentum (as in ADAM) can help
     prevent divergence, but it also requires more memory.  Gradient clipping
     can also help prevent divergence, but it is irritating in that setting
     the right threshold depends on the knowing the scale of the gradients.

     Adafactor uses a relatively long memory (setting the decay rate to
     step_num^-0.8), but detects and corrects for anomalies.   An anomaly
     is detected if the mean-square gradient for the current step
     (across the entire weight matrix) is much greater than the historical
     average.  When this occurs, we increase estimated_mean_square_grad
     for the current step for all weights in the matrix.  Note: it is important
     to detect anomalies based on entire matrices, rather than individual
     weights, since any individual weight may legitimately have a pattern
     of many small gradients and occasional very large ones.

  HYPERPARAMETERS:
    learning_rate: desired magnitude of variable updates.  a scalar - can be a
      constant, but more likely should have a warmup and then decay
      proportionally to rsqrt(step_num)
    epsilon: 1e-20 - a small floating point value to avoid division by zero.
    horizon_exponent: 0.8 - a value between 0 and 1 - The effective decay
      horizon of the second-moment estimator is step_num^horizon_exponent.
    anomaly_threshold: 2.0 - a value greater than 1.  Suppress anomalies
      where the mean-square-gradients for a step exceed the long-term average
      by at least this factor.

  ALGORITHM:

  We initialize
  ```
  t <- 0
  if var is 2-dimensional:
    v_r <- zeros([num_rows])
    v_c <- zeros([num_cols])
  else:
    v <- zeros(shape(var))
  ```

  The update rule is as follows:
  ```
  t <- t + 1
  decay_rate = 1 - t ^ (-horizon_exponent)
  grad_squared = tf.square(grad) + epsilon
  if var is 2-dimensional:
    v_r <- decay_rate * v_r + (1 - decay_rate) * reduce_mean(grad_squared, 1)
    v_c <- decay_rate * v_c + (1 - decay_rate) * reduce_mean(grad_squared, 0)
    anomaly_factor = max(1.0,
      reduce_mean(grad_squared) / reduce_mean(v_r) / anomaly_threshold)
    est_v = anomaly_factor * outer_prod(v_r, v_c) / reduce_mean(v_r)
  else:
    v <- decay_rate * v + (1 - decay_rate) * grad_squared
    anomaly_factor = max(1.0,
      reduce_mean(grad_squared) / reduce_mean(v) / anomaly_threshold)
    est_v = v * anomaly_factor
  var <- var - lr * grad / sqrt(est_v)
  ```
  TODO(noam): write a paper.
  TODO(noam): we should also apply the 2d logic to the two final dimensions.
    of >2d convolutional kernels.
  """

  def __init__(self,
               learning_rate=0.001,
               epsilon=1e-20,
               horizon_exponent=0.8,
               anomaly_threshold=2.0,
               use_locking=False,
               name="Adafactor"):
    """Construct a new Adafactor optimizer.

    See class comment.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      epsilon: A small constant for numerical stability.
      horizon_exponent: a floating point value between 0 and 1
      anomaly_threshold: a floating point value >= 1.0
      use_locking: If True use locks for update operations.
      name: Optional name for the operations created when applying gradients.
        Defaults to "AdafactorOptimizer".
    """
    super(AdafactorOptimizer, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._epsilon = epsilon
    self._horizon_exponent = horizon_exponent
    self._anomaly_threshold = anomaly_threshold

  def _should_use_factored_second_moment_estimate(self, shape):
    """Should we use a factored second moment estimator.

    Based on the shape of the variable.

    Args:
      shape: a list of integers
    Returns:
      a boolean
    """
    return len(shape) == 2

  def _create_slots(self, var_list):
    for v in var_list:
      shape = v.get_shape().as_list()
      if self._should_use_factored_second_moment_estimate(shape):
        r_val = tf.zeros([shape[0]], dtype=tf.float32)
        c_val = tf.zeros([shape[1]], dtype=tf.float32)
        self._get_or_make_slot(v, r_val, "vr", self._name)
        self._get_or_make_slot(v, c_val, "vc", self._name)
      else:
        self._zeros_slot(v, "v", self._name)

  def _apply_dense(self, grad, var):
    return self._resource_apply_dense(grad, var)

  def _resource_apply_dense(self, grad, var):
    grad_squared = tf.square(grad) + self._epsilon
    grad_squared_mean = tf.reduce_mean(grad_squared)
    lr = tf.to_float(self._lr)
    global_step = tf.to_float(tf.train.get_or_create_global_step()) + 1.0
    # HACK: Make lr and global_step dependent on grad.
    # This confounds the XLA rewriter and keeps it from fusing computations
    # across different variables.  This fusion is a bad for HBM usage, since
    # it causes the gradients to persist in memory.
    lr += grad_squared_mean * 1e-30
    global_step += grad_squared_mean * 1e-30
    # END HACK
    mixing_rate = tf.pow(global_step, -self._horizon_exponent)
    decay_rate = 1.0 - mixing_rate
    shape = var.get_shape().as_list()
    updates = []
    if self._should_use_factored_second_moment_estimate(shape):
      grad_squared_row_mean = tf.reduce_mean(grad_squared, 1)
      grad_squared_col_mean = tf.reduce_mean(grad_squared, 0)
      vr = self.get_slot(var, "vr")
      new_vr = (decay_rate * vr + mixing_rate * grad_squared_row_mean)
      vc = self.get_slot(var, "vc")
      new_vc = (decay_rate * vc + mixing_rate * grad_squared_col_mean)
      vr_update = tf.assign(vr, new_vr, use_locking=self._use_locking)
      vc_update = tf.assign(vc, new_vc, use_locking=self._use_locking)
      updates = [vr_update, vc_update]
      long_term_mean = tf.reduce_mean(new_vr)
      anomaly_factor = self._anomaly_factor(grad_squared_mean, long_term_mean)
      # This is the computation we should do.
      # est_v = (tf.expand_dims(new_vr, 1) * tf.expand_dims(new_vc, 0)
      #          * anomaly_factor / long_term_mean)
      # subtrahend = grad * lr / tf.sqrt(est_v)
      # Instead we do the following, which is mathematically equivalent.
      r_factor = lr * tf.rsqrt(new_vr * anomaly_factor / long_term_mean)
      c_factor = tf.rsqrt(new_vc)
      subtrahend = (
          grad * tf.expand_dims(r_factor, 1) * tf.expand_dims(c_factor, 0))
    else:
      v = self.get_slot(var, "v")
      new_v = decay_rate * v + mixing_rate * grad_squared
      v_update = tf.assign(v, new_v, use_locking=self._use_locking)
      updates = [v_update]
      long_term_mean = tf.reduce_mean(new_v)
      anomaly_factor = self._anomaly_factor(grad_squared_mean, long_term_mean)
      # This is the computation we should do.
      # est_v = (new_v * anomaly_factor)
      # subtrahend = grad * lr / tf.sqrt(est_v)
      # Instead we do the following, which is mathematically equivalent.
      subtrahend = grad * (lr / tf.sqrt(anomaly_factor)) * tf.rsqrt(new_v)
    var_update = tf.assign_sub(var, subtrahend, use_locking=self._use_locking)
    updates = [var_update] + updates
    return tf.group(*updates)

  def _anomaly_factor(self, grad_squared_mean, long_term_mean):
    """Multiplier for second-moment estimator, due to short-term anomalies.

    A step may have gradients with magnitudes much larger than the long-term
    average.  This can cause the model to diverge.  In these cases, we want to
    temoporarily increase the second-moment estimators to reflect that these
    steps are anomalous.

    It is important to make these calculations on whole weight matrices, rather
    than on individual parameters, since we want to allow individual parameters
    to have occasional large updates.

    Args:
      grad_squared_mean: A scalar.  The mean square gradient on the varaible
         for the current step.
      long_term_mean: A scalar.  The mean of the long-term second-moment
         estimator.
    Returns:
      a scalar that should be multiplied into the second-moment-estimator for
      this step.
    """
    ratio = grad_squared_mean / long_term_mean
    return tf.maximum(1.0, ratio / self._anomaly_threshold)
