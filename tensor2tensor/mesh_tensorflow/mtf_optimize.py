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
"""Mesh-Tensorflow Optimizers."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.mesh_tensorflow import mesh_tensorflow as mtf
import tensorflow as tf


def make_optimizer(hparams, lr):
  if hparams.optimizer == "SGD":
    return SgdOptimizer(lr)
  elif hparams.optimizer == "Adafactor":
    return adafactor_optimizer_from_hparams(hparams, lr)
  else:
    raise ValueError("Unknown Optimizer")


class Optimizer(object):
  """Base optmizer class."""

  def apply_grad(self, grad, var):
    raise ValueError("Apply_Grad not implemented %s %s" % (grad, var))


class SgdOptimizer(Optimizer):
  """oOptimizer implementing SGD."""

  def __init__(self, lr):
    self._lr = lr

  @property
  def lr(self):
    return self._lr

  def apply_grad(self, grad, var):
    return [mtf.assign(var, var.outputs[0] - (grad * self.lr))]


class AdafactorOptimizer(Optimizer):
  """Adafactor."""

  def __init__(self,
               multiply_by_parameter_scale=True,
               learning_rate=None,
               decay_rate=None,
               beta1=0.0,
               clipping_threshold=1.0,
               factored=True,
               epsilon1=1e-30,
               epsilon2=1e-3):
    """Construct a new Adafactor optimizer.

    See class comment.

    Args:
      multiply_by_parameter_scale: a boolean
      learning_rate: an optional Scalar.
      decay_rate: an optional Scalar.
      beta1: a float value between 0 and 1
      clipping_threshold: an optional float >= 1
      factored: a boolean - whether to use factored second-moment estimator
        for 2d variables
      epsilon1: Regularization constant for squared gradient.
      epsilon2: Regularization constant for parameter scale.

    Raises:
      ValueError: if absolute_update_scale and relative_update_scale_fn are both
        present or both absent.
    """
    self._multiply_by_parameter_scale = multiply_by_parameter_scale
    if learning_rate is None:
      learning_rate = self._learning_rate_default(multiply_by_parameter_scale)
    self._learning_rate = learning_rate
    if decay_rate is None:
      decay_rate = self._decay_rate_default()
    self._decay_rate = decay_rate
    self._beta1 = beta1
    self._clipping_threshold = clipping_threshold
    self._factored = factored
    self._epsilon1 = epsilon1
    self._epsilon2 = epsilon2

  def _factored_dims(self, shape):
    """Should we use a factored second moment estimator.

    Based on the shape of the variable.
    If we factor the accumulator, then this function returns a list of two
    mtf.Dimensions to reduce over.  We always pick the two largest dimensions.
    If there are not two dimensions of size >=128, then we do not factor.

    Args:
      shape: a Shape
    Returns:
      either a list of 2 Dimensions or None
    """
    if not self._factored or shape.ndims < 2:
      return None
    sorted_dims = sorted(shape.dims, key=lambda d: -d.size)
    if sorted_dims[1].size < 128:
      return None
    return sorted_dims[:2]

  def _parameter_scale(self, var):
    """Estimate the scale of the parameters from the current values.

    We include a minimum value of 0.001 to give it a chance to escape 0
    if it was zero-initialized.

    Instead of using the value, we could impute the scale from the shape,
    as initializers do.

    Args:
      var: a variable or Tensor.
    Returns:
      a Scalar
    """
    return mtf.maximum(reduce_rms(var), self._epsilon2)

  def apply_grad(self, grad, var):
    # create slots
    factored_dims = self._factored_dims(var.shape)
    if factored_dims:
      d0, d1 = factored_dims
      vr_shape = var.shape - d0
      vc_shape = var.shape - d1
      vr = mtf.get_variable(
          var.mesh, var.name + "_slot_vr", vr_shape,
          initializer=tf.zeros_initializer(), trainable=False)
      vc = mtf.get_variable(
          var.mesh, var.name + "_slot_vc", vc_shape,
          initializer=tf.zeros_initializer(), trainable=False)
    else:
      v = mtf.get_variable(
          var.mesh, var.name + "_slot_v", var.shape,
          initializer=tf.zeros_initializer(), trainable=False)
    if self._beta1:
      m = mtf.get_variable(
          var.mesh, var.name + "_slot_m", var.shape,
          iniitalizer=tf.zeros_initializer(), trainable=False)

    with tf.variable_scope(var.name + "/adafactor"):
      grad_squared = mtf.square(grad) + self._epsilon1
      decay_rate = self._decay_rate
      old_val = var.value
      if self._multiply_by_parameter_scale:
        update_scale = self._parameter_scale(old_val) * self._learning_rate
      else:
        update_scale = self._learning_rate
      mixing_rate = 1.0 - decay_rate
      updates = []
      if factored_dims:
        grad_squared_row_mean = mtf.reduce_mean(
            grad_squared, output_shape=vr_shape)
        grad_squared_col_mean = mtf.reduce_mean(
            grad_squared, output_shape=vc_shape)
        new_vr = vr * decay_rate + grad_squared_row_mean * mixing_rate
        new_vc = vc * decay_rate + grad_squared_col_mean * mixing_rate
        vr_update = mtf.assign(vr, new_vr)
        vc_update = mtf.assign(vc, new_vc)
        updates.extend([vr_update, vc_update])
        long_term_mean = mtf.reduce_mean(new_vr, reduced_dim=d1)
        r_factor = mtf.rsqrt(new_vr / long_term_mean)
        c_factor = mtf.rsqrt(new_vc)
        x = grad * r_factor * c_factor
      else:
        new_v = v * decay_rate + grad_squared * mixing_rate
        v_update = mtf.assign(v, new_v)
        updates.append(v_update)
        x = grad * mtf.rsqrt(new_v)
      if self._clipping_threshold is not None:
        clipping_denom = mtf.maximum(
            1.0, reduce_rms(x) / self._clipping_threshold)
        x /= clipping_denom
      subtrahend = x * update_scale
      if self._beta1:
        new_m = self._beta1 * m.value + (1.0 - self._beta1) * subtrahend
        subtrahend = new_m
        updates.append(mtf.assign(m, new_m))
      new_val = old_val - subtrahend
      var_update = mtf.assign(var, new_val)
      updates.append(var_update)
      return updates

  def _decay_rate_default(self):
    return adafactor_decay_rate_pow(0.8)

  def _learning_rate_default(self, multiply_by_parameter_scale):
    learning_rate = tf.minimum(tf.rsqrt(step_num() + 1.0), 0.01)
    if not multiply_by_parameter_scale:
      learning_rate *= 0.05
    return learning_rate


def adafactor_decay_rate_adam(beta2):
  """Second-moment decay rate like Adam, subsuming the correction factor.

  Args:
    beta2: a float between 0 and 1
  Returns:
    a scalar
  """
  t = tf.to_float(tf.train.get_or_create_global_step()) + 1.0
  decay = beta2 * (1.0 - tf.pow(beta2, t - 1.0)) / (1.0 - tf.pow(beta2, t))
  return decay


def adafactor_decay_rate_pow(exponent):
  """Second moment decay rate where memory-length grows as step_num^exponent.

  Args:
    exponent: a float between 0 and 1
  Returns:
    a scalar
  """
  return 1.0 - tf.pow((step_num() + 1.0), -exponent)


def step_num():
  return tf.to_float(tf.train.get_or_create_global_step())


def adafactor_optimizer_from_hparams(hparams, lr):
  """Create an Adafactor optimizer based on model hparams.

  Args:
    hparams: model hyperparameters
    lr: learning rate scalar.
  Returns:
    an AdafactorOptimizer
  Raises:
    ValueError: on illegal values
  """
  if hparams.optimizer_adafactor_decay_type == "Adam":
    decay_rate = adafactor_decay_rate_adam(
        hparams.optimizer_adafactor_beta2)
  elif hparams.optimizer_adafactor_decay_type == "pow":
    decay_rate = adafactor_decay_rate_pow(
        hparams.optimizer_adafactor_memory_exponent)
  else:
    raise ValueError("unknown optimizer_adafactor_decay_type")
  return AdafactorOptimizer(
      multiply_by_parameter_scale=(
          hparams.optimizer_adafactor_multiply_by_parameter_scale),
      learning_rate=lr,
      decay_rate=decay_rate,
      beta1=hparams.optimizer_adafactor_beta1,
      clipping_threshold=hparams.optimizer_adafactor_clipping_threshold,
      factored=hparams.optimizer_adafactor_factored)


def reduce_rms(x):
  return mtf.sqrt(mtf.reduce_mean(mtf.square(x)))
