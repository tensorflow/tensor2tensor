# coding=utf-8
# Copyright 2023 The Tensor2Tensor Authors.
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
"""Multi-step optimizers simulating large batches.

Optimizer variants which make it possible to use very large batch sizes with
limited GPU memory. Optimizers in this module accumulate the gradients for n
batches, and call the optimizer's update rule every n batches with the
accumulated gradients.

See [Saunders et al., 2018](https://arxiv.org/abs/1805.00456) for details.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import training_ops
# pylint: enable=g-direct-tensorflow-import


class MultistepAdamOptimizer(tf.train.Optimizer):
  """Adam with SGD updates every n steps with accumulated gradients."""

  def __init__(self,
               learning_rate=0.001,
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-8,
               use_locking=False,
               name="Adam",
               n=1):
    super(MultistepAdamOptimizer, self).__init__(
        use_locking=use_locking, name=name)
    self._lr = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon
    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None
    self._n = n  # Call Adam optimizer every n batches with accumulated grads
    self._n_t = None  # n as tensor

  def _get_beta_accumulators(self):
    with tf.init_scope():
      if tf.executing_eagerly():
        graph = None
      else:
        graph = tf.get_default_graph()
      return (self._get_non_slot_variable("beta1_power", graph=graph),
              self._get_non_slot_variable("beta2_power", graph=graph))

  def _create_slots(self, var_list):
    """Create slot variables for Adam with accumulated gradients."""
    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(
        initial_value=self._beta1, name="beta1_power", colocate_with=first_var)
    self._create_non_slot_variable(
        initial_value=self._beta2, name="beta2_power", colocate_with=first_var)
    # if iter is initialized as an int32, this optimizer could not run
    # with tensorflow_hub with a tensorflow-gpu version
    self._create_non_slot_variable(
        initial_value=0.0 if self._n == 1 else 1.0,
        name="iter",
        colocate_with=first_var)
    # Create slots for the first and second moments, as well as grad_acc.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)
      self._zeros_slot(v, "grad_acc", self._name)

  def _get_iter_variable(self):
    graph = (None if tf.executing_eagerly() else tf.get_default_graph())
    return self._get_non_slot_variable("iter", graph=graph)

  def _prepare(self):
    lr = self._call_if_callable(self._lr)
    beta1 = self._call_if_callable(self._beta1)
    beta2 = self._call_if_callable(self._beta2)
    epsilon = self._call_if_callable(self._epsilon)
    self._beta1_t = tf.convert_to_tensor(beta1, name="beta1")
    self._beta2_t = tf.convert_to_tensor(beta2, name="beta2")
    self._lr_t = tf.convert_to_tensor(lr, name="learning_rate")
    self._epsilon_t = tf.convert_to_tensor(epsilon, name="epsilon")
    self._n_t = tf.convert_to_tensor(self._n, name="n")

  def _apply_cond(self, apply_fn, grad, var, *args, **kwargs):
    """Apply conditionally if counter is zero."""
    grad_acc = self.get_slot(var, "grad_acc")

    def apply_adam(grad_acc, apply_fn, grad, var, *args, **kwargs):
      total_grad = (grad_acc + grad) / tf.cast(self._n_t, grad.dtype)
      adam_op = apply_fn(total_grad, var, *args, **kwargs)
      with tf.control_dependencies([adam_op]):
        grad_acc_to_zero_op = grad_acc.assign(
            tf.zeros_like(grad_acc), use_locking=self._use_locking)
      return tf.group(adam_op, grad_acc_to_zero_op)

    def accumulate_gradient(grad_acc, grad):
      assign_op = tf.assign_add(grad_acc, grad, use_locking=self._use_locking)
      return tf.group(assign_op)  # Strip return value

    return tf.cond(
        tf.equal(self._get_iter_variable(), 0),
        lambda: apply_adam(grad_acc, apply_fn, grad, var, *args, **kwargs),
        lambda: accumulate_gradient(grad_acc, grad))

  def _apply_dense(self, grad, var):
    return self._apply_cond(self._apply_dense_in_action, grad, var)

  def _apply_dense_in_action(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    beta1_power, beta2_power = self._get_beta_accumulators()
    return training_ops.apply_adam(
        var,
        m,
        v,
        tf.cast(beta1_power, var.dtype.base_dtype),
        tf.cast(beta2_power, var.dtype.base_dtype),
        tf.cast(self._lr_t, var.dtype.base_dtype),
        tf.cast(self._beta1_t, var.dtype.base_dtype),
        tf.cast(self._beta2_t, var.dtype.base_dtype),
        tf.cast(self._epsilon_t, var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, var):
    return self._apply_cond(self._resource_apply_dense_in_action, grad, var)

  def _resource_apply_dense_in_action(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    beta1_power, beta2_power = self._get_beta_accumulators()
    return training_ops.resource_apply_adam(
        var.handle,
        m.handle,
        v.handle,
        tf.cast(beta1_power, grad.dtype.base_dtype),
        tf.cast(beta2_power, grad.dtype.base_dtype),
        tf.cast(self._lr_t, var.dtype.base_dtype),
        tf.cast(self._beta1_t, grad.dtype.base_dtype),
        tf.cast(self._beta2_t, grad.dtype.base_dtype),
        tf.cast(self._epsilon_t, grad.dtype.base_dtype),
        grad,
        use_locking=self._use_locking)

  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    beta1_power, beta2_power = self._get_beta_accumulators()
    beta1_power = tf.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = tf.cast(beta2_power, var.dtype.base_dtype)
    lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = tf.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
    epsilon_t = tf.cast(self._epsilon_t, var.dtype.base_dtype)
    lr = (lr_t * tf.sqrt(1 - beta2_power) / (1 - beta1_power))
    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, "m")
    m_scaled_g_values = grad * (1 - beta1_t)
    m_t = tf.assign(m, m * beta1_t, use_locking=self._use_locking)
    with tf.control_dependencies([m_t]):
      m_t = scatter_add(m, indices, m_scaled_g_values)
    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, "v")
    v_scaled_g_values = (grad * grad) * (1 - beta2_t)
    v_t = tf.assign(v, v * beta2_t, use_locking=self._use_locking)
    with tf.control_dependencies([v_t]):
      v_t = scatter_add(v, indices, v_scaled_g_values)
    v_sqrt = tf.sqrt(v_t)
    var_update = tf.assign_sub(
        var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
    return tf.group(*[var_update, m_t, v_t])

  def _apply_sparse(self, grad, var):
    # TODO(fstahlberg): Implement a sparse version
    tf.logging.warning("MultistepAdamOptimizer does not support sparse updates")
    dense_grad = tf.convert_to_tensor(grad)
    return self._apply_cond(self._apply_dense_in_action, dense_grad, var)

  def _resource_apply_sparse_duplicate_indices(self, grad, var, indices):
    tf.logging.warning("MultistepAdamOptimizer does not support sparse updates")
    # Note that conversion to a dense Tensor handles duplicate `indices`
    # correctly (summing them). A real sparse implementation will probably want
    # to override _resource_apply_sparse instead so it gets them de-duplicated
    # automatically.
    dense_grad = tf.convert_to_tensor(
        tf.IndexedSlices(
            values=grad, indices=indices, dense_shape=tf.shape(var)))
    return self._apply_cond(self._resource_apply_dense_in_action, dense_grad,
                            var)

  def _resource_scatter_add(self, x, i, v):
    with tf.control_dependencies(
        [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
      return x.value()

  def _resource_apply_sparse(self, grad, var, indices):
    return self._apply_sparse_shared(grad, var, indices,
                                     self._resource_scatter_add)

  def _finish(self, update_ops, name_scope):
    """Updates beta_power variables every n batches and incrs counter."""
    iter_ = self._get_iter_variable()
    beta1_power, beta2_power = self._get_beta_accumulators()
    with tf.control_dependencies(update_ops):
      with tf.colocate_with(iter_):

        def update_beta_op():
          update_beta1 = beta1_power.assign(
              beta1_power * self._beta1_t, use_locking=self._use_locking)
          update_beta2 = beta2_power.assign(
              beta2_power * self._beta2_t, use_locking=self._use_locking)
          return tf.group(update_beta1, update_beta2)

        maybe_update_beta = tf.cond(
            tf.equal(iter_, 0), update_beta_op, tf.no_op)
        with tf.control_dependencies([maybe_update_beta]):
          # TODO(cuong): It is suboptimal here because we have to cast twice
          # (float to int, and then int to float)
          update_iter = iter_.assign(
              tf.cast(
                  tf.mod(tf.cast(iter_ + 1.0, tf.int32), self._n_t),
                  tf.float32),
              use_locking=self._use_locking)
    return tf.group(
        *update_ops + [update_iter, maybe_update_beta], name=name_scope)
