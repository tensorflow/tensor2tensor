# coding=utf-8
# Copyright 2021 The Tensor2Tensor Authors.
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


class MultistepAdamOptimizer(tf.train.AdamOptimizer):
  """Adam with SGD updates every n steps with accumulated gradients."""

  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
               use_locking=False, name="Adam", n=1):
    super(MultistepAdamOptimizer, self).__init__(
        learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon,
        use_locking=use_locking, name=name)
    self._n = n  # Call Adam optimizer every n batches with accumulated grads
    self._n_t = None  # n as tensor

  def _create_slots(self, var_list):
    """Create slot variables for Adam with accumulated gradients."""
    super(MultistepAdamOptimizer, self)._create_slots(var_list)
    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(initial_value=0 if self._n == 1 else 1,
                                   name="iter",
                                   colocate_with=first_var)
    for v in var_list:
      self._zeros_slot(v, "grad_acc", self._name)

  def _get_iter_variable(self):
    graph = (
        None if tf.executing_eagerly() else tf.get_default_graph())
    return self._get_non_slot_variable("iter", graph=graph)

  def _prepare(self):
    super(MultistepAdamOptimizer, self)._prepare()
    self._n_t = tf.convert_to_tensor(self._n, name="n")

  def _apply_cond(self, apply_fn, grad, var, *args, **kwargs):
    """Apply conditionally if counter is zero."""
    grad_acc = self.get_slot(var, "grad_acc")

    def apply_adam(grad_acc, apply_fn, grad, var, *args, **kwargs):
      total_grad = (grad_acc + grad) / tf.cast(self._n_t, grad.dtype)
      adam_op = apply_fn(total_grad, var, *args, **kwargs)
      with tf.control_dependencies([adam_op]):
        grad_acc_to_zero_op = grad_acc.assign(tf.zeros_like(grad_acc),
                                              use_locking=self._use_locking)
      return tf.group(adam_op, grad_acc_to_zero_op)

    def accumulate_gradient(grad_acc, grad):
      assign_op = tf.assign_add(grad_acc, grad, use_locking=self._use_locking)
      return tf.group(assign_op)  # Strip return value

    return tf.cond(
        tf.equal(self._get_iter_variable(), 0),
        lambda: apply_adam(grad_acc, apply_fn, grad, var, *args, **kwargs),
        lambda: accumulate_gradient(grad_acc, grad))

  def _apply_dense(self, grad, var):
    return self._apply_cond(
        super(MultistepAdamOptimizer, self)._apply_dense, grad, var)

  def _resource_apply_dense(self, grad, var):
    return self._apply_cond(
        super(MultistepAdamOptimizer, self)._resource_apply_dense, grad, var)

  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    return self._apply_cond(
        super(MultistepAdamOptimizer, self)._apply_sparse_shared, grad, var,
        indices, scatter_add)

  def _apply_sparse(self, grad, var):
    # TODO(fstahlberg): Implement a sparse version
    tf.logging.warning("MultistepAdamOptimizer does not support sparse updates")
    dense_grad = tf.convert_to_tensor(grad)
    return self._apply_cond(
        super(MultistepAdamOptimizer, self)._apply_dense, dense_grad, var)

  def _resource_apply_sparse_duplicate_indices(self, grad, var, indices):
    tf.logging.warning("MultistepAdamOptimizer does not support sparse updates")
    # Note that conversion to a dense Tensor handles duplicate `indices`
    # correctly (summing them). A real sparse implementation will probably want
    # to override _resource_apply_sparse instead so it gets them de-duplicated
    # automatically.
    dense_grad = tf.convert_to_tensor(
        tf.IndexedSlices(values=grad, indices=indices,
                         dense_shape=tf.shape(var)))
    return self._apply_cond(
        super(MultistepAdamOptimizer, self)._resource_apply_dense,
        dense_grad, var)

  def _finish(self, update_ops, name_scope):
    """Updates beta_power variables every n batches and incrs counter."""
    iter_ = self._get_iter_variable()
    beta1_power, beta2_power = self._get_beta_accumulators()
    with tf.control_dependencies(update_ops):
      with tf.colocate_with(iter_):

        def update_beta_op():
          update_beta1 = beta1_power.assign(
              beta1_power * self._beta1_t,
              use_locking=self._use_locking)
          update_beta2 = beta2_power.assign(
              beta2_power * self._beta2_t,
              use_locking=self._use_locking)
          return tf.group(update_beta1, update_beta2)
        maybe_update_beta = tf.cond(
            tf.equal(iter_, 0), update_beta_op, tf.no_op)
        with tf.control_dependencies([maybe_update_beta]):
          update_iter = iter_.assign(tf.mod(iter_ + 1, self._n_t),
                                     use_locking=self._use_locking)
    return tf.group(
        *update_ops + [update_iter, maybe_update_beta], name=name_scope)
