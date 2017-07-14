# Copyright 2017 The Tensor2Tensor Authors.
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

"""YellowFin for TensorFlow. Thanks Jian Zhang: zjian [@] stanford [.] edu."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


# Values for gate_gradients.
GATE_NONE = 0
GATE_OP = 1
GATE_GRAPH = 2


class YellowFinOptimizer(tf.train.Optimizer):
  """Optimizer that implements the YellowFin algorithm.

  See [Zhang et. al., 2017](https://arxiv.org/abs/1706.03471) for details.
  """

  def __init__(self,
               learning_rate=1.0,
               momentum=0.0,
               clip_thresh=None,
               beta=0.999,
               curvature_window_width=20,
               zero_debias=True,
               delta_mu=0.0):
    """Construct a new YellowFin optimizer.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      momentum: A Tensor or a floating point value.  The momentum.
      clip_thresh: A Tensor or a floating point value. The cliping threshold for
        tf.clip_by_global_norm.  If None, no clipping will be carried out.
      beta: A float value or a constant float tensor.  The smoothing parameter
        for estimations.
      curvature_window_width: A int value or a constant int tensor.
        The curvature window width.
      zero_debias: A boolean, zero debias moving-averages.
      delta_mu: For extensions. Not necessary in the basic use.

    Note:
      clip_thresh is the threshold value on ||lr * gradient||,
      delta_mu can be place holder/variable/tensor scalar.
      They are used for additional momentum in situations such as
      asynchronous-parallel training.
      The default is 0.0(or None) for basic usage of the optimizer.

    Other features:
      If you want to manually control the learning rates, self.lr_factor is
      an interface to the outside, it is an multiplier for the internal
      learning rate in YellowFin. It is helpful when you want to do additional
      hand tuning or some decaying scheme to the tuned learning rate in
      YellowFin.
      Example on using lr_factor can be found here:
      https://github.com/JianGoForIt/YellowFin/blob/master/char-rnn-tensorflow/train_YF.py#L140
    """
    # Set lr and mu
    self._lr = learning_rate
    self._mu = momentum

    # Set lr and mu tensor.
    self._lr_var = tf.Variable(learning_rate,
                               dtype=tf.float32,
                               name="YF_lr",
                               trainable=False)
    self._mu_var = tf.Variable(momentum,
                               dtype=tf.float32,
                               name="YF_mu",
                               trainable=False)

    # Tuning factor for learning rates step or decaying scheme.
    self.lr_factor = tf.Variable(1.0,
                                 dtype=tf.float32,
                                 name="YF_lr_factor",
                                 trainable=False)

    # Gradient Clipping Threshold.
    if clip_thresh is not None:
      self._clip_thresh_var = tf.Variable(clip_thresh,
                                          dtype=tf.float32,
                                          name="YF_clip_thresh",
                                          trainable=False)
    else:
      self._clip_thresh_var = None

    # Set initial lr and mu for momentum.
    self._lr_m = self._lr_var * self.lr_factor
    self._mu_m = self._mu_var + delta_mu

    # Init momentum optimizer.
    self._momentum_optimizer = tf.train.MomentumOptimizer(
        self._lr_m, self._mu_m)

    # Moving average for statistics.
    self._beta = beta
    self._moving_averager = None

    # Step counting.
    self._step = tf.Variable(0,
                             dtype=tf.int32,
                             name="YF_step",
                             trainable=False)
    # YF_step + 1 op.
    self._increment_step_op = None

    # For conditional tuning.
    self._do_tune = tf.greater(self._step, tf.constant(0))

    # Moving-averages.
    self._zero_debias = zero_debias

    # For curvature range.
    self.curvature_window_width = curvature_window_width
    self._curv_win = None

    # Gradients and Variables.
    self._grad = None
    self._vars = None

    # Get per var g**2, norm**2 and mean(norm**2).
    self._grad_squared = None
    self._grad_norm_squared = None
    self._grad_norm_squared_avg = None

    # Mean(grad) and Mean(grad**2) to compute Variance.
    self._grad_avg = None
    self._grad_avg_squared = None

    # Max and Min curvature variations.
    self._h_max_t = None
    self._h_min_t = None
    self._h_min = None
    self._h_max = None

    # Gradient Expected Variance.
    self._grad_var = None

    # Gradient Norm and Mean(Gradient Norm).
    self._grad_norm = None
    self._grad_norm_avg = None

    # Distance to optimum and Mean(Distance to optimum).
    self._d_t = None
    self._dist_to_opt_avg = None

    # Maintains moving averages of variables
    # by employing an exponential decay(Beta),
    # and (zero_devias) moving-averages.
    self._moving_averager = None

  def _curvature_range(self):
    """Curvature range.

    Returns:
      h_max_t, h_min_t ops
    """
    self._curv_win = tf.Variable(np.zeros([self.curvature_window_width,]),
                                 dtype=tf.float32,
                                 name="curv_win",
                                 trainable=False)

    self._curv_win = tf.scatter_update(self._curv_win,
                                       self._step % self.curvature_window_width,
                                       self._grad_norm_squared)
    # Note here the iterations start from iteration 0
    valid_window = tf.slice(self._curv_win,
                            tf.constant([0,]),
                            tf.expand_dims(
                                tf.minimum(
                                    tf.constant(self.curvature_window_width),
                                    self._step + 1), axis=0))
    self._h_min_t = tf.reduce_min(valid_window)
    self._h_max_t = tf.reduce_max(valid_window)

    curv_range_ops = []
    with tf.control_dependencies([self._h_min_t, self._h_max_t]):
      avg_op = self._moving_averager.apply([self._h_min_t, self._h_max_t])
      with tf.control_dependencies([avg_op]):
        self._h_min = tf.identity(self._moving_averager.average(self._h_min_t))
        self._h_max = tf.identity(self._moving_averager.average(self._h_max_t))
    curv_range_ops.append(avg_op)
    return curv_range_ops  # h_max_t, h_min_t

  def _grad_variance(self):
    """Estimate of gradient Variance.

    Returns:
      C_t ops.
    """
    grad_var_ops = []
    tensor_to_avg = []
    for t, g in zip(self._vars, self._grad):
      if isinstance(g, tf.IndexedSlices):
        tensor_to_avg.append(
            tf.reshape(tf.unsorted_segment_sum(g.values,
                                               g.indices,
                                               g.dense_shape[0]),
                       shape=t.get_shape()))
      else:
        tensor_to_avg.append(g)
    avg_op = self._moving_averager.apply(tensor_to_avg)
    grad_var_ops.append(avg_op)
    with tf.control_dependencies([avg_op]):
      self._grad_avg = [self._moving_averager.average(val)
                        for val in tensor_to_avg]
      self._grad_avg_squared = [tf.square(val) for val in self._grad_avg]
      self._grad_avg_squared = tf.add_n([tf.reduce_sum(val)
                                         for val in self._grad_avg_squared])
    # Compute Variance
    self._grad_var = self._grad_norm_squared_avg - self._grad_avg_squared
    return grad_var_ops  # C_t

  def _dist_to_opt(self):
    """Distance to optimum.

    Returns:
      D_t ops
    """
    dist_to_opt_ops = []
    # Running average of the norm of gradeint
    self._grad_norm = tf.sqrt(self._grad_norm_squared)
    avg_op = self._moving_averager.apply([self._grad_norm,])
    dist_to_opt_ops.append(avg_op)
    with tf.control_dependencies([avg_op]):
      self._grad_norm_avg = self._moving_averager.average(self._grad_norm)
      # Single iteration distance estimation, note here
      # self._grad_norm_avg is per variable
      self._d_t = self._grad_norm_avg / self._grad_norm_squared_avg
    # Running average of distance
    avg_op = self._moving_averager.apply([self._d_t])
    dist_to_opt_ops.append(avg_op)
    with tf.control_dependencies([avg_op]):
      self._dist_to_opt_avg = tf.identity(
          self._moving_averager.average(self._d_t))
    return dist_to_opt_ops  # D_t

  def _prepare_variables(self):
    """Prepare Variables for YellowFin.

    Returns:
      Grad**2, Norm, Norm**2, Mean(Norm**2) ops
    """
    self._moving_averager = tf.train.ExponentialMovingAverage(
        decay=self._beta, zero_debias=self._zero_debias)
    assert self._grad
    # List for the returned Operations
    prepare_variables_op = []

    # Get per var g**2 and norm**2
    self._grad_squared = []
    self._grad_norm_squared = []

    # Gradient squared
    for v, g in zip(self._vars, self._grad):
      if g is None: continue
      with ops.colocate_with(v):
        self._grad_squared.append(tf.square(g))

    # Norm squared.
    self._grad_norm_squared = [tf.reduce_sum(g_sq)
                               for g_sq in self._grad_squared]

    # The following running average on squared norm of gradient
    # is shared by grad_var and dist_to_opt
    avg_op = self._moving_averager.apply(self._grad_norm_squared)

    with tf.control_dependencies([avg_op]):
      self._grad_norm_squared_avg = [self._moving_averager.average(val)
                                     for val in self._grad_norm_squared]
      self._grad_norm_squared = tf.add_n(self._grad_norm_squared)
      self._grad_norm_squared_avg = tf.add_n(self._grad_norm_squared_avg)

    prepare_variables_op.append(avg_op)
    return tf.group(*prepare_variables_op)

  def _get_lr_tensor(self):
    """Get lr minimzing the surrogate.

    Returns:
      The lr_t.
    """
    lr = (1.0 - tf.sqrt(self._mu))**2 / self._h_min
    return lr

  def _get_mu_tensor(self):
    """Get the min mu which minimize the surrogate.

    Returns:
      The mu_t.
    """
    const_fact = self._dist_to_opt_avg**2 * self._h_min**2 / 2 / self._grad_var
    coef = tf.Variable([-1.0, 3.0, 0.0, 1.0],
                       dtype=tf.float32,
                       name="cubic_solver_coef")
    coef = tf.scatter_update(coef,
                             tf.constant(2),
                             -(3 + const_fact))
    roots = tf.py_func(np.roots,
                       [coef],
                       Tout=tf.complex64,
                       stateful=False)

    # Filter out the correct root
    root_idx = tf.logical_and(
        tf.logical_and(
            tf.greater(tf.real(roots), tf.constant(0.0)),
            tf.less(tf.real(roots), tf.constant(1.0))),
        tf.less(tf.abs(tf.imag(roots)), 1e-5))

    # In case there are two duplicated roots satisfying the above condition
    root = tf.reshape(tf.gather(tf.gather(roots, tf.where(root_idx)),
                                tf.constant(0)),
                      shape=[])

    dr = self._h_max / self._h_min
    mu = tf.maximum(tf.real(root)**2, ((tf.sqrt(dr) - 1)/(tf.sqrt(dr) + 1))**2)
    return mu

  def _yellowfin(self):
    """YellowFin auto-tuning optimizer based on momentum SGD.

    Returns:
      YF ops
        (Curvature range,
         Grad_variance,
         Dist_to_opt,
         Single-Step,
         Auto-Tuning)
    """
    # List for the returned Operations.
    yellowfin_ops = []

    # Curvature range ops.
    curv_range_ops = self._curvature_range()
    yellowfin_ops += curv_range_ops
    # Estimate of gradient Variance ops.
    grad_var_ops = self._grad_variance()
    yellowfin_ops += grad_var_ops
    # Distance to optimum ops.
    dist_to_opt_ops = self._dist_to_opt()
    yellowfin_ops += dist_to_opt_ops

    # Single-Step: minimizes the surrogate for the expected
    # squared distance from the optimum of a local quadratic
    # approximation after a single step while keeping all directions in the
    # robust region.
    self._mu = tf.identity(tf.cond(self._do_tune, self._get_mu_tensor,
                                   lambda: self._mu_var))
    with tf.control_dependencies([self._mu]):
      self._lr = tf.identity(tf.cond(self._do_tune,
                                     self._get_lr_tensor,
                                     lambda: self._lr_var))

    # Tune learning rate and momentum.
    with tf.control_dependencies([self._mu, self._lr]):
      self._mu = self._beta * self._mu_var + (1 - self._beta) * self._mu
      self._lr = self._beta * self._lr_var + (1 - self._beta) * self._lr
      yellowfin_ops.append(tf.assign(self._mu_var, self._mu))
      yellowfin_ops.append(tf.assign(self._lr_var, self._lr))

    yellowfin_ops = tf.group(*yellowfin_ops)
    return yellowfin_ops

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Applying gradients aand tune hyperparams with YellowFin.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name:  Optional name for the returned operation. Default to the
        name passed to the Optimizer constructor.

    Returns:
        (A group of operations)
        Variable Update with Momentum ops,
        YellowFin ops(Curvature, Variance, Distance) ops,
        SingleStep and lr_mu tuning ops,
        Step increment ops.

    """
    self._grad, self._vars = zip(*[(g, t)
                                   for g, t in grads_and_vars if g is not None])

    # Var update with Momentum.
    with tf.variable_scope("apply_updates"):
      # Gradient Clipping?
      if self._clip_thresh_var is not None:
        self._grads_clip, self._grads_norm = tf.clip_by_global_norm(
            self._grad, self._clip_thresh_var)

        apply_grad_op = self._momentum_optimizer.apply_gradients(
            zip(self._grads_clip, self._vars), global_step=global_step)
      else:
        apply_grad_op = self._momentum_optimizer.apply_gradients(
            zip(self._grad, self._vars), global_step=global_step)

    # Begin lr and mu tuning.
    with tf.variable_scope("prepare_yellowFin_variables"):
      prepare_variables_op = self._prepare_variables()

    with tf.variable_scope("yellowfin"):
      with tf.control_dependencies([prepare_variables_op]):
        yellowfin_op = self._yellowfin()

    # Update YellowFin step variable.
    with tf.control_dependencies([yellowfin_op]):
      self._increment_step_op = tf.assign_add(self._step, 1).op

    return tf.group(apply_grad_op,
                    prepare_variables_op,
                    yellowfin_op,
                    self._increment_step_op)

  def compute_gradients(self,
                        loss,
                        var_list,
                        global_step=None,
                        gate_gradients=GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        name=None,
                        grad_loss=None):
    """Compute gradients through momentum optimizer.

    Args:
      loss: A Tensor containing the value to minimize.
      var_list: Optional list or tuple of tf.Variable to update
        to minimize loss. Defaults to the list of variables collected
        in the graph under the key GraphKey.TRAINABLE_VARIABLES.
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      gate_gradients: How to gate the computation of gradients.
        Can be GATE_NONE, GATE_OP, or GATE_GRAPH.
      aggregation_method: Specifies the method used to combine
        gradient terms. Valid values are defined in the class AggregationMethod.
      colocate_gradients_with_ops: If True, try colocating gradients with
        the corresponding op.
      name: Optional name for the returned operation. Default to the name
        passed to the Optimizer constructor.
      grad_loss: Optional. A Tensor holding the gradient computed for loss.

    Returns:
      A list of (gradient, variable) pairs. Variable is always present,
        but gradient can be None.
    """
    return self._momentum_optimizer.compute_gradients(
        loss,
        var_list=var_list,
        gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)

  def minimize(self,
               loss,
               global_step=None,
               var_list=None,
               gate_gradients=GATE_OP,
               aggregation_method=None,
               colocate_gradients_with_ops=False,
               name=None,
               grad_loss=None):
    """Adapted from Tensorflow Optimizer base class member function.

    Add operations to minimize `loss` by updating `var_list`.
    This method simply combines calls `compute_gradients()` and
    `apply_gradients()`. If you want to process the gradient before applying
    them call `tf.gradients()` and `self.apply_gradients()` explicitly instead
    of using this function.

    Args:
      loss: A Tensor containing the value to minimize.
      global_step: Optional Variable to increment by one after the variables
        have been updated.
      var_list: Optional list or tuple of Variable objects to update to
        minimize loss. Defaults to the list of variables collected in
        the graph under the key GraphKeys.TRAINABLE_VARIABLES.
      gate_gradients: How to gate the computation of gradients.
        Can be GATE_NONE, GATE_OP, or GATE_GRAPH.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class AggregationMethod.
      colocate_gradients_with_ops: If True, try colocating gradients with
        the corresponding op.
      name: Optional name for the returned operation.
      grad_loss: Optional. A Tensor holding the gradient computed for loss.

    Returns:
      An Operation that updates the variables in var_list.
        If global_step was not None, that operation also increments global_step.

    Raises:
      ValueError: if no gradients are provided for any variable.
    """
    grads_and_vars = self._optimizer.compute_gradients(
        loss,
        var_list=var_list,
        gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)

    vars_with_grad = [v for g, v in grads_and_vars if g is not None]
    if not vars_with_grad:
      raise ValueError(
          "No gradients provided for any variable, check your graph for ops"
          " that do not support gradients, between variables %s and loss %s." %
          ([str(v) for _, v in grads_and_vars], loss))
    for g, v in grads_and_vars:
      print("g ", g)
      print("v ", v)

    return self.apply_gradients(grads_and_vars, global_step=global_step)
