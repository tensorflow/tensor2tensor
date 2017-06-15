# Copyright 2017 Google Inc.
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

"""Utilities for creating Sparsely-Gated Mixture-of-Experts Layers.

See the most recent draft of our ICLR paper:
https://openreview.net/pdf?id=B1ckMDqlg
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

# Dependency imports

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.framework import function


def NoisyTopKGatingParams():
  """Hyperparams defining NoisyTopK Gating Network.

  Returns:
    a tf.contrib.training.HParams object
  """
  return tf.contrib.training.HParams(
      gating_class=NoisyTopKGating,
      num_experts=16,  # The number of experts
      k=2,  # 'The number of experts to use per example
      input_size=None,  # size of input to MoE.  Set by MoE class
      dtype=tf.float32,  # floating point data type
      initializer=tf.zeros_initializer(),  # initializer for weight matrices
      noisy_gating=True,  # Add tunable noise (necessary for load-balancing)
      noise_epsilon=1e-2,  # Added to noise stddev for numerical stability
  )


def FeedForwardExpertParams():
  """Hyperparameters defining feed-forward expert networks.

  Returns:
    a tf.contrib.training.HParams object
  """
  return tf.contrib.training.HParams(
      # The class that implements the expert network
      expert_class=FeedForwardExpert,
      input_size=None,  # Size of input to MoE.  Set by MoE class.
      # List of hidden layer sizes, or None for no hidden layers.
      # The length of this list determines the number of hidden layers
      hidden_layer_sizes=None,
      output_size=None,  # Size of output from MoE.  Set by MoE class.
      dtype=tf.float32,  # Floating point data type)
      # Activation function applied at each hidden layer)
      hidden_activation=tf.nn.relu,
      initializer=None,  # Optional initializer for weight matrices.)
      # If autoscale=True, At each hidden/output layer, multiply by
      # rsqrt(prev_layer_size / input_size).  This scaling happens
      # before application of hidden_activation)
      autoscale=True,)


def _SetInputOutputSizes(hp, input_size, output_size):
  """Fill in the input_size and output_size hyperparameters.

  This is used by LocalMixtureOfExperts and DistributedMixtureOfExperts to
  fill in the input_size and output_size on the gating parameters and expert
  parameters so that the user does not have to set them in multiple places.

  Args:
    hp: a hyperparameters
    input_size: an integer
    output_size: an integer
  """
  if hp.input_size is None:
    hp.input_size = input_size
  else:
    assert hp.input_size == input_size
  if output_size is not None:
    if hp.output_size is None:
      hp.output_size = output_size
    else:
      assert hp.output_size == output_size


class FeedForwardExpert(object):
  """An object representing a feed forward network (used as an expert).
  """

  def __init__(self, hp, name):
    """Creates a FeedForwardExpert.

    Args:
      hp: hyperparameters.  Call FeedForwardExpertParams() to create these.
      name: a string.
    """
    self._hp = hp
    hidden_layer_sizes = hp.hidden_layer_sizes or []
    num_layers = 1 + len(hidden_layer_sizes)
    layer_sizes = [hp.input_size] + hidden_layer_sizes + [hp.output_size]
    self._layer_sizes = layer_sizes
    self._w = []
    for layer in range(num_layers):
      shape = layer_sizes[layer:layer + 2]
      self._w.append(
          tf.get_variable('%s_layer_%d' % (name, layer), shape, hp.dtype,
                          hp.initializer))

  def Eval(self, x):
    """Evaluate the FeedForwardExpert on the given input.

    Args:
      x: a `Tensor` of shape `[batch_size, hp.input_size]`

    Returns:
      a `Tensor` of shape `[batch_size, hp.output_size]`
    """
    hp = self._hp
    num_layers = len(self._w)
    for i in xrange(num_layers):
      x = tf.matmul(x, self._w[i])
      if hp.autoscale and self._layer_sizes[i] != hp.input_size:
        x *= (self._layer_sizes[i] / hp.input_size)**-0.5
      if i + 1 < num_layers and hp.hidden_activation:
        x = hp.hidden_activation(x)
    return x

  @property
  def vars(self):
    return self._w


@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def ConvertGradientToTensor(x):
  """Identity operation whose gradient is converted to a `Tensor`.

  Currently, the gradient to `tf.concat` is particularly expensive to
  compute if dy is an `IndexedSlices` (a lack of GPU implementation
  forces the gradient operation onto CPU).  This situation occurs when
  the output of the `tf.concat` is eventually passed to `tf.gather`.
  It is sometimes faster to convert the gradient to a `Tensor`, so as
  to get the cheaper gradient for `tf.concat`.  To do this, replace
  `tf.concat(x)` with `ConvertGradientToTensor(tf.concat(x))`.

  Args:
    x: A `Tensor`.

  Returns:
    The input `Tensor`.
  """
  return x


class Parallelism(object):
  """Helper class for creating sets of parallel function calls.

  The purpose of this class is to replace this code:

      e = []
      f = []
      for i in xrange(len(devices)):
        with tf.device(devices[i]):
          e_, f_ = func(a[i], b[i], c)
          e.append(e_)
          f.append(f_)

  with this code:

      e, f = expert_utils.Parallelism(devices)(func, a, b, c)
  """

  def __init__(self,
               device_names_or_functions,
               reuse=None,
               caching_devices=None,
               daisy_chain_variables=False):
    """Create a Parallelism.

    Args:
      device_names_or_functions: A list of of length n, containing device names
        or device functions (see `tf.device`)
      reuse: True or None.  Whether to reuse variables created in the first
        replica in the subsequent replicas.
      caching_devices: Either `None`, or a list of length n containing device
        names.
      daisy_chain_variables: a boolean - if true, then copies variables in a
        daisy chain between devices.

    Returns:
      a Parallelism.
    """
    assert device_names_or_functions
    self._devices = device_names_or_functions
    self._n = len(device_names_or_functions)
    self._reuse = reuse
    self._caching_devices = self._MaybeRepeat(caching_devices)
    self._daisy_chain_variables = daisy_chain_variables

  def __call__(self, fn, *args, **kwargs):
    """A parallel set of function calls (using the specified devices).

    Args:
      fn: a function or a list of n functions.
      *args: additional args.  Each arg should either be not a list, or a list
         of length n.
      **kwargs: additional keyword args.  Each arg should either be not a
         list, or a list of length n.

    Returns:
      either a single list of length n (if fn does not return a tuple), or a
      tuple of lists of length n (if fn returns a tuple).
    """
    # Construct lists or args and kwargs for each function.
    if args:
      my_args = TransposeListOfLists([self._MaybeRepeat(arg) for arg in args])
    else:
      my_args = [[] for _ in xrange(self.n)]
    my_kwargs = [{} for _ in xrange(self.n)]
    for k, v in six.iteritems(kwargs):
      vals = self._MaybeRepeat(v)
      for i in xrange(self.n):
        my_kwargs[i][k] = vals[i]

    # Construct lists of functions.
    fns = self._MaybeRepeat(fn)

    # Now make the parallel call.
    outputs = []
    cache = {}
    for i in xrange(self.n):

      def DaisyChainGetter(getter, name, *args, **kwargs):
        """Get a variable and cache in a daisy chain."""
        device_var_key = (self._devices[i], name)
        if device_var_key in cache:
          # if we have the variable on the correct device, return it.
          return cache[device_var_key]
        if name in cache:
          # if we have it on a different device, copy it from the last device
          v = tf.identity(cache[name])
        else:
          var = getter(name, *args, **kwargs)
          v = tf.identity(var._ref())  # pylint: disable=protected-access
        # update the cache
        cache[name] = v
        cache[device_var_key] = v
        return v

      # Variable scope will not reset caching_device on reused variables,
      # so we make a custom getter that uses identity to cache the variable.
      # pylint: disable=cell-var-from-loop
      def CachingGetter(getter, name, *args, **kwargs):
        v = getter(name, *args, **kwargs)
        key = (self._caching_devices[i], name)
        if key in cache:
          return cache[key]
        with tf.device(self._caching_devices[i]):
          ret = tf.identity(v._ref())  # pylint: disable=protected-access
        cache[key] = ret
        return ret

      if self._daisy_chain_variables:
        custom_getter = DaisyChainGetter
      elif self._caching_devices:
        custom_getter = CachingGetter
      else:
        custom_getter = None
      # pylint: enable=cell-var-from-loop
      with tf.name_scope('parallel_%d' % i):
        with tf.variable_scope(
            tf.get_variable_scope(),
            reuse=True if i > 0 and self._reuse else None,
            caching_device=self._caching_devices[i],
            custom_getter=custom_getter):
          with tf.device(self._devices[i]):
            outputs.append(fns[i](*my_args[i], **my_kwargs[i]))
    if isinstance(outputs[0], tuple):
      outputs = list(zip(*outputs))
      outputs = tuple([list(o) for o in outputs])
    return outputs

  @property
  def n(self):
    return self._n

  @property
  def devices(self):
    return self._devices

  def _MaybeRepeat(self, x):
    """Utility function for processing arguments that are singletons or lists.

    Args:
      x: either a list of self.n elements, or not a list.

    Returns:
      a list of self.n elements.
    """
    if isinstance(x, list):
      assert len(x) == self.n
      return x
    else:
      return [x] * self.n


def Parallel(device_names_or_functions, fn, *args):
  """Deprecated interface.

  Use `Parallelism(device_names_or_functions)(fn, *args)` instead.

  Args:
    device_names_or_functions: A list of length n.
    fn: a function or a list of n functions.
    *args: additional args.  Each arg should either be not a list, or a list
       of length n.

  Returns:
    either a single list of length n (if fn does not return a tuple), or a
    tuple of lists of length n (if fn returns a tuple).
  """
  return Parallelism(device_names_or_functions)(fn, *args)


def _RowwiseUnsortedSegmentSum(values, indices, n):
  """UnsortedSegmentSum on each row.

  Args:
    values: a `Tensor` with shape `[batch_size, k]`.
    indices: an integer `Tensor` with shape `[batch_size, k]`.
    n: an integer.
  Returns:
    A `Tensor` with the same type as `values` and shape `[batch_size, n]`.
  """
  batch, k = tf.unstack(tf.shape(indices), num=2)
  indices_flat = tf.reshape(indices, [-1]) + tf.div(tf.range(batch * k), k) * n
  ret_flat = tf.unsorted_segment_sum(
      tf.reshape(values, [-1]), indices_flat, batch * n)
  return tf.reshape(ret_flat, [batch, n])


def _NormalDistributionCDF(x, stddev):
  """Evaluates the CDF of the normal distribution.

  Normal distribution with mean 0 and standard deviation stddev,
  evaluated at x=x.

  input and output `Tensor`s have matching shapes.

  Args:
    x: a `Tensor`
    stddev: a `Tensor` with the same shape as `x`.

  Returns:
    a `Tensor` with the same shape as `x`.

  """
  return 0.5 * (1.0 + tf.erf(x / (math.sqrt(2) * stddev + 1e-20)))


def _ProbInTopK(clean_values, noisy_values, noise_stddev, noisy_top_values, k):
  """Helper function to NoisyTopKGating.

  Computes the probability that value is in top k, given different random noise.

  This gives us a way of backpropagating from a loss that balances the number
  of times each expert is in the top k experts per example.

  In the case of no noise, pass in None for noise_stddev, and the result will
  not be differentiable.

  Args:
    clean_values: a `Tensor` of shape [batch, n].
    noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
      normally distributed noise with standard deviation noise_stddev.
    noise_stddev: a `Tensor` of shape [batch, n], or None
    noisy_top_values: a `Tensor` of shape [batch, m].
       'values' Output of tf.top_k(noisy_top_values, m).  m >= k+1
    k: an integer.

  Returns:
    a `Tensor` of shape [batch, n].
  """
  batch = tf.shape(clean_values)[0]
  m = tf.shape(noisy_top_values)[1]
  top_values_flat = tf.reshape(noisy_top_values, [-1])
  # we want to compute the threshold that a particular value would have to
  # exceed in order to make the top k.  This computation differs depending
  # on whether the value is already in the top k.
  threshold_positions_if_in = tf.range(batch) * m + k
  threshold_if_in = tf.expand_dims(
      tf.gather(top_values_flat, threshold_positions_if_in), 1)
  is_in = tf.greater(noisy_values, threshold_if_in)
  if noise_stddev is None:
    return tf.to_float(is_in)
  threshold_positions_if_out = threshold_positions_if_in - 1
  threshold_if_out = tf.expand_dims(
      tf.gather(top_values_flat, threshold_positions_if_out), 1)
  # is each value currently in the top k.
  prob_if_in = _NormalDistributionCDF(clean_values - threshold_if_in,
                                      noise_stddev)
  prob_if_out = _NormalDistributionCDF(clean_values - threshold_if_out,
                                       noise_stddev)
  prob = tf.where(is_in, prob_if_in, prob_if_out)
  return prob


def CVSquared(x):
  """The squared coefficient of variation of a sample.

  Useful as a loss to encourage a positive distribution to be more uniform.
  Epsilons added for numerical stability.
  Returns 0 for an empty Tensor.

  Args:
    x: a `Tensor`.

  Returns:
    a `Scalar`.
  """
  epsilon = 1e-10
  float_size = tf.to_float(tf.size(x)) + epsilon
  mean = tf.reduce_sum(x) / float_size
  variance = tf.reduce_sum(tf.square(x - mean)) / float_size
  return variance / (tf.square(mean) + epsilon)


def MaxOverload(load):
  """The load of the hardest-hit device relative to average.

  This is useful for monitoring the performance of MoEs.

  The load of an expert is the number of examples assigned to that expert.
  The load of a device is the sum of the loads of all experts on that device.

  The input to this function is generally the 'load' output of
  DistributedMixtureOfExperts.Eval(), which is either a 1d or 2d `Tensor` of
  per-expert loads.  In either case, the fist dimension corresponds to devices.

  This function sums over all dimensions other than dimension zero, then
  computes the ratio of the maxmium value to the mean value.

  Args:
    load: a 1d or 2d `Tensor`.

  Returns:
    a `Scalar`.
  """
  per_device_load = tf.reduce_sum(tf.reshape(load, [tf.shape(load)[0], -1]), 1)
  return (tf.reduce_max(per_device_load) /
          (tf.reduce_mean(per_device_load) + 1e-10))


def _GatesToLoad(gates):
  """Compute the true load per expert, given the gates.

  The load is the number of examples for which the corresponding gate is >0.

  Args:
    gates: a `Tensor` of shape [batch_size, n]
  Returns:
    a float32 `Tensor` of shape [n]
  """
  return tf.reduce_sum(tf.to_float(gates > 0), 0)


def _MyTopK(x, k):
  """GPU-compatible version of top-k that works for very small constant k.

  Calls argmax repeatedly.

  Args:
    x: a 2d Tensor.
    k: a small integer.

  Returns:
    values: a Tensor of shape [batch_size, k]
    indices: a int32 Tensor of shape [batch_size, k]
  """
  if k > 10:
    return tf.nn.top_k(x, k)
  values = []
  indices = []
  depth = tf.shape(x)[1]
  for i in xrange(k):
    values.append(tf.reduce_max(x, 1))
    argmax = tf.argmax(x, 1)
    indices.append(argmax)
    if i + 1 < k:
      x += tf.one_hot(argmax, depth, -1e9)
  return tf.stack(values, axis=1), tf.to_int32(tf.stack(indices, axis=1))


class NoisyTopKGating(object):
  """Noisy top-k gating network.

  See paper: https://arxiv.org/abs/1701.06538.
  """

  def __init__(self, hp, name):
    """Create a NoisyTopKGating network.

    Args:
      hp: a hyperparameters created by NoisyTopKGatingParams()
      name: a string
    """
    self._vars = []
    self._hp = hp
    self._w_gate = tf.get_variable('%s_gate' % name,
                                   [hp.input_size,
                                    hp.num_experts], hp.dtype, hp.initializer)
    self._vars.append(self._w_gate)
    if hp.noisy_gating:
      self._w_noise = tf.get_variable('%s_noise' % name,
                                      [hp.input_size, hp.num_experts], hp.dtype,
                                      hp.initializer)
      self._vars.append(self._w_noise)

  def Eval(self, x, train=True, summaries=False):
    """Compute noisy top-k gating.

    Args:
      x: a `Tensor` of shape `[batch_size, input_size]`.
      train: a boolean `Scalar`.   Setting this to false turns off noise.
      summaries: a boolean.  Whether to add summaries.
    Returns:
      gates: a `Tensor` of shape `[batch_size, n]`
      load: a `Tensor` of shape `[n]`.
        If we are using noise, this is a smooth approximation of the load,
        and you can define a loss in terms of it to help with load-balancing.
    """
    with tf.variable_scope('NoisyTopKGating'):
      hp = self._hp
      clean_logits = tf.matmul(x, self._w_gate)
      if hp.noisy_gating:
        raw_noise_stddev = tf.matmul(x, self._w_noise)
        noise_stddev = ((tf.nn.softplus(raw_noise_stddev) + hp.noise_epsilon) *
                        (tf.to_float(train)))
        noisy_logits = clean_logits + (
            tf.random_normal(tf.shape(clean_logits)) * noise_stddev)
        logits = noisy_logits
        if summaries:
          tf.summary.histogram('noisy_logits', noisy_logits)
          tf.summary.histogram('noise_stddev', noise_stddev)
      else:
        logits = clean_logits
      top_logits, top_indices = _MyTopK(logits, min(hp.k + 1, hp.num_experts))
      top_k_logits = tf.slice(top_logits, [0, 0], [-1, hp.k])
      top_k_indices = tf.slice(top_indices, [0, 0], [-1, hp.k])
      top_k_gates = tf.nn.softmax(top_k_logits)
      # This will be a `Tensor` of shape `[batch_size, n]`, with zeros in the
      # positions corresponding to all but the top k experts per example.
      gates = _RowwiseUnsortedSegmentSum(top_k_gates, top_k_indices,
                                         hp.num_experts)
      if hp.noisy_gating and hp.k < hp.num_experts:
        load = tf.reduce_sum(
            _ProbInTopK(clean_logits, noisy_logits, noise_stddev, top_logits,
                        hp.k), 0)
      else:
        load = _GatesToLoad(gates)
      if summaries:
        tf.summary.histogram('importance', tf.reduce_sum(gates, 0))
        tf.summary.histogram('load', load)
      return gates, load

  @property
  def vars(self):
    return self._vars


class LocalMixtureOfExperts(object):
  """A MoE on a single device.
  """

  def __init__(self, gating_hp, expert_hp, input_size, output_size, name):
    """Create a LocalMixtureOfExperts.

    Args:
      gating_hp: hyperparameters for the gating network.
        e.g. NoisyTopKGatingParams()
      expert_hp: hyperparameters for the expert networks.
        e.g. FeedForwardExpertParams()
      input_size: an integer.
      output_size: an integer.
      name: a string.
    """
    self._name = name
    _SetInputOutputSizes(gating_hp, input_size, None)
    _SetInputOutputSizes(expert_hp, input_size, output_size)
    self._gating_hp = gating_hp
    self._gating = gating_hp.gating_class(gating_hp, name + '_gating')
    self._expert_hp = expert_hp
    self._experts = [
        expert_hp.expert_class(expert_hp, name + '_%d' % i)
        for i in xrange(gating_hp.num_experts)
    ]

  def Eval(self,
           x,
           train=True,
           per_example_multiplier=None,
           summaries=False,
           identifiers=None):
    """Evaluate mixture of experts.

    We provide a convenient debugging tool for determining the set of examples
    that we passed to each expert.  The caller may provide a `Tensor` of
    "identifiers", of any type whose first dimension matches the number of
    input examples. The function will then return a list
    "expert_to_identifiers", with one `Tensor` for each expert containing the
    identifiers for all examples assigned to that expert.  A parallel list of
    `Tensor`s, "expert_to_gates", is also returned, containing the
    corresponding gate values.

    Args:
      x: a `Tensor` of shape `[batch_size, input_size]`
      train: a boolean Scalar.  Are we in training mode?
      per_example_multiplier: an optional `Tensor` of shape `[batch_size]` which
        gets multiplied into the gate values.  If this LocalMixtureOfExperts
        represents one secondary MoE in a hierarchical MoE, then we pass in
        in the gate values from the primary gating function here.  This causes
        the computed values (`y`, `importance` and `expert_to_gates`) to also
        reflect the primary gate values.
      summaries: an boolean.  Enable summaries.
      identifiers: an optional `Tensor` whose first dimension is equal to
        batch_size.

    Returns:
      y: a `Tensor` of shape `[batch_size, output_size]`.  Output of the MoE.
      importance: a `Tensor` of shape `[n]`.  Batchwise sum of gates.
      load: a `Tensor` of shape `[n]`.  Smooth estimator of the number of
        examples passed to each expert.  This is useful for load-balancing,
        as any gradient on this `Tensor` will back-propagate to the gating
        network.
      expert_to_identifiers:  if `identifiers` was passed in, a list of
        length `num_experts`.  Each element is a `Tensor` whose shape matches
        that of `identifiers` in all but the first dimension.  Contains the
        slices of `identifiers` corresponding to the batch elements that were
        dispatched to that expert.
      expert_to_gates:  A list of length `num_experts`.  Each element contains
        a 1-dimensional tensor
    """
    gating_hp = self._gating_hp
    gates, load = self._gating.Eval(x, train, summaries)
    if per_example_multiplier is not None:
      gates *= tf.expand_dims(per_example_multiplier, 1)
    dispatcher = SparseDispatcher(gating_hp.num_experts, gates)
    expert_input = dispatcher.Dispatch(x)
    expert_output = [
        self._experts[i].Eval(expert_input[i])
        for i in xrange(gating_hp.num_experts)
    ]
    y = dispatcher.Combine(expert_output)
    if identifiers is not None:
      expert_to_identifiers = dispatcher.Dispatch(identifiers)
    else:
      expert_to_identifiers = None
    return (y, tf.reduce_sum(gates, 0), load, expert_to_identifiers,
            dispatcher.ExpertToGates())

  @property
  def vars(self):
    ret = []
    for x in self._experts:
      ret.extend(x.vars)
    ret.extend(self._gating.vars)
    return ret


class DistributedMixtureOfExperts(object):
  """Distributed (optionally Hierarchical) Mixture of Experts.

  This class implements the scheme described in our paper.
  See link at the top of this file.

  The model is trained synchronously using one large TF graph using
  multiple devices.

  The conventional (non-MoE) layers use data-parallelism, with each device
  processing a subset of the training batch.   We call these datashards.

  The MoE layer (this object) uses model parallelism.  Each expert is assigned
  to a particular device, which hosts the expert parameters and performs the
  expert computation for all examples assigned to that expert.  In the case
  of a hierarchical MoE, each second-level MoE is assigned to a device.
  """

  def __init__(self, primary_gating_hp, secondary_gating_hp, expert_hp,
               input_size, output_size, expert_devices, name):
    """Create a DistributedMixtureOfExperts.

    If `secondary_gating_hp` is `None`, then this is a flat MoE with
    `primary_gating_hp.num_experts` experts. Otherwise, this is a hierarchical
    MoE with `primary_gating_hp.num_experts` groups of
    `secondary_gating_hp.num_experts` experts.

    The assignemnt of experts (or groups of experts) to devices is by
    round-robin.   So to make equal use of all the devices, one should set
    `primary_gating_hp.num_experts` to the number of devices or a multiple
    thereof.

    Args:
      primary_gating_hp: hyperparameters for the primary gating network.
        e.g. NoisyTopKGatingParams().
      secondary_gating_hp: hyperparameters for the secondary gating network.
        e.g. NoisyTopKGatingParams().  None indicates a flat MoE.
      expert_hp: hyperparameters for the expert networks.
        e.g. FeedForwardExpertParams()
      input_size: an integer.
      output_size: an integer.
      expert_devices: a list of device strings.  The devices to be used for
        the experts.
      name: a string.
    """
    self._name = name
    # fill in the missing values in the hyperparameters
    _SetInputOutputSizes(primary_gating_hp, input_size, None)
    _SetInputOutputSizes(expert_hp, input_size, output_size)
    self._is_hierarchical = secondary_gating_hp is not None
    self._primary_gating_hp = primary_gating_hp
    self._primary_gating = primary_gating_hp.gating_class(
        primary_gating_hp, name + '_primary_gating')
    n1 = self._primary_gating_hp.num_experts
    # round robin assignment of experts to devices.
    expert_devices = [
        expert_devices[i % len(expert_devices)] for i in xrange(n1)
    ]
    self._expert_devices = expert_devices
    self._all_vars = []
    self._all_vars.extend(self._primary_gating.vars)
    if self._is_hierarchical:
      # hierarchical MoE
      self._secondary_moe = []
      for i in xrange(n1):
        with tf.device(expert_devices[i]):
          secondary_moe = LocalMixtureOfExperts(secondary_gating_hp, expert_hp,
                                                input_size, output_size,
                                                '%s_secondary_%d' % (name, i))
          self._secondary_moe.append(secondary_moe)
          self._all_vars.extend(secondary_moe.vars)
    else:
      # flat MoE
      self._experts = []
      for i in xrange(n1):
        with tf.device(expert_devices[i]):
          expert = expert_hp.expert_class(expert_hp, name + '_%d' % i)
          self._experts.append(expert)
          self._all_vars.extend(expert.vars)

  def Eval(self,
           datashard_devices,
           xs,
           train=True,
           summaries=False,
           identifiers=None,
           shadow_xs=None):
    """Evaluate MoE on given inputs.

    This class is designed for the case where the rest of the model is using
    data parallelism.   We receive an array of input `Tensor`s, one per
    datashard, and we produce a list of output Tensors, one per datashard.

    We provide a convenient debugging tool for determining the set of examples
    that we passed to each expert.  The caller may provide a `Tensor` of
    "identifiers", of any type whose first dimension matches the number of
    input examples. The function will then return a list
    "expert_to_identifiers", with one `Tensor` for each expert containing the
    identifiers for all examples assigned to that expert.  A parallel list of
    `Tensor`s, "expert_to_gates", is also returned, containing the
    corresponding gate values.

    Args:
      datashard_devices: a `list` of device strings of length `num_datashards`.
        Which devices to use for the output tensors.
      xs: A `list` of `Tensor`s of length `num_datashards`.  Each has shape
        `[batch_size[d], input_size].
      train: a boolean `Scalar`.   When train=`True`, noise is added to the
        gating function.
      summaries: a boolean.  Whether to write summaries.
      identifiers: an optional list of tensors.
        Each tensor has shape [<batch_size[datashard]>, extra_dims]
      shadow_xs: Optional `list` of `Tensor`s of length `num_datashards`.  Each
        has shape `[batch_size[d], input_size]. Shadow_xs is useful if you want
        to dispatch a transformed version of xs to the experts, but you want
        untransformed xs for the gating network.

    Returns:
      ys: the output (a list of one tensor per datashard).  Each has shape
         `[batch_size[d], output_size].
      importance: a `Tensor` of shape `[n]` for a flat MoE or `[n1, n2]` for a
         hierarchical MoE.  Batchwise sum of gates.
      load:  a `Tensor` of shape `[n]` for a flat MoE or `[n1, n2]` for a
         hierarchical MoE.  Smooth estimator of the number of
         examples passed to each expert.  This is useful for load-balancing,
         as any gradient on this `Tensor` will back-propagate to the gating
         network.
      expert_to_identifiers:  if `identifiers` was passed in, a list of
         length `num_experts`.  Each element is a `Tensor` whose shape matches
         that of `identifiers` in all but the first dimension.  Contains the
         slices of `identifiers` corresponding to the batch elements that were
         dispatched to that expert.
      expert_to_gates: a list of one tensor per expert.
         Each tensor has shape [<num_examples[expert]>]

    """
    n1 = self._primary_gating_hp.num_experts
    epsilon = 1e-10
    assert len(datashard_devices) == len(xs)
    num_datashards = len(xs)
    expert_devices = self._expert_devices
    has_identifiers = identifiers is not None
    # pylint: disable=unbalanced-tuple-unpacking
    primary_gates, primary_smooth_load = Parallel(
        datashard_devices, self._primary_gating.Eval, xs, train,
        [summaries] + [False] * (num_datashards - 1))
    primary_importance = tf.add_n(
        Parallel(datashard_devices, tf.reduce_sum, primary_gates, 0))
    primary_smooth_load = tf.add_n(primary_smooth_load)
    primary_true_load = tf.add_n(
        Parallel(datashard_devices, _GatesToLoad, primary_gates))
    primary_dispatcher = DistributedSparseDispatcher(
        datashard_devices, expert_devices, primary_gates)

    if shadow_xs is None:
      secondary_input = primary_dispatcher.Dispatch(xs)
    else:
      secondary_input = primary_dispatcher.Dispatch(shadow_xs)

    primary_expert_to_identifiers = (primary_dispatcher.Dispatch(identifiers)
                                     if has_identifiers else None)
    primary_expert_to_gates = primary_dispatcher.ExpertToGates()
    if not self._is_hierarchical:
      # one-level distributed mixture of experts
      secondary_output = Parallel(expert_devices, lambda a, b: a.Eval(b),
                                  self._experts, secondary_input)
      ys = primary_dispatcher.Combine(secondary_output)
      return (ys, primary_importance, primary_smooth_load,
              primary_expert_to_identifiers, primary_expert_to_gates)
    # two-level hierarchical MoE
    (secondary_output, secondary_importance, secondary_load,
     secondary_expert_to_identifiers, secondary_expert_to_gates) = (Parallel(
         expert_devices, [m.Eval for m in self._secondary_moe], secondary_input,
         train, primary_expert_to_gates, [summaries] + [False] * (n1 - 1),
         primary_expert_to_identifiers))
    # pylint: enable=unbalanced-tuple-unpacking
    ys = primary_dispatcher.Combine(secondary_output, multiply_by_gates=False)
    importance = tf.stack(secondary_importance)
    load = tf.stack(secondary_load) * tf.expand_dims(primary_smooth_load / (
        primary_true_load + epsilon), 1)
    expert_to_identifiers = []
    if identifiers is not None:
      for el in secondary_expert_to_identifiers:
        expert_to_identifiers.extend(el)
    expert_to_gates = []
    for el in secondary_expert_to_gates:
      expert_to_gates.extend(el)
    return (ys, importance, load, expert_to_identifiers, expert_to_gates)

  @property
  def vars(self):
    return self._all_vars


class SparseDispatcher(object):
  """Helper for implementing a mixture of experts.

  Example use:

  gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
  inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
  experts: a list of length `num_experts` containing sub-networks.

    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.Dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.Combine(expert_outputs)

  The preceding code sets the output for a particular example b to:
  output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))

  This class takes advantage of sparsity in the gate matrix by including in the
  `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
  """

  def __init__(self, num_experts, gates):
    """Create a SparseDispatcher.

    Args:
      num_experts: an integer.
      gates: a `Tensor` of shape `[batch_size, num_experts]`.

    Returns:
      a SparseDispatcher
    """
    self._gates = gates
    self._num_experts = num_experts

    where = tf.to_int32(tf.where(tf.transpose(gates) > 0))
    self._expert_index, self._batch_index = tf.unstack(where, num=2, axis=1)
    self._part_sizes_tensor = tf.reduce_sum(tf.to_int32(gates > 0), [0])
    self._nonzero_gates = tf.gather(
        tf.reshape(self._gates, [-1]),
        self._batch_index * num_experts + self._expert_index)

  def Dispatch(self, inp):
    """Create one input Tensor for each expert.

    The `Tensor` for a expert `i` contains the slices of `inp` corresponding
    to the batch elements `b` where `gates[b, i] > 0`.

    Args:
      inp: a `Tensor` of shape '[batch_size, <extra_input_dims>]`
    Returns:
      a list of `num_experts` `Tensor`s with shapes
        `[expert_batch_size_i, <extra_input_dims>]`.
    """
    inp = tf.gather(inp, self._batch_index)
    return tf.split(inp, self._part_sizes_tensor, 0)

  def Combine(self, expert_out, multiply_by_gates=True):
    """Sum together the expert output, weighted by the gates.

    The slice corresponding to a particular batch element `b` is computed
    as the sum over all experts `i` of the expert output, weighted by the
    corresponding gate values.  If `multiply_by_gates` is set to False, the
    gate values are ignored.

    Args:
      expert_out: a list of `num_experts` `Tensor`s, each with shape
        `[expert_batch_size_i, <extra_output_dims>]`.
      multiply_by_gates: a boolean

    Returns:
      a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
    """
    # see comments on ConvertGradientToTensor
    stitched = ConvertGradientToTensor(tf.concat(expert_out, 0))
    if multiply_by_gates:
      stitched *= tf.expand_dims(self._nonzero_gates, 1)
    combined = tf.unsorted_segment_sum(stitched, self._batch_index,
                                       tf.shape(self._gates)[0])
    return combined

  def ExpertToGates(self):
    """Gate values corresponding to the examples in the per-expert `Tensor`s.

    Returns:
      a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
          and shapes `[expert_batch_size_i]`
    """
    return tf.split(self._nonzero_gates, self._part_sizes_tensor, 0)

  @property
  def part_sizes(self):
    return self._part_sizes_tensor


class DistributedSparseDispatcher(object):
  """A distributed version of SparseDispatcher.

  Instead of one batch of input examples, we simultaneously process
  num_datashards batches of input examples.  The per-expert `Tensor`s contain
  a combination of examples from the different datashards.

  Each datashard is associated with a particular device and each expert is
  associated with a particular device.  All per-datashard and per-expert
  `Tensor`s are created on those devices.  There is no single-device bottleneck.
  """

  def __init__(self, datashard_devices, expert_devices, gates):
    """Create a DistributedSparseDispatcher.

    Args:
      datashard_devices: a list of num_datashards device strings.
      expert_devices: a list of num_experts device strings.
      gates: a list of num_datashards `Tensor`s of shapes
        `[batch_size[d], num_experts]`.

    Returns:
      a DistributedSparseDispatcher
    """
    self._gates = gates
    self._num_experts = len(expert_devices)
    assert len(gates) == len(datashard_devices)
    self._num_datashards = len(gates)
    self._datashard_devices = datashard_devices
    self._expert_devices = expert_devices
    self._dispatchers = Parallel(self._datashard_devices, SparseDispatcher,
                                 self._num_experts, gates)

  def Dispatch(self, inp):
    """Create one input Tensor for each expert.

    Args:
      inp: a list of length num_datashards `Tensor`s with shapes
        `[batch_size[d], <extra_input_dims>]`.
    Returns:
      a list of `num_experts` `Tensor`s with shapes
        `[num_examples[i], <extra_input_dims>]`.
    """
    dispatched = Parallel(self._datashard_devices, lambda a, b: a.Dispatch(b),
                          self._dispatchers, inp)
    ret = Parallel(self._expert_devices, tf.concat,
                   TransposeListOfLists(dispatched), 0)
    if ret[0].dtype == tf.float32:
      # see comments on ConvertGradientToTensor
      ret = Parallel(self._expert_devices, ConvertGradientToTensor, ret)
    return ret

  def Combine(self, expert_out, multiply_by_gates=True):
    """Sum together the expert output, multiplied by the corresponding gates.

    Args:
      expert_out: a list of `num_experts` `Tensor`s, each with shape
        `[expert_batch_size_i, <extra_output_dims>]`.
      multiply_by_gates: a boolean.

    Returns:
      a list of num_datashards `Tensor`s with shapes
        `[batch_size[d], <extra_output_dims>]`.
    """
    expert_part_sizes = tf.unstack(
        tf.stack([
            self._dispatchers[d].part_sizes
            for d in xrange(self._num_datashards)
        ]),
        num=self._num_experts,
        axis=1)
    # list of lists of shape [num_experts][num_datashards]
    expert_output_parts = Parallel(self._expert_devices, tf.split, expert_out,
                                   expert_part_sizes)
    expert_output_parts_t = TransposeListOfLists(expert_output_parts)
    ret = []
    for d in xrange(self._num_datashards):
      with tf.device(self._datashard_devices[d]):
        ret.append(self._dispatchers[d].Combine(
            # see comments on ConvertGradientToTensor
            ConvertGradientToTensor(tf.concat(expert_output_parts_t[d], 0)),
            multiply_by_gates=multiply_by_gates))
    return ret

  def ExpertToGates(self):
    """Gate values corresponding to the examples in the per-expert `Tensor`s.

    Returns:
      a list of `num_experts` one-dimensional `Tensor`s of type `tf.float32`.
    """
    return Parallel(self._expert_devices, tf.concat,
                    TransposeListOfLists(
                        Parallel(self._datashard_devices, [
                            self._dispatchers[d].ExpertToGates
                            for d in xrange(self._num_datashards)
                        ])), 0)


def TransposeListOfLists(lol):
  """Transpose a list of equally-sized python lists.

  Args:
    lol: a list of lists
  Returns:
    a list of lists
  """
  assert lol, 'cannot pass the empty list'
  return [list(x) for x in zip(*lol)]


class DistributedSingleDispatcher(object):
  """Dispatches to experts according to gates.

  Each example goes to one expert.

  Unlike SparseDispatcher, the gates are one-dimensional `Tensor`s of integer
  expert ids.  There are no weights.
  """

  def __init__(self, data_parallelism, model_parallelism, gates):
    """Constructs a Dispatcher.

    Args:
      data_parallelism: a Parallelism object.
      model_parallelism: a Parallelism object.
      gates: a list of 1d integer `Tensor`s, one per datashard.
        Says which expert to use for each batch element.

    Returns:
      a DistributedSingleDispatcher
    """
    gates = data_parallelism(tf.to_int32, gates)
    self._gates = gates
    self._data_parallelism = data_parallelism
    self._model_parallelism = model_parallelism

    # Compute the sizes number of examples going from each datashard to each
    # expert.
    def _PartSizes(gates):
      return tf.unsorted_segment_sum(
          tf.ones_like(gates), gates, model_parallelism.n)

    part_sizes_by_datashard = data_parallelism(_PartSizes, gates)
    self._part_sizes_by_expert = tf.unstack(
        tf.stack(part_sizes_by_datashard), num=model_parallelism.n, axis=1)

    # These indices will be used to combine the output on the datashards.
    def _StitchIndices(gates):
      return tf.dynamic_partition(
          tf.range(tf.size(gates)), gates, model_parallelism.n)

    self._stitch_indices = data_parallelism(_StitchIndices, gates)

  def Dispatch(self, d_tensors):
    """Reshuffles input `Tensor`s to produce output `Tensor`s.

    The dimensions of all input and output `Tensor`s match, except for
    dimension 0.  In dimension 0, the input `Tensor`s match the corresponding
    `gates` `Tensor`s which were passed to the constructor.

    Args:
      d_tensors: a list of `Tensor`s, one per datashard.

    Returns:
      a list of `Tensor`s, one per expert.

    """
    parts = self._data_parallelism(tf.dynamic_partition, d_tensors, self._gates,
                                   self._model_parallelism.n)
    parts_by_expert = TransposeListOfLists(parts)
    x_tensors = self._model_parallelism(tf.concat, parts_by_expert, 0)
    return x_tensors

  def Combine(self, x_tensors):
    """Reshuffles per-expert `Tensor`s to produce per-datashard `Tensor`s.

    Dispatch must have been called at least once first.

    The dimensions of all input and output `Tensor`s match, except for
    dimension 0.  In dimension 0, the input `Tensor`s match the corresponding
    outputs of `Dispatch`, and the output `Tensor`s match the corresponding
    `gates` `Tensor`s which were passed to the constructor.

    Args:
      x_tensors: a list of `Tensor`s, one per expert.

    Returns:
      a list of `Tensor`s, one per datashard.
    """
    parts = self._model_parallelism(tf.split, x_tensors,
                                    self._part_sizes_by_expert)
    d_tensors = self._data_parallelism(tf.dynamic_stitch, self._stitch_indices,
                                       TransposeListOfLists(parts))
    return d_tensors


def ParallelEmbeddingLookup(params, ids, data_parallelism):
  """Mod-sharded embedding lookup with multiple datashards.

  TODO(noam): does this work when vocab_size is not a multiple of `num_shards`?

  Args:
    params:  A list of `num_shards` `Tensors`, each with shapes
       `[vocab_size / num_params, depth]`.
    ids: A list of `num_datashards` one-dimensional ineger `Tensors`,
       with shapes `[batch_size[i]]`
    data_parallelism: A Parallelism object.

  Returns:
    a list of `num_datashards` `Tensors`, each with shape
       `[batch_size[i], depth]`.
  """
  param_devices = [x.device for x in params]
  model_parallelism = Parallelism(param_devices)
  num_shards = len(param_devices)
  # pylint: disable=unbalanced-tuple-unpacking
  ids, unique_idx = data_parallelism(tf.unique, ids)
  # pylint: enable=unbalanced-tuple-unpacking
  gates = data_parallelism(tf.mod, ids, num_shards)
  ids_div = data_parallelism(tf.div, ids, num_shards)
  dispatcher = DistributedSingleDispatcher(data_parallelism, model_parallelism,
                                           gates)
  x_ids_div = dispatcher.Dispatch(ids_div)
  params = model_parallelism(ConvertGradientToTensor, params)
  x_emb = model_parallelism(tf.gather, params, x_ids_div)
  r_emb = dispatcher.Combine(x_emb)
  r_emb = data_parallelism(tf.gather, r_emb, unique_idx)
  return r_emb


def SampledSoftmaxLoss(features, sampler, num_classes, target_classes,
                       target_params, sampled_classes, sampled_params):
  """Loss for training softmax classifiers on large label vocabulary.

  This function assumes that we have already chosen the sampled classes and
  fetched the parameters for the target classes and the sampled classes.

  Args:
    features: a Tensor with shape [batch_size, hidden_size]
    sampler: a candidate sampler object
      (see learning/brain/google/python/ops/candidate_sampling.py)
    num_classes: an integer
    target_classes: an integer Tensor with shape [batch_size]
    target_params: a Tensor with shape [batch_size, hidden_size]
      The parameters corresponding to the target classes.
    sampled_classes: an integer tensor with shape [num_sampled_classes]
    sampled_params: a Tensor with shape [num_sampled_classes, hidden_size]
      The parameters corresponding to the sampled classes.

  Returns:
    a Tensor with shape [batch_size]
  """
  sampled_logits = (tf.matmul(features, sampled_params, transpose_b=True) -
                    sampler.log_expected_count(sampled_classes))
  target_logits = (tf.reduce_sum(target_params * features, 1) -
                   sampler.log_expected_count(target_classes))
  sampled_log_denominator = tf.reduce_logsumexp(
      sampled_logits, [1], name='SampledLogDenominator')
  sampled_classes_mask = tf.unsorted_segment_sum(
      tf.fill(tf.shape(sampled_classes), float('-inf')), sampled_classes,
      num_classes)
  target_log_denominator = (
      target_logits + tf.gather(sampled_classes_mask, target_classes))
  combined_log_denominator = tf.reduce_logsumexp(
      tf.stack([sampled_log_denominator, target_log_denominator]), [0])
  loss = combined_log_denominator - target_logits
  return loss


def ParallelSampledSoftmaxLoss(params,
                               features,
                               target_classes,
                               sampler,
                               num_classes,
                               data_parallelism,
                               target_weights=None):
  """Computes sampled softmax loss across many datashards.

  This is used during training to efficiently train a softmax classifier layer.

  Args:
    params: A list of num_param_shards Tensors, each with shape
      [num_classes / num_param_shards, num_features].
      The parameters are assumed to be mod-sharded by class.
    features: a list of num_datashards Tensors, each with shape
      [batch_size_i, num_features]
    target_classes: A list of num_datashards integer Tensors each with shape
       [batch_size_i]
    sampler: a candidate sampler object
      (see learning/brain/google/python/ops/candidate_sampling.py)
    num_classes: an Integer
    data_parallelism: a Parallelism object
    target_weights: an optional list of num_datashards Tensors each with
      shape [batch_size_i]
  Returns:
     a Scalar.
  """
  sampled_classes = data_parallelism(sampler.sample)
  sampled_params = ParallelEmbeddingLookup(params, sampled_classes,
                                           data_parallelism)
  target_params = ParallelEmbeddingLookup(params, target_classes,
                                          data_parallelism)
  ret = data_parallelism(SampledSoftmaxLoss, features, sampler, num_classes,
                         target_classes, target_params, sampled_classes,
                         sampled_params)
  if target_weights is not None:
    ret = data_parallelism(tf.multiply, ret, target_weights)
  ret = data_parallelism(tf.reduce_sum, ret)
  ret = tf.add_n(ret)
  return ret
