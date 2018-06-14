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
"""Utilities for creating Sparsely-Gated Mixture-of-Experts Layers.

See "Outrageously Large Neural Networks"
https://arxiv.org/abs/1701.06538
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import six
from six.moves import range  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_layers

import tensorflow as tf

DEFAULT_DEV_STRING = "existing_device"


def add_scope(scope=None, scope_fn=None):
  """Return a decorator which add a TF name/variable scope to a function.

  Note that the function returned by the decorator accept an additional 'name'
  parameter, which can overwrite the name scope given when the function is
  created.

  Args:
    scope (str): name of the scope. If None, the function name is used.
    scope_fn (fct): Either tf.name_scope or tf.variable_scope

  Returns:
    fct: the add_scope decorator
  """
  def decorator(f):

    @functools.wraps(f)
    def decorated(*args, **kwargs):
      name = kwargs.pop("name", None)  # Python 2 hack for keyword only args
      with scope_fn(name or scope or f.__name__):
        return f(*args, **kwargs)

    return decorated

  return decorator


def add_var_scope(scope=None):
  return add_scope(scope, scope_fn=tf.variable_scope)


def add_name_scope(scope=None):
  return add_scope(scope, scope_fn=tf.name_scope)


def _add_variable_proxy_methods(var, proxy_tensor):
  """Proxy methods of underlying variable.

  This enables our custom getters to still work with, e.g., batch norm.

  Args:
    var: Variable to proxy
    proxy_tensor: Tensor that is identity of var
  """
  proxy_tensor.read_value = lambda: tf.identity(proxy_tensor)
  proxy_tensor.assign_sub = var.assign_sub
  proxy_tensor.assign = var.assign
  proxy_tensor.initialized_value = var.initialized_value


class Parallelism(object):
  """Helper class for creating sets of parallel function calls.

  The purpose of this class is to replace this code:

      e = []
      f = []
      for i in range(len(devices)):
        with tf.device(devices[i]):
          e_, f_ = func(a[i], b[i], c)
          e.append(e_)
          f.append(f_)

  with this code:

      e, f = expert_utils.Parallelism(devices)(func, a, b, c)
  """

  def __init__(self,
               device_names_or_functions,
               reuse=True,
               caching_devices=None,
               daisy_chain_variables=False,
               ps_devices=None):
    """Create a Parallelism.

    Args:
      device_names_or_functions: A list of length n, containing device names
        or device functions (see `tf.device`)
      reuse: True or None.  Whether to reuse variables created in the first
        replica in the subsequent replicas.
      caching_devices: Either `None`, or a list of length n containing device
        names.
      daisy_chain_variables: a boolean - if true, then copies variables in a
        daisy chain between devices.
      ps_devices: list<str>, list of devices for experts.

    Returns:
      a Parallelism.
    """
    assert device_names_or_functions
    self._devices = device_names_or_functions
    self._n = len(device_names_or_functions)
    self._reuse = reuse
    self._caching_devices = self._maybe_repeat(caching_devices)
    self._daisy_chain_variables = daisy_chain_variables
    self._ps_devices = ps_devices or [""]

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
      my_args = transpose_list_of_lists(
          [self._maybe_repeat(arg) for arg in args])
    else:
      my_args = [[] for _ in range(self.n)]
    my_kwargs = [{} for _ in range(self.n)]
    for k, v in six.iteritems(kwargs):
      vals = self._maybe_repeat(v)
      for i in range(self.n):
        my_kwargs[i][k] = vals[i]

    # Construct lists of functions.
    fns = self._maybe_repeat(fn)

    # Now make the parallel call.
    outputs = []
    cache = {}
    tensor_to_var = {}
    for i in range(self.n):

      def daisy_chain_getter(getter, name, *args, **kwargs):
        """Get a variable and cache in a daisy chain."""
        device_var_key = (self._devices[i], name)
        if device_var_key in cache:
          # if we have the variable on the correct device, return it.
          return cache[device_var_key]
        if name in cache:
          # if we have it on a different device, copy it from the last device
          last_device_v = cache[name]
          var = tensor_to_var[last_device_v]
          v = tf.identity(last_device_v)
        else:
          var = getter(name, *args, **kwargs)
          # v = tf.identity(var._ref())  # pylint: disable=protected-access
          v = var.read_value()

        # keep track of the original variable
        tensor_to_var[v] = var
        _add_variable_proxy_methods(tensor_to_var[v], v)
        # update the cache
        cache[name] = v
        cache[device_var_key] = v
        return v

      # Variable scope will not reset caching_device on reused variables,
      # so we make a custom getter that uses identity to cache the variable.
      # pylint: disable=cell-var-from-loop
      def caching_getter(getter, name, *args, **kwargs):
        """Cache variables on device."""
        key = (self._caching_devices[i], name)
        if key in cache:
          return cache[key]

        v = getter(name, *args, **kwargs)
        with tf.device(self._caching_devices[i]):
          # ret = tf.identity(v._ref())  # pylint: disable=protected-access
          ret = v.read_value()
        _add_variable_proxy_methods(v, ret)
        cache[key] = ret
        return ret

      if self._daisy_chain_variables:
        custom_getter = daisy_chain_getter
      elif self._caching_devices[i]:
        custom_getter = caching_getter
      else:
        custom_getter = None
      # pylint: enable=cell-var-from-loop
      with tf.name_scope("parallel_%d" % i):
        with tf.variable_scope(
            tf.get_variable_scope() if self._reuse else "parallel_%d" % i,
            reuse=True if i > 0 and self._reuse else None,
            caching_device=self._caching_devices[i],
            custom_getter=custom_getter):
          # TODO(noam, epot, avaswani)
          # Allows for passing no device in case you want to default to the
          # existing device. This is needed when we put all experts on a single
          # device, for example in local_moe.
          if self._devices[i] != DEFAULT_DEV_STRING:
            with tf.device(self._devices[i]):
              outputs.append(fns[i](*my_args[i], **my_kwargs[i]))
          else:
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

  @property
  def ps_devices(self):
    return self._ps_devices

  def _maybe_repeat(self, x):
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


def _rowwise_unsorted_segment_sum(values, indices, n):
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


def _normal_distribution_cdf(x, stddev):
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


def _prob_in_top_k(
    clean_values, noisy_values, noise_stddev, noisy_top_values, k):
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
       "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
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
  prob_if_in = _normal_distribution_cdf(clean_values - threshold_if_in,
                                        noise_stddev)
  prob_if_out = _normal_distribution_cdf(clean_values - threshold_if_out,
                                         noise_stddev)
  prob = tf.where(is_in, prob_if_in, prob_if_out)
  return prob


def cv_squared(x):
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


def _gates_to_load(gates):
  """Compute the true load per expert, given the gates.

  The load is the number of examples for which the corresponding gate is >0.

  Args:
    gates: a `Tensor` of shape [batch_size, n]
  Returns:
    a float32 `Tensor` of shape [n]
  """
  return tf.reduce_sum(tf.to_float(gates > 0), 0)


def _my_top_k(x, k):
  """GPU-compatible version of top-k that works for very small constant k.

  Calls argmax repeatedly.

  tf.nn.top_k is implemented for GPU, but the gradient, sparse_to_dense,
  seems not to be, so if we use tf.nn.top_k, then both the top_k and its
  gradient go on cpu.  Once this is not an issue, this function becomes
  obsolete and should be replaced by tf.nn.top_k.

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
  for i in range(k):
    values.append(tf.reduce_max(x, 1))
    argmax = tf.argmax(x, 1)
    indices.append(argmax)
    if i + 1 < k:
      x += tf.one_hot(argmax, depth, -1e9)
  return tf.stack(values, axis=1), tf.to_int32(tf.stack(indices, axis=1))


def noisy_top_k_gating(x,
                       num_experts,
                       train,
                       k=2,
                       initializer=tf.zeros_initializer(),
                       noisy_gating=True,
                       noise_epsilon=1e-2,
                       name=None):
  """Noisy top-k gating.

  See paper: https://arxiv.org/abs/1701.06538.

  Args:
    x: input Tensor with shape [batch_size, input_size]
    num_experts: an integer
    train: a boolean - we only add noise at training time.
    k: an integer - number of experts per example
    initializer: an initializer
    noisy_gating: a boolean
    noise_epsilon: a float
    name: an optional string

  Returns:
    gates: a Tensor with shape [batch_size, num_experts]
    load: a Tensor with shape [num_experts]
  """
  with tf.variable_scope(name, default_name="noisy_top_k_gating"):
    input_size = x.get_shape().as_list()[-1]
    w_gate = tf.get_variable(
        "w_gate", [input_size, num_experts], tf.float32, initializer)
    if noisy_gating:
      w_noise = tf.get_variable("w_noise",
                                [input_size, num_experts], tf.float32,
                                initializer)
    clean_logits = tf.matmul(x, w_gate)
    if noisy_gating:
      raw_noise_stddev = tf.matmul(x, w_noise)
      noise_stddev = ((tf.nn.softplus(raw_noise_stddev) + noise_epsilon) *
                      (tf.to_float(train)))
      noisy_logits = clean_logits + (
          tf.random_normal(tf.shape(clean_logits)) * noise_stddev)
      logits = noisy_logits
      if common_layers.should_generate_summaries():
        tf.summary.histogram("noisy_logits", noisy_logits)
        tf.summary.histogram("noise_stddev", noise_stddev)
    else:
      logits = clean_logits
    top_logits, top_indices = _my_top_k(logits, min(k + 1, num_experts))
    top_k_logits = tf.slice(top_logits, [0, 0], [-1, k])
    top_k_indices = tf.slice(top_indices, [0, 0], [-1, k])
    top_k_gates = tf.nn.softmax(top_k_logits)
    # This will be a `Tensor` of shape `[batch_size, n]`, with zeros in the
    # positions corresponding to all but the top k experts per example.
    gates = _rowwise_unsorted_segment_sum(top_k_gates, top_k_indices,
                                          num_experts)
    if noisy_gating and k < num_experts:
      load = tf.reduce_sum(
          _prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits,
                         k), 0)
    else:
      load = _gates_to_load(gates)
    if common_layers.should_generate_summaries():
      tf.summary.histogram("importance", tf.reduce_sum(gates, 0))
      tf.summary.histogram("load", load)
    return gates, load


class PadRemover(object):
  """Helper to remove padding from a tensor before sending to the experts.

  The padding is computed for one reference tensor containing the padding mask
  and then can be applied to any other tensor of shape [dim_origin,...].

  Ex:
      input = [
        [tok1, tok2],
        [tok3, tok4],
        [0, 0],
        [0, 0],
        [tok5, tok6],
        [0, 0],
      ]
      output = [
        [tok1, tok2],
        [tok3, tok4],
        [tok5, tok6],
      ]
  """

  def __init__(self, pad_mask):
    """Compute and store the location of the padding.

    Args:
      pad_mask (tf.Tensor): Reference padding tensor of shape
        [batch_size,length] or [dim_origin] (dim_origin=batch_size*length)
        containing non-zeros positive values to indicate padding location.
    """
    self.nonpad_ids = None
    self.dim_origin = None

    with tf.name_scope("pad_reduce/get_ids"):
      pad_mask = tf.reshape(pad_mask, [-1])  # Flatten the batch
      # nonpad_ids contains coordinates of zeros rows (as pad_mask is
      # float32, checking zero equality is done with |x| < epsilon, with
      # epsilon=1e-9 as standard, here pad_mask only contains positive values
      # so tf.abs would be redundant)
      self.nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))
      self.dim_origin = tf.shape(pad_mask)[:1]

  def remove(self, x):
    """Remove padding from the given tensor.

    Args:
      x (tf.Tensor): of shape [dim_origin,...]

    Returns:
      a tensor of shape [dim_compressed,...] with dim_compressed <= dim_origin
    """
    with tf.name_scope("pad_reduce/remove"):
      x_shape = x.get_shape().as_list()
      x = tf.gather_nd(
          x,
          indices=self.nonpad_ids,
      )
      if not tf.contrib.eager.in_eager_mode():
        # This is a hack but for some reason, gather_nd return a tensor of
        # undefined shape, so the shape is set up manually
        x.set_shape([None] + x_shape[1:])
    return x

  def restore(self, x):
    """Add padding back to the given tensor.

    Args:
      x (tf.Tensor): of shape [dim_compressed,...]

    Returns:
      a tensor of shape [dim_origin,...] with dim_compressed >= dim_origin. The
      dim is restored from the original reference tensor
    """
    with tf.name_scope("pad_reduce/restore"):
      x = tf.scatter_nd(
          indices=self.nonpad_ids,
          updates=x,
          shape=tf.concat([self.dim_origin, tf.shape(x)[1:]], axis=0),
      )
    return x


@add_name_scope("map_ids")
def map_ids(x, indices, map_fn):
  """Apply a function to each coordinate ids of a multidimensional tensor.

  This allows to process each sequence of a batch independently. This is
  similar to tf.map_fn but with tensor where the batch dim has been flatten.

  Warning: The indices ids have to be contiguous and ordered in memory as the
  output vector for each of the ids are simply concatenated after being
  processed.
  Ex: if your indices are [0,2,2,1,2,0], the output will contains the processed
  rows in the following order: [0,0,1,2,2,2]

  Args:
    x (Tensor): The tensor to be dispatched of shape [length,...]
    indices (Tensor): A int32 tensor of size [length, 1] containing the batch
      coordinate of x
    map_fn (fct): Function called for every ids of the original tensor. Take
      as input a tensor of same rank than x and from shape [length_id,...] with
      length_id <= length. Isn't called if length_id == 0

  Returns:
    a tensor of same shape as x, where each elements has been processed
  """
  indices = tf.reshape(indices, [-1])

  t_i = tf.constant(0)
  # batch_coordinates start at 0
  t_batch_size = tf.reduce_max(indices) + 1

  # ta_stack_out will store the intermediate results for each individual id
  # As alternative to tf.TensorArray, scatter_update could potentially be used
  # but that would require an additional mutable tensor.
  ta_stack_out = tf.TensorArray(
      x.dtype,
      size=t_batch_size,
  )

  # Then we iterate over each sequence individually and compute the
  # transformation for each id
  while_condition = lambda t_i, *args: tf.less(t_i, t_batch_size)
  def body(t_i, ta_stack_out):
    """Loop body."""
    # Gather the ids
    current_ids = tf.to_int32(tf.where(tf.equal(indices, t_i)))
    t_row = tf.gather_nd(x, indices=current_ids)

    # TODO(epot): Should not call map_fn if t_row size is 0

    # Apply transformation to each id
    # Restore batch_dim=1 as most function expect [batch_dim, length, ...] as
    # input
    t_row = tf.expand_dims(t_row, axis=0)
    t_row = map_fn(t_row)
    t_row = tf.squeeze(t_row, axis=0)  # Squeeze for concatenation
    ta_stack_out = ta_stack_out.write(t_i, t_row)

    return [tf.add(t_i, 1), ta_stack_out]  # ++i

  # Run the loop, equivalent to:
  # stack_out = []
  # while i < batch_size:
  #   stack_out.expand(map_fn(x[indices==i]))
  _, ta_stack_out = tf.while_loop(while_condition, body, [t_i, ta_stack_out])

  # Merge all results
  return ta_stack_out.concat()


class SparseDispatcher(object):
  """Helper for implementing a mixture of experts.

  The purpose of this class is to create input minibatches for the
  experts and to combine the results of the experts to form a unified
  output tensor.

  There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".

  The class is initialized with a "gates" Tensor, which specifies which
  batch elements go to which experts, and the weights to use when combining
  the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.

  The inputs and outputs are all two-dimensional [batch, depth].
  Caller is responsible for collapsing additional dimensions prior to
  calling this class and reshaping the output to the original shape.
  See common_layers.reshape_like().

  Example use:

  gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
  inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
  experts: a list of length `num_experts` containing sub-networks.

    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)

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

  @add_name_scope()
  def dispatch(self, inp):
    """Create one input Tensor for each expert.

    The `Tensor` for a expert `i` contains the slices of `inp` corresponding
    to the batch elements `b` where `gates[b, i] > 0`.

    Args:
      inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
    Returns:
      a list of `num_experts` `Tensor`s with shapes
        `[expert_batch_size_i, <extra_input_dims>]`.
    """
    inp = tf.gather(inp, self._batch_index)
    return tf.split(inp, self._part_sizes_tensor, 0, num=self._num_experts)

  @add_name_scope()
  def combine(self, expert_out, multiply_by_gates=True):
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
    # see comments on convert_gradient_to_tensor
    stitched = common_layers.convert_gradient_to_tensor(
        tf.concat(expert_out, 0))
    if multiply_by_gates:
      stitched *= tf.expand_dims(self._nonzero_gates, 1)
    combined = tf.unsorted_segment_sum(stitched, self._batch_index,
                                       tf.shape(self._gates)[0])
    return combined

  def expert_to_gates(self):
    """Gate values corresponding to the examples in the per-expert `Tensor`s.

    Returns:
      a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
          and shapes `[expert_batch_size_i]`
    """
    return tf.split(
        self._nonzero_gates, self._part_sizes_tensor, 0, num=self._num_experts)

  def expert_to_batch_indices(self):
    """Batch indices corresponding to the examples in the per-expert `Tensor`s.

    Returns:
      a list of `num_experts` one-dimensional `Tensor`s with type `tf.int64`
          and shapes `[expert_batch_size_i]`
    """
    return tf.split(
        self._batch_index, self._part_sizes_tensor, 0, num=self._num_experts)

  @property
  def part_sizes(self):
    return self._part_sizes_tensor


class DistributedSparseDispatcher(object):
  """A distributed version of SparseDispatcher.

  Instead of one batch of input examples, we simultaneously process
  a list of num_datashards batches of input examples.  The per-expert
  `Tensor`s contain a combination of examples from the different datashards.

  Each datashard is associated with a particular device and each expert is
  associated with a particular device.  All per-datashard and per-expert
  `Tensor`s are created on those devices.  There is no single-device bottleneck.
  """

  def __init__(self, data_parallelism, expert_parallelism, gates):
    """Create a DistributedSparseDispatcher.

    Args:
      data_parallelism: a Parallelism object.
      expert_parallelism: a Parallelism object.
      gates: a list of datashard_parallelism.n `Tensor`s of shapes
        `[batch_size[d], num_experts]`.

    Returns:
      a DistributedSparseDispatcher
    """
    self._gates = gates
    self._dp = data_parallelism
    self._ep = expert_parallelism
    assert len(gates) == self._dp.n
    self._dispatchers = self._dp(SparseDispatcher, self._ep.n, gates)

  def dispatch(self, inp):
    """Create one input Tensor for each expert.

    Args:
      inp: a list of length num_datashards `Tensor`s with shapes
        `[batch_size[d], <extra_input_dims>]`.
    Returns:
      a list of `num_experts` `Tensor`s with shapes
        `[num_examples[i], <extra_input_dims>]`.
    """
    dispatched = self._dp(lambda a, b: a.dispatch(b), self._dispatchers, inp)
    ret = self._ep(tf.concat, transpose_list_of_lists(dispatched), 0)
    if ret[0].dtype == tf.float32:
      # see comments on common_layers.convert_gradient_to_tensor
      ret = self._ep(common_layers.convert_gradient_to_tensor, ret)
    return ret

  def combine(self, expert_out, multiply_by_gates=True):
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
        tf.stack([d.part_sizes for d in self._dispatchers]),
        num=self._ep.n,
        axis=1)
    # list of lists of shape [num_experts][num_datashards]
    expert_output_parts = self._ep(tf.split, expert_out, expert_part_sizes)
    expert_output_parts_t = transpose_list_of_lists(expert_output_parts)
    def my_combine(dispatcher, parts):
      return dispatcher.combine(
          common_layers.convert_gradient_to_tensor(tf.concat(parts, 0)),
          multiply_by_gates=multiply_by_gates)
    return self._dp(my_combine, self._dispatchers, expert_output_parts_t)

  def expert_to_gates(self):
    """Gate values corresponding to the examples in the per-expert `Tensor`s.

    Returns:
      a list of `num_experts` one-dimensional `Tensor`s of type `tf.float32`.
    """
    return self._ep(
        tf.concat,
        transpose_list_of_lists(
            self._dp(lambda d: d.expert_to_gates(), self._dispatchers)), 0)


def transpose_list_of_lists(lol):
  """Transpose a list of equally-sized python lists.

  Args:
    lol: a list of lists
  Returns:
    a list of lists
  """
  assert lol, "cannot pass the empty list"
  return [list(x) for x in zip(*lol)]


def ffn_expert_fn(input_size,
                  hidden_sizes,
                  output_size,
                  hidden_activation=tf.nn.relu):
  """Returns a function that creates a feed-forward network.

  Use this function to create the expert_fn argument to distributed_moe.

  Args:
    input_size: an integer
    hidden_sizes: a list of integers
    output_size: an integer
    hidden_activation: a unary function.

  Returns:
    a unary function
  """
  def my_fn(x):
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    for i in range(1 + len(hidden_sizes)):
      w = tf.get_variable("w_%d" % i, layer_sizes[i:i+2], tf.float32)
      x = tf.matmul(x, w)
      if i < len(hidden_sizes):
        x = hidden_activation(x)
      if layer_sizes[i] != input_size:
        x *= (layer_sizes[i] / float(input_size))**-0.5
    return x
  return my_fn


def flatten_all_but_last(a):
  """Flatten all dimensions of a except the last."""
  ret = tf.reshape(a, [-1, tf.shape(a)[-1]])
  if not tf.contrib.eager.in_eager_mode():
    ret.set_shape([None] + a.get_shape().as_list()[-1:])
  return ret


def distributed_moe(data_parallelism,
                    expert_devices,
                    xs,
                    train,
                    input_size,
                    expert_fn,
                    num_experts,
                    k=2,
                    loss_coef=1e-2,
                    name=None):
  """Call a distributed mixture of experts.

  Args:
    data_parallelism: a expert_utils.Parallelism object.
    expert_devices: a list of strings.  We round-robin the experts across these
      devices.
    xs: a list of input tensors, each with shape [... , input_size]
    train: a boolean scalar.
    input_size: an integer (input size for this layer)
    expert_fn: a unary function for each expert to run
       It should take a Tensor with shape [batch_size, input_size]
       and return a Tensor with shape [batch_size, output_size].
       e.g. ffn_expert_fn(...)
    num_experts: an integer - number of experts
    k: an integer - how many experts to use for each batch element
    loss_coef: a scalar - multiplier on load-balancing losses
    name: a string

  Returns:
    ys: a list of tensors.  Each Tensor has the same shape as the corresponding
      Tensor in xs, except for the last dimension, which is output_size.
    extra_training_loss: a scalar.  This should be added into the overall
      training loss of the model.  The backpropagation of this loss
      encourages all experts to be approximately equally used across a batch.
  """
  dp = data_parallelism
  # create a parallelism object for running the experts.
  #   We use the default of reuse=False.  Otherwise, the experts would all
  #   use the same variables.
  ep = Parallelism(
      [expert_devices[i % len(expert_devices)] for i in range(num_experts)],
      reuse=None)
  # Experts expect 2d input tensors, so flatten the batch dimension and all
  # spatial dimensions together.
  xs_flat = dp(tf.reshape, xs, [[-1, input_size]] * dp.n)
  with tf.variable_scope(name, default_name="moe"):
    # The gates indicate which batch elements go to which tensors.
    # load is a measure of approximately how many examples go to each expert
    gates, load = dp(noisy_top_k_gating,
                     xs_flat,
                     num_experts,
                     train,
                     k,
                     initializer=tf.zeros_initializer(),
                     noisy_gating=True,
                     noise_epsilon=1e-2)
    # This magic object helps us shuffle data between datashards and experts.
    dispatcher = DistributedSparseDispatcher(dp, ep, gates)
    expert_in = dispatcher.dispatch(xs_flat)
    expert_out = ep(expert_fn, expert_in)
    ys_flat = dispatcher.combine(expert_out)
    ys = dp(common_layers.reshape_like, ys_flat, xs)
    # compute some load-balancing losses.
    load = tf.add_n(load)
    importance = tf.add_n(dp(tf.reduce_sum, gates, 0))
    loss = loss_coef * (cv_squared(importance) + cv_squared(load))
    return ys, loss


def local_moe(x,
              train,
              expert_fn,
              num_experts,
              k=2,
              loss_coef=1e-2,
              pass_x=True,
              pass_gates=False,
              additional_dispatch_params=None,
              name=None):
  """Call a local mixture of experts.

  Args:
    x: a tensors with shape [... , input_size]
    train: a boolean scalar.
    expert_fn: a function.
    num_experts: an integer - number of experts
    k: an integer - how many experts to use for each batch element
    loss_coef: a scalar - multiplier on load-balancing losses
    pass_x: a boolean. If true, x will also be dispatched to the experts.
    pass_gates: a boolean. If true, gates will be passed to experts. Might be
      necessary when dealing with sparse encoder-encoder decoder attention
    additional_dispatch_params: The extra tensors that need to be sent to each
      expert. Examples include batch batch coordinates (see
      common_attention.local_expert_attention)
    name: a string

  Returns:
    y: a tensor.  Has the same shape as x, except for the last dimension,
      which is output_size.
    extra_training_loss: a scalar.  This should be added into the overall
      training loss of the model.  The backpropagation of this loss
      encourages all experts to be approximately equally used across a batch.
  """

  with tf.variable_scope(name, default_name="local_moe"):
    x_flat = flatten_all_but_last(x)

    # The gates indicate which batch elements go to which tensors.
    # load is a measure of approximately how many examples go to each expert
    gates, load = noisy_top_k_gating(
        x_flat,
        num_experts,
        train,
        k,
        initializer=tf.zeros_initializer(),
        noisy_gating=True,
        noise_epsilon=1e-2)
    # This magic object helps us shuffle data between datashards and experts.
    dispatcher = SparseDispatcher(num_experts, gates)

    # Set up expert_fn arguments
    expert_kwargs = {}
    if pass_x:
      expert_kwargs["x"] = dispatcher.dispatch(x_flat)
    if pass_gates:
      expert_kwargs["gates"] = dispatcher.expert_to_gates()
    for key, val in six.iteritems(additional_dispatch_params or {}):
      val = flatten_all_but_last(val)
      expert_kwargs[key] = dispatcher.dispatch(val)

    ep = Parallelism([DEFAULT_DEV_STRING] * num_experts, reuse=None)
    expert_outputs = ep(expert_fn, **expert_kwargs)

    y_flat = dispatcher.combine(expert_outputs)
    y = common_layers.reshape_like(y_flat, x)

    importance = tf.reduce_sum(gates, 0)
    loss = loss_coef * (cv_squared(importance) + cv_squared(load))
    return y, loss


class TruncatingDispatcher(object):
  """Helper for implementing a mixture of experts.

  A TruncatingDispatcher is useful when you need to deal with
  fixed-sized Tensors.  As opposed to a SparseDispatcher, which
  produces batches of different sizes for the different experts, the
  TruncatingDispatcher always produces batches of the same given size,
  and the results are returned stacked in one big tensor.

  In the case where an expert is over-capacity, the last items that
  should have gone to that expert are dropped.

  Confusingly, the inputs to a TruncatingDispatcher have both a
  "batch" and a "length" dimension.  Not only does each expert receive
  the same total number of examples, it also receives the same number
  of examples for each element of "batch".  This behavior is necessary
  for applications such as grouped attention, where we have a batch of
  sequences, and we want each sequence to be divided evenly among
  experts.  For simpler applications like mixture-of-experts, you can
  reshape the input so that the "batch" dimension is 1, and only the
  "length" dimension is used.
  """

  @add_name_scope("truncating_dispatcher")
  def __init__(self, requests, expert_capacity):
    """Create a TruncatingDispatcher.

    Args:
      requests: a boolean `Tensor` of shape `[batch, length, num_experts]`.
        Alternatively, a float or int Tensor containing zeros and ones.
      expert_capacity: a Scalar - maximum number of examples per expert per
        batch element.

    Returns:
      a TruncatingDispatcher
    """
    self._requests = tf.to_float(requests)
    self._expert_capacity = expert_capacity
    expert_capacity_f = tf.to_float(expert_capacity)
    self._batch, self._length, self._num_experts = tf.unstack(
        tf.shape(self._requests), num=3)

    # [batch, length, num_experts]
    position_in_expert = tf.cumsum(self._requests, axis=1, exclusive=True)
    # [batch, length, num_experts]
    self._gates = self._requests * tf.to_float(
        tf.less(position_in_expert, expert_capacity_f))
    batch_index = tf.reshape(
        tf.to_float(tf.range(self._batch)), [self._batch, 1, 1])
    length_index = tf.reshape(
        tf.to_float(tf.range(self._length)), [1, self._length, 1])
    expert_index = tf.reshape(
        tf.to_float(tf.range(self._num_experts)), [1, 1, self._num_experts])
    # position in a Tensor with shape [batch * num_experts * expert_capacity]
    flat_position = (
        position_in_expert +
        batch_index * (tf.to_float(self._num_experts) * expert_capacity_f) +
        expert_index * expert_capacity_f)
    # Tensor of shape [batch * num_experts * expert_capacity].
    # each element is an integer in [0, length)
    self._indices = tf.unsorted_segment_sum(
        data=tf.reshape((length_index + 1.0) * self._gates, [-1]),
        segment_ids=tf.to_int32(tf.reshape(flat_position, [-1])),
        num_segments=self._batch * self._num_experts * expert_capacity)
    self._indices = tf.reshape(
        self._indices,
        [self._batch, self._num_experts, expert_capacity])
    # Tensors of shape [batch, num_experts, expert_capacity].
    # each element is 0.0 or 1.0
    self._nonpadding = tf.minimum(self._indices, 1.0)
    # each element is an integer in [0, length)
    self._indices = tf.nn.relu(self._indices - 1.0)
    # self._flat_indices is [batch, num_experts, expert_capacity], with values
    # in [0, batch * length)
    self._flat_indices = tf.to_int32(
        self._indices +
        (tf.reshape(tf.to_float(tf.range(self._batch)), [-1, 1, 1])
         * tf.to_float(self._length)))
    self._indices = tf.to_int32(self._indices)

  @add_name_scope("truncating_dispatcher_dispatch")
  def dispatch(self, inp):
    """Send the inputs to the experts.

    Args:
      inp: a `Tensor` of shape "[batch, length, depth]`
    Returns:
      a tensor with shape [batch, num_experts, expert_capacity, depth]
    """
    inp = tf.reshape(inp, [self._batch * self._length, -1])
    # [batch, num_experts, expert_capacity, depth]
    ret = tf.gather(inp, self._flat_indices)
    return ret

  @add_name_scope("truncating_dispatcher_combine")
  def combine(self, x):
    """Return the output from the experts.

    When one example goes to multiple experts, the outputs are summed.

    Args:
      x: a Tensor with shape [batch, num_experts, expert_capacity, depth]

    Returns:
      a `Tensor` with shape `[batch, length, depth]
    """
    depth = tf.shape(x)[-1]
    x *= tf.expand_dims(self._nonpadding, -1)
    ret = tf.unsorted_segment_sum(
        x, self._flat_indices, num_segments=self._batch * self._length)
    ret = tf.reshape(ret, [self._batch, self._length, depth])
    return ret

  def nonpadding(self):
    """Which elements of a dispatched Tensor are not padding.

    Returns:
      a Zero/One float tensor with shape [batch, num_experts, expert_capacity].
    """
    return self._nonpadding

  def gates(self):
    """A Tensor indicating which examples go to which experts.

    Returns:
      A float32 Tensor with shape [batch, length, num_experts], where each value
      is 0.0 or 1.0.
    """
    return self._gates

  def length_coordinate(self):
    """Length coordinate of dispatched tensor.

    Returns:
      a tensor with shape [batch, num_experts, expert_capacity] containing
       integers in the range [0, length)
    """
    return self._indices


def local_moe_tpu(inputs,
                  hidden_size,
                  output_size,
                  num_experts,
                  loss_coef=1e-3,
                  overhead=1.0):
  """Local mixture of experts that works well on TPU.

  See https://arxiv.org/abs/1701.06538

  There are num_experts expert networks, each containing a relu-activated
  hidden layer of size hidden_size, followed by an output projection.

  The number of parameters is thus:
    num_experts * (input_size * hidden_size + hidden_size * output_size)

  The input is 3d: [batch, length, depth], consisting of the representations
  of all positions in a batch of sequences.

  Each position of each sequence is sent to 0-2 experts.  The expert
  choices and the combination weights are determined by a learned gating
  function.

  This function returns a small auxiliary loss that should be added to the
  training loss of the model.  This loss helps to balance expert usage.
  Without the loss, it is very likely that a few experts will be trained and
  the rest will starve.

  Several hacks are necessary to get around current TPU limitations:

  - To ensure static shapes, we enforce (by truncation/padding)
    that each sequence send the same number of elements to each expert.

    It would make more sense to enforce this equality over the entire batch,
    as opposed to on individual sequences.  This would allow more freedom
    for individual sequences to be unbalanced.  Unfortunately, that would
    slow down our hacked-up gather-by-matmul implementation.

    TODO(noam): There is no real reason for a single sequence to be the unit
      of equal allocation.  Reshaping the inputs would allow us to pick a
      different unit of equal allocation.

  TODO(noam): Factor this code better.  We want to be able to substitute
  different code for the experts themselves.  We also want to integrate this
  gating/dispatching logic into multi-device mixtures-of-experts.

  Args:
    inputs: a Tensor with shape [batch, length, depth]
    hidden_size: an integer
    output_size: an integer
    num_experts: an integer
    loss_coef: a float scalar
    overhead: multiplicative factor of how much spare capacity to assign

  Returns:
    outputs: a Tensor with shape [batch, length, output_size]
    loss: a scalar
  """
  batch, length, input_size = common_layers.shape_list(inputs)[:]
  # Each sequence sends expert_capacity positions to each expert.
  if isinstance(length, int):
    expert_capacity = min(
        length, int((length * 2 * overhead) / num_experts))
  else:
    expert_capacity = tf.minimum(
        length, tf.to_int32(
            tf.to_float(length) * 2 * overhead / num_experts))
  expert_capacity_f = tf.to_float(expert_capacity)

  # This is the learned gating function.
  gates = tf.nn.softmax(
      tf.to_float(common_layers.dense(inputs, num_experts, name="logits")))

  # Find the top expert for each position.
  gate_1, index_1 = common_layers.top_1_tpu(gates)
  # [batch, length, num_experts]
  mask_1 = tf.one_hot(index_1, num_experts)
  # [batch, length, num_experts]
  # This is the position within the expert's mini-batch for this sequence
  position_in_expert_1 = common_layers.cumsum(
      mask_1, axis=1, exclusive=True) * mask_1
  # Remove the elements that don't fit.
  mask_1 *= tf.to_float(tf.less(position_in_expert_1, expert_capacity_f))
  # [batch, 1, num_experts]
  # How many examples in this sequence go to this expert
  mask_1_count = tf.reduce_sum(mask_1, axis=1, keepdims=True)
  # [batch, length] - mostly ones, but zeros where something didn't fit
  mask_1_flat = tf.reduce_sum(mask_1, axis=2)
  position_in_expert_1 = tf.reduce_sum(position_in_expert_1, axis=2)
  # Weight assigned to first expert.
  gate_1 *= mask_1_flat

  # Pick a second-place expert for each position.
  # We first mask out the experts that we expect to be over-capacity
  space_remaining = expert_capacity_f - mask_1_count
  use_rate = (mask_1_count + 1.0) / tf.to_float(length)
  # At what point in the sequence do we expect the expert to be full.
  expected_exhaustion_pos = space_remaining / use_rate
  # A Tensor with shape [batch, length, num_experts] representing a boolean
  #   - whether we expect that the expert will already be full.
  expected_exhausted = tf.to_float(tf.greater(
      tf.reshape(tf.to_float(tf.range(length)), [1, length, 1]),
      expected_exhaustion_pos))
  masked_gates = gates - mask_1 - expected_exhausted
  # This section is similar to the section above.
  gate_2, index_2 = common_layers.top_1_tpu(masked_gates)
  # [batch, length, num_experts]
  mask_2 = tf.one_hot(index_2, num_experts)
  position_in_expert_2 = (
      common_layers.cumsum(mask_2, axis=1, exclusive=True) + mask_1_count)
  position_in_expert_2 *= mask_2
  mask_2 *= tf.to_float(tf.less(position_in_expert_2, expert_capacity_f))
  mask_2_count = tf.reduce_sum(mask_2, axis=1, keepdims=True)
  mask_2_flat = tf.reduce_sum(mask_2, axis=2)
  position_in_expert_2 = tf.reduce_sum(position_in_expert_2, axis=2)
  gate_2 *= mask_2_flat

  # What fraction didn't fit - show summaries
  miss_rate_1 = 1.0 - tf.reduce_sum(mask_1_count) / tf.to_float(batch * length)
  miss_rate_2 = 1.0 - tf.reduce_sum(mask_2_count) / tf.to_float(batch * length)
  tf.summary.scalar("miss_rate_1", miss_rate_1)
  tf.summary.scalar("miss_rate_2", miss_rate_2)

  # renormalize the two gate values to add up to 1
  denom = gate_1 + gate_2 + 1e-9
  gate_1 /= denom
  gate_2 /= denom

  # inputs: [batch, length, input_size]
  # forward_assignment: [batch, length, num_experts * expert_capacity]
  # expert_inputs: [batch, num_experts * expert_capacity, input_size]

  segment_ids_forward_1 = (
      (index_1 * expert_capacity) +
      tf.to_int32(position_in_expert_1) +
      tf.to_int32(1.0 - mask_1_flat) * (num_experts * expert_capacity))

  segment_ids_forward_2 = (
      (index_2 * expert_capacity) +
      tf.to_int32(position_in_expert_2) +
      tf.to_int32(1.0 - mask_2_flat) * (num_experts * expert_capacity))

  # Gather and scatter are painfully slow on TPU.
  # We will use one_hot and matmul instead.

  # [batch, length, num_experts * expert_capacity]
  one_hot_1 = tf.one_hot(
      segment_ids_forward_1, num_experts * expert_capacity, dtype=inputs.dtype)
  one_hot_2 = tf.one_hot(
      segment_ids_forward_2, num_experts * expert_capacity, dtype=inputs.dtype)

  forward_assignment = (one_hot_1 + one_hot_2)

  # [batch, num_experts * expert_capacity, input_size]
  expert_inputs = tf.matmul(forward_assignment, inputs, transpose_a=True)

  # [batch, num_experts, expert_capacity, input_size]
  expert_inputs = tf.reshape(
      expert_inputs, [batch, num_experts, expert_capacity, input_size])
  # [num_experts, batch, expert_capacity, input_size]
  expert_inputs = tf.transpose(expert_inputs, [1, 0, 2, 3])

  # [num_experts, batch * expert_capacity, input_size]
  expert_inputs = tf.reshape(
      expert_inputs, [num_experts, batch * expert_capacity, input_size])

  # Now feed the expert inputs through the experts.
  h = common_layers.batch_dense(
      expert_inputs, hidden_size, activation=tf.nn.relu, name="x0")
  expert_output = common_layers.batch_dense(h, output_size, name="x1")
  expert_output = tf.reshape(
      expert_output, [num_experts, batch, expert_capacity, output_size])

  # [batch, num_experts, expert_capacity, output_size]
  expert_output = tf.transpose(expert_output, [1, 0, 2, 3])
  expert_output = tf.reshape(
      expert_output, [batch, num_experts * expert_capacity, output_size])

  # Again, use matmul instead of unsorted_segment_sum.  This time, we need
  # to multiply by the combination weights gate_1 and gate_2.

  # expert_output: [batch, num_experts * expert_capacity, output_size]
  # backward_assigmnent: [batch, length, num_experts * expert_capacity]
  # output: [batch, length, output_size]
  backward_assigmnent = (
      one_hot_1 * tf.cast(tf.expand_dims(gate_1, 2), inputs.dtype) +
      one_hot_2 * tf.cast(tf.expand_dims(gate_2, 2), inputs.dtype))
  output = tf.matmul(backward_assigmnent, expert_output)

  # Compute a loss equal to the coefficient ov variation of the
  # total gate value per expert per sequence.
  # This loss causes the experts to be used about equally used per sequence.
  importance = tf.reduce_sum(gates * (mask_1 + mask_2), 1)
  loss = loss_coef * cv_squared(importance)
  return output, loss


def reduce_by_device(parallelism, data, reduce_fn):
  """Reduces data per device.

  This can be useful, for example, if we want to all-reduce n tensors on k<n
  devices (like during eval when we have only one device).  We call
  reduce_by_device() to first sum the tensors per device, then call our usual
  all-reduce operation to create one sum per device, followed by
  expand_by_device, to create the appropriate number of pointers to these
  results.  See all_reduce_ring() below for an example of how this is used.

  Args:
    parallelism: a expert_utils.Parallelism object
    data: a list of Tensors with length parallelism.n
    reduce_fn: a function taking a list of Tensors.  e.g. tf.add_n

  Returns:
    device_parallelism: a Parallelism object with each device listed only once.
    reduced_data: A list of Tensors, one per device.
  """
  unique_devices = []
  device_to_data = {}
  for dev, datum in zip(parallelism.devices, data):
    if dev not in device_to_data:
      unique_devices.append(dev)
      device_to_data[dev] = [datum]
    else:
      device_to_data[dev].append(datum)
  device_parallelism = Parallelism(unique_devices)
  grouped_data = [device_to_data[dev] for dev in unique_devices]
  return device_parallelism, device_parallelism(reduce_fn, grouped_data)


def expand_by_device(original_parallelism, device_parallelism, data):
  """Opposite of reduce_by_device().

  Args:
    original_parallelism: a expert_utils.Parallelism object.
    device_parallelism: a expert_utils.Parallelism object.
    data: a list of tensors with length device_parallelism.n

  Returns:
    a list of Tensors with length original_parallelism.n
  """
  device_to_datum = {
      device_parallelism.devices[i]: data[i]
      for i in range(device_parallelism.n)}
  return [device_to_datum[d] for d in original_parallelism.devices]


def all_reduce_ring(x, parallelism, maybe_reduce=True, use_bfloat16=True):
  """Compute the sum of all Tensors and put the result everywhere.

  Assumes that the devices are connected in a ring.

  Args:
    x: a list of Tensors with length parallelism.n
    parallelism: a expert_utils.Parallelism object.
    maybe_reduce: a boolean - first reduce per device.
    use_bfloat16: a boolean - saves bandwidth but loses precision

  Returns:
    a list of Tensors with length parallelism.n
  """
  if parallelism.n == 1:
    return x

  if maybe_reduce:
    original_parallelism = parallelism
    parallelism, x = reduce_by_device(parallelism, x, tf.add_n)

  if parallelism.n == 1:
    y = x
  else:
    # first shard the input:
    x_flat = parallelism(tf.reshape, x, [[-1]] * parallelism.n)
    # [device, shard]
    x_split = parallelism(
        common_layers.approximate_split, x_flat, parallelism.n, 0)
    def _step(source_replica, target_replica, x_split, op="plus_eq"):
      """Helper function - one step of summing or copying.

      If op == "plus_eq", then adds source_replica into target_replica
      If op == "copy", then copies source_replica onto target_replica

      These operations happen for all shards.  The replica numbers are offset
      by the shard numbers to keep all physical links busy.

      Args:
        source_replica: an integer
        target_replica: an integer
        x_split: a list of lists of tensors
        op: a string
      """
      for shard in range(parallelism.n):
        source_device = (shard + source_replica) % parallelism.n
        target_device = (shard + target_replica) % parallelism.n
        source = x_split[source_device][shard]
        if use_bfloat16:
          with tf.device(parallelism.devices[source_device]):
            source = tf.to_bfloat16(source)
        with tf.device(parallelism.devices[target_device]):
          source = tf.to_float(source)
          if op == "plus_eq":
            x_split[target_device][shard] += source
          else:
            assert op == "copy"
            x_split[target_device][shard] = tf.identity(source)
    center = parallelism.n // 2

    # accumulate everything towards the center.
    for i in reversed(range(center, parallelism.n - 1)):
      _step(i + 1, i, x_split, op="plus_eq")
    for i in range(center):
      _step(i, i + 1, x_split, op="plus_eq")
    # copy everything away from the center.
    for i in range(center, parallelism.n - 1):
      _step(i, i + 1, x_split, op="copy")
    for i in reversed(range(center)):
      _step(i + 1, i, x_split, op="copy")
    x_concat = parallelism(tf.concat, x_split, 0)
    y = parallelism(common_layers.reshape_like_all_dims, x_concat, x)
  if maybe_reduce:
    y = expand_by_device(original_parallelism, parallelism, y)
  return y
