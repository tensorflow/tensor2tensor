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
"""Placement Mesh Implementation (for CPU/GPU clusters)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.mesh_tensorflow import mesh_tensorflow as mtf
import tensorflow as tf


class PlacementMeshImpl(mtf.MeshImpl):
  """Mesh implemented using explicit device placement."""

  def __init__(self, shape, layout, devices):
    super(PlacementMeshImpl, self).__init__(shape, layout)
    self._devices = devices

  class LaidOutTensor(object):
    """One Slice for each processor."""

    def __init__(self, tensor_list):
      self._tensor_list = tensor_list

    def __repr__(self):
      return "[" + ",".join([str(t) for t in self._tensor_list]) + "]"

    @property
    def tensor_list(self):
      return self._tensor_list

    @classmethod
    def from_tensor_list(cls, tensor_list):
      return cls(tensor_list)

    @property
    def all_slices(self):
      return self._tensor_list

    @property
    def slice_shape(self):
      return self.tensor_list[0].shape.as_list()

    def to_laid_out_tensor(self):
      return self

  class LaidOutVariable(object):
    """Maintains slice-variables and copy operations."""

    def __init__(self, variable, mesh_impl):
      """Create a LaidOutVariable.

      Args:
        variable: a Variable (Operation)
        mesh_impl: a MeshImpl
      """
      self._variable = variable
      self._mesh_impl = mesh_impl
      shape = variable.outputs[0].shape
      dtype = variable.outputs[0].dtype
      slice_shape = mesh_impl.slice_shape(shape)
      base_name = variable.name
      slices = []
      for pnum in xrange(mesh_impl.size):
        with tf.device(mesh_impl.devices[pnum]):
          slices.append(tf.get_variable(
              base_name + "_slice_%d" % pnum,
              slice_shape,
              dtype=dtype, collections=[]))
      self._laid_out_tensor = mesh_impl.LaidOutTensor(slices)
      self._copy_master_to_slices = self.assign_to_slices(
          mesh_impl.make_slices(variable.master, shape))
      self._copy_slices_to_master = tf.assign(
          variable.master,
          mesh_impl.combine_slices(self._laid_out_tensor.all_slices, shape))

    def assign_to_slices(self, slices):
      """Assign to the slice variables.

      Args:
        slices: a list of tf.Tensor

      Returns:
        a tf.operation
      """
      return tf.group(mtf.parallel(
          self._mesh_impl.devices, tf.assign,
          self.laid_out_tensor.all_slices, slices))

    @property
    def laid_out_tensor(self):
      return self._laid_out_tensor

    @property
    def copy_master_to_slices(self):
      return self._copy_master_to_slices

    @property
    def copy_slices_to_master(self):
      return self._copy_slices_to_master

  def slicewise(self, fn, *inputs):
    """Execute a function in parallel on all slices.

    Args:
      fn: a function from tf.Tensors to tf.Tensor or a tuple of tf.Tensors.
      *inputs: a list of inputs.  Each input is either a LaidOutTensor or
        is convertible to a tf.Tensor.
    Returns:
      a LaidOutTensor, or a tuple of LaidOutTensors if fn returns a tuple.
    """
    if fn == tf.add:
      assert len(inputs) == 2
      if isinstance(inputs[0], mtf.LazyAllreduceSum):
        # sum of LazyAllreduceSum (keep delaying the allreduce)
        return inputs[0] + inputs[1]
    # convert all inputs to LaidOutTensor where possible
    inputs = mtf.convert_args_to_laid_out_tensors(inputs)
    inputs = [x.tensor_list if isinstance(x, self.LaidOutTensor)
              else [x] * len(self.devices) for x in inputs]
    ret = mtf.parallel(self.devices, fn, *inputs)
    if isinstance(ret[0], tuple):
      ret = mtf.transpose_list_of_lists(ret)
      return tuple([self.LaidOutTensor(t) for t in ret])
    else:
      return self.LaidOutTensor(ret)

  def Print(self, x, data, message, **kwargs):  # pylint: disable=invalid-name
    """call tf.Print.

    Args:
      x: a LaidOutTensor
      data: a list of LaidOutTensor
      message: a string
      **kwargs: keyword arguments to tf.print
    Returns:
      a LaidOutTensor
    """
    tf.logging.info("PlacementMeshImpl::Print")
    new_slices = x.tensor_list[:]
    with tf.device(self._devices[0]):
      new_slices[0] = tf.Print(
          new_slices[0], [t for d in data for t in d.tensor_list],
          message, **kwargs)
    return self.LaidOutTensor(new_slices)

  def allreduce(self, x, mesh_axes, reduction_fn_string):
    """Grouped allreduce, (across the given dimensions).

    Args:
      x: a LaidOutTensor
      mesh_axes: a list of integers - the mesh dimensions to be reduced
      reduction_fn_string: "SUM" or "MAX"
    Returns:
      a LaidOutTensor
    """
    return self._collective_with_groups(
        x, mesh_axes, functools.partial(
            allreduce_ring, reduction_fn_string=reduction_fn_string))

  def allconcat(self, x, mesh_axis, concat_axis):
    """Grouped allconcat (like MPI allgather followed by concat).

    Args:
      x: a LaidOutTensor
      mesh_axis: an integer - the mesh axis along which to group
      concat_axis: an integer (the Tensor axis along which to concatenate)
    Returns:
      a LaidOutTensor
    """
    return self._collective_with_groups(
        x, [mesh_axis],
        functools.partial(allconcat_ring, concat_axis=concat_axis))

  def alltoall(self, x, mesh_axis, split_axis, concat_axis):
    """Grouped alltoall.

    Args:
      x: a LaidOutTensor
      mesh_axis: an integer the mesh axis along which to group
      split_axis: an integer (the Tensor axis along which to split)
      concat_axis: an integer (the Tensor axis along which to concatenate)
    Returns:
      a LaidOutTensor
    """
    return self._collective_with_groups(
        x, [mesh_axis],
        functools.partial(
            alltoall_ring, split_axis=split_axis, concat_axis=concat_axis))

  def _collective_with_groups(self, x, mesh_axes, collective):
    """Grouped collective, (across the given dimensions).

    Args:
      x: a LaidOutTensor
      mesh_axes: a list of integers - the mesh dimensions to be reduced
      collective: fn from list(tf.Tensor), list(device) -> list(tf.Tensor)
    Returns:
      a LaidOutTensor
    """
    if not mesh_axes:
      return x
    x = x.to_laid_out_tensor()
    if len(mesh_axes) == self.ndims:
      return self.LaidOutTensor(collective(x.tensor_list, self._devices))
    else:
      groups = mtf.processor_groups(self.shape, mesh_axes)
      ret = [None] * self.size
      for g in groups:
        inputs = [x.tensor_list[pnum] for pnum in g]
        devices = [self._devices[pnum] for pnum in g]
        reduced = collective(inputs, devices)
        for pnum, y in zip(g, reduced):
          ret[pnum] = y
      return self.LaidOutTensor(ret)

  def random(self, shape, tf_fn, kwargs):
    """Call a random tf operation (e.g. random_uniform).

    Args:
      shape: a Shape
      tf_fn: a function such as tf.random_uniform
      kwargs: kwargs to pass to tf_fn, except for seed

    Returns:
      a LaidOutTensor
    """
    slice_shape = self.slice_shape(shape)
    var_scope = tf.get_variable_scope().name
    def my_fn(pnum):
      # seeds are necessary to make sure that slices that should have the
      # same values actually do have the same values.
      seed = hash("%s%s" % (var_scope, self.slice_begin(shape, pnum)))
      return tf_fn(slice_shape, seed=seed, **kwargs)
    return self.slicewise(my_fn, self.laid_out_pnum())

  def laid_out_pnum(self):
    """Returns a LaidOutTensor containing the processor number."""
    return self.LaidOutTensor(list(range(self.size)))

  @property
  def devices(self):
    return self._devices

  def export_to_tf_tensor(self, x, laid_out_x):
    """Turn a Tensor into a tf.Tensor.

    Args:
      x: a Tensor
      laid_out_x: a LaidOutTensor
    Returns:
      a tf.Tensor
    """
    return self.combine_slices(laid_out_x.all_slices, x.shape)

  def import_tf_tensor(self, x, tf_x):
    """Import a tf.Tensor, producing a LaidOutTensor.

    Args:
      x: a Tensor
      tf_x: a tf.Tensor
    Returns:
      a LaidOutTensor
    """
    return self.LaidOutTensor(self.make_slices(tf_x, x.shape))


def allreduce_ring_single_shard(xs, devices, reduction_fn_string="SUM"):
  """Compute the reduction of all Tensors and put the result everywhere.

  Performance-optimized for a ring of devices.

  Args:
    xs: a list of n tf.Tensors
    devices: a list of strings
    reduction_fn_string: "SUM" or "MAX"

  Returns:
    a list of n Tensors
  Raises:
    ValueError: if devices is not a list of n strings
  """
  n = len(xs)
  binary_reduction = mtf.binary_reduction_fn(reduction_fn_string)
  assert len(devices) == n, "devices must be a list of length len(xs)"
  if n == 1:
    return xs
  result = [None] * n
  if n % 2 == 0:
    left_center = n // 2 - 1
    right_center = left_center + 1
  else:
    left_center = n // 2
    right_center = left_center
  left_sum = xs[0]
  for i in xrange(1, left_center + 1):
    with tf.device(devices[i]):
      left_sum = binary_reduction(left_sum, xs[i])
  right_sum = xs[n-1]
  for i in reversed(xrange(left_center + 1, n - 1)):
    with tf.device(devices[i]):
      right_sum = binary_reduction(xs[i], right_sum)
  with tf.device(devices[left_center]):
    result[left_center] = binary_reduction(left_sum, right_sum)
  if n % 2 == 0:
    with tf.device(devices[right_center]):
      result[right_center] = binary_reduction(left_sum, right_sum)
  for i in reversed(xrange(left_center)):
    with tf.device(devices[i]):
      result[i] = tf.identity(result[i + 1])
  for i in xrange(right_center + 1, n):
    with tf.device(devices[i]):
      result[i] = tf.identity(result[i - 1])
  return result


def allreduce_ring(xs, devices, reduction_fn_string="SUM"):
  """Compute the reduction of all Tensors and put the result everywhere.

  Performance-optimized for a ring of devices.

  Args:
    xs: a list of n tf.Tensors
    devices: a list of strings
    reduction_fn_string: "SUM" or "MAX"

  Returns:
    a list of n Tensors
  Raises:
    ValueError: if devices is not a list of n strings
  """
  n = len(xs)
  if len(devices) != n:
    raise ValueError("devices must be a list of length len(xs)")
  if n == 1:
    return xs
  shape = xs[0].shape.as_list()
  # tf.logging.info("allreduce_ring shape = %s" % shape)
  size = None if None in shape else mtf.list_product(shape)
  if size is None or size < 1024 or size % n != 0:
    return allreduce_ring_single_shard(xs, devices, reduction_fn_string)

  def _circular_shift(l, n):
    n %= len(l)
    return l[-n:] + l[:-n]
  def _flatten_and_split(x):
    return tf.split(tf.reshape(x, [size]), n)
  def _concat_and_reshape(xs):
    return tf.reshape(tf.concat(xs, 0), shape)

  # [device, shard]
  x_split = mtf.parallel(devices, _flatten_and_split, xs)
  x_split_t = mtf.transpose_list_of_lists(x_split)

  y_split_t = []
  for shard in xrange(n):
    shard_xs = _circular_shift(x_split_t[shard], shard)
    shard_devices = _circular_shift(devices, shard)
    shard_ys = allreduce_ring_single_shard(
        shard_xs, shard_devices, reduction_fn_string)
    y_split_t.append(_circular_shift(shard_ys, -shard))
  y_split = mtf.transpose_list_of_lists(y_split_t)
  ys = mtf.parallel(devices, _concat_and_reshape, y_split)
  return ys


def allconcat_ring(xs, devices, concat_axis):
  """Concatenate all Tensors everywhere.

  Performance-optimized for a ring of devices.

  Args:
    xs: a list of n tf.Tensors
    devices: a list of n strings
    concat_axis: an integer

  Returns:
    a list of n Tensors
  """
  n = len(xs)
  if n == 1:
    return xs
  # [target, source]
  parts = [[xs[target] if target == source else None for source in xrange(n)]
           for target in xrange(n)]
  for distance in xrange(1, n // 2 + 1):
    for target in xrange(n):
      source = (target + distance) % n
      if parts[target][source] is None:
        with tf.device(devices[target]):
          parts[target][source] = tf.identity(parts[(target + 1) % n][source])
      source = (target - distance) % n
      if parts[target][source] is None:
        with tf.device(devices[target]):
          parts[target][source] = tf.identity(parts[(target - 1) % n][source])
  return mtf.parallel(devices, tf.concat, parts, axis=[concat_axis] * n)


def alltoall_pointtwise(xs, devices, split_axis, concat_axis):
  """MPI alltoall operation.

  Implementation of alltoall using pointwise communication.

  Args:
    xs: a list of n tf.Tensors
    devices: a list of n strings
    split_axis: an integer
    concat_axis: an integer

  Returns:
    a list of n Tensors
  """
  n = len(xs)
  if n == 1:
    return xs
  # [target, source]
  parts = mtf.transpose_list_of_lists(
      mtf.parallel(devices, tf.split, xs, [n] * n, axis=[split_axis] * n))
  return mtf.parallel(devices, tf.concat, parts, axis=[concat_axis] * n)


def alltoall_ring(xs, devices, split_axis, concat_axis):
  """MPI alltoall operation.

  Performance-optimized for a ring of devices.

  Args:
    xs: a list of n tf.Tensors
    devices: a list of n strings
    split_axis: an integer
    concat_axis: an integer

  Returns:
    a list of n Tensors
  """
  n = len(xs)
  if n == 1:
    return xs
  # set up
  # [target, source]
  parts = [[None] * n for i in xrange(n)]
  def my_split(x, size_splits):
    total_size = tf.shape(x)[split_axis]
    part_size = total_size // sum(size_splits)
    return tf.split(x, [s * part_size for s in size_splits], axis=split_axis)
  forward_message_size = (n - 1) // 2
  backward_message_size = (n - 1) - forward_message_size
  forward_messages = [None] * n
  backward_messages = [None] * n
  for i in xrange(n):
    with tf.device(devices[i]):
      if i >= backward_message_size:
        a, b, c, d = my_split(
            xs[i], [i - backward_message_size,
                    backward_message_size, 1, n - i - 1])
        backward_messages[i] = b
        parts[i][i] = c
        forward_messages[i] = tf.concat([d, a], axis=split_axis)
      else:
        a, b, c, d = my_split(
            xs[i], [i, 1, forward_message_size, backward_message_size - i])
        backward_messages[i] = tf.concat([d, a], axis=split_axis)
        parts[i][i] = b
        forward_messages[i] = c
  for step in xrange(1, max(forward_message_size, backward_message_size) + 1):
    new_forward_messages = [None] * n
    new_backward_messages = [None] * n
    for i in xrange(n):
      with tf.device(devices[i]):
        if forward_message_size > 0:
          parts[i][(i - step) % n], new_forward_messages[i] = my_split(
              forward_messages[(i - 1) % n], [1, forward_message_size - 1])
        if backward_message_size > 0:
          new_backward_messages[i], parts[i][(i + step) % n] = my_split(
              backward_messages[(i + 1) % n], [backward_message_size - 1, 1])
    forward_message_size -= 1
    backward_message_size -= 1
    forward_messages = new_forward_messages
    backward_messages = new_backward_messages
  return mtf.parallel(devices, tf.concat, parts, axis=[concat_axis] * n)
