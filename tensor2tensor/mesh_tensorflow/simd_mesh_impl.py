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
"""SIMD Mesh implementation (for TPU/XLA)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.mesh_tensorflow import mesh_tensorflow as mtf
from tensor2tensor.mesh_tensorflow import mtf_utils
from tensor2tensor.mesh_tensorflow import tpu_variables

import tensorflow as tf

from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.python.framework import ops


class SimdMeshImpl(mtf.MeshImpl):
  """Mesh implementation for TPU using SIMD and MPI operations."""

  def __init__(self, shape, layout, devices, device_assignment):
    super(SimdMeshImpl, self).__init__(shape, layout)
    self._devices = devices
    self._device_assignment = device_assignment
    tf.logging.info("SimdMeshImpl init: {0} {1}".format(shape, layout))
    self._pnum_tensor = None

  @property
  def pnum_tensor(self):
    if self._pnum_tensor is not None:
      return self._pnum_tensor
    with mtf_utils.outside_all_rewrites():
      tf.logging.info("Create pnum_tensor")
      self._pnum_tensor = tpu_ops.tpu_replicated_input(
          list(range(self.size)), name="pnum_constants")
      return self._pnum_tensor

  class LaidOutTensor(object):
    """One Slice."""

    def __init__(self, tensor_list):
      assert isinstance(tensor_list, list)
      self._tensor_list = tensor_list

    def __repr__(self):
      return "[" + ",".join([str(t) for t in self._tensor_list]) + "]"

    @property
    def tensor_list(self):
      return self._tensor_list

    @property
    def one_slice(self):
      return self._tensor_list[0]

    @classmethod
    def from_tensor_list(cls, tensor_list):
      return cls(tensor_list)

    @property
    def all_slices(self):
      return self._tensor_list

    @property
    def slice_shape(self):
      return self.one_slice.shape.as_list()

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
        slice_var_name = base_name + "_slice_%d" % pnum
        tpu_device = mesh_impl.device_assignment.tpu_device(replica=pnum)
        # The initializer is unimportant, since the slice variables will be
        # overwritten.  zeros_initializer() is here to avoid the default
        # initialization which adds lots of useless operations to the TF graph.
        with ops.device(tpu_device):
          slices.append(
              tf.get_variable(
                  slice_var_name,
                  slice_shape,
                  dtype=dtype,
                  collections=[],
                  initializer=tf.zeros_initializer()))
      self._laid_out_tensor = mesh_impl.LaidOutTensor(
          [tpu_variables.ReplicatedVariable(base_name, slices)])
      with tf.device("cpu:0"), mtf_utils.outside_all_rewrites():
        self._copy_master_to_slices = self.assign_to_slices(
            mesh_impl.make_slices(variable.master, shape),
            assign_to_tensor_list=slices)
        self._copy_slices_to_master = tf.assign(
            variable.master,
            mesh_impl.combine_slices(slices, shape, device="cpu:0"))

    def assign_to_slices(self, slice_values, assign_to_tensor_list=None):
      """Assign to the slice variables.

      Args:
        slice_values: a list of tf.Tensor
        assign_to_tensor_list: an optional list of tf.Variable

      Returns:
        a tf.operation
      """
      if assign_to_tensor_list is None:
        assign_to_tensor_list = self._laid_out_tensor.all_slices
      # Handle both N -> 1 and N -> N cases.
      num_slices = min(
          len(assign_to_tensor_list), len(slice_values))
      devices = [""] * num_slices
      return tf.group(
          mtf.parallel(devices, tf.assign, assign_to_tensor_list[:num_slices],
                       slice_values[:num_slices]))

    @property
    def laid_out_tensor(self):
      return self._laid_out_tensor

    @property
    def copy_master_to_slices(self):
      return self._copy_master_to_slices

    @property
    def copy_slices_to_master(self):
      return self._copy_slices_to_master

  def laid_out_pnum(self):
    """Returns a LaidOutTensor containing the processor number.

    Returns:
      a LaidOutTensor where each slice is an integer scalar
    """
    return self.LaidOutTensor([self.pnum_tensor])

  def allreduce(self, x, mesh_axes, reduction_fn_string):
    """Grouped allreduce, (summed across the given dimensions).

    Args:
      x: a LaidOutTensor
      mesh_axes: a list of integers
      reduction_fn_string: "SUM"
    Returns:
      a LaidOutTensor
    Raises:
      ValueError: if the reduction is not yet implemented.
    """
    if not mesh_axes:
      return x
    x = x.to_laid_out_tensor()
    if reduction_fn_string == "SUM":
      partitioning = [
          mtf.pnum_to_group(self.shape, mesh_axes, pnum)
          for pnum in xrange(self.size)]
      return self.LaidOutTensor(
          [tpu_ops.cross_replica_sum(x.one_slice, partitioning)])
    else:
      for axis in mesh_axes:
        x = self.allconcat(x, axis, 0, stack=True)
        x = self.LaidOutTensor(
            [mtf.reduction_fn(reduction_fn_string)(x.one_slice, 0)])
      return x

  def allconcat(self, x, mesh_axis, concat_axis, stack=False):
    """Grouped allconcat (like MPI allgather followed by concat).

    Args:
      x: a LaidOutTensor
      mesh_axis: an integer - the mesh axis along which to group
      concat_axis: an integer (the Tensor axis along which to concatenate)
      stack: a boolean - whether to stack instead of concat
    Returns:
      a LaidOutTensor
    """
    x = x.to_laid_out_tensor()
    coord = self.laid_out_pcoord(mesh_axis)
    t = x.one_slice
    old_shape = t.shape.as_list()
    num_parts = self.shape[mesh_axis].size
    t = tf.expand_dims(t, concat_axis)
    t *= tf.reshape(
        tf.one_hot(coord.one_slice, num_parts, dtype=t.dtype),
        [num_parts if i == concat_axis else 1
         for i in xrange(len(old_shape) + 1)])
    if not stack:
      new_shape = old_shape[:]
      new_shape[concat_axis] *= num_parts
      t = tf.reshape(t, new_shape)
    return self.allreduce(self.LaidOutTensor([t]), [mesh_axis], "SUM")

  def alltoall(self, x, mesh_axis, split_axis, concat_axis):
    """Grouped alltoall (like MPI alltoall with splitting and concatenation).

    TODO(noam): this is a terribly inefficient implementation using allreduce.
    Replace this with a native xla alltoall once it is ready.

    Args:
      x: a LaidOutTensor
      mesh_axis: an integer the mesh axis along which to group
      split_axis: an integer (the Tensor axis along which to split)
      concat_axis: an integer (the Tensor axis along which to concatenate)
    Returns:
      a LaidOutTensor
    """
    x = x.to_laid_out_tensor()
    x = self.allconcat(x, mesh_axis, concat_axis)
    x = self.allsplit(x, mesh_axis, split_axis)
    return x

  def slice(self, tf_tensor, tensor_shape):
    """"Slice out the correspoding part of tensor given the pnum variable."""
    tensor_layout = self.tensor_layout(tensor_shape)

    if tensor_layout.is_fully_replicated:
      return self.LaidOutTensor([tf_tensor])
    else:
      slice_shape = self.slice_shape(tensor_shape)
      slice_begins = [
          self.slice_begin(tensor_shape, pnum) for pnum in xrange(self.size)
      ]
      slice_begins_tensor = tf.stack(slice_begins)
      # slice on source device
      selected_slice_begin = tf.gather(slice_begins_tensor, self.pnum_tensor)
      return self.LaidOutTensor(
          [tf.slice(tf_tensor, selected_slice_begin, slice_shape)])

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
    ret = fn(*[x.one_slice if isinstance(x, self.LaidOutTensor)
               else x for x in inputs])
    if isinstance(ret, tuple):
      return tuple([self.LaidOutTensor([t]) for t in ret])
    else:
      return self.LaidOutTensor([ret])

  @property
  def device_assignment(self):
    return self._device_assignment

  @property
  def devices(self):
    return self._devices

  def random(self, shape, tf_fn, kwargs):
    """Call a random tf operation (e.g. random_uniform).

    Args:
      shape: a Shape
      tf_fn: a function such as tf.random_uniform
      kwargs: kwargs to pass to tf_fn, except for seed

    Returns:
      a LaidOutTensor
    """
    # TODO(noam): can we make things better with stateless_random?
    slice_shape = self.slice_shape(shape)
    x = tf_fn(slice_shape, **kwargs)
    # TPU does not have seeds enabled.  Sync up the
    # random choices by zeroing out all but the first core per group of
    # identical slices, then allreducing by group.
    layout = self.tensor_layout(shape)
    # we need to sync across these axes.
    mesh_axes = [i for i in xrange(self.ndims)
                 if i not in layout.tensor_axis_to_mesh_axis]
    multiplier = 1.0
    for axis in mesh_axes:
      multiplier *= tf.cast(
          tf.equal(self.laid_out_pcoord(axis).one_slice, 0), x.dtype)
    x *= multiplier
    x = self.LaidOutTensor([x])
    x = self.allreduce(x, mesh_axes, "SUM")
    return x

  def export_to_tf_tensor(self, x, laid_out_x):
    """Turn a Tensor into a tf.Tensor.

    Args:
      x: a Tensor
      laid_out_x: a LaidOutTensor
    Returns:
      a tf.Tensor
    """
    tensor_layout = self.tensor_layout(x.shape)
    if not tensor_layout.is_fully_replicated:
      raise NotImplementedError(
          "SimdMeshImpl only supports export_to_tf_tensor of fully-replicated "
          "Tensors.  Try reshaping to new dimension names. "
          " x.shape = %s tensor_layout=%s"
          % (x.shape, tensor_layout))
    return laid_out_x.one_slice

  def import_tf_tensor(self, x, tf_x):
    """Import a tf.Tensor, producing a LaidOutTensor.

    Args:
      x: a Tensor
      tf_x: a tf.Tensor
    Returns:
      a LaidOutTensor
    """
    return self.slice(tf_x, x.shape)

  @property
  def supports_control_dependencies(self):
    return False
