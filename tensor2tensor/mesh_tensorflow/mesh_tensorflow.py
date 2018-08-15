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
"""Mesh-TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from functools import reduce  # pylint: disable=redefined-builtin; for py3
from operator import mul
import re
import six
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.mesh_tensorflow import mtf_utils
import tensorflow as tf


Dimension = collections.namedtuple("Dimension", ["name", "size"])


def convert_to_dimension(d):
  """Converts input to a Dimension.

  Args:
    d: Dimension, tuple (string, int), or None.

  Returns:
    Dimension or None.

  Raises:
    ValueError: If d cannot be converted to a Dimension.
  """
  if d is None:
    return None
  if isinstance(d, Dimension):
    return d
  name, size = d
  if isinstance(name, str) and isinstance(size, int):
    return Dimension(name, size)
  else:
    raise ValueError("could not convert %s to Dimension" % (d,))


class Shape(object):
  """Shape of a Tensor or Mesh.

  #### Examples

  ```python
  # Create shape [4, 8] with names "x" and "y" respectively.
  shape = mtf.Shape([mtf.Dimension("x", 4), mtf.Dimension("y", 8)])
  ```
  """

  def __init__(self, dims):
    """Constructs a shape for a Tensor or Mesh.

    Args:
      dims: List-like of Dimensions.

    Raises:
      ValueError: If Dimensions are repeated.
    """
    self._dims = [convert_to_dimension(d) for d in tuple(dims)]
    if len(set(dims)) != len(dims):
      raise ValueError("Shape must not have repeated dimensions %s" % dims)

  @property
  def dims(self):
    return list(self._dims)

  @property
  def ndims(self):
    return len(self._dims)

  def __repr__(self):
    return self.to_string

  def __eq__(self, other):
    return self.dims == other.dims

  def __ne__(self, other):
    return self.dims != other.dims

  def __add__(self, other):
    if isinstance(other, Shape):
      other = other.dims
    if isinstance(other, Dimension):
      other = [other]
    return Shape(self.dims + other)

  def __sub__(self, other):
    if other is None:
      return self
    if isinstance(other, Shape):
      other = other.dims
    if isinstance(other, Dimension):
      other = [other]
    return Shape([d for d in self.dims if d not in other])

  def __len__(self):
    return len(self._dims)

  def __getitem__(self, key):
    return self._dims[key]

  def __iter__(self):
    return iter(self._dims)

  @property
  def to_integer_list(self):
    return [d.size for d in self.dims]

  @property
  def size(self):
    return list_product(self.to_integer_list)

  @property
  def to_string(self):
    return "Shape[%s]" % ", ".join(
        ["%s=%d" % (d.name, d.size) for d in self.dims])

  @property
  def cumprod(self):
    """Cumulative product (exclusive) of Dimension sizes."""
    return _cumprod(self.to_integer_list)[::-1]

  def cumprod_to_tensor_axis(self, cumprod):
    """Tensor axis i such that self.cumprod[i] == cumprod, or None."""
    try:
      return self.cumprod.index(cumprod)
    except ValueError:
      return None

  @property
  def dimension_names(self):
    return [d.name for d in self.dims]

  def rename_dimension(self, old_name, new_name):
    """Returns a copy where one dimension is renamed."""
    if old_name not in self.dimension_names:
      raise ValueError("Shape %s does not have dimension named %s"
                       % (self, old_name))
    return Shape(
        [Dimension(new_name, d.size) if d.name == old_name else d
         for d in self.dims])

  def resize_dimension(self, name, new_size):
    """Returns a copy where one dimension has a different size."""
    if name not in self.dimension_names:
      raise ValueError("Shape %s does not have dimension named %s"
                       % (self, name))
    return Shape(
        [Dimension(name, new_size) if d.name == name else d
         for d in self.dims])


def convert_to_shape(x):
  """Converts input to a Shape.

  Args:
    x: Shape, str, or None.

  Returns:
    Shape or None.

  Raises:
    ValueError: If x cannot be converted to a Shape.
  """
  if x is None:
    return None
  if isinstance(x, Shape):
    return x
  if isinstance(x, str):
    x = _parse_string_to_list_of_pairs(x, seconds_to_int=True)
  return Shape(x)


class LayoutRules(object):
  """Represents layout of a computation.

  #### Examples

  ```python
  # Map "d_ff" and "heads" Tensor Dimensions to the "model" Mesh Dimension.
  layout_rules = mtf.LayoutRules([("d_ff", "model"), ("heads", "model")])
  ```
  """

  def __init__(self, pairs):
    """Constructs a layout.

    Args:
      pairs: Set-like of string pairs (tensor_dim_name, mesh_dim_name).
    """
    self._pairs = set(pairs)

  def __repr__(self):
    return "LayoutRules%s" % self._pairs

  def tensor_dimension_to_mesh_axis(self, tensor_dimension, mesh_shape):
    """Mesh axis associated with tensor dimension (or None).

    Args:
      tensor_dimension: Dimension.
      mesh_shape: Shape.

    Returns:
      Integer or None.

    Raises:
      ValueError: If one Tensor dimension maps to two mesh dimensions.
    """
    val = [i for i, mesh_dimension in enumerate(mesh_shape)
           if (tensor_dimension.name, mesh_dimension.name) in self._pairs]
    if len(val) > 1:
      raise ValueError(
          "Tensor dimension maps to multiple mesh dimensions"
          " tensor_dimension=%s mesh_shape=%s layout=%s"
          % (tensor_dimension, mesh_shape, self._pairs))
    return val[0] if val else None

  def tensor_layout(self, tensor_shape, mesh_shape):
    """Computes TensorLayout given a Tensor Shape and a Mesh Shape.

    Args:
      tensor_shape: Shape.
      mesh_shape: Shape.

    Returns:
      TensorLayout.

    Raises:
      ValueError: If two Tensor Dimensions map to the same Mesh Dimensions.
    """
    ret = [self.tensor_dimension_to_mesh_axis(d, mesh_shape)
           for d in tensor_shape]
    not_nones = [a for a in ret if a is not None]
    if len(not_nones) != len(set(not_nones)):
      raise ValueError(
          "Two Tensor Dimensions may not map to the same Mesh Dimension:"
          " layout=%s tensor_shape=%s mesh_shape=%s " %
          (self, tensor_shape, mesh_shape))
    return TensorLayout(ret)


def convert_to_layout_rules(x):
  """Converts input to a LayoutRules.

  Args:
    x: LayoutRules, str, or set-like of string pairs.

  Returns:
    LayoutRules.
  """
  if isinstance(x, LayoutRules):
    return x
  if isinstance(x, str):
    x = _parse_string_to_list_of_pairs(x)
  return LayoutRules(x)


class TensorLayout(object):
  """Injective partial map between Tensor axes and Mesh axes.

  TensorLayout is a tuple of optional integers with length tensor.ndims. Each
  item is either a unique integer indicating the mesh axis over which that
  tensor dimension is split or None, indicating that this tensor dimension is
  not split.

  #### Examples

  ```python
  # Split first and last Tensor dimensions according to mesh axes 0 and 1.
  tensor_layout = mtf.TensorLayout([0, None, 1])
  ```
  """

  def __init__(self, tensor_axis_to_mesh_axis):
    """Creates a TensorLayout.

    Args:
      tensor_axis_to_mesh_axis: List-like where each element is an int or None.
    """
    self._tensor_axis_to_mesh_axis = tuple(tensor_axis_to_mesh_axis)

  def __eq__(self, other):
    return self.tensor_axis_to_mesh_axis == other.tensor_axis_to_mesh_axis

  def __ne__(self, other):
    return self.tensor_axis_to_mesh_axis != other.tensor_axis_to_mesh_axis

  def __repr__(self):
    return "TensorLayout%s" % (self.tensor_axis_to_mesh_axis,)

  def __len__(self):
    return len(self._tensor_axis_to_mesh_axis)

  def __getitem__(self, key):
    return self._tensor_axis_to_mesh_axis[key]

  def __iter__(self):
    return iter(self._tensor_axis_to_mesh_axis)

  @property
  def tensor_axis_to_mesh_axis(self):
    """Converts to a tuple of optional integers."""
    return self._tensor_axis_to_mesh_axis

  @property
  def is_fully_replicated(self):
    """Whether all tensor dimensions map to None."""
    return self.tensor_axis_to_mesh_axis == (None,) * len(self)

  def mesh_axis_to_tensor_axis(self, mesh_ndims):
    """For each mesh axis, which Tensor axis maps to it.

    Args:
      mesh_ndims: int.

    Returns:
      Tuple of optional integers, with length mesh_ndims.
    """
    return tuple(
        [self._tensor_axis_to_mesh_axis.index(mesh_axis)
         if mesh_axis in self._tensor_axis_to_mesh_axis else None
         for mesh_axis in xrange(mesh_ndims)])


class Graph(object):
  """Mesh-TensorFlow graph."""

  def __init__(self):
    self._operations = []
    self._tensors = []
    self._trainable_variables = []
    self._all_variables = []

  def __repr__(self):
    return self.to_string

  @property
  def operations(self):
    return self._operations

  @property
  def tensors(self):
    return self._tensors

  @property
  def trainable_variables(self):
    return self._trainable_variables

  @property
  def all_variables(self):
    return self._all_variables

  @property
  def to_string(self):
    return "\n".join([op.to_string for op in self.operations])


class Lowering(object):
  """Lowering of a Graph from Mesh-TensorFlow to TensorFlow.

  #### Examples

  Below we form a Graph with one Tensor and lower it to recover the original
  tf.Tensor.

  ```python
  from tensor2tensor.mesh_tensorflow import placement_mesh_impl

  graph = mtf.Graph()
  mesh = mtf.Mesh(graph, "my_mesh")
  inputs = tf.constant(0.)
  mtf_inputs = mtf.import_tf_tensor(mesh,
                                    inputs=inputs,
                                    shape=mtf.Shape([]))
  mesh_impl = placement_mesh_impl.PlacementMeshImpl(
      shape=[], layout={}, devices=[""])
  lowering = mtf.Lowering(graph, {mesh: mesh_impl})
  outputs = lowering.export_to_tf_tensor(mtf_inputs)  # tf.constant(0.)
  ```
  """

  def __init__(self, graph, mesh_to_impl):
    """Creates a Lowering of a Graph.

    Args:
      graph: Graph.
      mesh_to_impl: {Mesh: MeshImpl}. Keys are the Mesh's in the graph and
        their values are MeshImpl's, which map Tensor Dimension names to
        Mesh Dimension names.
    """
    # tf.logging.info("LOWERING GRAPH:\n%s" % graph.to_string)
    self.mesh_to_impl = mesh_to_impl   # {Mesh: MeshImpl}
    self.graph = graph
    self._counters = []
    self.tensors = {}                  # {Tensor: Mesh.LaidOutTensor}
    self.operations = {}               # {Operation: tf.Operation}
    self.variables = {}                # {Variable: LaidOutVariable}
    for op in graph.operations:
      # tf.logging.info("Lowering operation %s" % op.to_string)
      with tf.name_scope(op.name):
        op.lower(self)
      for out in op.outputs:
        self.add_counter(
            "output/%s" % type(op).__name__, self.laid_out_size(out))
        self.add_counter("output_unique/%s" % type(op).__name__, out.size)
    log_variable_sizes(
        graph.trainable_variables, "Trainable Variables", verbose=True)
    tf.logging.info("Counters:\n" + pretty_print_counters(self._counters))

  def mesh_impl(self, m):
    if not isinstance(m, Mesh):
      m = m.mesh
    return self.mesh_to_impl[m]

  def export_to_tf_tensor(self, x):
    """Turn a Tensor into a tf.Tensor.

    Args:
      x: Tensor.

    Returns:
      tf.Tensor.
    """
    mesh_impl = self.mesh_impl(x)
    return mesh_impl.export_to_tf_tensor(
        x, self.tensors[x].to_laid_out_tensor())

  def lowered_operation(self, op):
    return self.operations[op]

  def copy_masters_to_slices(self):
    return tf.group(
        [v.copy_master_to_slices for v in six.itervalues(self.variables)])

  def copy_slices_to_masters(self):
    return tf.group(
        [v.copy_slices_to_master for v in six.itervalues(self.variables)])

  def add_counter(self, key, value):
    assert isinstance(value, int)
    self._counters.append((key, value))

  @property
  def counters(self):
    return self._counters

  def laid_out_size(self, tensor):
    """Total size of all slices.

    Args:
      tensor: Tensor.

    Returns:
      int.
    """
    return self.mesh_impl(tensor).laid_out_size(tensor.shape)

  def set_tensor_lowering(self, tensor, laid_out_tensor):
    self.verify_slice_shapes(tensor, laid_out_tensor)
    self.tensors[tensor] = laid_out_tensor

  def verify_slice_shapes(self, tensor, laid_out_tensor):
    mesh_impl = self.mesh_impl(tensor)
    correct_shape = mesh_impl.slice_shape(tensor.shape)
    actual_shape = laid_out_tensor.slice_shape
    if actual_shape != correct_shape:
      raise ValueError(
          "Wrong slice shape: correct_shape = %s actual shape = %s"
          % (correct_shape, actual_shape))


class Mesh(object):
  """A placeholder with no functionality.

  A Graph is built with each Tensor assigned to a Mesh. The Mesh does not
  know its shape or its implementation.

  A Lowering assigns each Mesh to a MeshImpl.
  """

  def __init__(self, graph, name):
    self._graph = graph
    self._name = name

  @property
  def graph(self):
    return self._graph


class MeshImpl(object):
  """Implementation of a Mesh.

  Unlike Mesh, MeshImpl carries Shape and LayoutRules. Subclasses of MeshImpl
  also carry devices.

  #### Examples

  ```python
  shape = mtf.Shape([mtf.Dimension("batch", 4),
                     mtf.Dimension("model", 8)])
  layout_rules = mtf.LayoutRules([("batch", "batch"),
                                  ("d_ff", "model"),
                                  ("heads", "model")])
  mesh_impl = mtf.MeshImpl(shape=shape, layout_rules=layout_rules)
  ```
  """

  def __init__(self, shape, layout_rules):
    """Creates a mesh implementation.

    Args:
      shape: Shape.
      layout_rules: LayoutRules.
    """
    self._shape = convert_to_shape(shape)
    self._layout_rules = convert_to_layout_rules(layout_rules)

  @property
  def shape(self):
    return self._shape

  @property
  def ndims(self):
    return len(self._shape)

  @property
  def layout_rules(self):
    return self._layout_rules

  @property
  def size(self):
    return self.shape.size

  @property
  def supports_control_dependencies(self):
    return True

  def tensor_dimension_to_mesh_axis(self, tensor_dimension):
    """Mesh axis associated with tensor dimension (or None).

    Args:
      tensor_dimension: Dimension.

    Returns:
      int or None.
    """
    return self.layout_rules.tensor_dimension_to_mesh_axis(
        tensor_dimension, self.shape)

  def tensor_layout(self, arg):
    """Compute TensorLayout for a Tensor or a Shape.

    Args:
      arg: Tensor or Shape.

    Returns:
      TensorLayout.
    """
    if isinstance(arg, Tensor):
      arg = arg.shape
    return self.layout_rules.tensor_layout(arg, self.shape)

  def mesh_axis_to_cumprod(self, tensor_shape):
    """For each mesh axis, give the product of previous tensor axes.

    Args:
      tensor_shape: Shape.

    Returns:
      list with length self.ndims where each element is an integer or None.
    """
    tensor_layout = self.tensor_layout(tensor_shape)
    ma2ta = tensor_layout.mesh_axis_to_tensor_axis(self.ndims)
    ta2cumprod = tensor_shape.cumprod
    return [None if ta is None else ta2cumprod[ta] for ta in ma2ta]

  def slice_shape(self, tensor_shape):
    """Shape of each slice of the Tensor.

    Args:
      tensor_shape: Shape.

    Returns:
      list of integers with length tensor_shape.ndims.

    Raises:
      ValueError: If a Tensor dimension is not divisible by the corresponding
        Mesh dimension.
    """
    tensor_layout = self.tensor_layout(tensor_shape)
    ret = []
    for tensor_dim, mesh_axis in zip(
        tensor_shape, tensor_layout.tensor_axis_to_mesh_axis):
      if mesh_axis is None:
        ret.append(tensor_dim.size)
      else:
        mesh_dim = self.shape[mesh_axis]
        if tensor_dim.size % mesh_dim.size != 0:
          raise ValueError(
              "Tensor dimension size not divisible by mesh dimension size:"
              " tensor_shape=%s tensor_layout=%s"
              % (tensor_shape, tensor_layout))
        ret.append(tensor_dim.size // mesh_dim.size)
    return ret

  def slice_begin(self, tensor_shape, pnum):
    """Begin position for the tensor slice for the given processor.

    Args:
      tensor_shape: Shape.
      pnum: int <= self.size.

    Returns:
      list of integers with length tensor_shape.ndims.
    """
    tensor_layout = self.tensor_layout(tensor_shape)
    coordinates = pnum_to_processor_coordinates(self.shape, pnum)
    ret = []
    for dim_size, mesh_axis in zip(
        tensor_shape.to_integer_list, tensor_layout.tensor_axis_to_mesh_axis):
      if mesh_axis is None:
        ret.append(0)
      else:
        ret.append(
            dim_size // self.shape[mesh_axis].size * coordinates[mesh_axis])
    return ret

  def laid_out_size(self, tensor_shape):
    """Total size of all slices.

    Args:
      tensor_shape: Shape.

    Returns:
      int.
    """
    return list_product(self.slice_shape(tensor_shape)) * self.size

  def slicewise(self, fn, *inputs):
    """Executes a function in parallel on all slices.

    Args:
      fn: function from tf.Tensors to tf.Tensor or a tuple of tf.Tensors.
      *inputs: list of inputs.  Each input is either a LaidOutTensor or
        is convertible to a tf.Tensor.

    Returns:
      LaidOutTensor, or a tuple of LaidOutTensors if fn returns a tuple.
    """
    raise NotImplementedError("Slicewise not implemented")

  def Print(self, x, data, message, **kwargs):  # pylint: disable=invalid-name
    """Calls tf.Print.

    Args:
      x: LaidOutTensor.
      data: list of LaidOutTensor.
      message: str.
      **kwargs: keyword arguments to tf.print.

    Returns:
      LaidOutTensor.
    """
    del data, message, kwargs
    tf.logging.warning("Warning - mtf.Print not implemented for this mesh type")
    return x

  def allreduce(self, x, mesh_axes, reduction_fn_string):
    """Grouped allreduce, (summed across the given dimensions).

    Args:
      x: LaidOutTensor.
      mesh_axes: list of integers, the mesh dimensions to be reduced.
      reduction_fn_string: "SUM" or "MAX".

    Returns:
      LaidOutTensor.
    """
    raise NotImplementedError("Allreduce not implemented")

  def allsplit(self, x, mesh_axis, split_axis):
    """Inverse of allconcat - split each slice and keep only one piece of it.

    The number of ways to split is the number of processors in the group.
    The part that is kept corresponds to the processor's index in the group.

    Args:
      x: LaidOutTensor.
      mesh_axis: int, the mesh axis along which to split.
      split_axis: int, the Tensor axis along which to split.

    Returns:
      LaidOutTensor.
    """
    num_splits = self.shape[mesh_axis].size
    def my_fn(x, coordinate):
      slice_begin = [
          dimsize // num_splits * coordinate if i == split_axis
          else 0 for i, dimsize in enumerate(x.shape.as_list())]
      slice_size = [
          dimsize // num_splits if i == split_axis
          else dimsize for i, dimsize in enumerate(x.shape.as_list())]
      return tf.slice(x, slice_begin, slice_size)
    return self.slicewise(my_fn, x, self.laid_out_pcoord(mesh_axis))

  def allconcat(self, x, mesh_axis, concat_axis):
    """Grouped allconcat (like MPI allgather followed by concat).

    Args:
      x: LaidOutTensor.
      mesh_axis: int, the mesh axis along which to group.
      concat_axis: int, the Tensor axis along which to concatenate.

    Returns:
      LaidOutTensor.
    """
    raise NotImplementedError("Allconcat not implemented")

  def alltoall(self, x, mesh_axis, split_axis, concat_axis):
    """Grouped alltoall (like MPI alltoall with splitting and concatenation).

    Args:
      x: LaidOutTensor.
      mesh_axis: int, the mesh axis along which to group.
      split_axis: int, the Tensor axis along which to split.
      concat_axis: int, the Tensor axis along which to concatenate.

    Returns:
      LaidOutTensor.
    """
    raise NotImplementedError("Alltoall not implemented")

  def laid_out_pnum(self):
    """Returns a LaidOutTensor containing the processor number.

    Returns:
      LaidOutTensor where each slice is an integer scalar.
    """
    raise NotImplementedError("laid_out_pnum not implemented")

  def laid_out_pcoord(self, mesh_axis):
    """Returns a LaidOutTensor containing the processor coordinate.

    Args:
      mesh_axis: int.

    Returns:
      LaidOutTensor where each slice is an integer scalar.
    """
    divisor = list_product(self.shape.to_integer_list[mesh_axis + 1:])
    modulus = self.shape[mesh_axis].size
    def my_fn(pnum):
      return (pnum // divisor) % modulus
    return self.slicewise(my_fn, self.laid_out_pnum())

  def broadcast_impl(self, old_slices, old_shape, new_shape):
    """Implementation of a broadcast operation.

    Args:
      old_slices: LaidOutTensor.
      old_shape: Shape.
      new_shape: Shape.

    Returns:
      LaidOutTensor.
    """
    new_slice_shape = self.slice_shape(new_shape)
    def tf_fn(x):
      return (tf.zeros(new_slice_shape, dtype=x.dtype) +
              _expand_dims(x, old_shape, new_shape))
    return self.slicewise(tf_fn, old_slices)

  def make_slices(self, tf_tensor, tensor_shape):
    """Turns a single tf.Tensor into a list of slices, one for each processor.

    Args:
      tf_tensor: tf.Tensor.
      tensor_shape: Shape.

    Returns:
      list of tf.tensor with length self.size.
    """
    tensor_layout = self.tensor_layout(tensor_shape)
    slice_shape = self.slice_shape(tensor_shape)
    def my_fn(pnum):
      if tensor_layout.is_fully_replicated:
        return tf_tensor
      else:
        slice_begin = self.slice_begin(tensor_shape, pnum)
        return tf.slice(tf_tensor, slice_begin, slice_shape)

    return parallel([tf_tensor.device] * self.size, my_fn,
                    list(xrange(self.size)))

  def combine_slices(self, slices, tensor_shape, device=None):
    """Turns a set of slices into a single tensor.

    Args:
      slices: list of tf.Tensor with length self.size.
      tensor_shape: Shape.
      device: optional str. If absent, we use the devices of the slices.

    Returns:
      tf.Tensor.
    """
    if tensor_shape.ndims == 0:
      return slices[0]

    ret = slices[:]
    tensor_layout = self.tensor_layout(tensor_shape)
    for mesh_dim, tensor_axis in zip(
        self.shape, tensor_layout.mesh_axis_to_tensor_axis(self.ndims)):
      slice_size = len(ret) // mesh_dim.size
      if tensor_axis is None:
        ret = ret[:slice_size]
      else:
        if device:
          devices = [device] * slice_size
        else:
          devices = [ret[i].device for i in xrange(slice_size)]
        concat_inputs = [[ret[i + slice_size * j]
                          for j in xrange(mesh_dim.size)]
                         for i in xrange(slice_size)]
        ret = parallel(
            devices, tf.concat, concat_inputs,
            axis=[tensor_axis] * len(devices))
    assert len(ret) == 1
    return ret[0]

  def export_to_tf_tensor(self, x, laid_out_x):
    """Turns a Tensor into a tf.Tensor.

    Args:
      x: Tensor.
      laid_out_x: LaidOutTensor.

    Returns:
      tf.Tensor.
    """
    raise NotImplementedError("export_to_tf_tensor not implemented")

  def import_tf_tensor(self, x, tf_x):
    """Imports a tf.Tensor, producing a LaidOutTensor.

    Args:
      x: Tensor.
      tf_x: tf.Tensor.

    Returns:
      LaidOutTensor.
    """
    raise NotImplementedError("Import not implemented")


class LazyAllreduceSum(object):
  """Represents a LaidOutTensor with a lazy allreduce.

  The purpose of delaying allreduce is that it saves bandwidth to first add
  and then allreduce, as opposed to the other way around.
  """

  def __init__(self,
               mesh_impl,
               laid_out_input,
               mesh_axes,
               add_counter_fn=None):
    """Create a LazyAllreduceSum.

    Args:
      mesh_impl: a mesh_impl
      laid_out_input: a LaidOutTensor
      mesh_axes: a list of mesh axes
      add_counter_fn: a function taking no arguments which calls
        lowering.add_counter if and when the allreduce executes.
    Returns:
      a LazyAllreduceSum
    """
    self.mesh_impl = mesh_impl
    self.laid_out_input = laid_out_input
    self.mesh_axes = mesh_axes
    self._add_counter_fn = add_counter_fn
    self._reduced = None

  def to_laid_out_tensor(self):
    if not self._reduced:
      self._reduced = self.mesh_impl.allreduce(
          self.laid_out_input, self.mesh_axes, "SUM")
      if self._add_counter_fn:
        self._add_counter_fn()
    return self._reduced

  def __add__(self, other):
    """Add to another LazyAllreduceSum.

    Args:
      other: a LazyAllreduceSum or a LaidOutTensor
    Returns:
      a LazyAllreduceSum or a LaidOutTensor
    """
    if (isinstance(other, LazyAllreduceSum) and
        self.mesh_impl == other.mesh_impl and
        self.mesh_axes == other.mesh_axes):
      return LazyAllreduceSum(
          self.mesh_impl,
          self.mesh_impl.slicewise(
              tf.add, self.laid_out_input, other.laid_out_input),
          self.mesh_axes,
          add_counter_fn=self._add_counter_fn)
    else:
      return self.mesh_impl.slicewise(
          tf.add, self.to_laid_out_tensor(), other.to_laid_out_tensor())

  @property
  def slice_shape(self):
    return self.laid_out_input.slice_shape


def convert_args_to_laid_out_tensors(xs):
  """Convert list elements to laid-out-tensors when possible.

  Args:
    xs: a list
  Returns:
    a list
  """
  ret = []
  for x in xs:
    try:
      ret.append(x.to_laid_out_tensor())
    except AttributeError:
      ret.append(x)
  return ret


class Tensor(object):
  """A Distributed Tensor."""

  def __init__(self, operation, shape, dtype, name=None):
    if not isinstance(shape, Shape):
      raise ValueError("shape must be a Shape got %s" % shape.to_string)
    if not isinstance(dtype, tf.DType):
      raise ValueError("dtype must be a tf.DType got %s" % dtype)
    self._mesh = operation.mesh
    self._operation = operation
    self._shape = shape
    self._dtype = dtype
    if name is None:
      name = self.operation.name
    self._name = name
    self._mesh.graph.tensors.append(self)

  @property
  def shape(self):
    return self._shape

  @property
  def size(self):
    return self.shape.size

  @property
  def mesh(self):
    return self._mesh

  @property
  def graph(self):
    return self._mesh.graph

  @property
  def operation(self):
    return self._operation

  @property
  def dtype(self):
    return self._dtype

  @property
  def name(self):
    return self._name

  def __repr__(self):
    return self.to_string

  def __add__(self, other):
    return add(self, other)

  def __radd__(self, other):
    return add(self, other)

  def __sub__(self, other):
    return sub(self, other)

  def __rsub__(self, other):
    return sub(other, self)

  def __mul__(self, other):
    return multiply(self, other)

  def __rmul__(self, other):
    return multiply(self, other)

  def __neg__(self):
    return negative(self)

  def __truediv__(self, other):
    return divide(self, other)

  def __rtruediv__(self, other):
    return divide(other, self)

  def __floordiv__(self, other):
    return floordiv(self, other)

  def __rfloordiv__(self, other):
    return floordiv(other, self)

  def __mod__(self, other):
    return mod(self, other)

  def __rmod__(self, other):
    return mod(other, self)

  @property
  def to_string(self):
    return "Tensor[%s, %s, %s]" % (self.name, self.shape.to_string, self.dtype)


class Operation(object):
  """A Distributed Operation."""

  def __init__(self, inputs, mesh=None, name=None):
    if mesh is None:
      if not inputs:
        raise ValueError("mesh must be specified if no inputs")
      mesh = inputs[0].mesh
    self._inputs = inputs
    self._outputs = []
    self._mesh = mesh
    assert name is not None
    scope_name = tf.get_variable_scope().name
    if scope_name:
      name = scope_name + "/" + name
    self._name = name
    mesh.graph.operations.append(self)

  @property
  def graph(self):
    return self._mesh.graph

  @property
  def mesh(self):
    return self._mesh

  @property
  def name(self):
    return self._name

  @property
  def inputs(self):
    return self._inputs[:]

  @property
  def outputs(self):
    return self._outputs[:]

  @property
  def to_string(self):
    return "%s[Inputs=(%s) Outputs=(%s)]" % (
        type(self).__name__,
        ", ".join([t.to_string for t in self.inputs]),
        ", ".join([t.to_string for t in self.outputs]))

  @property
  def has_gradient(self):
    return (
        [t for t in self.inputs if t.dtype.is_floating] and
        [t for t in self.outputs if t.dtype.is_floating])

  def gradient(self, unused_grad_ys):
    raise NotImplementedError("Gradient not implemented")

  def lower(self, lowering):
    raise NotImplementedError("Lower not implemented")


class SlicewiseOperation(Operation):
  """Apply any tensorflow function slice-wise.

  Calls the Tensorflow function on each slice of the inputs to produce the
  corresponding slice of the outputs.  Gradients are computed through
  tensorflow.

  The user must specify "splittable_dims": a list of Dimensions which can
  be split while still keeping this computation valid.  For example, for
  component-wise functions, all the dimensions are splittable, but if the
  function is a reduction, the reduced dimensions are not splittable.
  """

  def __init__(self,
               tf_fn,
               inputs,
               output_shape,
               output_dtype,
               splittable_dims,
               grad_function=None,
               name=None):
    """Create a SlicewiseOperation.

    grad_function is a python function taking this operation and a gradients
    Tensor and producing input gradients tensors.
    e.g.
    def _square_grad(op, dy):
      return [dy * op.inputs[0] * 2]

    Args:
      tf_fn: a function taking n tf.Tensors and returning a tf.Tensor
      inputs: a list of n Tensors
      output_shape: a Shape
      output_dtype: a dtype
      splittable_dims: a list of Dimensions which are ok to split
      grad_function: an optional python function. Default to using tf.gradients
      name: an optional string
    """
    super(SlicewiseOperation, self).__init__(inputs, name=name or "slicewise")
    self._tf_fn = tf_fn
    self._outputs = [Tensor(self, output_shape, output_dtype)]
    self._splittable_dims = splittable_dims
    self._grad_function = grad_function

  def gradient(self, grad_ys):
    if self._grad_function is not None:
      return self._grad_function(self, grad_ys[0])
    return GenericGradOperation(self, grad_ys).outputs

  def lower(self, lowering):
    # Check that only splittable dims are split
    mesh_impl = lowering.mesh_impl(self)
    for t in self.inputs + self.outputs:
      layout = mesh_impl.tensor_layout(t)
      for d, mesh_axis in zip(t.shape.dims, layout.tensor_axis_to_mesh_axis):
        if mesh_axis is not None and d not in self._splittable_dims:
          raise ValueError("dimension %s is not declared as splittable" % d)
    lowering.set_tensor_lowering(
        self.outputs[0],
        mesh_impl.slicewise(
            self._tf_fn, *[lowering.tensors[x] for x in self.inputs]))


def slicewise(tf_fn,
              xs,
              output_shape=None,
              output_dtype=None,
              splittable_dims=None,
              grad_function=None,
              name=None):
  """Slice-wise call to any tensorflow function.

  The output shape and dtype default to those of the first input.
  splittable_dims is a list of Dimensions which can be split while keeping the
  computation valid.

  Args:
    tf_fn: a function taking n tf.Tensors and returning a tf.Tensor
    xs: a list of n Tensors
    output_shape: a Shape
    output_dtype: a dtype
    splittable_dims: a list of Dimensions which are ok to split
    grad_function: an optional gradients function.  If None, use tf gradient.
    name: an optional string

  Returns:
    a Tensor
  """
  return SlicewiseOperation(
      tf_fn,
      xs,
      convert_to_shape(output_shape) or xs[0].shape,
      output_dtype or xs[0].dtype,
      splittable_dims,
      grad_function,
      name=name).outputs[0]


def cwise(tf_fn, xs, output_dtype=None, grad_function=None, name=None):
  """Component-wise operation with no broadcasting.

  Args:
    tf_fn: a component-wise function taking n tf.Tensor inputs and producing
      a tf.Tensor output
    xs: n Tensors
    output_dtype: an optional dtype
    grad_function: an optional python function
    name: an optional string

  Returns:
    a Tensor
  """
  return slicewise(
      tf_fn, xs, output_dtype=output_dtype, splittable_dims=xs[0].shape.dims,
      grad_function=grad_function, name=name or "cwise")


def square(x, name="square"):
  return cwise(
      tf.square, [x], name=name,
      grad_function=lambda op, dy: [dy * op.inputs[0] * 2])


def sqrt(x, name="sqrt"):
  return cwise(
      tf.sqrt, [x], name=name,
      grad_function=lambda op, dy: [dy * 0.5 / op.outputs[0]])


def _rsqrt_grad(op, dy):
  return [dy * -0.5 * op.outputs[0] * op.outputs[0] * op.outputs[0]]


def rsqrt(x, name="rsqrt"):
  return cwise(
      tf.rsqrt, [x], name=name, grad_function=_rsqrt_grad)


def log(x, name="log"):
  return cwise(
      tf.log, [x], name=name,
      grad_function=lambda op, dy: [dy / op.inputs[0]])


def exp(x, name="exp"):
  return cwise(tf.exp, [x], name=name,
               grad_function=lambda op, dy: [dy * op.outputs[0]])


def pow(x, y):  # pylint: disable=redefined-builtin
  return exp(log(x) * y)


def negative(x, name="negative"):
  return cwise(tf.negative, [x], name=name,
               grad_function=lambda op, dy: [negative(dy)])


def logical_not(x, name="logical_not"):
  return cwise(tf.logical_not, [x], name=name)


def reciprocal(x, name="reciprocal"):
  return cwise(
      tf.reciprocal, [x], name=name,
      grad_function=lambda op, dy: [negative(dy * square(op.outputs[0]))])


def _relu_grad(op, dy):
  return [dy * cast(greater(op.inputs[0], 0), op.inputs[0].dtype)]


def relu(x, name="relu"):
  return cwise(tf.nn.relu, [x], name=name, grad_function=_relu_grad)


def cast(x, dtype, name="cast"):
  if dtype == x.dtype:
    return x
  return cwise(
      lambda x: tf.cast(x, dtype), [x], output_dtype=dtype, name=name,
      grad_function=lambda op, dy: [cast(dy, op.inputs[0].dtype)])


def to_float(x, name="to_float"):
  return cast(x, tf.float32, name=name)


def to_int32(x, name="to_int32"):
  return cast(x, tf.int32, name=name)


class GenericGradOperation(Operation):
  """Gradients that follow regular TF.

  Calling tf.gradients multiple times seems really slow in python.
  TODO(noam): can we speed this up using functions or some other method?
  """

  def __init__(self, forward_op, grad_ys, name=None):
    # tf.logging.info("forward inp %s, operations %s, grad_ys: %s",
    #                 forward_op.inputs, forward_op.outputs, grad_ys)
    super(GenericGradOperation, self).__init__(
        forward_op.inputs + forward_op.outputs + grad_ys,
        name=name or "generic_grad")
    self._grad_ys = grad_ys
    self._forward_op = forward_op
    self._outputs = [Tensor(self, x.shape, x.dtype) for x in forward_op.inputs]

  def lower(self, lowering):
    # lists of lists of tf.Tensor
    all_ys = transpose_list_of_lists(
        [lowering.tensors[y].tensor_list for y in self._forward_op.outputs])
    all_xs = transpose_list_of_lists(
        [lowering.tensors[x].tensor_list for x in self._forward_op.inputs])
    all_grad_ys = transpose_list_of_lists(
        [lowering.tensors[dy].tensor_list for dy in self._grad_ys])
    all_grad_xs = [tf.gradients(ys=ys, xs=xs, grad_ys=grad_ys) for
                   ys, xs, grad_ys in zip(all_ys, all_xs, all_grad_ys)]
    grad_xs = transpose_list_of_lists(all_grad_xs)
    for out, grad_x in zip(self.outputs, grad_xs):
      lowering.set_tensor_lowering(
          out,
          lowering.mesh_impl(self).LaidOutTensor.from_tensor_list(grad_x))


class ScalarMultiplyOperation(Operation):
  """Multiply by a tf Scalar (no backprop to scalar)."""

  def __init__(self, x, scalar, name=None):
    super(ScalarMultiplyOperation, self).__init__(
        [x], name=name or "scalar_mul")
    self._outputs = [Tensor(self, x.shape, x.dtype)]
    self._scalar = scalar

  def gradient(self, grad_ys):
    dy = grad_ys[0]
    return [dy * self._scalar]

  def lower(self, lowering):
    lowering.set_tensor_lowering(
        self.outputs[0],
        lowering.mesh_impl(self).slicewise(
            lambda x: x * self._scalar, lowering.tensors[self.inputs[0]]))


class ScalarAddOperation(Operation):
  """Add a tf Scalar (no backprop to scalar)."""

  def __init__(self, x, scalar, name=None):
    super(ScalarAddOperation, self).__init__([x], name=name or "scalar_add")
    self._outputs = [Tensor(self, x.shape, x.dtype)]
    self._scalar = scalar

  def gradient(self, grad_ys):
    return grad_ys

  def lower(self, lowering):
    lowering.set_tensor_lowering(
        self.outputs[0],
        lowering.mesh_impl(self).slicewise(
            lambda x: x + self._scalar, lowering.tensors[self.inputs[0]]))


class BinaryOpWithBroadcasting(Operation):
  """Binary operation with broadcasting."""

  def __init__(self, tf_fn, x1, x2, output_shape, output_dtype, name=None):
    super(BinaryOpWithBroadcasting, self).__init__(
        [x1, x2], name=name or "binary_op")
    assert isinstance(output_dtype, tf.DType)
    self._outputs = [Tensor(self, output_shape, output_dtype)]
    self._tf_fn = tf_fn

  def gradient(self, unused_grad_ys):
    raise ValueError("Gradient not implememnted")

  def lower(self, lowering):
    x1 = self.inputs[0]
    x2 = self.inputs[1]
    output = self.outputs[0]
    laid_out_x1 = lowering.tensors[x1]
    laid_out_x2 = lowering.tensors[x2]
    mesh_impl = lowering.mesh_impl(self)
    if x1.shape != output.shape:
      laid_out_x1 = mesh_impl.slicewise(
          _expand_dims, laid_out_x1, x1.shape, output.shape)
    if x2.shape != output.shape:
      laid_out_x2 = mesh_impl.slicewise(
          _expand_dims, laid_out_x2, x2.shape, output.shape)
    lowering.set_tensor_lowering(
        self.outputs[0],
        mesh_impl.slicewise(
            self._tf_fn, laid_out_x1, laid_out_x2))


def binary_arguments_to_tensors(x1, x2):
  """Convert argument of a binary operation to Tensors.

  Args:
    x1: a Tensor or something convertible to a tf Scalar
    x2: a Tensor or something convertible to a tf Scalar

  Returns:
    new_x1: a Tensor
    new_x2: a Tensor

  Raises:
    ValueError: on failure
  """
  if not isinstance(x1, Tensor) and not isinstance(x2, Tensor):
    raise ValueError("at least one of x1 and x2 must be an mtf Tensor")
  elif isinstance(x1, Tensor) and isinstance(x2, Tensor):
    return x1, x2
  elif isinstance(x1, Tensor):
    return x1, import_tf_tensor(
        x1.mesh, tf.convert_to_tensor(x2, dtype=x1.dtype), Shape([]))
  else:
    return import_tf_tensor(x2.mesh, tf.convert_to_tensor(x1, dtype=x2.dtype),
                            Shape([])), x2


def binary_op_with_broadcasting(
    tf_fn, x1, x2, output_shape=None, output_dtype=None):
  x1, x2 = binary_arguments_to_tensors(x1, x2)
  output_shape = _infer_binary_broadcast_shape(x1.shape, x2.shape, output_shape)
  output_dtype = output_dtype or x1.dtype
  assert isinstance(output_dtype, tf.DType)
  return BinaryOpWithBroadcasting(
      tf_fn, x1, x2, convert_to_shape(output_shape),
      output_dtype).outputs[0]


def maximum(x1, x2, output_shape=None):
  return binary_op_with_broadcasting(
      tf.maximum, x1, x2, output_shape=output_shape)


def minimum(x1, x2, output_shape=None):
  return binary_op_with_broadcasting(
      tf.minimum, x1, x2, output_shape=output_shape)


def less(x1, x2, output_shape=None):
  return binary_op_with_broadcasting(
      tf.less, x1, x2, output_dtype=tf.bool, output_shape=output_shape)


def greater(x1, x2, output_shape=None):
  return binary_op_with_broadcasting(
      tf.greater, x1, x2, output_dtype=tf.bool, output_shape=output_shape)


def less_equal(x1, x2, output_shape=None):
  return binary_op_with_broadcasting(
      tf.less_equal, x1, x2, output_dtype=tf.bool, output_shape=output_shape)


def greater_equal(x1, x2, output_shape=None):
  return binary_op_with_broadcasting(
      tf.greater_equal, x1, x2, output_dtype=tf.bool, output_shape=output_shape)


def equal(x1, x2, output_shape=None):
  return binary_op_with_broadcasting(
      tf.equal, x1, x2, output_dtype=tf.bool, output_shape=output_shape)


def not_equal(x1, x2, output_shape=None):
  return binary_op_with_broadcasting(
      tf.not_equal, x1, x2, output_dtype=tf.bool, output_shape=output_shape)


def logical_and(x1, x2, output_shape=None):
  return binary_op_with_broadcasting(
      tf.logical_and, x1, x2, output_dtype=tf.bool, output_shape=output_shape)


def logical_or(x1, x2, output_shape=None):
  return binary_op_with_broadcasting(
      tf.logical_or, x1, x2, output_dtype=tf.bool, output_shape=output_shape)


def floordiv(x1, x2, output_shape=None):
  output_dtype = x1.dtype if isinstance(x1, Tensor) else x2.dtype
  return binary_op_with_broadcasting(
      tf.floordiv, x1, x2, output_dtype=output_dtype, output_shape=output_shape)


def mod(x1, x2, output_shape=None):
  output_dtype = x1.dtype if isinstance(x1, Tensor) else x2.dtype
  return binary_op_with_broadcasting(
      tf.mod, x1, x2, output_dtype=output_dtype, output_shape=output_shape)


class AddOperation(BinaryOpWithBroadcasting):
  """Binary addition with broadcasting."""

  def __init__(self, x1, x2, output_shape, name=None):
    super(AddOperation, self).__init__(
        tf.add, x1, x2, output_shape, x1.dtype, name=name or "add")
    if x1.dtype != x2.dtype:
      raise ValueError("Dtypes must be equal.")

  def gradient(self, grad_ys):
    dy = grad_ys[0]
    return [reduce_sum(dy, output_shape=self.inputs[0].shape),
            reduce_sum(dy, output_shape=self.inputs[1].shape)]


class BroadcastOperation(Operation):
  """Broadcast - output dims are a superset of input dims, in any order."""

  def __init__(self, x, output_shape, name=None):
    super(BroadcastOperation, self).__init__([x], name=name or "broadcast")
    self._outputs = [Tensor(self, output_shape, x.dtype)]

  def gradient(self, grad_ys):
    return [reduce_sum(grad_ys[0], output_shape=self.inputs[0].shape)]

  def lower(self, lowering):
    ret = lowering.mesh_impl(self).broadcast_impl(
        lowering.tensors[self.inputs[0]], self.inputs[0].shape,
        self.outputs[0].shape)
    lowering.set_tensor_lowering(self.outputs[0], ret)


def broadcast(x, new_shape):
  return BroadcastOperation(x, new_shape).outputs[0]


def _reduce_helper(input_shape,
                   output_shape,
                   input_tensor_layout,
                   reduction_fn_string="SUM"):
  """Returns slicewise function and reduced mesh dimensions.

  Args:
    input_shape: a Shape
    output_shape: a Shape
    input_tensor_layout: a TensorLayout
    reduction_fn_string: "SUM" or "MAX"
  Returns:
    reduce_slice_fn: a function from tf.Tensor to tf.Tensor
    reduced_mesh_axes: a list of integers
  """
  reduce_dims_indices = [
      i for i, d in enumerate(input_shape.dims) if d not in output_shape.dims]
  reduced_input_shape = Shape([
      d for d in input_shape.dims if d in output_shape.dims])
  perm = [reduced_input_shape.dims.index(d) for d in output_shape.dims]
  def reduce_slice_fn(xslice):
    ret = xslice
    if reduce_dims_indices:
      ret = reduction_fn(reduction_fn_string)(xslice, reduce_dims_indices)
    if perm != list(xrange(len(perm))):
      ret = tf.transpose(ret, perm)
    return ret
  reduced_mesh_axes = []
  for i in reduce_dims_indices:
    mesh_axis = input_tensor_layout[i]
    if mesh_axis is not None:
      reduced_mesh_axes.append(mesh_axis)
  return reduce_slice_fn, reduced_mesh_axes


class ReduceOperation(Operation):
  """Reduction - output dims are a subset of input dims, in any order."""

  def __init__(self, x, output_shape, reduction_fn_string, name=None):
    super(ReduceOperation, self).__init__([x], name=name or "reduce")
    self._outputs = [Tensor(self, output_shape, x.dtype)]
    self._reduction_fn_string = reduction_fn_string

  def gradient(self, grad_ys):
    if self._reduction_fn_string == "SUM":
      return [broadcast(grad_ys[0], self.inputs[0].shape)]
    elif (self._reduction_fn_string == "MAX" or
          self._reduction_fn_string == "MIN"):
      return [cast(equal(self.inputs[0], self.outputs[0]), self.inputs[0].dtype)
              * grad_ys[0]]
    else:
      raise ValueError("Gradients to other reductions not implemented")

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    slicewise_fn, reduced_mesh_axes = _reduce_helper(
        self.inputs[0].shape, self.outputs[0].shape,
        mesh_impl.tensor_layout(self.inputs[0]),
        self._reduction_fn_string)
    y = mesh_impl.slicewise(slicewise_fn, lowering.tensors[self.inputs[0]])
    if reduced_mesh_axes:
      def add_counter_fn():
        lowering.add_counter("allreduce/%s/reduce_op" % reduced_mesh_axes,
                             lowering.laid_out_size(self.outputs[0]))
      if self._reduction_fn_string == "SUM":
        y = LazyAllreduceSum(
            mesh_impl, y, reduced_mesh_axes, add_counter_fn=add_counter_fn)
      else:
        y = mesh_impl.allreduce(
            y, reduced_mesh_axes, self._reduction_fn_string)
        add_counter_fn()
    lowering.set_tensor_lowering(self.outputs[0], y)


class ConcatOperation(Operation):
  """tf.concat.

  All inputs have the same shape, except for the size of the dimension named
  dim_name.
  """

  def __init__(self, xs, concat_dim_name, name=None):
    super(ConcatOperation, self).__init__(xs, name=name or "concat")
    # verify that the shapes are all compatible
    dim_names = [dim.name for dim in xs[0].shape.dims]
    self._concat_dim_name = concat_dim_name

    if concat_dim_name not in dim_names:
      raise ValueError("xs[0] does not contain a dimension named dim_name")
    self._axis = dim_names.index(concat_dim_name)

    should_be_equal = [
        x.shape.resize_dimension(concat_dim_name, 0) for x in xs]
    if not all(s == should_be_equal[0] for s in should_be_equal):
      raise ValueError("shapes are not compatible %s" % xs)

    self._input_sizes = [x.shape.dims[self._axis].size for x in xs]
    output_size = sum(self._input_sizes)
    self._outputs = [
        Tensor(self, xs[0].shape.resize_dimension(concat_dim_name, output_size),
               xs[0].dtype)]

  def gradient(self, grad_ys):
    dy = grad_ys[0]
    return split(dy, self.outputs[0].shape.dims[self._axis], self._input_sizes)

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    if mesh_impl.tensor_dimension_to_mesh_axis(
        Dimension(self._concat_dim_name, 0)) is not None:
      raise ValueError("can't concat along split axis")
    def slicewise_fn(*args):
      return tf.concat(args, axis=self._axis, name="concat")
    y = mesh_impl.slicewise(
        slicewise_fn, *[lowering.tensors[x] for x in self._inputs])
    lowering.set_tensor_lowering(self.outputs[0], y)


def concat(xs, concat_dim_name, name=None):
  """Like tf.concat.

  All inputs must have equal shape except for the sizes in the concatenated
  dimension.  The dimension names should be the same, even that of the
  concatenated dimension.

  Args:
    xs: a list of Tensors
    concat_dim_name: a string
    name: an optional string
  Returns:
    a Tensor
  """
  return ConcatOperation(xs, concat_dim_name, name).outputs[0]


class SplitOperation(Operation):
  """like tf.split.

  TODO(noam, nikip): this code has never been run.  Run it and test it.
  """

  def __init__(self, x, split_dim, num_or_size_splits, name=None):
    super(SplitOperation, self).__init__([x], name=name or "concat")

    self._split_dim = split_dim
    if split_dim not in x.shape.dims:
      raise ValueError("%s does not contain dimension %s" % (x, split_dim))
    self._axis = x.shape.dims.index(split_dim)

    if isinstance(num_or_size_splits, list):
      self._output_sizes = num_or_size_splits
      if sum(num_or_size_splits) != split_dim.size:
        raise ValueError(
            "Sizes do not add up %s %s" % (num_or_size_splits, split_dim))
    else:
      assert isinstance(num_or_size_splits, int)
      assert split_dim.size % num_or_size_splits == 0
      self._output_sizes = (
          [split_dim.size / num_or_size_splits] * num_or_size_splits)

    self._outputs = [
        Tensor(self, x.shape.resize_dimension(split_dim.name, output_size),
               x.dtype) for output_size in self._output_sizes]

  def gradient(self, grad_ys):
    return concat(grad_ys, self._split_dim.name)

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    if mesh_impl.tensor_dimension_to_mesh_axis(self._split_dim) is not None:
      raise ValueError("can't split along split axis")
    def slicewise_fn(x):
      # Since we return a tuple of tf.Tensor, slicewise will collate the
      # outputs and return a tuple of LaidOutTensors.
      return tuple(tf.split(x, self._output_sizes, axis=self._axis))
    values = mesh_impl.slicewise(
        slicewise_fn, lowering.tensors[self.inputs[0]])
    for t, v in zip(self._outputs, values):
      lowering.set_tensor_lowering(t, v)


def split(x, split_dim, num_or_size_splits, name=None):
  """Like tf.split.

  Args:
    x: a Tensor
    split_dim: a Dimension in x.shape.dims
    num_or_size_splits: either an integer dividing split_dim.size
       or a list of integers adding up to split_dim.size
    name: an optional string
  Returns:
    a list of Tensors.
  """
  return SplitOperation(x, split_dim, num_or_size_splits, name=name).outputs


class StackOperation(Operation):
  """Like tf.stack."""

  def __init__(self, xs, dim_name, axis, name=None):
    super(StackOperation, self).__init__(xs, name=name or "stack")
    self._axis = axis
    self._new_dim = Dimension(dim_name, len(xs))
    input_shape = xs[0].shape
    for x in xs:
      if x.shape != xs[0].shape:
        raise ValueError(
            "inputs to stack must have the same shape, got %s" % xs)
    output_shape = Shape(
        input_shape.dims[:axis] + [self._new_dim]+ input_shape.dims[axis:])
    self._outputs = [Tensor(self, output_shape, xs[0].dtype)]

  def gradient(self, grad_ys):
    return unstack(grad_ys[0], self._new_dim)

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    if mesh_impl.tensor_dimension_to_mesh_axis(self._new_dim) is not None:
      raise ValueError("can't stack along split axis")
    inputs = [lowering.tensors[t] for t in self._inputs]
    def slicewise_fn(*args):
      return tf.stack(args, axis=self._axis)
    ret = mesh_impl.slicewise(slicewise_fn, *inputs)
    lowering.set_tensor_lowering(self.outputs[0], ret)


def stack(xs, dim_name, axis, name=None):
  """Stack multiple Tensors to make a new dimension.

  Args:
    xs: a list of Tensors with identical shapes.
    dim_name: a string (name of the new dimension)
    axis: an integer (index of the new dimension in the output shape)
    name: an optional string

  Returns:
    a Tensor
  """
  ret = StackOperation(xs, dim_name, axis, name).outputs[0]
  return ret


class UnstackOperation(Operation):
  """Split into multiple Tensors, eliminating a dimension."""

  def __init__(self, x, dim, name=None):
    super(UnstackOperation, self).__init__([x], name=name or "unstack")
    self._dim = dim
    self._axis = x.shape.dims.index(dim)
    output_shape = x.shape - dim
    self._outputs = [
        Tensor(self, output_shape, x.dtype) for _ in xrange(dim.size)]

  def gradient(self, grad_ys):
    return [stack(grad_ys, self._dim.name, self._axis)]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    if mesh_impl.tensor_dimension_to_mesh_axis(self._dim) is not None:
      raise ValueError("can't unstack along split axis")
    def slicewise_fn(x):
      return tuple(tf.unstack(x, num=self._dim.size, axis=self._axis))
    output_values = mesh_impl.slicewise(
        slicewise_fn, lowering.tensors[self._inputs[0]])
    for t, v in zip(self.outputs, list(output_values)):
      lowering.set_tensor_lowering(t, v)


def unstack(x, dim, name=None):
  """Split into multiple Tensors, eliminating a dimension.

  Args:
    x: a Tensor
    dim: a Dimension
    name: an optional string

  Returns:
    a list of dim.size Tensors, each with shape (x.shape - dim)
  """
  return UnstackOperation(x, dim, name).outputs


def cumsum(x, dim, exclusive=False):
  """Cumulative sum.

  Args:
    x: a Tensor
    dim: a Dimension
    exclusive: a boolean

  Returns:
    a Tensor with the same shape as x.
  """
  new_name = "tmp_dim_cumsum"
  new_dim = Dimension(new_name, dim.size)
  new_shape = x.shape.rename_dimension(dim.name, new_name)
  comparator = less if exclusive else less_equal
  m = cast(
      comparator(range(x.mesh, dim, dtype=tf.float32),
                 range(x.mesh, new_dim, dtype=tf.float32)), x.dtype)
  ret = einsum([x, m], output_shape=new_shape)
  return reshape(ret, x.shape)


def _einsum_helper(input_shapes, output_shape, mesh_impl):
  """Returns slicewise function and reduced mesh dimensions.

  Assumes the output shape contains no new dimensions.

  Args:
    input_shapes: a list of Shapes
    output_shape: a Shape
    mesh_impl: a MeshImpl
  Returns:
    einsum_slice_fn: a function from tf.Tensors to tf.Tensor
    reduced_mesh_axes: a list of integers
  """
  input_shape_set = set(sum([s.dims for s in input_shapes], []))
  total_num_dims = len(input_shape_set)
  # list of input shapes that contain all dimensions.
  full_shapes = [
      s for s in input_shapes + [output_shape] if s.ndims == total_num_dims]
  full_shape = (
      full_shapes[0] if full_shapes else Shape(list(input_shape_set)))
  reduce_slice_fn, reduced_mesh_axes = _reduce_helper(
      full_shape, output_shape, mesh_impl.tensor_layout(full_shape))
  def einsum_slice_fn_naive(*slices):
    # naive einsum implementation where we broadcst all inputs to the full
    # shape, multiply componentwise, then reduce.
    return reduce_slice_fn(reduce(tf.multiply, [
        _expand_dims(x, input_shape, full_shape)
        for x, input_shape in zip(slices, input_shapes)]))
  if full_shapes:
    # it is not wasteful of space to broadcast fully and then reduce.
    # this helps to avoid some inefficient GPU implementations.
    einsum_slice_fn = einsum_slice_fn_naive
  else:
    # call tf.einsum
    equation = _einsum_equation(input_shapes, output_shape)
    def einsum_slice_fn(*slices):
      if slices[0].dtype.is_floating:
        return tf.einsum(equation, *slices)
      else:
        return einsum_slice_fn_naive(*slices)
  return einsum_slice_fn, reduced_mesh_axes


class EinsumOperation(Operation):
  """Einstein summation (matmul, etc).

  The equation follows the dimensions in the input and output shapes.

  Every dimension must occur in at least two of the input/output Tensors.
  i.e. no new dimensions in the output, and no reduction of dimensions that
  occur in only one input.
  """

  def __init__(self, inputs, output_shape, name=None):
    super(EinsumOperation, self).__init__(inputs, name=name or "einsum")
    if not inputs:
      raise ValueError("Einsum needs at least one input")
    for x in inputs:
      if x.dtype != inputs[0].dtype:
        raise ValueError("Input dtypes must be equal")
    self._outputs = [Tensor(self, output_shape, inputs[0].dtype)]

  def gradient(self, grad_ys):
    dy = grad_ys[0]
    xs = self.inputs
    return [
        einsum([dy] + [xs[j] for j in xrange(len(xs)) if j != i], xs[i].shape)
        for i in xrange(len(self.inputs))]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    xs = self.inputs
    input_shape_set = set(sum([x.shape.dims for x in xs], []))
    output_shape = self.outputs[0].shape
    intersection_shape = Shape(
        [d for d in output_shape.dims if d in input_shape_set])
    einsum_slice_fn, reduced_mesh_axes = _einsum_helper(
        [x.shape for x in self.inputs], intersection_shape, mesh_impl)
    y = mesh_impl.slicewise(
        einsum_slice_fn, *[lowering.tensors[x] for x in self.inputs])
    if reduced_mesh_axes:
      def add_counter_fn():
        lowering.add_counter(
            "allreduce/%s/einsum_op" % reduced_mesh_axes,
            mesh_impl.laid_out_size(intersection_shape))
      y = LazyAllreduceSum(
          mesh_impl, y, reduced_mesh_axes, add_counter_fn=add_counter_fn)
    # broadcast from intersection_shape to output_shape
    if intersection_shape != output_shape:
      y = mesh_impl.broadcast_impl(y, intersection_shape, output_shape)
    lowering.set_tensor_lowering(self.outputs[0], y)
    computation_shape = Shape(list(input_shape_set))
    lowering.add_counter("einsum", mesh_impl.laid_out_size(computation_shape))
    lowering.add_counter("einsum_unique", computation_shape.size)


class SliceOperation(Operation):
  """tf.slice.

  We support the slice operation along one axis. Similar to tf.slice, specify
  the begin and size values for the slice_dim.
  """

  def __init__(self, x, begin, size, slice_dim_name, name=None):
    super(SliceOperation, self).__init__([x], name=name or "slice")
    dim_names = x.shape.dimension_names
    self._axis = axis = dim_names.index(slice_dim_name)
    self._begin = begin
    self._slice_dim = Dimension(slice_dim_name, size)
    input_shape = self._inputs[0].shape
    output_shape = Shape(
        input_shape.dims[:axis] + [self._slice_dim] + input_shape.dims[axis+1:])
    self._outputs = [Tensor(self, output_shape, x.dtype)]

  def gradient(self, grad_ys):
    actual_size = self._inputs[0].shape.dims[self._axis].size
    return [
        pad(grad_ys[0],
            [self._begin, actual_size - self._slice_dim.size - self._begin],
            self._slice_dim.name)]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    if mesh_impl.tensor_dimension_to_mesh_axis(self._slice_dim) is not None:
      raise ValueError("can't slice along split axis")
    inputs = self._inputs[0]
    ndims = self._inputs[0].shape.ndims
    axis = self._axis
    begin = [0] * axis + [self._begin] + [0] * (ndims - axis - 1)
    size = [-1] * axis + [self._slice_dim[1]] + [-1] * (ndims - axis - 1)

    def slicewise_fn(x, begin, size):
      return tf.slice(x, begin, size, name="slice")
    y = mesh_impl.slicewise(
        slicewise_fn, lowering.tensors[inputs], begin, size)
    lowering.set_tensor_lowering(self.outputs[0], y)


class PadOperation(Operation):
  """tf.pad.

  Similar to tf.pad but we only pad along one axis given by pad_dim_name
  with values specified by paddings. paddings is a list of two
  values, giving the padding value before and after pad_dim.
  """

  def __init__(self, x, paddings, pad_dim_name, name=None):
    super(PadOperation, self).__init__([x], name=name or "pad")
    assert len(paddings) == 2
    input_shape = self._inputs[0].shape
    dim_names = [dim.name for dim in x.shape.dims]
    if pad_dim_name not in dim_names:
      raise ValueError("Padding dim name %s not found in input." % pad_dim_name)
    self._paddings = paddings
    self._axis = axis = dim_names.index(pad_dim_name)
    output_size = input_shape.dims[axis].size + sum(paddings)
    self._output_dim = Dimension(pad_dim_name, output_size)
    output_shape = Shape(
        input_shape.dims[:axis] +
        [self._output_dim] + input_shape.dims[axis+1:])
    self._outputs = [Tensor(self, output_shape, x.dtype)]

  def gradient(self, grad_ys):
    # slice_dim = self._inputs[0].shape.dims[self._axis]
    slice_dim_name = self._output_dim.name
    slice_size = self._inputs[0].shape.dims[self._axis].size
    return [slice(grad_ys[0], self._paddings[0], slice_size, slice_dim_name)]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    if mesh_impl.tensor_dimension_to_mesh_axis(self._output_dim) is not None:
      raise ValueError("can't pad along split axis")
    inputs = self._inputs[0]
    ndims = self._inputs[0].shape.ndims
    axis = self._axis
    paddings = [[0, 0]] * axis + [self._paddings] + [[0, 0]]* (ndims - axis - 1)

    def slicewise_fn(x, paddings):
      return tf.pad(x, paddings, name="pad")
    y = mesh_impl.slicewise(
        slicewise_fn, lowering.tensors[inputs], paddings)
    lowering.set_tensor_lowering(self.outputs[0], y)


class OneHotOperation(Operation):
  """one_hot.
  """

  def __init__(self, indices, output_dim, on_value, off_value, dtype,
               name=None):
    super(OneHotOperation, self).__init__([indices], name=name or "one_hot")
    if not indices.dtype.is_integer:
      raise ValueError("indices requires an integer dtype got %s" % indices)
    self._output_dim = output_dim
    self._on_value = on_value
    self._off_value = off_value
    self._dtype = dtype
    output_shape = Shape(indices.shape.dims + [output_dim])
    self._outputs = [Tensor(self, output_shape, dtype)]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    indices = self.inputs[0]
    output_shape = self.outputs[0].shape
    output_slice_shape = mesh_impl.slice_shape(output_shape)
    mesh_axis = mesh_impl.tensor_dimension_to_mesh_axis(self._output_dim)
    depth = output_slice_shape[-1]
    if mesh_axis is None:
      offset = 0
    else:
      offset = mesh_impl.slicewise(
          tf.multiply, mesh_impl.laid_out_pcoord(mesh_axis), depth)

    def slicewise_fn(indices_slice, offset):
      return tf.one_hot(indices_slice - offset,
                        depth,
                        on_value=tf.cast(self._on_value, self._dtype),
                        off_value=tf.cast(self._off_value, self._dtype),
                        dtype=self._dtype)
    y = mesh_impl.slicewise(
        slicewise_fn, lowering.tensors[indices], offset)
    lowering.set_tensor_lowering(self.outputs[0], y)


class ImportOperation(Operation):
  """Import a tf.Tensor onto a mesh."""

  def __init__(self, mesh, tf_tensor, shape, name=None):
    super(ImportOperation, self).__init__([], mesh=mesh, name=name or "import")
    self._outputs = [Tensor(self, shape, tf_tensor.dtype)]
    self._tf_tensor = tf_tensor

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    lowering.set_tensor_lowering(
        self.outputs[0],
        mesh_impl.import_tf_tensor(self.outputs[0], self._tf_tensor))


def anonymous_shape(shape):
  shape = convert_to_shape(shape)
  return Shape([Dimension("_anonymous_%i" % i, d.size)
                for i, d in enumerate(shape)])


def anonymize(x):
  return reshape(x, anonymous_shape(x.shape))


def import_tf_tensor(mesh, tf_tensor, shape=None, name=None):
  tf_tensor = tf.convert_to_tensor(tf_tensor)
  if shape is None:
    shape = Shape([])
    assert not tf_tensor.shape.as_list()
  return ImportOperation(
      mesh, tf_tensor, convert_to_shape(shape), name=name).outputs[0]


def import_fully_replicated(mesh, tf_tensor, shape, name=None):
  return reshape(import_tf_tensor(
      mesh, tf_tensor, anonymous_shape(shape), name), shape)


class Variable(Operation):
  """Variable."""

  def __init__(self, mesh, name, shape, dtype, initializer,
               trainable, **kwargs):
    super(Variable, self).__init__([], mesh, name="name_will_be_set_later")
    self._trainable = trainable
    with tf.device("cpu:0"), mtf_utils.outside_all_rewrites():
      self.master = tf.get_variable(
          name, shape.to_integer_list, dtype=dtype, initializer=initializer,
          **kwargs)
    self._name = self.master.name[:self.master.name.find(":")]
    self._outputs = [Tensor(self, shape, dtype)]
    self.graph.all_variables.append(self)
    if trainable:
      self.graph.trainable_variables.append(self)

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    with mtf_utils.outside_all_rewrites():
      sv = mesh_impl.LaidOutVariable(self, mesh_impl)
    lowering.variables[self] = sv
    lowering.set_tensor_lowering(self.outputs[0], sv.laid_out_tensor)
    if self._trainable:
      lowering.add_counter("variables/trainable", self.outputs[0].size)
    else:
      lowering.add_counter("variables/untrainable", self.outputs[0].size)

  @property
  def value(self):
    return self.outputs[0]

  @property
  def shape(self):
    return self.value.shape

  @property
  def dtype(self):
    return self.value.dtype


def get_variable(mesh, name, shape, dtype=tf.float32,
                 initializer=None, trainable=True,
                 activation_dtype=None, **kwargs):
  ret = Variable(
      mesh, name, convert_to_shape(shape), dtype, initializer,
      trainable, **kwargs).outputs[0]
  if activation_dtype and activation_dtype != dtype:
    ret = cast(ret, activation_dtype)
  return ret


class Assign(Operation):
  """Assign to a variable."""

  def __init__(self, var, new_val, name=None):
    super(Assign, self).__init__([new_val], var.mesh, name=name or "assign")
    self._var = var
    self._outputs = []

  def lower(self, lowering):
    lowering.operations[self] = lowering.variables[self._var].assign_to_slices(
        lowering.tensors[self.inputs[0]].to_laid_out_tensor().all_slices)


def assign(var, new_val):
  """Assign a new value to a variable.

  Args:
    var: either a Variable operation or its output Tensor.
    new_val: a Tensor
  Returns:
    an Operation
  Raises:
    ValueError: if var is not a Variable and var.operation is not a Variable
  """
  if isinstance(var, Tensor):
    var = var.operation
  if not isinstance(var, Variable):
    raise ValueError("var must be a mtf.Variable or its output Tensor.")
  return Assign(var, new_val)


class Depend(Operation):
  """Control dependency."""

  def __init__(self, x, dependencies, name=None):
    super(Depend, self).__init__([x], x.mesh, name=name or "depend")
    for d in dependencies:
      if not isinstance(d, Operation):
        raise ValueError("dependencies must be mtf.Operations. got %s" % d)
    self._dependencies = dependencies
    self._outputs = [Tensor(self, x.shape, x.dtype)]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    if not mesh_impl.supports_control_dependencies:
      raise ValueError("Mesh does not suppport control dependencies.")
    with tf.control_dependencies(
        [lowering.operations[d] for d in self._dependencies]):
      lowering.set_tensor_lowering(
          self.outputs[0],
          mesh_impl.slicewise(tf.identity,
                              lowering.tensors[self.inputs[0]]))

  def gradient(self, grad_ys):
    return grad_ys


def depend(x, dependencies):
  """Identity of Tensor x that dependes on operations dependencies.

  Args:
    x: a Tensor
    dependencies: a list of Operations
  Returns:
    an tensor
  """
  return Depend(x, dependencies).outputs[0]


class Constant(Operation):
  """A tensor where every element is the same constant value."""

  def __init__(self, mesh, value, shape, dtype, name=None):
    super(Constant, self).__init__([], mesh, name=name or "constant")
    self._outputs = [Tensor(self, shape, dtype)]
    self._value = value

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    slice_shape = mesh_impl.slice_shape(self.outputs[0].shape)
    def tf_fn():
      return tf.constant(value=self._value,
                         dtype=self.outputs[0].dtype,
                         shape=slice_shape)
    lowering.set_tensor_lowering(self.outputs[0], mesh_impl.slicewise(tf_fn))


def constant(mesh, value, shape=None, dtype=tf.float32):
  shape = convert_to_shape(shape)
  return Constant(mesh, value,
                  shape if shape is not None else Shape([]),
                  dtype).outputs[0]


def zeros(mesh, shape, dtype=tf.float32):
  return constant(mesh, 0, shape=convert_to_shape(shape), dtype=dtype)


def zeros_like(t):
  return zeros(t.mesh, t.shape, dtype=t.dtype)


class StopGradient(Operation):
  """Similar to tf.stop_gradient."""

  def __init__(self, x, name=None):
    super(StopGradient, self).__init__(
        [x], x.mesh, name=name or "stop_gradient")
    self._outputs = [Tensor(self, x.shape, x.dtype)]

  def lower(self, lowering):
    lowering.set_tensor_lowering(self.outputs[0],
                                 lowering.tensors[self.inputs[0]])

  @property
  def has_gradient(self):
    return False


def stop_gradient(x):
  return StopGradient(x).outputs[0]


class PrintOperation(Operation):
  """Similar to tf.stop_gradient."""

  def __init__(self, x, data, message, name=None, **kwargs):
    super(PrintOperation, self).__init__(
        [x], x.mesh, name=name or "Print")
    self._outputs = [Tensor(self, x.shape, x.dtype)]
    self._data = data
    self._message = message
    self._kwargs = kwargs

  def lower(self, lowering):
    lowering.set_tensor_lowering(
        self.outputs[0],
        lowering.mesh_impl(self).Print(
            lowering.tensors[self.inputs[0]],
            [lowering.tensors[d].to_laid_out_tensor() for d in self._data],
            self._message, **self._kwargs))

  def gradient(self, grad_ys):
    return grad_ys


def Print(x, data, message, **kwargs):  # pylint: disable=invalid-name
  """Call tf.Print.

  Args:
    x: a Tensor.
    data: a list of Tensor
    message: a string
    **kwargs: keyword arguments to tf.Print
  Returns:
    a Tensor which is identical in value to x
  """
  return PrintOperation(x, data, message, **kwargs).outputs[0]


class ReshapeOperation(Operation):
  """Similar to tf.stop_gradient."""

  def __init__(self, x, new_shape, name=None):
    super(ReshapeOperation, self).__init__([x], x.mesh, name=name or "reshape")
    self._outputs = [Tensor(self, new_shape, x.dtype)]

  def lower(self, lowering):
    """Lower the ReshapeOperation.

    Reshaping can require collective communication between processors.
    We haven't yet implemented all possible reshapes.  We try to handle the
    common cases here - otherwise we raise a NotImplementedError.

    Args:
      lowering: a Lowering
    Raises:
      NotImplementedError: if we haven't covered this case
    """
    old_shape = self.inputs[0].shape
    new_shape = self.outputs[0].shape
    mesh_impl = lowering.mesh_impl(self)
    slices = lowering.tensors[self.inputs[0]]

    mesh_axis_to_cumprod_old = mesh_impl.mesh_axis_to_cumprod(old_shape)
    mesh_axis_to_cumprod_new = mesh_impl.mesh_axis_to_cumprod(new_shape)
    # Figure out what needs to be done for different mesh-axes
    mesh_axes_allsplit = []
    mesh_axes_allconcat = []
    mesh_axes_alltoall = []
    for mesh_axis, (old_cumprod, new_cumprod) in enumerate(
        zip(mesh_axis_to_cumprod_old, mesh_axis_to_cumprod_new)):
      if new_cumprod != old_cumprod:
        if old_cumprod is None:
          # split in new layout but not in old layout - we need an allsplit
          mesh_axes_allsplit.append(mesh_axis)
        elif new_cumprod is None:
          # split in old layout but not in new layout - we need an allconcat
          mesh_axes_allconcat.append(mesh_axis)
        else:
          # split differently in old and new layouts - we need an alltoall
          mesh_axes_alltoall.append(mesh_axis)

    laid_out_size = mesh_impl.laid_out_size(old_shape)

    for mesh_axis in mesh_axes_allsplit:
      tensor_axis = old_shape.cumprod_to_tensor_axis(
          mesh_axis_to_cumprod_new[mesh_axis])
      if tensor_axis is None:
        # TODO(noam): try to handle this case
        raise NotImplementedError(
            "Try first reshaping to insert a new tf dimension,"
            " then changing layout.")
      slices = mesh_impl.allsplit(slices, mesh_axis, tensor_axis)
      laid_out_size //= mesh_impl.shape[mesh_axis].size
    for mesh_axis in mesh_axes_alltoall:
      split_tensor_axis = old_shape.cumprod_to_tensor_axis(
          mesh_axis_to_cumprod_new[mesh_axis])
      if split_tensor_axis is None:
        # TODO(noam): try to handle this case
        raise NotImplementedError(
            "Try first reshaping to insert a new tf dimension,"
            " then changing layout.")
      concat_tensor_axis = old_shape.cumprod_to_tensor_axis(
          mesh_axis_to_cumprod_old[mesh_axis])
      assert concat_tensor_axis is not None
      slices = mesh_impl.alltoall(
          slices, mesh_axis, split_tensor_axis, concat_tensor_axis)
      lowering.add_counter(
          "alltoall/%s/reshape_op" % mesh_axis, laid_out_size)

    for mesh_axis in mesh_axes_allconcat:
      tensor_axis = old_shape.cumprod_to_tensor_axis(
          mesh_axis_to_cumprod_old[mesh_axis])
      assert tensor_axis is not None
      slices = mesh_impl.allconcat(slices, mesh_axis, tensor_axis)
      laid_out_size *= mesh_impl.shape[mesh_axis].size
      lowering.add_counter(
          "allconcat/%s/reshape_op" % mesh_axis, laid_out_size)
    # now reshape the slices
    old_slice_shape = mesh_impl.slice_shape(old_shape)
    new_slice_shape = mesh_impl.slice_shape(new_shape)
    if new_slice_shape != old_slice_shape:
      def reshape_fn(x):
        return tf.reshape(x, new_slice_shape)
      slices = mesh_impl.slicewise(reshape_fn, slices)
    lowering.set_tensor_lowering(self.outputs[0], slices)

  def gradient(self, grad_ys):
    return [reshape(grad_ys[0], self.inputs[0].shape)]


def reshape(x, new_shape):
  return ReshapeOperation(x, convert_to_shape(new_shape)).outputs[0]


def rename_dimension(x, old_name, new_name):
  """Reshape a Tensor, renaming one dimension.

  Args:
    x: a Tensor
    old_name: a string
    new_name: a string

  Returns:
    a Tensor
  """
  return reshape(x, x.shape.rename_dimension(old_name, new_name))


def einsum(xs, output_shape=None, name=None):
  """Einstein summation.

  If output_shape is not specified and there are two inputs, reduce over
  all common dimensions and default the output shape to the unique dimensions
  of the first input followed by the unique dimensions of the second input.

  Args:
    xs: a list of Tensors
    output_shape: an optional Shape.
    name: an optional string
  Returns:
    a Tensor
  Raises:
    ValueError: if the output shape cannot be inferred
  """
  output_shape = convert_to_shape(output_shape)
  if output_shape is None:
    if len(xs) == 2:
      output_shape = Shape(
          [d for d in xs[0].shape.dims if d not in xs[1].shape.dims] +
          [d for d in xs[1].shape.dims if d not in xs[0].shape.dims])
    else:
      raise ValueError("could not infer einsum output_shape for inputs %s" %
                       [x.to_string for x in xs])
  return EinsumOperation(xs, output_shape, name=name).outputs[0]


def matmul(a, b, output_shape=None, name=None):
  return einsum([a, b], output_shape=output_shape, name=name)


def _reduction_output_shape(x, output_shape, reduced_dim):
  """Helper function to reduce_sum, etc."""
  if output_shape is None:
    if reduced_dim is None:
      return Shape([])
    else:
      if reduced_dim not in x.shape.dims:
        raise ValueError(
            "reduced_dim=%s not in x.shape.dims=%s" % (reduced_dim, x.shape))
      return x.shape - reduced_dim
  elif reduced_dim is not None:
    raise ValueError("do not specify both reduced_dim and output_shape")
  else:
    return output_shape


def reduce_sum(x,
               disable_positional_args=None,
               output_shape=None,
               reduced_dim=None,
               name=None):
  """Reduction on 1 or more axes.

  If reduced_dim is present, then only that dimension is reduced out.
  Alternatively, specify output_shape.
  Do not specify both reduced_dim and output_shape.
  If neither is specified, then all dimensions are reduced out.

  Args:
    x: a Tensor
    disable_positional_args: None
    output_shape: an optional Shape.  Must be a subsequence of x.shape.
    reduced_dim: a mtf.Dimension
    name: an optional string
  Returns:
    a Tensor
  """
  output_shape = convert_to_shape(output_shape)
  reduced_dim = convert_to_dimension(reduced_dim)
  assert disable_positional_args is None
  output_shape = _reduction_output_shape(x, output_shape, reduced_dim)
  if output_shape == x.shape:
    return x
  return ReduceOperation(x, output_shape, "SUM", name=name).outputs[0]


def reduce_mean(x,
                disable_positional_args=None,
                output_shape=None,
                reduced_dim=None,
                name=None):
  """Reduction on 1 or more axes.

  If reduced_dim is present, then only that dimension is reduced out.
  Alternatively, specify output_shape.
  Do not specify both reduced_dim and output_shape.
  If neither is specified, then all dimensions are reduced out.

  Args:
    x: a Tensor
    disable_positional_args: None
    output_shape: an optional Shape. Must be a subsequence of x.shape.
    reduced_dim: a mtf.Dimension
    name: an optional string

  Returns:
    a Tensor
  """
  output_shape = convert_to_shape(output_shape)
  reduced_dim = convert_to_dimension(reduced_dim)
  assert disable_positional_args is None
  output_shape = _reduction_output_shape(x, output_shape, reduced_dim)
  with tf.variable_scope(name, default_name="reduce_mean"):
    if output_shape == x.shape:
      return x
    return reduce_sum(
        x, output_shape=output_shape) * (output_shape.size / x.shape.size)


def reduce_max(x,
               disable_positional_args=None,
               output_shape=None,
               reduced_dim=None,
               name=None):
  """Reduction on 1 or more axes.

  Args:
    x: a Tensor
    disable_positional_args: None
    output_shape: an optional Shape.  Must be a subsequence of x.shape.
    reduced_dim: an optional Dimension
    name: an optional string
  Returns:
    a Tensor
  """
  output_shape = convert_to_shape(output_shape)
  reduced_dim = convert_to_dimension(reduced_dim)
  assert disable_positional_args is None
  output_shape = _reduction_output_shape(x, output_shape, reduced_dim)
  if output_shape is None:
    output_shape = Shape([])
  if output_shape == x.shape:
    return x
  return ReduceOperation(
      x, output_shape, "MAX", name=name or "reduce_max").outputs[0]


def reduce_min(x,
               disable_positional_args=None,
               output_shape=None,
               reduced_dim=None,
               name=None):
  """Reduction on 1 or more axes.

  Args:
    x: a Tensor
    disable_positional_args: None
    output_shape: an optional Shape.  Must be a subsequence of x.shape.
    reduced_dim: an optional Dimension
    name: an optional string
  Returns:
    a Tensor
  """
  output_shape = convert_to_shape(output_shape)
  reduced_dim = convert_to_dimension(reduced_dim)
  assert disable_positional_args is None
  output_shape = _reduction_output_shape(x, output_shape, reduced_dim)
  if output_shape is None:
    output_shape = Shape([])
  if output_shape == x.shape:
    return x
  return ReduceOperation(
      x, output_shape, "MIN", name=name or "reduce_min").outputs[0]


def reduce_all(x,
               disable_positional_args=None,
               output_shape=None,
               reduced_dim=None,
               name=None):
  output_shape = convert_to_shape(output_shape)
  reduced_dim = convert_to_dimension(reduced_dim)
  return cast(reduce_min(to_float(x),
                         disable_positional_args=disable_positional_args,
                         output_shape=output_shape,
                         reduced_dim=reduced_dim,
                         name=name or "reduce_all"), tf.bool)


def reduce_any(x,
               disable_positional_args=None,
               output_shape=None,
               reduced_dim=None,
               name=None):
  output_shape = convert_to_shape(output_shape)
  reduced_dim = convert_to_dimension(reduced_dim)
  return cast(reduce_max(to_float(x),
                         disable_positional_args=disable_positional_args,
                         output_shape=output_shape,
                         reduced_dim=reduced_dim,
                         name=name or "reduce_any"), tf.bool)


def top_1(x, reduced_dim, dtype=tf.int32, name=None):
  """Argmax and Max.

  Args:
    x: a Tensor
    reduced_dim: a Dimension in x.shape.dims
    dtype: a tf.dtype (for the output)
    name: an optional string
  Returns:
    indices: a Tensor with given dtype
    values: optional Tensor equal to mtf.reduce_max(x, reduced_dim=reduced_dim)
  """
  reduced_dim = convert_to_dimension(reduced_dim)
  with tf.name_scope(name, default_name="top_1"):
    max_val = reduce_max(x, reduced_dim=reduced_dim)
    is_max = to_float(equal(x, max_val))
    pos = range(x.mesh, reduced_dim, tf.float32)
    ret = reduce_max(is_max * pos, reduced_dim=reduced_dim)
    ret = cast(ret, dtype)
    return ret, max_val


def argmax(x, reduced_dim, dtype=tf.int32, name=None):
  reduced_dim = convert_to_dimension(reduced_dim)
  return top_1(x, reduced_dim, dtype, name)[0]


def top_k(x, reduced_dim, new_dim, dtype=tf.int32, name=None):
  """Like tf.top_k.

  This operation returns two tensors with the same shape.  The output shape
  is identical to the shape of x, except that reduced_dim is replaced by
  new_dim.

  Args:
    x: a Tensor
    reduced_dim: a Dimension in x.shape.dims.
    new_dim: a Dimension.  The size determines k.
    dtype: optional dtype for indices.
    name: optional string.
  Returns:
    indices: a Tensor with given dtype.
    values: a Tensor with same type as x.
  """
  reduced_dim = convert_to_dimension(reduced_dim)
  new_dim = convert_to_dimension(new_dim)
  indices = []
  values = []
  k = new_dim.size
  with tf.name_scope(name, default_name="top_k"):
    for i in xrange(k):
      max_index, max_val = top_1(x, reduced_dim, dtype)
      indices.append(max_index)
      values.append(max_val)
      if i + 1 < k:
        x += one_hot(max_index, reduced_dim, on_value=-1e9)
  axis = x.shape.dims.index(reduced_dim)
  return stack(indices, new_dim.name, axis), stack(values, new_dim.name, axis)


def sample_with_temperature(x, dim, temperature=1.0, dtype=tf.int32, name=None):
  dim = convert_to_dimension(dim)
  with tf.name_scope(name, default_name="sample_with_temperature"):
    if temperature != 0.0:
      # gumbel trick
      g = -log(-log(random_uniform(x.mesh, x.shape, dtype=x.dtype)))
      x += g * temperature
    return argmax(x, dim, dtype, name)


def add(x1, x2, output_shape=None, name=None):
  """Binary addition with broadcsting.

  Args:
    x1: a Tensor
    x2: a Tensor
    output_shape: an optional Shape
    name: an optional string
  Returns:
    a Tensor
  """
  output_shape = convert_to_shape(output_shape)
  if not isinstance(x2, Tensor):
    return ScalarAddOperation(x1, x2).outputs[0]
  with tf.name_scope(name, default_name="add"):
    x1, x2 = binary_arguments_to_tensors(x1, x2)
    return AddOperation(
        x1, x2, output_shape=_infer_binary_broadcast_shape(
            x1.shape, x2.shape, output_shape)).outputs[0]


def sub(x1, x2, output_shape=None, name=None):
  """Binary subtraction with broadcsting.

  Args:
    x1: a Tensor
    x2: a Tensor
    output_shape: an optional Shape
    name: an optional string
  Returns:
    a Tensor
  """
  output_shape = convert_to_shape(output_shape)
  if not isinstance(x2, Tensor):
    return ScalarAddOperation(x1, -x2).outputs[0]
  with tf.name_scope(name, default_name="sub"):
    x1, x2 = binary_arguments_to_tensors(x1, x2)
    return add(x1, negative(x2), output_shape=output_shape)


def multiply(x1, x2, output_shape=None, name=None):
  """Binary multiplication with broadcsting.

  Args:
    x1: a Tensor
    x2: a Tensor
    output_shape: an optional Shape
    name: an optional string
  Returns:
    a Tensor
  """
  if not isinstance(x2, Tensor):
    return ScalarMultiplyOperation(x1, x2).outputs[0]
  with tf.name_scope(name, default_name="mul"):
    x1, x2 = binary_arguments_to_tensors(x1, x2)
    return einsum(
        [x1, x2],
        output_shape=_infer_binary_broadcast_shape(
            x1.shape, x2.shape, output_shape))


def divide(x1, x2, output_shape=None, name=None):
  """Binary division with broadcsting.

  Args:
    x1: a Tensor
    x2: a Tensor
    output_shape: an optional Shape
    name: an optional string
  Returns:
    a Tensor
  """
  output_shape = convert_to_shape(output_shape)
  if not isinstance(x2, Tensor):
    return ScalarMultiplyOperation(x1, 1.0 / x2).outputs[0]
  with tf.name_scope(name, default_name="divide"):
    x1, x2 = binary_arguments_to_tensors(x1, x2)
    return multiply(x1, reciprocal(x2), output_shape=output_shape)


def slice(x, begin, size, slice_dim_name, name=None):  # pylint: disable=redefined-builtin
  """Slice operation.

  Args:
    x: a list of Tensors
    begin: integer, where to begin slicing from along the axis
    size: integer, size to slice from axis.
    slice_dim_name: string, dimension name of slicing axis.
    name: an optional string
  Returns:
    a Tensor with shape extended by output_shape for the last axis.
  """
  return SliceOperation(
      x, begin, size, slice_dim_name, name=name).outputs[0]


def pad(x, paddings, dim_name, name=None):
  """Slice operation.

  Args:
    x: a list of Tensors
    paddings: list of integers of size 2, padding size before and after for dim.
    dim_name: string, name for the padding dim
    name: an optional string
  Returns:
    a Tensor with shape extended by output_shape for the last axis.
  """
  return PadOperation(
      x, paddings, dim_name, name=name).outputs[0]


def one_hot(indices, output_dim, on_value=1.0,
            off_value=0.0, dtype=tf.float32, name=None):
  """One hot operation.

  Args:
    indices: a Tensor
    output_dim: a Dimension
    on_value: Value taken when indices are on at a location, default 1
    off_value: Value taken when indices are off at a location, default 0
    dtype: a tf.DType
    name: an optional string
  Returns:
    a Tensor with shape extended by output_dim for the last axis.
  """
  return OneHotOperation(
      indices, output_dim, on_value, off_value, dtype, name=name).outputs[0]


def gather(weights, indices, dim, output_shape=None):
  """Shorthand for einsum([one_hot(indices, dim)], weights).

  Args:
    weights: a Tensor
    indices: a Tensor with integer type
    dim: a Dimension
    output_shape: an optional mtf.Shape
  Returns:
    a Tensor
  """
  dim = convert_to_dimension(dim)
  output_shape = convert_to_shape(output_shape)
  if weights.dtype == tf.bool:
    return cast(gather(to_float(weights), indices, dim, output_shape), tf.bool)
  return einsum([one_hot(indices, dim, dtype=weights.dtype), weights],
                output_shape=output_shape)


def gradients(ys, xs, grad_ys=None):
  """Compute gradients in dtf.

  Args:
    ys: a list of Tensors
    xs: a list of Tensors
    grad_ys: an optional list of Tensors

  Returns:
    grad_xs: a list of Tensors
  """
  graph = ys[0].graph
  if not grad_ys:
    grad_ys = [Constant(y.mesh, 1.0, y.shape, y.dtype).outputs[0] for y in ys]
  # figure out what Tensors are downstream of xs
  downstream = set(xs)
  for op in graph.operations:
    if op.has_gradient:
      if set(op.inputs) & downstream:
        downstream |= set(op.outputs)
  tensor_to_gradient = dict(zip(ys, grad_ys))
  for op in graph.operations[::-1]:
    grad_outputs = [tensor_to_gradient.get(out) for out in op.outputs]
    if op.has_gradient and any(grad_outputs) and (set(op.inputs) & downstream):
      with tf.variable_scope(op.name + "/gradients"):
        input_grads = op.gradient(grad_outputs)
        for inp, grad in zip(op.inputs, input_grads):
          if inp in downstream and grad is not None:
            if inp in tensor_to_gradient:
              tensor_to_gradient[inp] += grad
            else:
              tensor_to_gradient[inp] = grad
  return [tensor_to_gradient.get(x, None) for x in xs]


def _infer_binary_broadcast_shape(shape1, shape2, given_output_shape=None):
  """Infer shape of the output of a binary op with broadcasting.

  If the output shape is not given with given_output_shape, then we check
  to see if one of the shapes is a subsequence of the other one, and we
  return the one that is the supersequence.  Otherwise, we list the dimensions
  of shape1, followed by all new dimensions in shape2.

  Args:
    shape1: a Shape
    shape2: a Shape
    given_output_shape: an optional Shape
  Returns:
    a Shape
  """
  shape1 = convert_to_shape(shape1)
  shape2 = convert_to_shape(shape2)
  given_output_shape = convert_to_shape(given_output_shape)
  if given_output_shape is not None:
    return given_output_shape
  if is_subsequence(shape1.dims, shape2.dims):
    return shape2
  if is_subsequence(shape2.dims, shape1.dims):
    return shape1
  return Shape(
      shape1.dims + [d for d in shape2.dims if d not in shape1.dims])


def _expand_dims(x, input_shape, output_shape):
  """Expand dimensions and transpose if necessary.

  Args:
    x: a tf.Tensor
    input_shape: a Shape
    output_shape: a Shape whose dimensions are a superset of
      those in input_shape

  Returns:
    a tf.Tensor
  """
  verify_no_new_dims([output_shape], input_shape)
  if input_shape == output_shape or input_shape.ndims == 0:
    return x
  perm = [input_shape.dims.index(d) for d in output_shape.dims
          if d in input_shape.dims]
  x = tf.transpose(x, perm)
  for i, d in enumerate(output_shape.dims):
    if d not in input_shape.dims:
      x = tf.expand_dims(x, i)
  return x


def _einsum_equation(input_shapes, output_shape):
  """Turn shapes into an einsum equation.

  e.g. "ij,jk->ik"

  Args:
    input_shapes: a list of Shapes
    output_shape: a Shape
  Returns:
    a string
  """
  ret = []
  next_letter = ord("a")
  dim_to_letter = {}
  for shape_num, shape in enumerate(input_shapes + [output_shape]):
    if shape_num == len(input_shapes):
      ret.append("->")
    elif shape_num > 0:
      ret.append(",")
    for d in shape.dims:
      if d not in dim_to_letter:
        dim_to_letter[d] = chr(next_letter)
        next_letter += 1
      ret.append(dim_to_letter[d])

  return "".join(ret)


def is_subsequence(short_seq, long_seq):
  """Is short_seq a subsequence of long_seq."""
  if not short_seq:
    return True
  pos = 0
  for x in long_seq:
    if pos == len(short_seq):
      return True
    if short_seq[pos] == x:
      pos += 1
  if pos == len(short_seq):
    return True
  return False


def verify_no_new_dims(input_shapes, output_shape):
  """Verifies that all dimensions in the output are in at least one input.

  Args:
    input_shapes: a list of Shapes
    output_shape: a Shape
  Raises:
    ValueError: if there are new dimensions in the output.
  """
  all_input_dims = set(sum([s.dims for s in input_shapes], []))
  all_output_dims = set(output_shape.dims)
  if not all_output_dims.issubset(all_input_dims):
    raise ValueError(
        "No new dimensions allowed in output"
        " input_shapes = %s output_shape= %s"
        % ([s.dims for s in input_shapes], output_shape.dims))


def pnum_to_processor_coordinates(mesh_shape, pnum):
  """Coordinates of a processor in the mesh.

  Args:
    mesh_shape: a Shape
    pnum: an integer less than len(mesh_shape)

  Returns:
    a list of integers with length len(mesh_shape)
  """
  ret = []
  for dimsize in mesh_shape.to_integer_list[::-1]:
    ret.append(pnum % dimsize)
    pnum //= dimsize
  return ret[::-1]


def processor_coordinates_to_pnum(mesh_shape, coord):
  """Inverse of pnum_to_processor_coordinates.

  Args:
    mesh_shape: a Shape
    coord: a list of integers with length len(mesh_shape)

  Returns:
    an integer less than len(mesh_shape)
  """
  ret = 0
  multiplier = 1
  for c, d in zip(coord[::-1], mesh_shape.to_integer_list[::-1]):
    ret += multiplier * c
    multiplier *= d
  return ret


def pnum_to_group(mesh_shape, group_dims, pnum):
  """Group number for grouped allreduce.

  Args:
    mesh_shape: a Shape
    group_dims: a list of integers (the dimensions reduced over)
    pnum: an integer

  Returns:
    an integer
  """
  coord = pnum_to_processor_coordinates(mesh_shape, pnum)
  remaining_shape = Shape(
      [d for i, d in enumerate(mesh_shape) if i not in group_dims])
  remaining_coord = [d for i, d in enumerate(coord) if i not in group_dims]
  return processor_coordinates_to_pnum(remaining_shape, remaining_coord)


def processor_groups(mesh_shape, group_dims):
  """Groups of processors which differ only in the given dimensions.

  Args:
    mesh_shape: a Shape
    group_dims: a list of integers

  Returns:
    a list of lists of integers (processor numbers)
  """
  group_numbers = [
      pnum_to_group(mesh_shape, group_dims, pnum)
      for pnum in xrange(mesh_shape.size)]
  ret = []
  for pnum, g in enumerate(group_numbers):
    while len(ret) <= g:
      ret.append([])
    ret[g].append(pnum)
  return ret


def list_product(l):
  return reduce(mul, l, 1)


def log_softmax(x, reduced_dim, name=None):
  """log(softmax(x)).

  Args:
    x: a Tensor whose shape contains vocab_dim
    reduced_dim: a Dimension
    name: an optional string

  Returns:
    a Tensor with the same shape as x
  """
  reduced_dim = convert_to_dimension(reduced_dim)
  with tf.variable_scope(name, default_name="log_softmax"):
    reduced_shape = x.shape - reduced_dim
    max_logit = reduce_max(stop_gradient(x), output_shape=reduced_shape)
    x -= max_logit
    exp_x = exp(x)
    sum_exp_x = reduce_sum(exp_x, output_shape=reduced_shape)
    log_denom = log(sum_exp_x)
    return x - log_denom


def softmax(x, reduced_dim, name=None):
  with tf.variable_scope(name, default_name="softmax"):
    return exp(log_softmax(x, reduced_dim))


def range(mesh, dim, dtype, name=None):  # pylint: disable=redefined-builtin
  """Create a 1d mesh tensor with a range from [0, dim.size).

  Args:
    mesh: a Mesh
    dim: a Dimension
    dtype: a tf.DType
    name: an optional string

  Returns:
    a Tensor
  """
  dim = convert_to_dimension(dim)
  with tf.variable_scope(name, default_name="range"):
    return import_tf_tensor(
        mesh, tf.range(dim.size, dtype=dtype), shape=Shape([dim]))


def pretty_print_counters(counters):
  """print counters hierarchically.

  Each counter is a pair of a string and a number.
  The string can have slashes, meaning that the number also counts towards
  each prefix.  e.g.  "parameters/trainable" counts towards both "parameters"
  and "parameters/trainable".

  Args:
    counters: a list of (string, number) pairs

  Returns:
    a string
  """
  totals = collections.defaultdict(int)
  for (name, val) in counters:
    prefixes = [name[:i] for i in xrange(len(name)) if name[i] == "/"] + [name]
    for p in prefixes:
      totals[p] += val
  parts = []
  for name, val in sorted(six.iteritems(totals)):
    parts.append(" " * name.count("/") + "%s: %.3g" % (name, val))
  return "\n".join(parts)


def _parse_string_to_list_of_pairs(s, seconds_to_int=False):
  r"""Parses a string into a list of pairs.

  In the input string, each pair is separated by a colon, and the delimiters
  between pairs are any of " ,.;".

  e.g. "rows:32,cols:32"

  Args:
    s: str to parse.
    seconds_to_int: Boolean. If True, then the second elements are returned
      as integers;  otherwise they are strings.

  Returns:
    List of tuple pairs.

  Raises:
    ValueError: Badly formatted string.
  """
  ret = []
  for p in [s.split(":") for s in re.sub("[,.;]", " ", s).split()]:
    if len(p) != 2:
      raise ValueError("bad input to _parse_string_to_list_of_pairs %s" % s)
    if seconds_to_int:
      ret.append((p[0], int(p[1])))
    else:
      ret.append(tuple(p))
  return ret


def parallel(devices, fn, *args, **kwargs):
  """Call a function once on each device.

  Args:
    devices: a list of n devices
    fn: a function
    *args: arguments, each of which is a list of length n
    **kwargs: keyword-args, each of which is a list of length n
  Returns:
    a list of length n
  Raises:
    ValueError: if the arguments are not all lists of length n
  """
  if not isinstance(devices, list):
    raise ValueError("devices must be a list")
  for x in list(args) + list(six.itervalues(kwargs)):
    if not isinstance(x, list) or len(x) != len(devices):
      raise ValueError(
          "Argument not a list with same length as devices "
          "arg=%s devices=%s %s %s" % (x, devices, len(x), len(devices)))
  ret = []
  for i, device in enumerate(devices):
    with tf.device(device):
      with tf.variable_scope("parallel_%d" % i):
        my_args = [x[i] for x in args]
        my_kwargs = {k: v[i] for k, v in six.iteritems(kwargs)}
        ret.append(fn(*my_args, **my_kwargs))
  return ret


def transpose_list_of_lists(lol):
  """Transpose a list of equally-sized python lists.

  Args:
    lol: a list of lists
  Returns:
    a list of lists
  Raises:
    ValueError: if list is empty
  """
  if not lol:
    raise ValueError("cannot transpose the empty list")
  return [list(x) for x in zip(*lol)]


def binary_reduction_fn(reduction_fn_string):
  if reduction_fn_string == "SUM":
    return tf.add
  elif reduction_fn_string == "MAX":
    return tf.maximum
  elif reduction_fn_string == "MIN":
    return tf.minimum
  else:
    raise ValueError("Unknown reduction_fn_string %s" % reduction_fn_string)


def reduction_fn(reduction_fn_string):
  if reduction_fn_string == "SUM":
    return tf.reduce_sum
  elif reduction_fn_string == "MAX":
    return tf.reduce_max
  elif reduction_fn_string == "MIN":
    return tf.reduce_min
  else:
    raise ValueError("Unknown reduction_fn_string %s" % reduction_fn_string)


class MtfCheckpointSaverListener(tf.train.CheckpointSaverListener):
  """Copy slices to masters before saving."""

  def __init__(self, lowering):
    self._op = lowering.copy_slices_to_masters()

  def begin(self):
    # You can add ops to the graph here.
    tf.logging.info("Starting the session.")

  def before_save(self, session, global_step_value):
    # assigns
    tf.logging.info("Before Save.")
    session.run(self._op)
    tf.logging.info("About to write a checkpoint")

  def after_save(self, session, global_step_value):
    tf.logging.info("Done writing checkpoint.")

  def end(self, session, global_step_value):
    tf.logging.info("Done with the session.")


class MtfRestoreHook(tf.train.SessionRunHook):
  """Copy masters to slices after restoring."""

  def __init__(self, lowering):
    self._lowering = lowering

  def begin(self):
    self._op = self._lowering.copy_masters_to_slices()

  def after_create_session(self, session, coord):
    session.run(self._op)


class RandomOperation(Operation):
  """Random operation such as tf.random_uniform."""

  def __init__(self, mesh, shape, tf_fn, **kwargs):
    super(RandomOperation, self).__init__(
        [], mesh=mesh, name=kwargs.get("name", "random"))
    self._tf_fn = tf_fn
    self._kwargs = kwargs
    self._outputs = [Tensor(self, shape, kwargs.get("dtype", tf.float32))]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    output_shape = self.outputs[0].shape
    lowering.set_tensor_lowering(self.outputs[0], (
        mesh_impl.random(output_shape, self._tf_fn, self._kwargs)))


def random_uniform(mesh, shape, **kwargs):
  """Random uniform.

  Args:
    mesh: a Mesh
    shape: a Shape
    **kwargs: keyword args for tf.random_uniform, except seed

  Returns:
    a Tensor
  """
  shape = convert_to_shape(shape)
  return RandomOperation(mesh, shape, tf.random_uniform, **kwargs).outputs[0]


def dropout(x, keep_prob, noise_shape=None, name=None):
  """Dropout layer.

  Args:
    x: a Tensor
    keep_prob: a float between 0.0 and 1.0
    noise_shape: an optional Shape (a subset of x.shape)
    name: an optional string

  Returns:
    a Tensor
  """
  noise_shape = convert_to_shape(noise_shape)
  if noise_shape is None:
    noise_shape = x.shape
  with tf.variable_scope(name, default_name="dropout"):
    if keep_prob == 1.0:
      return x
    noise = cast(less(random_uniform(
        x.mesh, noise_shape, dtype=x.dtype), keep_prob), x.dtype)
    noise /= keep_prob
    return x * noise


def _cumprod(l):
  """Cumulative product of a list.

  Args:
    l: a list of integers
  Returns:
    a list with one more element (starting with 1)
  """
  ret = [1]
  for item in l:
    ret.append(ret[-1] * item)
  return ret


def log_variable_sizes(var_list, tag, verbose=True):
  """Log the sizes and shapes of variables, and the total size.

  Args:
    var_list: a list of variables; defaults to trainable_variables
    tag: a string; defaults to "Trainable Variables"
    verbose: bool, if True, log every weight; otherwise, log total size only.
  """
  if not var_list:
    return

  name_to_var = {v.name: v for v in var_list}
  total_size = 0
  for v_name in sorted(list(name_to_var)):
    v = name_to_var[v_name]
    v_size = v.shape.size
    if verbose:
      tf.logging.info("Weight    %s\tshape    %s\tsize    %d",
                      v.name.ljust(80),
                      str(v.shape).ljust(30), v_size)
    total_size += v_size
  tf.logging.info("%s Total size: %d", tag, total_size)


class WhileLoopOperation(Operation):
  """While loop."""

  def __init__(self, cond_fn, body_fn, inputs,
               tf_kwargs=None, name="while_loop"):
    super(WhileLoopOperation, self).__init__(
        inputs, mesh=inputs[0].mesh, name=name)
    self._cond_fn = cond_fn
    self._body_fn = body_fn
    self._tf_kwargs = tf_kwargs or {}
    assert not self._tf_kwargs.get("back_prop", False)
    ops = self.graph.operations
    before = len(ops)
    def make_placeholders(name):
      return [Tensor(self, t.shape, t.dtype, name="%s_%d" % (name, i))
              for i, t in enumerate(inputs)]
    self._cond_inputs = make_placeholders("cond_input")
    self._cond_output = self._cond_fn(*self._cond_inputs)
    self._cond_ops = ops[before:]
    del ops[before:]
    self._body_inputs = make_placeholders("body_input")
    self._body_outputs = self._body_fn(*self._body_inputs)
    for (i, (inp, body_out)) in enumerate(zip(inputs, self._body_outputs)):
      if inp.shape != body_out.shape:
        raise ValueError(
            "shape mismatch i=%d inp=%s body_out=%s" % (i, inp, body_out))
    self._body_ops = ops[before:]
    del ops[before:]
    self._outputs = make_placeholders("output")

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    def tf_cond_fn(*tf_inputs):
      for tf_inp, mtf_inp in zip(tf_inputs, self._cond_inputs):
        lowering.tensors[mtf_inp] = mesh_impl.LaidOutTensor(tf_inp)
      for op in self._cond_ops:
        with tf.name_scope(op.name):
          op.lower(lowering)
      lowered_output = lowering.tensors[self._cond_output]
      ret = lowered_output.to_laid_out_tensor().tensor_list[0]
      return ret

    def tf_body_fn(*tf_inputs):
      for tf_inp, mtf_inp in zip(tf_inputs, self._body_inputs):
        lowering.tensors[mtf_inp] = mesh_impl.LaidOutTensor(tf_inp)
      for op in self._body_ops:
        with tf.name_scope(op.name):
          op.lower(lowering)
      return [
          lowering.tensors[mtf_out].to_laid_out_tensor().tensor_list
          for mtf_out in self._body_outputs]

    lowered_inputs = [
        lowering.tensors[t].to_laid_out_tensor().tensor_list
        for t in self.inputs]

    tf_outs = tf.while_loop(tf_cond_fn,
                            tf_body_fn,
                            lowered_inputs,
                            back_prop=False,
                            **self._tf_kwargs)
    for tf_out, mtf_out in zip(tf_outs, self._outputs):
      lowering.set_tensor_lowering(mtf_out, mesh_impl.LaidOutTensor(tf_out))


def while_loop(cond_fn, body_fn, inputs, num_loop_vars=None, **kwargs):
  """While Loop.

  num_loop_vars is a hack for the multi-gpu setup.  In this case, loops
  are generally slow, as all loop variables are placed on device.  By setting
  num_loop_vars=k, then all of the loop variables except for the first k
  are handled as mtf Variables instead of loop variables, using explicit
  updates and control dependencies.  In this case, we only return the
  first num_loop_vars outputs.  Do not use this option on TPU, since it
  is unnecessary and also produces incorrect results, since xla does not
  respect control dependencies.

  Args:
    cond_fn: a function from n Tensors to scalar boolean Tensor
    body_fn: a function from n Tensors to n Tensors
    inputs: a list of n Tensors
    num_loop_vars: an optional integer.
    **kwargs: additional kwargs passed to tf.while_loop

  Returns:
    a list of n Tensors.
  """
  if num_loop_vars is None:
    return WhileLoopOperation(cond_fn, body_fn, inputs, kwargs).outputs
  # Turn all loop vars except for the first ones into non-loop vars.
  # see comments in docstring.
  assert num_loop_vars > 0
  extra_inputs = inputs[num_loop_vars:]
  my_vars = tuple([get_variable(
      x.mesh, "loop_var_%d" % i,
      x.shape, initializer=tf.zeros_initializer(),
      dtype=x.dtype,
      collections=[tf.GraphKeys.LOCAL_VARIABLES])
                   for i, x in enumerate(extra_inputs)])
  first_input = depend(
      inputs[0], [assign(var, x) for var, x in zip(my_vars, extra_inputs)])
  inputs = [first_input] + inputs[1:num_loop_vars]
  def my_cond_fn(*inputs):
    return cond_fn(*(inputs + my_vars))
  def my_body_fn(*inputs):
    outputs = tuple(body_fn(*(inputs + my_vars)))
    extra_outputs = outputs[num_loop_vars:]
    first_output = depend(
        outputs[0], [assign(var, x) for var, x in zip(my_vars, extra_outputs)])
    outputs = (first_output,) + outputs[1:num_loop_vars]
    return outputs
  return WhileLoopOperation(
      my_cond_fn, my_body_fn, inputs, kwargs).outputs


def where(condition, if_true, if_false):
  dtype = if_true.dtype
  return (
      if_true * cast(condition, dtype) +
      if_false * cast(logical_not(condition), dtype))
