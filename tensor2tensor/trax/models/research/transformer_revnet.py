# coding=utf-8
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

"""Transformer Models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
import numpy as onp

from tensor2tensor.trax import backend
from tensor2tensor.trax import layers as tl
from tensor2tensor.trax.backend import numpy as np
from tensor2tensor.trax.layers.combinators import _pop_rng_and_split


# Layers are always CamelCase, but functions in general are snake_case
# pylint: disable=invalid-name


class Map(tl.Layer):
  """Combinator for applying a layer to a list or tuple."""

  def __init__(self, layer, n_sections=1, check_shapes=True):
    """Initialize the combinator.

    Args:
      layer: a layer to apply to each element.
      n_sections: how many sections to map to (default: 1).
      check_shapes: whether to check that shapes are identical (default: true).

    Returns:
      A new layer representing mapping layer to all elements of the input.
    """
    super(Map, self).__init__()
    if layer is None or isinstance(layer, (list, tuple)):
      layer = tl.Serial(layer)
    self._layer = layer
    # Generally a Map should be applied to lists where all elements have
    # the same shape -- because self._layer will only be initialized once
    # and it could have different parameters for different shapes. But there
    # are valid cases -- e.g., when self._layer has no parameters -- where we
    # can apply Map to different shapes -- set check_shapes=False in such cases.
    self._check_shapes = check_shapes
    self._n_sections = n_sections

  def n_inputs(self):
    """Specifies how many data tensors this layer expects as input."""
    return self._n_sections

  def n_outputs(self):
    """Specifies how many data tensors this layer promises as output."""
    return self._n_sections

  def call(self, inputs, params=(), state=(), **kwargs):
    rngs = _pop_rng_and_split(kwargs, len(inputs))
    results = [self._layer(x, params=params, state=state, rng=r, **kwargs)
               for x, r in zip(inputs, rngs)]
    result_outputs, result_states = zip(*results)
    return tuple(result_outputs), tuple(result_states)

  def new_parameters(self, input_shape, input_dtype, rng):
    first_shape = input_shape[0]
    if self._check_shapes:
      for shape in input_shape:
        if shape != first_shape:
          raise ValueError('Map layer can only be applied to list of elements '
                           'with the same shapes. Shapes: %s' % str(shape))
    return self._layer.initialize(first_shape, input_dtype[0], rng)


@tl.layer()
def BroadcastedDropout(x, params, rate=0.0, mode='train', broadcast_dims=(-2,),
                       rng=None, **kwargs):
  """Dropout, with broadcasting to save memory."""
  del params, kwargs
  if rng is None:
    raise ValueError('BroadcastedDropout requires rng kwarg.')
  if rate >= 1.0:
    raise ValueError('Dropout rate (%f) must be lower than 1.' % rate)
  if mode == 'train' and rate > 0.0:
    noise_shape = list(x.shape)
    for dim in broadcast_dims:
      noise_shape[dim] = 1
    keep_prob = jax.lax.tie_in(rng, 1.0 - rate)
    keep = backend.random.bernoulli(rng, keep_prob, tuple(noise_shape))
    multiplier = keep.astype(x.dtype) / jax.lax.tie_in(keep, keep_prob)
    return x * multiplier
  else:
    return x


def FeedForward(d_model, d_ff, dropout, mode):
  """Feed-forward block with layer normalization at start."""
  return [
      tl.LayerNorm(),
      tl.Dense(d_ff),
      BroadcastedDropout(rate=dropout, mode=mode),  # pylint: disable=no-value-for-parameter
      tl.Relu(),
      tl.Dense(d_model),
      BroadcastedDropout(rate=dropout, mode=mode),  # pylint: disable=no-value-for-parameter
  ]


class Split(tl.Layer):
  """Splits the input into sections along an axis."""

  def __init__(self, n_sections=2, axis=-1):
    super(Split, self).__init__()
    self._n_sections = n_sections
    self._axis = axis

  def call(self, inputs, params=(), state=(), **kwargs):
    del params, kwargs
    res = tuple(backend.numpy.split(inputs, self._n_sections, self._axis))
    return res, state

  def new_parameters(self, input_shapes, input_dtype, rng):
    return (), ()

  def n_inputs(self):
    """Specifies how many data tensors this layer expects as input."""
    return 1

  def n_outputs(self):
    """Specifies how many data tensors this layer promises as output."""
    return self._n_sections


class SplitForOutput(tl.ReversibleLayer):
  """Splits activations into sections (for use right before the output layer).

  After the reversible portion of the network, there is a final output portion
  that's non-reversible (which at minimum includes normalization, output
  projection, and log-softmax). The output portion needs to operate on chunks
  of the sequence to avoid running out of memory for large vocabulary sizes.

  This layer concatenates the two subparts of the activations along the feature
  dimension, and then splits into chunks along the time dimension. We implement
  it is a subclass of tl.ReversibleLayer because we want to ensure that multiple
  copies of the activations don't exist simultaneously except in the middle of a
  memory copy operation.
  """

  def __init__(self, n_sections=2, axis=-2):
    super(SplitForOutput, self).__init__()
    self._n_sections = n_sections
    self._axis = axis

  def n_inputs(self):
    """Specifies how many data tensors this layer expects as input."""
    return 2

  def n_outputs(self):
    """Specifies how many data tensors this layer promises as output."""
    return self._n_sections

  def new_parameters(self, input_shape, input_dtype, rng):
    return (), ()

  def call(self, inputs, params=(), state=(), **kwargs):
    del params, kwargs
    x1, x2 = inputs

    x1_split = backend.numpy.split(x1, self._n_sections, self._axis)
    x2_split = backend.numpy.split(x2, self._n_sections, self._axis)

    res = [backend.numpy.concatenate(ys, -1) for ys in zip(x1_split, x2_split)]
    return tuple(res), state

  def reverse(self, output, params=(), **kwargs):
    del params, kwargs

    x1_split = []
    x2_split = []
    for y in output:
      y1, y2 = backend.numpy.split(y, 2, -1)
      x1_split.append(y1)
      x2_split.append(y2)

    x1 = backend.numpy.concatenate(x1_split, self._axis)
    x2 = backend.numpy.concatenate(x2_split, self._axis)

    return (x1, x2)

  def reverse_and_grad(self, output, ct, params=(), **kwargs):
    del params, kwargs
    return self.reverse(output), (self.reverse(ct), ())


@tl.layer()
def Chunk(x, params, n_sections=2, **kwargs):
  del params, kwargs
  assert x.shape[1] % n_sections == 0
  return backend.numpy.reshape(x, (
      x.shape[0] * n_sections,
      x.shape[1] // n_sections,
      ) + x.shape[2:])


@tl.layer()
def Unchunk(x, params, n_sections=2, **kwargs):
  del params, kwargs
  assert x.shape[0] % n_sections == 0
  return backend.numpy.reshape(x, (
      x.shape[0] // n_sections,
      x.shape[1] * n_sections,
      ) + x.shape[2:])


class ReversibleHalfResidual(tl.ReversibleLayer, tl.Serial):
  """Half of a RevNet-style residual (only updates part of the hidden state)."""

  def __init__(self, residual_layers):
    self.compute_residual = tl.Serial([
        # (x1_or_y1, x2) -> (x2, x1_or_y1, x2)
        tl.Parallel([], tl.Dup()),
        tl.Swap(),
        tl.Parallel(residual_layers, [], []),
    ])

    layers = [
        self.compute_residual,
        tl.Parallel(tl.Add(), [])
    ]
    super(ReversibleHalfResidual, self).__init__(layers)

    self.subtract_top = tl.Parallel(tl.SubtractTop(), [])
    self.reverse_layers = [self.compute_residual, self.subtract_top]

  def reverse(self, output, params=(), **kwargs):
    reconstructed_x = output
    rng = kwargs.pop('rng', None)
    rngs = (None,) * self._n_layers
    if rng is not None:
      rngs = backend.random.split(rng, self._n_layers)
    # Note that self.sublayers() aligns exactly with self.reverse_layers in
    # terms of parameter and rng usage, so no re-ordering is required.
    for layer, p, rng in zip(self.reverse_layers, params, rngs):
      reconstructed_x = layer(reconstructed_x, p, rng=rng, **kwargs)
    return reconstructed_x

  def reverse_and_grad(self, output, ct, params=(), **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = (None,) * self._n_layers
    if rng is not None:
      rngs = backend.random.split(rng, self._n_layers)

    def call_compute_residual(x, params):
      return self.compute_residual(x, params, rng=rngs[0], **kwargs)

    assert len(ct) == 2
    ct = ((ct[0], ct[0], ct[1]))

    stack_with_residual, vjpfun = jax.vjp(
        call_compute_residual, output, params[0])
    reconstructed_x = self.subtract_top(
        stack_with_residual, params[-1], rng=rngs[-1], **kwargs)

    x_ct, residual_params_ct = vjpfun(ct)
    return reconstructed_x, (x_ct, (residual_params_ct, ()))


class ComputeAttentionHeads(tl.Layer):
  """Computes queries/keys/values via linear projection.

  The output shape is (n_batch * n_heads, seqlen, d_head); the batch and head
  dimensions are fused to allow for more efficient memory layouts.
  """

  def __init__(self, n_heads=1, d_head=64,
               kernel_initializer=tl.initializers.GlorotUniformInitializer()):
    super(ComputeAttentionHeads, self).__init__()
    self._n_heads = n_heads
    self._d_head = d_head
    self._kernel_initializer = kernel_initializer
    # The lack of a bias term here is consistent with the tensor2tensor
    # implementation, and shouldn't have an effect on modeling quality.

  def call(self, x, params, state, **kwargs):
    del kwargs
    seqlen = x.shape[1]
    res = np.dot(x, params)

    # n_batch, seqlen, n_heads*d_head -> n_batch, seqlen, n_heads, d_head
    res = np.reshape(res, (x.shape[0], seqlen, self._n_heads, self._d_head))
    # n_batch, seqlen, n_heads, d_head -> n_batch, n_heads, seqlen, d_head
    res = np.transpose(res, (0, 2, 1, 3))
    # n_batch, n_heads, seqlen, d_head -> n_batch*n_heads, seqlen, d_head
    res = np.reshape(res, (-1, seqlen, self._d_head))

    return res, state

  def new_parameters(self, input_shape, input_dtype, rng):
    del input_dtype
    w = self._kernel_initializer(
        (input_shape[-1], self._n_heads * self._d_head), rng)
    return w, ()


class ComputeAttentionOutput(tl.Layer):
  """Joins outputs from different heads via linear projection."""

  def __init__(self, n_heads=1, d_model=1024,
               kernel_initializer=tl.initializers.GlorotUniformInitializer()):
    super(ComputeAttentionOutput, self).__init__()
    self._n_heads = n_heads
    self._d_model = d_model
    self._kernel_initializer = kernel_initializer
    # The lack of a bias term here is consistent with the tensor2tensor
    # implementation, and shouldn't have an effect on modeling quality.

  def call(self, x, params, state, **kwargs):
    del kwargs
    seqlen = x.shape[1]
    d_head = x.shape[2]

    x = np.reshape(x, (-1, self._n_heads, seqlen, d_head))
    x = np.transpose(x, (0, 2, 1, 3))  # -> n_batch, seqlen, n_heads, d_head
    x = np.reshape(x, (-1, seqlen, self._n_heads * d_head))

    return np.dot(x, params), state

  def new_parameters(self, input_shape, input_dtype, rng):
    del input_dtype
    w = self._kernel_initializer(
        (input_shape[-1] * self._n_heads, self._d_model), rng)
    return w, ()


class ApplyAttentionWrapper(tl.Parallel):
  """Same as tl.Parallel(attention, [], []), but implements forward_and_vjp.

  See MemoryEfficientDotProductAttention for why this is needed.
  """

  def __init__(self, attention):
    assert hasattr(attention, 'forward_and_vjp')
    super(ApplyAttentionWrapper, self).__init__(attention, [], [])
    self.attention = attention

  def forward_and_vjp(self, inputs, ct, params=(), **kwargs):
    # Simultaneous forward pass and backprop through the attention mechanism.
    qkv = inputs[:3]
    passthrough = inputs[3:]
    out_ct = ct[0]
    passthrough_ct = ct[1:]

    out, qkv_ct = self.attention.forward_and_vjp(
        qkv, out_ct, params=(), **kwargs)
    return (out,) + passthrough, qkv_ct + passthrough_ct


class DotProductAttention(tl.Layer):
  """A standard (non-memory-efficient) dot product attention implementation.

  This class sets up the API that is required to implement
  MemoryEfficientDotProductAttention.
  """

  def __init__(self, dropout, mode):
    super(DotProductAttention, self).__init__()
    self._dropout = dropout
    self._mode = mode

  def call(self, inputs, params=(), state=(), rng=None, **kwargs):
    del params
    q, k, v = inputs
    mask_size = q.shape[-2]
    mask = np.tril(np.ones((1, mask_size, mask_size), dtype=onp.bool_), k=0)
    res = tl.DotProductAttention(
        q, k, v, mask, dropout=self._dropout, mode=self._mode, rng=rng)
    return res, state

  def forward_and_vjp(self, inputs, ct, params=(), **kwargs):
    # Simultaneous forward pass and backprop through the attention mechanism.
    def do_call(x):
      return self.call(x, params, **kwargs)
    output, vjpfun = jax.vjp(do_call, inputs)
    return output, vjpfun(ct)[0]

  def new_parameters(self, input_shapes, input_dtype, rng):
    return (), ()

  def n_inputs(self):
    return 3

  def n_outputs(self):
    return 1


class MemoryEfficientDotProductAttention(DotProductAttention):
  """Memory-efficient dot product attention."""

  def __init__(self, loop_stride, dropout, mode):
    super(MemoryEfficientDotProductAttention, self).__init__(dropout, mode)
    self._loop_stride = loop_stride
    if dropout >= 1.0:
      raise ValueError('Dropout rates must be lower than 1.')
    if mode == 'train':
      self.dropout = dropout
    else:
      self.dropout = None

  def call(self, inputs, params=(), state=(), **kwargs):
    output, _ = self.forward_and_vjp(inputs, None, params=params, **kwargs)
    return output, state

  def forward_and_vjp(self, inputs, ct, params=(), rng=None, **kwargs):
    # This is the core of the memory-efficient attention implementation, where
    # we use the jax.lax.while_loop primitive to compute attention for a small
    # set of query positions at a time. Note how in the backwards pass, we
    # compute both the forward direction (to recover the previous layer's
    # activations) and the backward direction simultaneously. This allows us to
    # only use a single loop, where the inner portion of the loop does a slice
    # of the forward+backward joint computation. Unfortunately we have had to
    # introduce a large number of wrapper classes (including
    # ReversibleAttentionHalfResidual and ApplyAttentionWrapper) for the sole
    # purpose of connecting this implementation of forward_and_vjp with the core
    # backprop implementation.

    query, key, value = inputs
    depth = np.shape(query)[-1]
    do_backprop = ct is not None

    def make_mask(N, M, k):
      x = np.arange(N, dtype=np.int32)
      y = np.arange(M, dtype=np.int32)
      mask = jax.lax.lt(
          (jax.lax.broadcast_in_dim(
              x, shape=(N, M), broadcast_dimensions=(0,)) + k),
          jax.lax.broadcast(y, [N]))
      mask = jax.lax.convert_element_type(mask, np.float32)
      return mask

    def forward_slice(query_slice, q_loop_idx, key, value):
      """Forward pass for a subset of the query vectors."""
      dots = np.matmul(
          query_slice, np.swapaxes(key, -1, -2)) / np.sqrt(depth)

      # Causal masking
      mask = make_mask(dots.shape[-2], dots.shape[-1], q_loop_idx)
      dots = dots - 1e9 * mask

      # Softmax.
      dots = np.exp(dots - dots.max(axis=-1, keepdims=True))
      dots = dots / dots.sum(axis=-1, keepdims=True)

      if self.dropout is not None and self.dropout > 0.0:
        # Dropout is broadcast across the batch+head dimension
        dropout_shape = (1, dots.shape[-2], dots.shape[-1])
        slice_rng = jax.random.fold_in(rng, q_loop_idx)
        keep_prob = jax.lax.tie_in(dots, 1.0 - self.dropout)
        keep = backend.random.bernoulli(slice_rng, keep_prob, dropout_shape)
        multiplier = keep.astype(dots.dtype) / jax.lax.tie_in(keep, keep_prob)
        dots = dots * multiplier

      out_slice = np.matmul(dots, value)
      return out_slice

    def forward_and_vjp_slice(query_slice, q_loop_idx, key, value, ct_slice):
      # Capture q_loop_idx to avoid calculated gradients wrt. it.
      def forward_slice_with_q_loop_idx(query_slice, key, value):
        return forward_slice(query_slice, q_loop_idx, key, value)

      output_slice, vjpfun = jax.vjp(
          forward_slice_with_q_loop_idx, query_slice, key, value)
      return output_slice, vjpfun(ct_slice)

    q_loop_idx = np.zeros((), dtype=np.int32)
    q_loop_max = query.shape[-2]
    q_loop_stride = self._loop_stride
    assert q_loop_max % q_loop_stride == 0, (
        'Stride must evenly divide the number of query elements.')

    out_accum = np.zeros_like(query)
    if do_backprop:
      query_ct_accum = np.zeros_like(query)
      key_ct_accum = np.zeros_like(key)
      value_ct_accum = np.zeros_like(value)
      init_vals = (
          q_loop_idx, out_accum,
          query_ct_accum, key_ct_accum, value_ct_accum)
    else:
      init_vals = (q_loop_idx, out_accum)

    def cond_fun(vals):
      q_loop_idx = vals[0]
      return jax.lax.lt(q_loop_idx, q_loop_max)

    def body_fun(vals):
      """Compute a slice of the attention mechanism."""
      if do_backprop:
        (q_loop_idx, out_accum,
         query_ct_accum, key_ct_accum, value_ct_accum) = vals
      else:
        q_loop_idx, out_accum = vals

      query_slice = jax.lax.dynamic_slice_in_dim(
          query, q_loop_idx, q_loop_stride, axis=-2)

      if do_backprop:
        ct_slice = jax.lax.dynamic_slice_in_dim(
            ct, q_loop_idx, q_loop_stride, axis=-2)
        out_slice, partial_ct = forward_and_vjp_slice(
            query_slice, q_loop_idx, key, value, ct_slice)
        query_ct_accum = jax.lax.dynamic_update_slice_in_dim(
            query_ct_accum, partial_ct[0], q_loop_idx, axis=-2)
        key_ct_accum = key_ct_accum + partial_ct[1]
        value_ct_accum = value_ct_accum + partial_ct[2]
      else:
        out_slice = forward_slice(query_slice, q_loop_idx, key, value)

      out_accum = jax.lax.dynamic_update_slice_in_dim(
          out_accum, out_slice, q_loop_idx, axis=-2)
      q_loop_idx = q_loop_idx + q_loop_stride

      if do_backprop:
        return (q_loop_idx, out_accum,
                query_ct_accum, key_ct_accum, value_ct_accum)
      else:
        return (q_loop_idx, out_accum)

    final_vals = jax.lax.while_loop(cond_fun, body_fun, init_vals)

    if not do_backprop:
      return final_vals[1], None
    else:
      return final_vals[1], final_vals[2:]


class DummyHashedAttention(DotProductAttention):
  """A stand-in for hash-based attention, but without a real hash function."""

  def __init__(self, dropout, mode, n_bins=64):
    super(DummyHashedAttention, self).__init__(dropout, mode)
    self.n_bins = n_bins

  def call(self, inputs, params=(), state=(), **kwargs):
    output, _ = self.forward_and_vjp(inputs, None, params=params, **kwargs)
    return output, state

  def forward_and_vjp(self, inputs, ct, params=(), **kwargs):
    del params, kwargs
    q, k, v = inputs
    # q/k/v are n_batch*n_heads, seqlen, d_head

    assert k.shape[-2] % self.n_bins == 0
    bin_size = int(k.shape[-2] // self.n_bins)

    # q_bins/kv_bins are n_batch*n_heads, seqlen
    # They specify which hash bucket the query/key/value vectors fall in. For
    # now, instead of hashing we just put consecutive items in the same bucket.
    q_bins = np.arange(q.shape[-2], dtype=np.int32) // bin_size
    q_bins = jax.lax.tie_in(q, q_bins)
    q_bins = q_bins[None, :]
    q_bins = np.broadcast_to(q_bins, q.shape[:-1])
    q_bins = -q_bins
    kv_bins = q_bins * 2

    # q_t/kv_t are n_batch*n_heads, seqlen
    q_t = jax.lax.tie_in(q, np.arange(q.shape[-2]))
    q_t = np.reshape(q_t, (1, q_t.shape[0]))
    q_t = np.broadcast_to(q_t, q.shape[:-1])
    kv_t = q_t

    def chunk_scalars(x):
      return np.reshape(x, (x.shape[0], self.n_bins, -1))

    def chunk_vectors(x):
      return np.reshape(
          x, (x.shape[0], self.n_bins, -1, x.shape[-1]))

    def unchunk_vectors(x):
      return np.reshape(x, (x.shape[0], -1, x.shape[-1]))

   # Sort everything by bin number (variables starting with "s" are sorted)
    _, sq_t = jax.lax.sort_key_val(q_bins, q_t, dimension=-1)

    sq = np.take_along_axis(q, sq_t[:, :, None], axis=-2)
    if ct is not None:
      so_ct = np.take_along_axis(ct, sq_t[:, :, None], axis=-2)

    _, skv_t = jax.lax.sort_key_val(kv_bins, kv_t, dimension=-1)
    sk = np.take_along_axis(k, skv_t[:, :, None], axis=-2)
    sv = np.take_along_axis(v, skv_t[:, :, None], axis=-2)

    @jax.jit
    def binned_attn(sq, sk, sv):
      """Performs attention on sorted queries/keys/values."""
      # Split off a "bin" axis so that attention only occurs whithin chunks.
      bq_t = chunk_scalars(sq_t)
      bkv_t = chunk_scalars(skv_t)
      bq = chunk_vectors(sq)
      bk = chunk_vectors(sk)
      bv = chunk_vectors(sv)

      dots = np.matmul(bq, np.swapaxes(bk, -1, -2)) / np.sqrt(bq.shape[-1])

      # Causal masking
      mask = jax.lax.convert_element_type(
          jax.lax.lt(bq_t[:, :, :, None], bkv_t[:, :, None, :]),
          np.float32)
      dots = dots - 1e9 * mask

      # Softmax.
      dots = np.exp(dots - dots.max(axis=-1, keepdims=True))
      dots = dots / dots.sum(axis=-1, keepdims=True)
      bo = np.matmul(dots, bv)

      so = unchunk_vectors(bo)
      return so

    @jax.jit
    def binned_attn_vjp(sq, sk, sv, so_ct):
      so, vjpfun = jax.vjp(binned_attn, sq, sk, sv)
      sqkv_ct = vjpfun(so_ct)
      return so, sqkv_ct

    if ct is None:
      so = binned_attn(sq, sk, sv)
      _, undo_q_sort = jax.lax.sort_key_val(sq_t, q_t, dimension=-1)
      out = np.take_along_axis(so, undo_q_sort[:, :, None], axis=-2)
      return out, None
    else:
      # Jax can construct a backward pass automatically, but it's about 2x
      # slower than writing our own. The main reason is that the backward pass
      # of gather is in general a scatter operation, but we know we're dealing
      # with permutations so we use gather for the backward pass too.
      so, (sq_ct, sk_ct, sv_ct) = binned_attn_vjp(sq, sk, sv, so_ct)

      _, undo_q_sort = jax.lax.sort_key_val(sq_t, q_t, dimension=-1)
      out = np.take_along_axis(so, undo_q_sort[:, :, None], axis=-2)
      q_ct = np.take_along_axis(sq_ct, undo_q_sort[:, :, None], axis=-2)

      _, undo_kv_sort = jax.lax.sort_key_val(skv_t, kv_t, dimension=-1)
      k_ct = np.take_along_axis(sk_ct, undo_kv_sort[:, :, None], axis=-2)
      v_ct = np.take_along_axis(sv_ct, undo_kv_sort[:, :, None], axis=-2)

      return out, (q_ct, k_ct, v_ct)


class ReversibleAttentionHalfResidual(tl.ReversibleLayer, tl.Serial):
  """Half of a RevNet-style residual that performs attention.

  If inputs are (x1, x2), then outputs are (x1 + z, x2) where:
  z = post_attention(attention(pre_attention(x1)))

  The post_attention layers must be linear in their input (typically they will
  consists of reshaping and dense linear layers). This allows back-propagating
  the gradient signal from the output of ReversibleAttentionHalfResidual to the
  output of the "attention" portion based only on the network parameters.

  The forward pass is equivalent to using
  ReversibleHalfResidual([pre_attention, attention, post_attention]), but the
  backward pass uses attention.forward_and_vjp. See
  MemoryEfficientDotProductAttention for why forward_and_vjp is helpful.
  """

  def __init__(self, pre_attention, attention, post_attention):
    self.pre_attention = tl.Serial([
        # (x1_or_y1, x2) -> (x2, x1_or_y1, x2)
        tl.Parallel([], tl.Dup()),
        tl.Swap(),
        tl.Parallel(pre_attention, [], []),
    ])
    assert hasattr(attention, 'forward_and_vjp')
    self.attention = ApplyAttentionWrapper(attention)
    self.post_attention = tl.Parallel(post_attention, [], [])

    layers = [
        self.pre_attention,
        self.attention,
        self.post_attention,
        tl.Parallel(tl.Add(), []),
    ]
    super(ReversibleAttentionHalfResidual, self).__init__(layers)

    self.subtract_top = tl.Parallel(tl.SubtractTop(), [])
    self.reverse_layers = [
        self.pre_attention,
        self.attention,
        self.post_attention,
        self.subtract_top,
    ]

  def reverse(self, output, params=(), **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = (None,) * self._n_layers
    if rng is not None:
      rngs = backend.random.split(rng, self._n_layers)

    reconstructed_x = output
    # Note that self.sublayers() aligns exactly with self.reverse_layers in
    # terms of parameter and rng usage, so no re-ordering is required.
    for layer, p, rng in zip(self.reverse_layers, params, rngs):
      reconstructed_x = layer.reverse(reconstructed_x, p, rng=rng, **kwargs)
    return reconstructed_x

  def reverse_and_grad(self, output, ct, params=(), **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = (None,) * self._n_layers
    if rng is not None:
      rngs = backend.random.split(rng, self._n_layers)

    # Forward pass through self.pre_attention, while preparing for
    # later backprop.
    def call_pre_attention(x, params):
      return self.pre_attention(x, params, rng=rngs[0], **kwargs)
    stack, pre_attention_vjpfun = jax.vjp(call_pre_attention, output, params[0])

    # Backprop through adding the residual
    assert len(ct) == 2
    ct = saved_ct = (ct[0], ct[0], ct[1])

    # Backprop through self.post_attention with respect to the inputs only
    def call_post_attention(x):
      return self.post_attention(x, params[2], rng=rngs[2], **kwargs)
    # Note: these are *not* the actual inputs to self.post_attention.
    # If self.post_attention is not linear, we will get incorrect gradients.
    dummy_inputs = (stack[-3], stack[-2], stack[-1])
    _, post_attention_vjpfun = jax.vjp(call_post_attention, dummy_inputs)
    (ct,) = post_attention_vjpfun(ct)

    # Simultaneous forward pass and backprop through the attention mechanism
    stack, ct = self.attention.forward_and_vjp(
        stack, ct, rng=rngs[1], **kwargs)
    attention_params_ct = ()

    # Backprop through self.pre_attention
    x_ct, pre_attention_params_ct = pre_attention_vjpfun(ct)

    # Forward pass for self.post_attention, and backprop with respect to the
    # parameters only
    def call_post_attention2(params):
      return self.post_attention(stack, params, rng=rngs[2], **kwargs)
    stack, post_attention_vjpfun = jax.vjp(call_post_attention2, params[2])
    (post_attention_params_ct,) = post_attention_vjpfun(saved_ct)

    # Forward pass through subtracting the residual
    reconstructed_x = self.subtract_top(
        stack, params[-1], rng=rngs[-1], **kwargs)

    params_ct = (
        pre_attention_params_ct,
        attention_params_ct,
        post_attention_params_ct,
        (),
        )

    return reconstructed_x, (x_ct, params_ct)


def DecoderBlock(d_model, d_ff, d_attention_key, d_attention_value,
                 n_heads, n_attention_chunks, attention_type,
                 dropout, mode):
  """Reversible transformer decoder layer.

  Args:
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    d_attention_key: int: depth of key vector for each attention head
    d_attention_value: int: depth of value vector for each attention head
    n_heads: int: number of attention heads
    n_attention_chunks: int: number of chunks for attention
    attention_type: class: attention class to use, such as DotProductAttention.
    dropout: float: dropout rate (how much to drop out)
    mode: str: 'train' or 'eval'

  Returns:
    the layer.
  """

  pre_attention = [
      Chunk(n_sections=n_attention_chunks),  # pylint: disable=no-value-for-parameter
      tl.LayerNorm(),
      tl.Dup(), tl.Dup(),
      tl.Parallel(
          [ComputeAttentionHeads(n_heads=n_heads, d_head=d_attention_key)],
          [ComputeAttentionHeads(n_heads=n_heads, d_head=d_attention_key)],
          [ComputeAttentionHeads(n_heads=n_heads, d_head=d_attention_value)],
      ),
  ]

  attention = attention_type(mode=mode)

  # ReversibleAttentionHalfResidual requires that post_attention be linear in
  # its input (so the backward pass can be computed without knowing the input)
  post_attention = [
      ComputeAttentionOutput(n_heads=n_heads, d_model=d_model),
      Unchunk(n_sections=n_attention_chunks),  # pylint: disable=no-value-for-parameter
      BroadcastedDropout(rate=dropout, mode=mode),  # pylint: disable=no-value-for-parameter
  ]

  feed_forward = [
      FeedForward(d_model, d_ff, dropout, mode=mode),
  ]
  return [
      ReversibleAttentionHalfResidual(pre_attention, attention, post_attention),
      tl.ReversibleSwap(),
      ReversibleHalfResidual(feed_forward),
      tl.ReversibleSwap(),
  ]


def TransformerRevnetLM(vocab_size,
                        d_model=512,
                        d_ff=2048,
                        d_attention_key=64,
                        d_attention_value=64,
                        n_layers=6,
                        n_heads=8,
                        dropout=0.1,
                        max_len=2048,
                        n_chunks=32,
                        n_attention_chunks=8,
                        attention_type=DotProductAttention,
                        mode='train'):
  """Reversible transformer language model (only uses a decoder, no encoder).

  Args:
    vocab_size: int: vocab size
    d_model: int:  depth of *each half* of the two-part features
    d_ff: int: depth of feed-forward layer
    d_attention_key: int: depth of key vector for each attention head
    d_attention_value: int: depth of value vector for each attention head
    n_layers: int: number of decoder layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    max_len: int: maximum symbol length for positional encoding
    n_chunks: int: number of chunks (must match input pipeline)
    n_attention_chunks: int: number of chunks for attention
    attention_type: class: attention class to use, such as DotProductAttention.
    mode: str: 'train' or 'eval'

  Returns:
    the layer.
  """
  positional_embedder = [
      tl.Embedding(d_model, vocab_size),
      BroadcastedDropout(rate=dropout, mode=mode),  # pylint: disable=no-value-for-parameter
      tl.PositionalEncoding(max_len=max_len),
  ]
  return tl.Model(
      tl.Concatenate(n_items=n_chunks),
      tl.ShiftRight(),
      positional_embedder,
      tl.Dup(),
      tl.ReversibleSerial([
          # pylint: disable=g-complex-comprehension
          DecoderBlock(d_model, d_ff,
                       d_attention_key, d_attention_value, n_heads,
                       n_attention_chunks, attention_type,
                       dropout, mode)
          for _ in range(n_layers)
      ] + [
          SplitForOutput(n_sections=n_chunks, axis=-2),  # pylint: disable=no-value-for-parameter
      ]),
      Map([
          # TODO(kitaev): Test whether dropout should go before or after the
          # LayerNorm, and whether dropout broadcasting is needed here.
          tl.LayerNorm(),
          BroadcastedDropout(rate=dropout, mode=mode),  # pylint: disable=no-value-for-parameter
          tl.Dense(vocab_size),
          tl.LogSoftmax(),
      ], n_sections=n_chunks),
  )
