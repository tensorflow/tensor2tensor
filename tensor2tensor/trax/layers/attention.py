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

"""Attention Layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random

import jax
import numpy as onp

from tensor2tensor.trax import backend
from tensor2tensor.trax.backend import numpy as np
from tensor2tensor.trax.layers import base
from tensor2tensor.trax.layers import combinators as cb
from tensor2tensor.trax.layers import core
from tensor2tensor.trax.layers import initializers as init


# Layers are always CamelCase, but functions in general are snake_case
# pylint: disable=invalid-name


@base.layer()
def ShiftRight(x, mode='train', **unused_kwargs):
  """Layer to shift the tensor to the right by padding on axis 1."""
  if mode == 'predict':
    # Do nothing in predict mode, as then the sequence length is 1.
    return x

  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[1] = (1, 0)  # Padding on axis=1
  padded = np.pad(x, pad_widths, mode='constant',
                  constant_values=x.dtype.type(0))
  return padded[:, :-1]


@base.layer()
def CausalMask(x, params, axis=-1, **kwargs):
  del params, kwargs
  size = x.shape[axis]
  return onp.tril(onp.ones((1, size, size), dtype=onp.bool_), k=0)


@base.layer()
def PaddingMask(x, params, pad=0, **kwargs):
  del params, kwargs
  return np.reshape(x != pad, (x.shape[0], 1, 1, x.shape[-1]))


@base.layer(n_inputs=2)
def EncoderDecoderMask(x, **unused_kwargs):
  """Makes encoder-decoder mask from decoder input and a padding mask."""
  decoder_input, padding_mask = x
  padding_mask = np.reshape(
      padding_mask, (padding_mask.shape[0], 1, 1, padding_mask.shape[-1]))
  # Final mask shape is [batch, 1 for heads, decoder-len, encoder-len].
  return padding_mask + np.zeros((1, 1, decoder_input.shape[1], 1))


class PositionalEncoding(base.Layer):
  """Implements bare positional encoding."""

  def __init__(self, max_len=2048, mode='train'):
    super(PositionalEncoding, self).__init__()
    self._max_len = max_len
    self._mode = mode

  def forward(self, inputs, params=(), state=(), **kwargs):
    if self._mode in ('train', 'eval'):
      x = inputs
      symbol_size = np.shape(x)[1]
      return (x + params[:, :symbol_size, :], state)
    else:
      assert self._mode == 'predict'
      # Fast inference: return consectutive elements of the encoding sequence,
      # storing the index in state.
      return (inputs + np.expand_dims(params[:, state, :], 1), state + 1)

  def new_params_and_state(self, input_shape, input_dtype, rng):
    del input_dtype, rng
    d_feature = input_shape[-1]
    pe = onp.zeros((self._max_len, d_feature), dtype=onp.float32)
    position = onp.arange(0, self._max_len)[:, onp.newaxis]
    div_term = onp.exp(
        onp.arange(0, d_feature, 2) * -(onp.log(10000.0) / d_feature))
    pe[:, 0::2] = onp.sin(position * div_term)
    pe[:, 1::2] = onp.cos(position * div_term)
    pe = pe[onp.newaxis, :, :]  # [1, self._max_len, d_feature]
    params = np.array(pe)  # These are trainable parameters, initialized above.
    state = 0 if self._mode == 'predict' else ()
    return params, state


def DotProductAttention(query, key, value, mask, dropout, mode, rng):
  """Core dot product self-attention.

  Args:
    query: array of representations
    key: array of representations
    value: array of representations
    mask: attention-mask, gates attention
    dropout: float: dropout rate
    mode: 'eval' or 'train': whether to use dropout
    rng: JAX PRNGKey: subkey for disposable use

  Returns:
    Self attention for q, k, v arrays.
  """
  depth = np.shape(query)[-1]
  dots = np.matmul(query, np.swapaxes(key, -1, -2)) / np.sqrt(depth)
  if mask is not None:
    # TODO(kitaev): workaround for https://github.com/google/jax/issues/850
    # We must ensure that both mask and the -1e9 constant have a data dependency
    # on the input. Broadcasted copies of these use a lot of memory, so they
    # should be computed at runtime (rather than being global constants).
    if backend.get_name() == 'jax':
      mask = jax.lax.tie_in(dots, mask)
    dots = np.where(mask, dots, np.full_like(dots, -1e9))
  # Softmax.
  dots = np.exp(dots - backend.logsumexp(dots, axis=-1, keepdims=True))
  if dropout >= 1.0:
    raise ValueError('Dropout rates must be lower than 1.')
  if dropout is not None and dropout > 0.0 and mode == 'train':
    keep = backend.random.bernoulli(rng, 1.0 - dropout, dots.shape)
    dots = np.where(keep, dots / (1.0 - dropout), np.zeros_like(dots))
  out = np.matmul(dots, value)
  return out


@base.layer(n_inputs=4, n_outputs=2)
def PureAttention(x, params, n_heads=1, dropout=0.0, mode='train', **kwargs):
  """Pure transformer-style multi-headed attention.

  Args:
    x: inputs (q, k, v, mask)
    params: parameters (none)
    n_heads: int: number of attention heads
    dropout: float: dropout rate
    mode: str: 'train' or 'eval'
    **kwargs: other arguments including the rng

  Returns:
    Pure Multi-headed attention result, and the mask.
  """
  del params
  rng = kwargs.get('rng', None)
  q, k, v, mask = x
  d_feature = q.shape[-1]
  assert d_feature % n_heads == 0
  d_head = d_feature // n_heads
  nbatch = np.shape(q)[0]
  # nbatch, seqlen, d_feature --> nbatch, n_heads, seqlen, d_head
  def SplitHeads(x):
    return np.transpose(
        np.reshape(x, (nbatch, -1, n_heads, d_head)), (0, 2, 1, 3))
  # nbatch, n_heads, seqlen, d_head --> nbatch, seqlen, d_feature
  def JoinHeads(x):  # pylint: disable=invalid-name
    return np.reshape(
        np.transpose(x, (0, 2, 1, 3)), (nbatch, -1, n_heads * d_head))
  # Split heads, dot-product attention, rejoin heads.
  res = JoinHeads(
      DotProductAttention(
          SplitHeads(q), SplitHeads(k), SplitHeads(v), mask,
          dropout=dropout, mode=mode, rng=rng))
  return res, mask  # Keep the mask.


def AttentionQKV(d_feature, n_heads=1, dropout=0.0, mode='train'):
  """Transformer-style multi-headed attention.

  Accepts inputs of the form q, k, v, mask.

  Args:
    d_feature: int:  dimensionality of feature embedding
    n_heads: int: number of attention heads
    dropout: float: dropout rate
    mode: str: 'train' or 'eval'

  Returns:
    Multi-headed self-attention result and the mask.
  """
  return [
      cb.Parallel(
          core.Dense(d_feature),
          core.Dense(d_feature),
          core.Dense(d_feature),
      ),
      PureAttention(  # pylint: disable=no-value-for-parameter
          n_heads=n_heads, dropout=dropout, mode=mode),
      core.Dense(d_feature),
  ]


def Attention(d_feature, n_heads=1, dropout=0.0, mode='train'):
  """Transformer-style multi-headed attention.

  Accepts inputs of the form (x, mask) and constructs (q, k, v) from x.

  Args:
    d_feature: int:  dimensionality of feature embedding
    n_heads: int: number of attention heads
    dropout: float: dropout rate
    mode: str: 'train' or 'eval'

  Returns:
    Multi-headed self-attention result and the mask.
  """
  return [
      cb.Dup(), cb.Dup(),
      AttentionQKV(d_feature, n_heads=n_heads, dropout=dropout, mode=mode),
  ]


def BasicCausalAttention(d_feature, n_heads=1, dropout=0.0, mode='train'):
  """Transformer-style multi-headed causal attention.

  This implementation is less configurable than the CausalAttention layer
  defined below, but it shares code with the non-causal attention.

  # TODO(jonni,lukaszkaiser): standardize and improve layer comments.
  Accepts inputs of the form x and constructs (q, k, v) and causal mask from x.

  Args:
    d_feature: int:  dimensionality of feature embedding
    n_heads: int: number of attention heads
    dropout: float: dropout rate
    mode: str: 'train' or 'eval'

  Returns:
    Multi-headed self-attention result.
  """
  return [
      cb.Dup(),
      cb.Parallel([], CausalMask(axis=-2)),  # pylint: disable=no-value-for-parameter
      Attention(d_feature, n_heads=n_heads, dropout=dropout, mode=mode),
      cb.Parallel([], cb.Drop()),  # x
  ]


class ShiftRightLearned(base.Layer):
  """Layer constructor function for shifting right by a learned vector."""

  def __init__(self, initializer=init.RandomNormalInitializer(0.01)):
    super(ShiftRightLearned, self).__init__()
    self._initializer = initializer

  def forward(self, x, params=(), state=(), **kwargs):
    del kwargs
    c = backend.numpy.reshape(params, [1, 1, -1])
    c += backend.numpy.zeros((x.shape[0], 1, x.shape[2]), dtype=x.dtype)
    return backend.numpy.concatenate([c, x], axis=1)[:, :-1, :], state

  def new_params_and_state(self, input_shape, input_dtype, rng):
    del input_dtype
    b = self._initializer((input_shape[-1],), rng)
    return b, ()


class ComputeAttentionHeads(base.Layer):
  """Computes queries/keys/values via linear projection.

  The output shape is (n_batch * n_heads, seqlen, d_head); the batch and head
  dimensions are fused to allow for more efficient memory layouts.
  """

  def __init__(self, n_heads=1, d_head=64,
               kernel_initializer=init.GlorotUniformInitializer()):
    super(ComputeAttentionHeads, self).__init__()
    self._n_heads = n_heads
    self._d_head = d_head
    self._kernel_initializer = kernel_initializer
    # The lack of a bias term here is consistent with the tensor2tensor
    # implementation, and shouldn't have an effect on modeling quality.
    # Note that AttentionQKV above is different in that it uses a bias term.

  def forward(self, x, params=(), state=(), **kwargs):
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

  def new_params_and_state(self, input_shape, input_dtype, rng):
    del input_dtype
    w = self._kernel_initializer(
        (input_shape[-1], self._n_heads * self._d_head), rng)
    return w, ()


class ComputeAttentionOutput(base.Layer):
  """Joins outputs from different heads via linear projection."""

  def __init__(self, n_heads=1, d_model=1024,
               kernel_initializer=init.GlorotUniformInitializer()):
    super(ComputeAttentionOutput, self).__init__()
    self._n_heads = n_heads
    self._d_model = d_model
    self._kernel_initializer = kernel_initializer
    # The lack of a bias term here is consistent with the tensor2tensor
    # implementation, and shouldn't have an effect on modeling quality.
    # Note that AttentionQKV above is different in that it uses a bias term.

  def forward(self, x, params=(), state=(), **kwargs):
    del kwargs
    seqlen = x.shape[1]
    d_head = x.shape[2]

    x = np.reshape(x, (-1, self._n_heads, seqlen, d_head))
    x = np.transpose(x, (0, 2, 1, 3))  # -> n_batch, seqlen, n_heads, d_head
    x = np.reshape(x, (-1, seqlen, self._n_heads * d_head))

    return np.dot(x, params), state

  def new_params_and_state(self, input_shape, input_dtype, rng):
    del input_dtype
    w = self._kernel_initializer(
        (input_shape[-1] * self._n_heads, self._d_model), rng)
    return w, ()


class BaseCausalAttention(base.Layer):
  """Base class for variants of causal self-attention."""

  def __init__(self, mode='train'):
    del mode
    super(BaseCausalAttention, self).__init__(n_inputs=3)

  def forward(self, inputs, params=(), state=(), rng=None, **kwargs):
    """Forward pass for the attention layer."""
    raise NotImplementedError()

  def forward_and_backward(self, inputs, grad, **kwargs):
    """Performs both forward and backward pass for the attention layer.

    This is used in reversible models: for the backward pass of a reversible
    model, we need to compute both the forward direction (to recover the
    previous layer's activations) and the backward direction simultaneously.
    Some computation can be shared between the forward and backward directions,
    which makes it more efficient to implement them jointly.

    This method assumes that the layer is stateless and has no parameters.

    Args:
      inputs: A tuple (q, k, v), where each element has shape
          n_batch*n_heads, seqlen, d_head
      grad: gradient signal for the layer output.
      **kwargs: kwargs for the layer

    Returns:
      A nested-tuple structure (output, (q_grad, k_grad, v_grad)) that contains
      the output of the forward pass and the gradient signal for each input.
    """
    raise NotImplementedError()


def _fast_inference_init_state(input_shapes, input_dtypes, buffer_length):
  """Initializes state of a causal attention layer for fast inference."""
  ((batch_size, _, _), _, _) = input_shapes
  def init_buffer(shape, dtype):
    (_, _, depth) = shape
    return np.zeros((batch_size, buffer_length, depth), dtype=dtype)
  (_, k, v) = tuple(
      init_buffer(shape, dtype)
      for (shape, dtype) in zip(input_shapes, input_dtypes)
  )
  mask = np.zeros((batch_size, 1, buffer_length))
  index = 0
  state = (k, v, mask, index)
  return state


def _fast_inference_update_state(inputs, state):
  """Updates state of a causal attention layer for fast inference."""
  assert backend.get_name() == 'jax', (
      'JAX backend is required to use the predict mode.')
  for x in inputs:
    assert x.shape[1] == 1, (
        'In predict mode the input sequence must be of length 1.')
  # Fast inference: run with only 1 query in each step, storing the sequence
  # of keys and values calculated so far in state.
  (_, new_k, new_v) = inputs
  (ks, vs, mask, index) = state
  ks = jax.ops.index_update(ks, jax.ops.index[:, index, :], new_k[:, 0, :])
  vs = jax.ops.index_update(vs, jax.ops.index[:, index, :], new_v[:, 0, :])
  mask = jax.ops.index_update(mask, jax.ops.index[:, :, index], 1)
  return (ks, vs, mask, index + 1)


class DotProductCausalAttention(BaseCausalAttention):
  """A standard (non-memory-efficient) dot product attention implementation."""

  def __init__(self, dropout=0.0, mode='train'):
    super(DotProductCausalAttention, self).__init__()
    self._dropout = dropout
    self._mode = mode

  def forward(self, inputs, params=(), state=(), rng=None, **kwargs):
    del params
    q, k, v = inputs
    if self._mode in ('train', 'eval'):
      mask_size = q.shape[-2]
      # Not all backends define np.tril. However, using onp.tril is inefficient
      # in that it creates a large global constant. TODO(kitaev): try to find an
      # alternative that works across all backends.
      if backend.get_name() == 'jax':
        mask = np.tril(
            np.ones((1, mask_size, mask_size), dtype=onp.bool_), k=0)
      else:
        mask = onp.tril(
            onp.ones((1, mask_size, mask_size), dtype=onp.bool_), k=0)
    else:
      assert self._mode == 'predict'
      state = _fast_inference_update_state(inputs, state)
      (k, v, mask, _) = state

    res = DotProductAttention(
        q, k, v, mask, dropout=self._dropout, mode=self._mode, rng=rng)
    return res, state

  def forward_and_backward(self, inputs, ct, **kwargs):
    assert backend.get_name() == 'jax', (
        'JAX backend is required to use forward_and_backward.')
    # Simultaneous forward pass and backprop through the attention mechanism.
    def _do_forward(x):  # pylint: disable=invalid-name
      res, _ = self.forward(x, **kwargs)
      return res
    output, vjpfun = jax.vjp(_do_forward, inputs)
    return output, vjpfun(ct)[0]

  def new_params_and_state(self, input_shapes, input_dtype, rng):
    if self._mode in ('train', 'eval'):
      return (), ()

    assert self._mode == 'predict'
    params = ()
    # Buffer length is hardcoded for now. TODO(pkozakowski): Pass it from the
    # model.
    max_len = 2048
    state = _fast_inference_init_state(input_shapes, input_dtype, max_len)
    return params, state


class MemoryEfficientCausalAttention(BaseCausalAttention):
  """Memory-efficient dot product attention.

  This layer performs causal attention on long sequences without running out
  of memory. Instead of computing dot products for all query-key pairs at once,
  it uses a loop to compute attention for a small set of query positions at a
  time. The "loop_stride" parameter controls how many query positions are
  considered at each iteration of the loop.

  Note that this class does not slice along the batch/head dimension. Looping
  over batch elements and heads instead of query positions is also a viable
  option. We haven't implemented it, but it may perform well, too.
  """

  def __init__(self, loop_stride, dropout, mode, share_qk=False, hard_k=0):
    assert backend.get_name() == 'jax', (
        'JAX backend is required to use MemoryEfficientCausalAttention.')
    super(MemoryEfficientCausalAttention, self).__init__()
    self._loop_stride = loop_stride
    if dropout >= 1.0:
      raise ValueError('Dropout rates must be lower than 1.')
    if mode == 'train':
      self.dropout = dropout
    else:
      self.dropout = None
    self._share_qk = share_qk
    self._hard_k = hard_k

  def forward(self, inputs, params=(), state=(), **kwargs):
    del params
    output, _ = self.forward_and_backward(inputs, None, **kwargs)
    return output, state

  def has_backward(self):
    return True

  def backward(self, inputs, output, ct, params=(), state=(), **kwargs):
    del output, params, state
    _, inputs_ct = self.forward_and_backward(inputs, ct, **kwargs)
    return inputs_ct, ()

  def make_unit_length(self, x, epsilon=1e-6):
    variance = np.mean(x**2, axis=-1, keepdims=True)
    norm_inputs = x / np.sqrt(variance + epsilon)
    return norm_inputs

  def forward_and_backward(self, inputs, ct, rng=None, **kwargs):
    del kwargs
    query, key, value = inputs
    depth = np.shape(query)[-1]
    do_backprop = ct is not None
    # jax uses the term cotangent (ct) to refer to gradient signals, and
    # vector-Jacobian product (vjp) for back-propagation through a layer.

    def make_mask(N, M, k):  # pylint: disable=invalid-name
      """Constructs a slice of the causal attention mask.

      Args:
        N: number of query positions
        M: number of key positions
        k: position of the initial query element

      Returns:
        N x M mask, where 1.0 indicates that attention is not allowed.
      """
      x = jax.lax.tie_in(k, np.arange(N, dtype=np.int32))
      y = jax.lax.tie_in(k, np.arange(M, dtype=np.int32))
      mask = jax.lax.lt(
          (jax.lax.broadcast_in_dim(
              x, shape=(N, M), broadcast_dimensions=(0,)) + k),
          jax.lax.broadcast(y, [N]))
      mask = jax.lax.convert_element_type(mask, np.float32)
      return mask

    def make_self_mask(N, M, k):  # pylint: disable=invalid-name
      """Masks out elements attending to self.

      Args:
        N: number of query positions
        M: number of key positions
        k: position of the initial query element

      Returns:
        N x M mask, where 1.0 indicates that attention is not allowed.
      """
      x = jax.lax.tie_in(k, np.arange(N, dtype=np.int32))
      y = jax.lax.tie_in(k, np.arange(M, dtype=np.int32))
      mask = jax.lax.eq(
          (jax.lax.broadcast_in_dim(
              x, shape=(N, M), broadcast_dimensions=(0,)) + k),
          jax.lax.broadcast(y, [N]))
      mask = jax.lax.convert_element_type(mask, np.float32)
      return mask

    def forward_slice(query_slice, q_loop_idx, key, value):  # pylint: disable=invalid-name
      """Forward pass for a subset of the query vectors."""
      if self._share_qk:
        key = self.make_unit_length(key)

      dots = np.matmul(
          query_slice, np.swapaxes(key, -1, -2)) / np.sqrt(depth)

      # Causal masking
      mask = make_mask(dots.shape[-2], dots.shape[-1], q_loop_idx)
      dots = dots - 1e9 * mask

      # Mask out attention to self except when no other targets are available.
      if self._share_qk:
        self_mask = make_self_mask(dots.shape[-2], dots.shape[-1], q_loop_idx)
        dots = dots - 1e5 * self_mask

      # Softmax.
      dots = np.exp(dots - backend.logsumexp(dots, axis=-1, keepdims=True))

      if self.dropout is not None and self.dropout > 0.0:
        # Dropout is broadcast across the batch+head dimension
        dropout_shape = (1, dots.shape[-2], dots.shape[-1])
        slice_rng = jax.random.fold_in(rng, q_loop_idx)
        keep_prob = jax.lax.tie_in(dots, 1.0 - self.dropout)
        keep = backend.random.bernoulli(slice_rng, keep_prob, dropout_shape)
        multiplier = keep.astype(dots.dtype) / jax.lax.tie_in(keep, keep_prob)
        dots = dots * multiplier

      if self._hard_k > 0:
        top_k = np.sort(dots)[..., -self._hard_k]  # Get the top-kth weight.
        top_k = jax.lax.stop_gradient(top_k)
        dots -= top_k[..., np.newaxis]  # Subtract (be 0 for lower ones).
        dots = np.maximum(dots, 0)
        dots_sum = np.sum(dots, axis=-1, keepdims=True)  # Re-normalize.
        dots /= dots_sum  # Re-normalize.

      out_slice = np.matmul(dots, value)
      return out_slice

    def forward_and_vjp_slice(query_slice, q_loop_idx, key, value, ct_slice):  # pylint: disable=invalid-name
      # Capture q_loop_idx to avoid calculated gradients wrt. it.
      def forward_slice_with_q_loop_idx(query_slice, key, value):  # pylint: disable=invalid-name
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

    def cond_fun(vals):  # pylint: disable=invalid-name
      q_loop_idx = vals[0]
      return jax.lax.lt(q_loop_idx, q_loop_max)

    def body_fun(vals):  # pylint: disable=invalid-name
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


class TimeBinCausalAttention(BaseCausalAttention):
  """Causal attention where only nearby chunks of items attend to each other."""

  def __init__(self, mode, dropout=0.0, bin_length=None, n_bins=None,
               share_qk=False):
    super(TimeBinCausalAttention, self).__init__()
    if (bin_length is None) == (n_bins is None):
      raise ValueError('Exactly one of {bin_length, n_bins} must be set.')
    self.bin_length = bin_length
    self.n_bins = n_bins
    self._share_qk = share_qk
    if dropout >= 1.0:
      raise ValueError('Dropout rates must be lower than 1.')
    if mode == 'train':
      self.dropout = dropout
    else:
      self.dropout = 0.0
    self._mode = mode

  def forward_and_backward(self, inputs, ct, **kwargs):
    assert backend.get_name() == 'jax', (
        'JAX backend is required to use forward_and_backward.')
    # Simultaneous forward pass and backprop through the attention mechanism.
    def _do_forward(x):  # pylint: disable=invalid-name
      res, _ = self.forward(x, **kwargs)
      return res
    output, vjpfun = jax.vjp(_do_forward, inputs)
    return output, vjpfun(ct)[0]

  def make_unit_length(self, x, epsilon=1e-6):
    variance = np.mean(x**2, axis=-1, keepdims=True)
    norm_inputs = x / np.sqrt(variance + epsilon)
    return norm_inputs

  def _pad_inputs(self, inputs):
    seq_len = inputs[0].shape[-2]
    n_bins = self.n_bins
    bin_length = self.bin_length
    if n_bins is None:
      n_bins = int(math.ceil(seq_len / bin_length))
    else:
      bin_length = int(math.ceil(seq_len / n_bins))
    pad_len = n_bins * bin_length - seq_len

    def pad_input(x):
      pad_widths = [(0, 0)] * len(x.shape)
      pad_widths[-2] = (0, pad_len)  # Padding on axis=-2
      return np.pad(x, pad_widths, mode='constant',
                    constant_values=x.dtype.type(0))

    padded_inputs = tuple(map(pad_input, inputs))
    return (padded_inputs, seq_len, n_bins)

  def forward(self, inputs, params=(), state=(), rng=None, **kwargs):
    del params, kwargs
    if self._mode in ('train', 'eval'):
      output = self._forward_train_eval(inputs, rng)
      return (output, state)
    else:
      assert self._mode == 'predict'
      return self._forward_predict(inputs, state, rng)

  def _forward_train_eval(self, inputs, rng):
    (inputs, original_len, n_bins) = self._pad_inputs(inputs)
    q, k, v = inputs
    seqlen = q.shape[-2]
    # q/k/v are n_batch*n_heads, seqlen, d_head
    # Time indices for causal masking.
    t = jax.lax.tie_in(q, np.arange(seqlen))

    # Split off a "bin" axis for chunks of consecutive items.
    bq_t = np.reshape(t, (n_bins, -1))
    bq = np.reshape(q, (q.shape[0], n_bins, -1, q.shape[-1]))
    if self._share_qk:
      bk = self.make_unit_length(bq)
    else:
      bk = np.reshape(k, (k.shape[0], n_bins, -1, k.shape[-1]))
    bv = np.reshape(v, (v.shape[0], n_bins, -1, v.shape[-1]))

    # Allow each chunk to attend within itself, and also one chunk back.
    def look_one_back(x):
      # Output: pairs [ bin_i bin_{i-1} ] concatenated on the time axis.
      if len(x.shape) == 2:
        x_extra = np.concatenate([x[-1:, :], x[:-1, :]], axis=0)
        return np.concatenate([x, x_extra], axis=1)
      else:
        assert len(x.shape) == 4
        x_extra = np.concatenate([x[:, -1:, :, :], x[:, :-1, :, :]], axis=1)
        return np.concatenate([x, x_extra], axis=2)

    bkv_t = look_one_back(bq_t)
    bk = look_one_back(bk)
    bv = look_one_back(bv)

    # Dot-product attention.
    dots = np.matmul(bq, np.swapaxes(bk, -1, -2)) / np.sqrt(bq.shape[-1])

    # Causal masking based on the time indices.
    mask = jax.lax.convert_element_type(
        jax.lax.lt(bq_t[None, :, :, None], bkv_t[None, :, None, :]),
        np.float32)
    dots = dots - 1e9 * mask

    # Mask out attention to self except when no other targets are available.
    if self._share_qk:
      self_mask = jax.lax.broadcasted_eye(dots.dtype, dots.shape, (2, 3))
      self_mask = jax.lax.tie_in(dots, self_mask)
      dots = dots - 1e5 * self_mask

    if self.dropout > 0.0:
      # Dropout is broadcast across the batch+head dimension
      dropout_shape = (1, dots.shape[-3], dots.shape[-2], dots.shape[-1])
      keep_prob = jax.lax.tie_in(dots, 1.0 - self.dropout)
      keep = backend.random.bernoulli(rng, keep_prob, dropout_shape)
      multiplier = keep.astype(dots.dtype) / jax.lax.tie_in(keep, keep_prob)
      dots = dots * multiplier

    # Softmax.
    dots = np.exp(dots - backend.logsumexp(dots, axis=-1, keepdims=True))
    bo = np.matmul(dots, bv)

    output = np.reshape(bo, (bo.shape[0], -1, bo.shape[-1]))
    assert output.shape == v.shape
    return output[..., :original_len, :]

  def _forward_predict(self, inputs, state, rng):
    state = _fast_inference_update_state(inputs, state)

    (q, _, _) = inputs
    (ks, vs, mask, index) = state
    output = DotProductAttention(
        q, ks, vs, mask, dropout=self.dropout, mode=self._mode, rng=rng
    )

    def roll_state(state):
      """Rolls the buffers backward to make space for new data."""
      (ks, vs, mask, index) = state
      # Move the second bin into the first one's place in both buffers.
      def roll_buffer(buf):
        return jax.ops.index_update(
            buf,
            jax.ops.index[:, :self.bin_length, :],
            buf[:, self.bin_length:, :],
        )
      (ks, vs) = map(roll_buffer, (ks, vs))
      # Zero out the second bin in the mask.
      mask = jax.ops.index_update(
          mask, jax.ops.index[:, :, self.bin_length:], 0
      )
      # Update the index to match the rolled buffers.
      index -= self.bin_length
      return (ks, vs, mask, index)

    # Once we get to the end of the buffer, move the second bin back to make
    # space for new data: [ bin_i bin_{i+1} | ] -> [ bin_{i+1} | bin_{i+1} ],
    # where | is where index points at in the buffer.
    state = jax.lax.cond(
        pred=(index == 2 * self.bin_length),
        true_operand=state,
        true_fun=roll_state,
        false_operand=state,
        false_fun=(lambda x: x),
    )
    return (output, state)

  def new_params_and_state(self, input_shapes, input_dtype, rng):
    if self._mode in ('train', 'eval'):
      return (), ()

    assert self._mode == 'predict'
    assert self.bin_length is not None, (
        'For fast inference, TimeBinCausalAttention must be parameterized by '
        'bin_length.'
    )
    params = ()
    state = _fast_inference_init_state(
        input_shapes, input_dtype, 2 * self.bin_length
    )
    return params, state


class LSHCausalAttention(BaseCausalAttention):
  """Causal attention based on locality-sensitive hashing."""

  def __init__(self, dropout, mode, n_bins=64, n_hashes=1, n_buckets=64,
               one_rng=False, allow_duplicate_attention=False,
               attend_across_buckets=False, hard_k=0,
               rehash_each_round=True, drop_for_hash_rate=0.0):
    del dropout
    self._mode = mode
    super(LSHCausalAttention, self).__init__()
    assert n_buckets >= n_bins, 'This setting is not recommended: too few bins.'
    assert rehash_each_round or allow_duplicate_attention, (
        'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
        ' is not implemented.')
    self.n_bins = n_bins
    self.n_hashes = n_hashes
    self.n_buckets = n_buckets
    self._drop_for_hash_rate = drop_for_hash_rate
    self._one_rng = one_rng
    self._prng = None
    if one_rng:
      seed = random.randint(0, 2**31 - 1)
      self._prng = backend.random.get_prng(seed)

    self._allow_duplicate_attention = allow_duplicate_attention
    self._attend_across_buckets = attend_across_buckets
    self._hard_k = hard_k
    self._rehash_each_round = rehash_each_round

  def forward(self, inputs, params=(), state=(), rng=None, **kwargs):
    del params, kwargs
    output, _ = self.batch_call_and_or_grad(inputs[0], inputs[2], rng=rng)
    return output, state

  def forward_and_backward(self, inputs, ct, rng=None, **kwargs):
    del kwargs
    output, (qk_ct, v_ct) = self.batch_call_and_or_grad(
        inputs[0], inputs[2], ct=ct, rng=rng)
    return output, (qk_ct, np.zeros_like(inputs[1]), v_ct)

  def has_backward(self):
    return True

  def backward(self, inputs, output, ct, params=(), state=(), rng=None,
               **kwargs):
    del output, params, state
    _, (qk_ct, v_ct) = self.batch_call_and_or_grad(
        inputs[0], inputs[2], return_output=False, ct=ct, rng=rng)
    inputs_ct = (qk_ct, np.zeros_like(inputs[1]), v_ct)
    return inputs_ct, ()

  def batch_call_and_or_grad(self, qk, v, ct=None, return_output=True,
                             rng=None):
    assert return_output or ct is not None, 'No work to perform!'
    # pylint: disable=protected-access
    stash_buckets = (return_output and ct is None
                     and base.Layer._STASH_IN is not None)
    if return_output and ct is not None and base.Layer._STASH_OUT is not None:
      buckets = base.Layer._STASH_OUT.pop(self)
    else:
      buckets = None
    # pylint: enable=protected-access

    # The approach here is to perform attention for one batch element and head
    # at a time. Note that there is absolutely no interaction across examples or
    # heads: this layer has no parameters, and hashing patterns are also
    # different across examples/heads. As a result, batching doesn't give any
    # performance gains except in the case of accelerator under-utilization. We
    # assume that hash-based attention will be applied primarily to long
    # sequences, where unbatched attention for a single head has sufficient
    # computation to fill up the accelerator.

    batch_loop_idx = np.zeros((), dtype=np.int32)
    batch_loop_max = qk.shape[0]

    init_vals = (batch_loop_idx,)
    if return_output:
      out_accum = np.zeros_like(qk)
      init_vals = init_vals + (out_accum,)
    if stash_buckets:
      buckets_accum = np.zeros(
          [qk.shape[0], self.n_hashes * qk.shape[1]], dtype=np.int32)
      init_vals = init_vals + (buckets_accum,)
    if ct is not None:
      qk_ct_accum = np.zeros_like(qk)
      v_ct_accum = np.zeros_like(v)
      init_vals = init_vals + (qk_ct_accum, v_ct_accum)

    def cond_fun(vals):
      batch_loop_idx = vals[0]
      return jax.lax.lt(batch_loop_idx, batch_loop_max)

    def body_fun(vals):
      """Performs attention for a single batch element and head."""
      batch_loop_idx = vals[0]
      if self._prng is None:
        hash_rng = jax.random.fold_in(rng, batch_loop_idx)
      else:
        # TODO(kitaev): Maybe use the same RNG across examples (but not heads)?
        hash_rng = jax.random.fold_in(self._prng, batch_loop_idx)
      qk_slice = jax.lax.dynamic_index_in_dim(
          qk, batch_loop_idx, axis=0, keepdims=False)
      v_slice = jax.lax.dynamic_index_in_dim(
          v, batch_loop_idx, axis=0, keepdims=False)

      if buckets is None:
        buckets_slice = self.hash_vectors(qk_slice, rng=hash_rng)
      else:
        buckets_slice = jax.lax.dynamic_index_in_dim(
            buckets, batch_loop_idx, axis=0, keepdims=False)

      if ct is None:
        out_slice = self.single_call(
            qk_slice, v_slice, buckets_slice, hash_rng=hash_rng)
      else:
        def _do_single_call(qk_slice, v_slice):
          return self.single_call(
              qk_slice, v_slice, buckets_slice, hash_rng=hash_rng)
        ct_slice = jax.lax.dynamic_index_in_dim(
            ct, batch_loop_idx, axis=0, keepdims=False)
        out_slice, vjpfun = jax.vjp(_do_single_call, qk_slice, v_slice)
        qk_ct_slice, v_ct_slice = vjpfun(ct_slice)

      new_vals = (batch_loop_idx + 1,)
      if return_output:
        out_accum = vals[1]
        out_accum = jax.lax.dynamic_update_index_in_dim(
            out_accum, out_slice, batch_loop_idx, axis=0)
        new_vals = new_vals + (out_accum,)
      if stash_buckets:
        buckets_accum = vals[2]
        buckets_accum = jax.lax.dynamic_update_index_in_dim(
            buckets_accum, buckets_slice, batch_loop_idx, axis=0)
        new_vals = new_vals + (buckets_accum,)
      if ct is not None:
        qk_ct_accum, v_ct_accum = vals[-2:]
        qk_ct_accum = jax.lax.dynamic_update_index_in_dim(
            qk_ct_accum, qk_ct_slice, batch_loop_idx, axis=0)
        v_ct_accum = jax.lax.dynamic_update_index_in_dim(
            v_ct_accum, v_ct_slice, batch_loop_idx, axis=0)
        new_vals = new_vals + (qk_ct_accum, v_ct_accum)

      return new_vals

    final_vals = jax.lax.while_loop(cond_fun, body_fun, init_vals)

    if return_output:
      out = final_vals[1]
    else:
      out = None

    if stash_buckets:
      base.Layer._STASH_IN[self] = final_vals[2]  # pylint: disable=protected-access

    if ct is not None:
      input_ct = final_vals[-2:]
    else:
      input_ct = None

    return out, input_ct

  def make_unit_length(self, x, epsilon=1e-6):
    variance = np.mean(x**2, axis=-1, keepdims=True)
    norm_inputs = x / np.sqrt(variance + epsilon)
    return norm_inputs

  def drop_for_hash(self, x, rng):
    rate = self._drop_for_hash_rate
    if self._mode == 'train' and rate > 0.0:
      keep = backend.random.bernoulli(rng, 1.0 - rate, x.shape)
      return np.where(keep, x / (1.0 - rate), np.zeros_like(x))
    return x

  def hash_vectors(self, vecs, rng):
    # See https://arxiv.org/pdf/1509.02897.pdf
    # We sample a different random rotation for each round of hashing to
    # decrease the probability of hash misses.
    assert self.n_buckets % 2 == 0
    random_rotations_shape = (
        vecs.shape[-1],
        self.n_hashes if self._rehash_each_round else 1,
        self.n_buckets // 2)

    rng = jax.lax.tie_in(vecs, rng)
    rng, subrng = backend.random.split(rng)
    random_rotations = jax.random.normal(
        rng, random_rotations_shape).astype('float32')
    # TODO(lukaszkaiser): the dropout mask will be used for all rounds of
    # hashing, so it's shared between them. Check if that's what we want.
    dropped_vecs = self.drop_for_hash(vecs, subrng)
    rotated_vecs = np.einsum('tf,fhb->htb', dropped_vecs, random_rotations)
    rotated_vecs = np.concatenate([rotated_vecs, -rotated_vecs], axis=-1)

    if self._rehash_each_round:
      buckets = np.argmax(rotated_vecs, axis=-1)
      # buckets is now (self.n_hashes, seqlen). Next we add offsets so that
      # bucket numbers from different hashing rounds don't overlap.
      offsets = jax.lax.tie_in(buckets, np.arange(self.n_hashes))
      offsets = np.reshape(offsets * self.n_buckets, (-1, 1))
      buckets = np.reshape(buckets + offsets, (-1,))
    else:
      # In this configuration, we map each item to the top self.n_hashes buckets
      rotated_vecs = np.squeeze(rotated_vecs, 0)
      bucket_range = jax.lax.tie_in(vecs, np.arange(rotated_vecs.shape[-1]))
      bucket_range = np.reshape(bucket_range, (1, -1))
      bucket_range = np.broadcast_to(bucket_range, rotated_vecs.shape)

      _, buckets = jax.lax.sort_key_val(
          rotated_vecs, bucket_range, dimension=-1)
      buckets = buckets[:, -self.n_hashes:]
      buckets = np.reshape(np.moveaxis(buckets, 0, -1), (-1,))

    return buckets

  def single_call(self, qk, v, buckets, hash_rng=None):
    # We use the same vector as both a query and a key.
    seqlen = qk.shape[-2]
    assert int(buckets.shape[0]) == self.n_hashes * seqlen

    ticker = jax.lax.tie_in(qk, np.arange(self.n_hashes * seqlen))
    buckets_and_t = seqlen * buckets + (ticker % seqlen)
    buckets_and_t = jax.lax.stop_gradient(buckets_and_t)

    # Hash-based sort ("s" at the start of variable names means "sorted")
    sbuckets_and_t, sticker = jax.lax.sort_key_val(
        buckets_and_t, ticker, dimension=-1)
    _, undo_sort = jax.lax.sort_key_val(sticker, ticker, dimension=-1)
    sbuckets_and_t = jax.lax.stop_gradient(sbuckets_and_t)
    sticker = jax.lax.stop_gradient(sticker)
    undo_sort = jax.lax.stop_gradient(undo_sort)

    st = (sticker % seqlen)
    sqk = np.take(qk, st, axis=0)
    sv = np.take(v, st, axis=0)

    # Split off a "bin" axis so that attention only occurs within chunks.
    bq_t = bkv_t = np.reshape(st, (self.n_hashes * self.n_bins, -1))
    bqk = np.reshape(sqk, (self.n_hashes * self.n_bins, -1, sqk.shape[-1]))
    bv = np.reshape(sv, (self.n_hashes * self.n_bins, -1, sv.shape[-1]))
    bq_buckets = bkv_buckets = np.reshape(
        sbuckets_and_t // seqlen, (self.n_hashes * self.n_bins, -1))

    # Hashing operates on unit-length vectors. Unnormalized query vectors are
    # fine because they effectively provide a learnable temperature for the
    # attention softmax, but normalizing keys is needed so that similarity for
    # the purposes of attention correctly corresponds to hash locality.
    bq = bqk
    bk = self.make_unit_length(bqk)

    # Allow each chunk to attend within itself, and also one chunk back. Chunk
    # boundaries might occur in the middle of a sequence of items from the
    # same bucket, so this increases the chances of attending to relevant items.
    # TODO(kitaev): benchmark whether XLA pad operation is noticeably faster.
    def look_one_back(x):
      if len(x.shape) == 2:
        x_extra = np.concatenate([x[-1:, :], x[:-1, :]], axis=0)
      else:
        x_extra = np.concatenate([x[-1:, :, :], x[:-1, :, :]], axis=0)
      return np.concatenate([x, x_extra], axis=1)

    bk = look_one_back(bk)
    bv = look_one_back(bv)
    bkv_t = look_one_back(bkv_t)
    bkv_buckets = look_one_back(bkv_buckets)

    # Dot-product attention.
    dots = np.matmul(bq, np.swapaxes(bk, -1, -2)) / np.sqrt(bq.shape[-1])

    # Causal masking
    mask = jax.lax.convert_element_type(
        jax.lax.lt(bq_t[:, :, None], bkv_t[:, None, :]),
        np.float32)
    dots = dots - 1e9 * mask

    # Mask out attention to self except when no other targets are available.
    self_mask = jax.lax.convert_element_type(
        jax.lax.eq(bq_t[:, :, None], bkv_t[:, None, :]),
        np.float32)
    dots = dots - 1e5 * self_mask

    # Mask out attention to other hash buckets.
    if not self._attend_across_buckets:
      bucket_mask = jax.lax.convert_element_type(
          jax.lax.ne(bq_buckets[:, :, None], bkv_buckets[:, None, :]),
          np.float32)
      dots = dots - 1e7 * bucket_mask

    # Don't double-count query-key pairs across multiple rounds of hashing.
    # There are two possible strategies here. (1) The default is to count how
    # many times a query-key pair is repeated, and to lower its log-prob
    # correspondingly at each repetition. (2) When hard_k is set, the code
    # instead masks all but the first occurence of each query-key pair.
    # TODO(kitaev): is one strategy faster or more numerically stable?
    if not self._allow_duplicate_attention:
      locs1 = undo_sort // bq_t.shape[-1]
      locs2 = (locs1 + 1) % (self.n_hashes * self.n_bins)
      if not self._attend_across_buckets:
        locs1 = buckets * (self.n_hashes * self.n_bins) + locs1
        locs2 = buckets * (self.n_hashes * self.n_bins) + locs2
      locs = np.moveaxis(np.concatenate([
          np.reshape(locs1, (self.n_hashes, seqlen)),
          np.reshape(locs2, (self.n_hashes, seqlen)),
      ], 0), 0, -1)  # produces shape (seqlen, 2 * self.n_hashes)
      slocs = np.take(locs, st, axis=0)
      b_locs = np.reshape(
          slocs, (self.n_hashes * self.n_bins, -1, 2 * self.n_hashes))
      # Queries always use the primary location (based on locs1).
      b_locs1 = b_locs[:, :, None, :self.n_hashes]
      if self._hard_k > 0:
        range_n_hashes = jax.lax.tie_in(b_locs, np.arange(self.n_hashes))
        nouse_locs = (range_n_hashes[:, None] > range_n_hashes[None, :])
        nouse_locs = 2 * nouse_locs - 1  # 1 = use, -1 = don't use
        nouse_locs = np.reshape(
            np.broadcast_to(nouse_locs[:, None, :],
                            (self.n_hashes, self.n_bins, self.n_hashes)),
            (self.n_hashes * self.n_bins, 1, 1, self.n_hashes))
        b_locs1 = b_locs1 * nouse_locs
      bq_locs = np.broadcast_to(
          b_locs1,
          b_locs.shape[:2] + (2, self.n_hashes))
      bq_locs = np.reshape(bq_locs, b_locs.shape)
      bkv_locs = look_one_back(b_locs)

      dup_counts = np.sum(
          jax.lax.convert_element_type(
              jax.lax.eq(bq_locs[:, :, None, :], bkv_locs[:, None, :, :]),
              np.float32),
          axis=-1)
      assert dup_counts.shape == dots.shape
      if self._hard_k > 0:
        dots = dots - 1e7 * jax.lax.stop_gradient(dup_counts)
      else:
        dots = dots - jax.lax.stop_gradient(np.log(dup_counts + 1e-9))

    # Each query only attends to the top k most relevant keys.
    if self._hard_k > 0:
      b_top_dots = np.sort(dots)[..., -self._hard_k:]  # Get the top k dots.
      b_top_dots = jax.lax.stop_gradient(b_top_dots)
      s_top_dots = np.reshape(b_top_dots, (-1, self._hard_k))
      top_dots = np.take(s_top_dots, undo_sort, axis=0)

      merged_top_dots = np.moveaxis(
          np.reshape(top_dots, (self.n_hashes, seqlen, self._hard_k)), 0, -1)
      merged_top_dots = np.reshape(merged_top_dots, (seqlen, -1))

      dots_thresh = np.sort(merged_top_dots)[:, -self._hard_k]
      # It's possible to compute the partition function at this point, but right
      # now this codepath isn't set up for backprop, and there might also be
      # issues computing it this way if two dot-products are exactly equal.

      sdots_thresh = dots_thresh[st]
      bdots_thresh = np.reshape(sdots_thresh, (self.n_hashes * self.n_bins, -1))
      bdots_thresh = jax.lax.stop_gradient(bdots_thresh)

      top_k_mask = jax.lax.convert_element_type(
          dots < bdots_thresh[..., None], np.float32)
      dots = dots - 1e7 * jax.lax.stop_gradient(top_k_mask)

    # Softmax.
    dots_logsumexp = backend.logsumexp(dots, axis=-1, keepdims=True)
    dots = np.exp(dots - dots_logsumexp)

    bo = np.matmul(dots, bv)
    so = np.reshape(bo, (-1, bo.shape[-1]))
    slogits = np.reshape(dots_logsumexp, (-1,))

    def unsort_for_output_impl(so, slogits):
      o = np.take(so, undo_sort, axis=0)
      # Sorting is considerably faster than gather, but first we need to get the
      # XLA compiler to abandon the idea of fusing this sort with the input sort
      # (which introduces a computation cycle and leads to a crash).
      # TODO(kitaev): remove "sticker_" variable if XLA is fixed.
      sticker_ = sticker + jax.lax.convert_element_type(
          slogits[0] > 0, sticker.dtype)
      _, logits = jax.lax.sort_key_val(sticker_, slogits, dimension=-1)
      return o, logits

    def unsort_for_output_vjp(so, slogits):
      """Custom gradient for unsort_for_output."""
      so = jax.lax.stop_gradient(so)
      slogits = jax.lax.stop_gradient(slogits)
      o, logits = unsort_for_output_impl(so, slogits)
      def vjpfun(o_logits_grads):
        so_grad = np.take(o_logits_grads[0], sticker, axis=0)
        # TODO(kitaev): this exists to match the forward pass, but I'm not sure
        # if it's actually required.
        buckets_and_t_ = buckets_and_t + jax.lax.convert_element_type(
            o_logits_grads[1][0] > 0, buckets_and_t.dtype)
        _, slogits_grad = jax.lax.sort_key_val(
            buckets_and_t_, o_logits_grads[1], dimension=-1)
        return (so_grad, slogits_grad)
      return (o, logits), vjpfun

    unsort_for_output = jax.custom_transforms(unsort_for_output_impl)
    jax.defvjp_all(unsort_for_output, unsort_for_output_vjp)
    o, logits = unsort_for_output_impl(so, slogits)

    if self.n_hashes == 1:
      out = o
    else:
      o = np.reshape(o, (self.n_hashes, seqlen, o.shape[-1]))
      logits = np.reshape(logits, (self.n_hashes, seqlen, 1))
      probs = np.exp(logits - backend.logsumexp(logits, axis=0, keepdims=True))
      out = np.sum(o * probs, axis=0)

    assert out.shape == v.shape
    return out


def CausalAttention(d_feature, n_heads=1,
                    d_attention_key=None, d_attention_value=None,
                    attention_type=DotProductCausalAttention,
                    share_qk=False, mode='train'):
  """Transformer-style multi-headed causal attention.

  Args:
    d_feature: int:  dimensionality of feature embedding
    n_heads: int: number of attention heads
    d_attention_key: int: depth of key vector for each attention head
        (default is d_feature // n_heads)
    d_attention_value: int: depth of value vector for each attention head
        (default is d_feature // n_heads)
    attention_type: subclass of BaseCausalAttention: attention class to use
    share_qk: bool, whether to share queries and keys
    mode: str: 'train' or 'eval'

  Returns:
    Multi-headed self-attention result.
  """
  if d_attention_key is None:
    assert d_feature % n_heads == 0
    d_attention_key = d_feature // n_heads
  if d_attention_value is None:
    assert d_feature % n_heads == 0
    d_attention_value = d_feature // n_heads

  if share_qk:
    pre_attention = [
        cb.Dup(),
        cb.Parallel(
            ComputeAttentionHeads(n_heads=n_heads, d_head=d_attention_key),
            ComputeAttentionHeads(n_heads=n_heads, d_head=d_attention_value),
        ),
        cb.Dup(),
    ]
  else:
    pre_attention = [
        cb.Dup(), cb.Dup(),
        cb.Parallel(
            ComputeAttentionHeads(n_heads=n_heads, d_head=d_attention_key),
            ComputeAttentionHeads(n_heads=n_heads, d_head=d_attention_key),
            ComputeAttentionHeads(n_heads=n_heads, d_head=d_attention_value),
        ),
    ]

  return pre_attention + [
      attention_type(mode=mode),
      ComputeAttentionOutput(n_heads=n_heads, d_model=d_feature),
  ]
