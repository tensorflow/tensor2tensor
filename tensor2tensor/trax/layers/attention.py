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
def ShiftRight(x, **unused_kwargs):
  """Layer to shift the tensor to the right by padding on axis 1."""
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


# Positional encoding.
def _positional_encoding_new_params(  # pylint: disable=invalid-name
    input_shape, input_dtype, rng, max_len=2048):
  """Helper: create positional encoding parameters."""
  del input_dtype, rng
  d_feature = input_shape[-1]
  pe = onp.zeros((max_len, d_feature), dtype=onp.float32)
  position = onp.arange(0, max_len)[:, onp.newaxis]
  div_term = onp.exp(
      onp.arange(0, d_feature, 2) * -(onp.log(10000.0) / d_feature))
  pe[:, 0::2] = onp.sin(position * div_term)
  pe[:, 1::2] = onp.cos(position * div_term)
  pe = pe[onp.newaxis, :, :]  # [1, max_len, d_feature]
  return np.array(pe)  # These are trainable parameters, initialized as above.


@base.layer(new_parameters=_positional_encoding_new_params)
def PositionalEncoding(x, params, **unused_kwargs):
  """Implements bare positional encoding."""
  symbol_size = np.shape(x)[1]
  return x + params[:, :symbol_size, :]


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

  def call(self, x, params, state, **kwargs):
    del kwargs
    c = backend.numpy.reshape(params, [1, 1, -1])
    c += backend.numpy.zeros((x.shape[0], 1, x.shape[2]), dtype=x.dtype)
    return backend.numpy.concatenate([c, x], axis=1)[:, :-1, :], state

  def new_parameters(self, input_shape, input_dtype, rng):
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


class BaseCausalAttention(base.Layer):
  """Base class for variants of causal self-attention."""

  def __init__(self):
    super(BaseCausalAttention, self).__init__(n_inputs=3)

  def call(self, inputs, params=(), state=(), rng=None, **kwargs):
    """Forward pass for the attention layer."""
    raise NotImplementedError()

  def call_and_grad(self, inputs, grad, **kwargs):
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

  def new_parameters(self, input_shapes, input_dtype, rng):
    return (), ()


class DotProductCausalAttention(BaseCausalAttention):
  """A standard (non-memory-efficient) dot product attention implementation."""

  def __init__(self, dropout=0.0, mode='train'):
    super(DotProductCausalAttention, self).__init__()
    self._dropout = dropout
    self._mode = mode

  def call(self, inputs, params=(), state=(), rng=None, **kwargs):
    del params
    q, k, v = inputs
    mask_size = q.shape[-2]
    # Not all backends define np.tril. However, using onp.tril is inefficient in
    # that it creates a large global constant. TODO(kitaev): try to find an
    # alternative that works across all backends.
    if backend.get_name() == 'jax':
      mask = np.tril(np.ones((1, mask_size, mask_size), dtype=onp.bool_), k=0)
    else:
      mask = onp.tril(onp.ones((1, mask_size, mask_size), dtype=onp.bool_), k=0)
    res = DotProductAttention(
        q, k, v, mask, dropout=self._dropout, mode=self._mode, rng=rng)
    return res, state

  def call_and_grad(self, inputs, ct, **kwargs):
    assert backend.get_name() == 'jax', (
        'JAX backend is required to use call_and_grad.')
    # Simultaneous forward pass and backprop through the attention mechanism.
    def do_call(x):  # pylint: disable=invalid-name
      res, _ = self.call(x, **kwargs)
      return res
    output, vjpfun = jax.vjp(do_call, inputs)
    return output, vjpfun(ct)[0]


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

  def __init__(self, loop_stride, dropout, mode):
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

  def call(self, inputs, params=(), state=(), **kwargs):
    del params
    output, _ = self.call_and_grad(inputs, None, **kwargs)
    return output, state

  def has_custom_grad(self):
    return True

  def custom_grad(self, inputs, output, ct, params=(), state=(), **kwargs):
    del output, params, state
    _, inputs_ct = self.call_and_grad(inputs, ct, **kwargs)
    return inputs_ct, ()

  def call_and_grad(self, inputs, ct, rng=None, **kwargs):
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
      x = np.arange(N, dtype=np.int32)
      y = np.arange(M, dtype=np.int32)
      mask = jax.lax.lt(
          (jax.lax.broadcast_in_dim(
              x, shape=(N, M), broadcast_dimensions=(0,)) + k),
          jax.lax.broadcast(y, [N]))
      mask = jax.lax.convert_element_type(mask, np.float32)
      return mask

    def forward_slice(query_slice, q_loop_idx, key, value):  # pylint: disable=invalid-name
      """Forward pass for a subset of the query vectors."""
      dots = np.matmul(
          query_slice, np.swapaxes(key, -1, -2)) / np.sqrt(depth)

      # Causal masking
      mask = make_mask(dots.shape[-2], dots.shape[-1], q_loop_idx)
      dots = dots - 1e9 * mask

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


class MergedHashedCausalAttention(BaseCausalAttention):
  """Hash-based causal attention."""

  def __init__(self, dropout, mode, n_bins=64, bin_by_time=False):
    del dropout, mode
    super(MergedHashedCausalAttention, self).__init__()
    self.n_bins = n_bins
    self.bin_by_time = bin_by_time

  def call(self, inputs, params=(), state=(), **kwargs):
    del params
    output, _ = self.call_and_grad(inputs, None, **kwargs)
    return output, state

  def has_custom_grad(self):
    return True

  def custom_grad(self, inputs, output, ct, params=(), state=(), **kwargs):
    del output, params, state
    _, inputs_ct = self.call_and_grad(inputs, ct, **kwargs)
    return inputs_ct, ()

  def bin_vectors_by_time(self, vecs):
    seqlen = vecs.shape[-2]
    assert seqlen % self.n_bins == 0
    bin_size = int(seqlen // self.n_bins)

    bins = np.arange(seqlen, dtype=np.int32) // bin_size
    bins = jax.lax.tie_in(vecs, bins)
    bins = bins[None, :]
    bins = np.broadcast_to(bins, vecs.shape[:-1])
    return bins

  def make_unit_length(self, x, epsilon=1e-6):
    variance = np.mean(x**2, axis=-1, keepdims=True)
    norm_inputs = x / np.sqrt(variance + epsilon)
    return norm_inputs

  def hash_vectors(self, vecs, rng):
    if self.bin_by_time:
      # Instead of hashing, put chunks of consecutive items in the same bin.
      # This exists as a sanity check for the other parts of this class.
      return self.bin_vectors_by_time(vecs)

    # See https://arxiv.org/pdf/1509.02897.pdf
    # It's not clear whether sampling a different random rotation for each head
    # and batch element matters here, but see MergedMultiHashedCausalAttention.
    assert self.n_bins % 2 == 0
    random_rotation = jax.random.normal(
        rng, (vecs.shape[0], vecs.shape[-1], self.n_bins//2)).astype('float32')

    # TODO(kitaev): making the vectors unit-length here is probably redundant.
    vecs = self.make_unit_length(vecs)
    rotated_vecs = np.matmul(vecs, random_rotation)
    rotated_vecs = self.make_unit_length(rotated_vecs)
    rotated_vecs = np.concatenate([rotated_vecs, -rotated_vecs], axis=-1)
    bins = np.argmax(rotated_vecs, axis=-1)
    return bins

  def call_and_grad(self, inputs, ct, rng=None, **kwargs):
    del kwargs
    # We use the same vector as both a query and a key. For now we haven't
    # adjusted any of the surrounding code, so we still get a separate "key"
    # input that we ignore.
    qk, ignored_k, v = inputs
    seqlen = qk.shape[-2]
    # qk/v are n_batch*n_heads, seqlen, d_head

    # bins are n_batch*n_heads, seqlen
    # They specify which hash bucket the query/key/value vectors fall in.
    bins = self.hash_vectors(qk, rng=rng)

    # joint_t is n_batch*n_heads, seqlen
    joint_t = jax.lax.tie_in(qk, np.arange(seqlen))
    joint_t = np.reshape(joint_t, (1, seqlen))
    joint_t = np.broadcast_to(joint_t, qk.shape[:-1])

    assert int((self.n_bins + 1) * seqlen) < 2 ** 31, (
        'Potential 32-bit integer overflow; please double-check the code.')
    joint_bins_and_t = seqlen * bins + joint_t

    def chunk_scalars(x):  # pylint: disable=invalid-name
      return np.reshape(x, (x.shape[0], self.n_bins, -1))

    def chunk_vectors(x):  # pylint: disable=invalid-name
      return np.reshape(
          x, (x.shape[0], self.n_bins, -1, x.shape[-1]))

    def unchunk_vectors(x):  # pylint: disable=invalid-name
      return np.reshape(x, (x.shape[0], -1, x.shape[-1]))

    # Sort everything by bin number, with a secondary sort by time
    # (variables starting with "s" are sorted)
    _, sjoint_t = jax.lax.sort_key_val(
        joint_bins_and_t, joint_t, dimension=-1)

    sqk = np.take_along_axis(qk, sjoint_t[:, :, None], axis=-2)
    sv = np.take_along_axis(v, sjoint_t[:, :, None], axis=-2)

    if ct is not None:
      so_ct = np.take_along_axis(ct, sjoint_t[:, :, None], axis=-2)

    @jax.jit
    def binned_attn(sqk, sv):  # pylint: disable=invalid-name
      """Performs attention on sorted queries/keys/values."""
      # Split off a "bin" axis so that attention only occurs whithin chunks.
      bq_t = bkv_t = chunk_scalars(sjoint_t)
      bqk = chunk_vectors(sqk)
      bv = chunk_vectors(sv)

      # Hashing operates on unit-length vectors. Unnormalized query vectors are
      # fine because they effectively provide a learnable temperature for the
      # attention softmax, but normalizing keys is needed so that similarity for
      # the purposes of attention correctly corresponds to hash locality.
      bq = bqk
      bk = self.make_unit_length(bqk)

      # Allow each chunk to attend within itself, and also one chunk back. Chunk
      # boundaries might occur in the middle of a sequence of items from the
      # same bin, so this increases the chances of attending to relevant items.
      # TODO(kitaev): benchmark whether XLA pad operation is noticeably faster.
      bk_extra = np.concatenate([bk[:, -1:, :, :], bk[:, :-1, :, :]], axis=1)
      bk = np.concatenate([bk, bk_extra], axis=2)
      bv_extra = np.concatenate([bv[:, -1:, :, :], bv[:, :-1, :, :]], axis=1)
      bv = np.concatenate([bv, bv_extra], axis=2)
      bkv_t_extra = np.concatenate([bkv_t[:, -1:, :], bkv_t[:, :-1, :]], axis=1)
      bkv_t = np.concatenate([bkv_t, bkv_t_extra], axis=2)

      # Dot-product attention.
      dots = np.matmul(bq, np.swapaxes(bk, -1, -2)) / np.sqrt(bq.shape[-1])

      # Causal masking
      mask = jax.lax.convert_element_type(
          jax.lax.lt(bq_t[:, :, :, None], bkv_t[:, :, None, :]),
          np.float32)
      dots = dots - 1e9 * mask

      # Mask out attention to self except when no other targets are available.
      self_mask = jax.lax.broadcasted_eye(dots.dtype, dots.shape, (2, 3))
      self_mask = jax.lax.tie_in(dots, self_mask)
      dots = dots - 32 * self_mask

      # Softmax.
      dots = np.exp(dots - backend.logsumexp(dots, axis=-1, keepdims=True))
      bo = np.matmul(dots, bv)

      so = unchunk_vectors(bo)
      return so

    @jax.jit
    def binned_attn_vjp(sqk, sv, so_ct):  # pylint: disable=invalid-name
      so, vjpfun = jax.vjp(binned_attn, sqk, sv)
      sqkv_ct = vjpfun(so_ct)
      return so, sqkv_ct

    if ct is None:
      so = binned_attn(sqk, sv)
      _, undo_sort = jax.lax.sort_key_val(sjoint_t, joint_t, dimension=-1)
      out = np.take_along_axis(so, undo_sort[:, :, None], axis=-2)
      return out, None
    else:
      # Jax can construct a backward pass automatically, but it's about 2x
      # slower than writing our own. The main reason is that the backward pass
      # of gather is in general a scatter operation, but we know we're dealing
      # with permutations so we use gather for the backward pass too.
      so, (sqk_ct, sv_ct) = binned_attn_vjp(sqk, sv, so_ct)

      _, undo_sort = jax.lax.sort_key_val(sjoint_t, joint_t, dimension=-1)
      out = np.take_along_axis(so, undo_sort[:, :, None], axis=-2)

      qk_ct = np.take_along_axis(sqk_ct, undo_sort[:, :, None], axis=-2)
      v_ct = np.take_along_axis(sv_ct, undo_sort[:, :, None], axis=-2)

      return out, (qk_ct, np.zeros_like(ignored_k), v_ct)


class MergedMultiHashedCausalAttention(BaseCausalAttention):
  """Hash-based causal attention, with multiple hashes."""

  def __init__(self, dropout, mode, n_bins=64, n_hashes=1, bin_by_time=False):
    del dropout, mode
    super(MergedMultiHashedCausalAttention, self).__init__()
    self.n_bins = n_bins
    self.n_hashes = n_hashes
    self.bin_by_time = bin_by_time

  def bin_vectors_by_time(self, vecs):
    seqlen = vecs.shape[-2]
    assert seqlen % self.n_bins == 0
    bin_size = int(seqlen // self.n_bins)

    bins = np.arange(seqlen, dtype=np.int32) // bin_size
    bins = jax.lax.tie_in(vecs, bins)
    bins = bins[None, :]
    bins = np.broadcast_to(bins, vecs.shape[:-1])
    return bins

  def make_unit_length(self, x, epsilon=1e-6):
    variance = np.mean(x**2, axis=-1, keepdims=True)
    norm_inputs = x / np.sqrt(variance + epsilon)
    return norm_inputs

  def hash_vectors(self, vecs, rng):
    if self.bin_by_time:
      # Instead of hashing, put chunks of consecutive items in the same bin.
      # This exists as a sanity check for the other parts of this class.
      return self.bin_vectors_by_time(vecs)

    # See https://arxiv.org/pdf/1509.02897.pdf
    # We sample a different random rotation for each batch element, head, and
    # (crucially) each round of hashing. All of these are part of dimension 0
    # of vecs. Applying multiple hashes to the same input is important because
    # it increases the probability of being in the same bin as relevant items.
    assert self.n_bins % 2 == 0
    random_rotation = jax.random.normal(
        rng, (vecs.shape[0], vecs.shape[-1], self.n_bins//2)).astype('float32')

    # TODO(kitaev): making the vectors unit-length here is probably redundant.
    vecs = self.make_unit_length(vecs)
    rotated_vecs = np.matmul(vecs, random_rotation)
    rotated_vecs = self.make_unit_length(rotated_vecs)
    rotated_vecs = np.concatenate([rotated_vecs, -rotated_vecs], axis=-1)
    bins = np.argmax(rotated_vecs, axis=-1)
    return bins

  def call(self, inputs, params=(), state=(), rng=None, **kwargs):
    del params, kwargs
    # We use the same vector as both a query and a key. For now we haven't
    # adjusted any of the surrounding code, so we still get a separate "key"
    # input that we ignore.
    qk, _, v = inputs
    seqlen = qk.shape[-2]

    # qk/v are n_hashes*n_batch*n_heads, seqlen, d_head
    # TODO(kitaev): is it faster to fuse this tiling into gather/scatter ops?
    qk = np.tile(qk, (self.n_hashes, 1, 1))
    v = np.tile(v, (self.n_hashes, 1, 1))

    # bins are n_hashes*n_batch*n_heads, seqlen
    # They specify which hash bucket the query/key/value vectors fall in.
    bins = self.hash_vectors(qk, rng=rng)

    # joint_t is n_hashes*n_batch*n_heads, seqlen
    joint_t = jax.lax.tie_in(qk, np.arange(seqlen))
    joint_t = np.reshape(joint_t, (1, seqlen))
    joint_t = np.broadcast_to(joint_t, qk.shape[:-1])

    assert int((self.n_bins + 1) * seqlen) < 2 ** 31, (
        'Potential 32-bit integer overflow; please double-check the code.')
    joint_bins_and_t = seqlen * bins + joint_t

    def chunk_scalars(x):  # pylint: disable=invalid-name
      return np.reshape(x, (x.shape[0], self.n_bins, -1))

    def chunk_vectors(x):  # pylint: disable=invalid-name
      return np.reshape(
          x, (x.shape[0], self.n_bins, -1, x.shape[-1]))

    def unchunk_vectors(x):  # pylint: disable=invalid-name
      return np.reshape(x, (x.shape[0], -1, x.shape[-1]))

    # Sort everything by bin number, with a secondary sort by time
    # (variables starting with "s" are sorted)
    _, sjoint_t = jax.lax.sort_key_val(
        joint_bins_and_t, joint_t, dimension=-1)
    _, undo_sort = jax.lax.sort_key_val(sjoint_t, joint_t, dimension=-1)
    # TODO(kitaev): why does jax flag integer indices as differentiable?
    # If we don't call stop_gradient here, custom gradients below won't work
    # because the primitive functions close over "differentiable" variables.
    sjoint_t = jax.lax.stop_gradient(sjoint_t)
    undo_sort = jax.lax.stop_gradient(undo_sort)

    # The backward pass of gather is in general a scatter operation, but we know
    # we're dealing with permutations so we use gather for the backward pass
    # too. This custom gradient should be about 2x faster than having jax infer
    # one that uses scatter ops instead.
    def permute_impl(vecs):
      assert len(vecs.shape) == 3
      return np.take_along_axis(vecs, sjoint_t[:, :, None], axis=-2)

    def unpermute_impl(vecs):
      assert len(vecs.shape) == 3
      return np.take_along_axis(vecs, undo_sort[:, :, None], axis=-2)

    @jax.custom_transforms
    def permute(vecs):
      return permute_impl(vecs)

    def permute_vjp(vecs):
      out_vecs = permute_impl(vecs)
      def vjpfun(grad):
        return (unpermute_impl(grad),)
      return out_vecs, vjpfun

    @jax.custom_transforms
    def unpermute(vecs):
      return unpermute_impl(vecs)

    def unpermute_vjp(vecs):
      out_vecs = unpermute_impl(vecs)
      def vjpfun(grad):
        return (permute_impl(grad),)
      return out_vecs, vjpfun

    jax.defvjp_all(permute, permute_vjp)
    jax.defvjp_all(unpermute, unpermute_vjp)

    sqk = permute(qk)
    sv = permute(v)

    # Split off a "bin" axis so that attention only occurs within chunks.
    bq_t = bkv_t = chunk_scalars(sjoint_t)
    bqk = chunk_vectors(sqk)
    bv = chunk_vectors(sv)

    # Hashing operates on unit-length vectors. Unnormalized query vectors are
    # fine because they effectively provide a learnable temperature for the
    # attention softmax, but normalizing keys is needed so that similarity for
    # the purposes of attention correctly corresponds to hash locality.
    bq = bqk
    bk = self.make_unit_length(bqk)

    # Allow each chunk to attend within itself, and also one chunk back. Chunk
    # boundaries might occur in the middle of a sequence of items from the
    # same bin, so this increases the chances of attending to relevant items.
    # TODO(kitaev): benchmark whether XLA pad operation is noticeably faster.
    bk_extra = np.concatenate([bk[:, -1:, :, :], bk[:, :-1, :, :]], axis=1)
    bk = np.concatenate([bk, bk_extra], axis=2)
    bv_extra = np.concatenate([bv[:, -1:, :, :], bv[:, :-1, :, :]], axis=1)
    bv = np.concatenate([bv, bv_extra], axis=2)
    bkv_t_extra = np.concatenate([bkv_t[:, -1:, :], bkv_t[:, :-1, :]], axis=1)
    bkv_t = np.concatenate([bkv_t, bkv_t_extra], axis=2)

    # Dot-product attention.
    dots = np.matmul(bq, np.swapaxes(bk, -1, -2)) / np.sqrt(bq.shape[-1])

    # Causal masking
    mask = jax.lax.convert_element_type(
        jax.lax.lt(bq_t[:, :, :, None], bkv_t[:, :, None, :]),
        np.float32)
    dots = dots - 1e9 * mask

    # Mask out attention to self except when no other targets are available.
    self_mask = jax.lax.broadcasted_eye(dots.dtype, dots.shape, (2, 3))
    self_mask = jax.lax.tie_in(dots, self_mask)
    dots = dots - 32 * self_mask

    # Softmax.
    dots_logsumexp = backend.logsumexp(dots, axis=-1, keepdims=True)
    dots = np.exp(dots - dots_logsumexp)

    bo = np.matmul(dots, bv)
    so = unchunk_vectors(bo)
    slogits = unchunk_vectors(dots_logsumexp)

    o = unpermute(so)
    logits = unpermute(slogits)

    o = np.reshape(o, (self.n_hashes, -1, seqlen, o.shape[-1]))
    logits = np.reshape(logits, (self.n_hashes, -1, seqlen, 1))
    probs = np.exp(logits - backend.logsumexp(logits, axis=0, keepdims=True))
    out = np.sum(o * probs, axis=0)
    assert out.shape == inputs[2].shape

    return out, state

  def call_and_grad(self, inputs, ct, rng=None, **kwargs):
    # TODO(kitaev): is there a manual implementation of call_and_grad that's
    # faster than having jax infer one? Or are the permute/unpermute custom
    # gradients defined in call() sufficient for reasonable speed?
    def _do_call(x):
      return self.call(x, params=(), state=(), rng=rng, **kwargs)[0]

    output, vjpfun = jax.vjp(_do_call, inputs)
    return output, vjpfun(ct)[0]


def CausalAttention(d_feature, n_heads=1,
                    d_attention_key=None, d_attention_value=None,
                    attention_type=DotProductCausalAttention,
                    share_kv=False, mode='train'):
  """Transformer-style multi-headed causal attention.

  Args:
    d_feature: int:  dimensionality of feature embedding
    n_heads: int: number of attention heads
    d_attention_key: int: depth of key vector for each attention head
        (default is d_feature // n_heads)
    d_attention_value: int: depth of value vector for each attention head
        (default is d_feature // n_heads)
    attention_type: subclass of BaseCausalAttention: attention class to use
    share_kv: bool, whether to share keys and values
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

  if share_kv:
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
