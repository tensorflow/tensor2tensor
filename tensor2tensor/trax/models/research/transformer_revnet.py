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
  """Combinator for applying a layer to a list or tuple.

  Args:
    layer: a layer to apply to each element.

  Returns:
    A new layer representing mapping layer to all elements of the input.
  """

  def __init__(self, layer, sections=1, check_shapes=True):
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
    self._sections = sections

  def n_inputs(self):
    """Specifies how many data tensors this layer expects as input."""
    return self._sections

  def n_outputs(self):
    """Specifies how many data tensors this layer promises as output."""
    return self._sections

  def call(self, inputs, params=(), **kwargs):
    rngs = _pop_rng_and_split(kwargs, len(inputs))
    result = [self._layer(x, params=params, rng=r, **kwargs)
              for x, r in zip(inputs, rngs)]
    return tuple(result)

  def new_parameters(self, input_shape, input_dtype, rng):
    first_shape = input_shape[0]
    if self._check_shapes:
      for shape in input_shape:
        if shape != first_shape:
          raise ValueError('Map layer can only be applied to list of elements '
                           'with the same shapes. Shapes: %s' % str(shape))
    return self._layer.initialize(first_shape, input_dtype[0], rng)


def FeedForward(d_feature, d_feedforward, dropout, mode):
  """Feed-forward block with layer normalization at start."""
  # TODO(kitaev): add dropout. Dropout is typically performed by adding noise to
  # the activations, but when the size of the activations is very large it is
  # more efficient to add noise to the *parameters* instead.
  del dropout, mode
  return [
      tl.LayerNorm(),
      tl.Dense(d_feedforward),
      tl.Relu(),
      tl.Dense(d_feature),
  ]


class ReversibleLayerMixin(object):
  """Reversible Layer Mixin."""

  def inverse_and_vjp(self, output, ct, params=(), **kwargs):
    """Backward pass: computes the inverse of a layer and propagates gradients.

    Args:
      output: Output activations; can be a (possibly nested) tuple.
      ct: gradient signal (cotangent) computed based on subsequent layers. If
          None, no gradients are propagated. Otherwise the structure and shape
          must match the output.
      params: layer parameters
      **kwargs: kwargs for the layer

    Returns:
      A tuple (x, x_ct), where x is the reconstructed input and x_ct is the
      gradient signal for the input. If ct is None, x_ct will also be None.
    """
    if ct is None:
      # Subclasses must override inverse_and_vjp, but in the case where ct is
      # not None there is an unoptimized implementation below that they can
      # delegate to.
      raise NotImplementedError

    # Note: jax.vjp does not allow us to use **kwargs in the signature here.
    def _do_call(x, params, kwargs):
      return super(ReversibleLayerMixin, self).__call__(x, params, **kwargs)

    reconstructed_x, must_be_none = self.inverse_and_vjp(
        output, None, params, **kwargs)
    assert must_be_none is None
    _, vjpfun = jax.vjp(_do_call, reconstructed_x, params, kwargs)
    input_ct = vjpfun(ct)
    return reconstructed_x, input_ct

  def __call__(self, x, params=(), **kwargs):
    assert backend.get_name() == 'jax', (
        'Reversible layers are only supported in JAX')

    if params is () and self._params:  # pylint: disable=literal-comparison
      # TODO(kitaev): Figure out why parameter sharing doesn't work (if this
      # explicit error isn't thrown, a jax tracer error occurs instead)
      raise NotImplementedError(
          'Parameter sharing between reversible layers is not implemented.')

    @jax.custom_transforms
    def do_call(x, params, kwargs):
      return super(ReversibleLayerMixin, self).__call__(x, params, **kwargs)

    def do_call_vjp(x, params, kwargs):
      output = super(ReversibleLayerMixin, self).__call__(x, params, **kwargs)
      def vjpfun(ct):
        _, input_ct = self.inverse_and_vjp(output, ct, params, **kwargs)
        return input_ct

      return output, vjpfun

    jax.defvjp_all(do_call, do_call_vjp)
    return do_call(x, params, kwargs)


class Split(tl.Layer):
  """Splits the input into sections along an axis."""

  def __init__(self, sections=2, axis=-1):
    super(Split, self).__init__()
    self._sections = sections
    self._axis = axis

  def call(self, inputs, params=(), **kwargs):
    del params, kwargs
    return tuple(backend.numpy.split(inputs, self._sections, self._axis))

  def new_parameters(self, input_shapes, input_dtype, rng):
    return ()

  def n_inputs(self):
    """Specifies how many data tensors this layer expects as input."""
    return 1

  def n_outputs(self):
    """Specifies how many data tensors this layer promises as output."""
    return self._sections


@tl.layer()
def Chunk(x, params, sections=2, **kwargs):
  del params, kwargs
  assert x.shape[1] % sections == 0
  return backend.numpy.reshape(x, (
      x.shape[0] * sections,
      x.shape[1] // sections,
      ) + x.shape[2:])


@tl.layer()
def Unchunk(x, params, sections=2, **kwargs):
  del params, kwargs
  assert x.shape[0] % sections == 0
  return backend.numpy.reshape(x, (
      x.shape[0] // sections,
      x.shape[1] * sections,
      ) + x.shape[2:])


class ReversibleHalfResidual(ReversibleLayerMixin, tl.Serial):
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

  def inverse_and_vjp(self, output, ct, params=(), **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = (None,) * self._n_layers
    if rng is not None:
      rngs = backend.random.split(rng, self._n_layers)

    if ct is None:
      reconstructed_x = output
      # Note that self.sublayers() aligns exactly with self.reverse_layers in
      # terms of parameter and rng usage, so no re-ordering is required.
      for layer, p, rng in zip(self.reverse_layers, params, rngs):
        reconstructed_x = layer(reconstructed_x, p, rng=rng, **kwargs)
      return reconstructed_x, None
    else:
      # Note: jax.vjp does not allow us to use **kwargs in the signature here.
      def call_compute_residual(x, params, kwargs):
        return self.compute_residual(x, params, **kwargs)

      assert len(ct) == 2
      ct = ((ct[0], ct[0], ct[1]))

      compute_residual_kwargs = kwargs.copy()
      compute_residual_kwargs['rng'] = rngs[0]
      stack_with_residual, vjpfun = jax.vjp(
          call_compute_residual, output, params[0], compute_residual_kwargs)
      reconstructed_x = self.subtract_top(
          stack_with_residual, params[-1], rng=rngs[-1], **kwargs)

      x_ct, residual_params_ct, kwargs_ct = vjpfun(ct)
      return reconstructed_x, (x_ct, (residual_params_ct, ()), kwargs_ct)


@tl.layer(n_inputs=1, n_outputs=1)
def SplitHeads(x, params, n_heads=1, **kwargs):
  del params, kwargs
  d_feature = x.shape[-1]
  assert d_feature % n_heads == 0
  d_head = d_feature // n_heads
  n_batch = np.shape(x)[0]
  # n_batch, seqlen, d_feature --> n_batch, n_heads, seqlen, d_head
  return np.transpose(
      np.reshape(x, (n_batch, -1, n_heads, d_head)), (0, 2, 1, 3))


@tl.layer(n_inputs=1, n_outputs=1)
def JoinHeads(x, params, **kwargs):
  del params, kwargs
  n_batch = np.shape(x)[0]
  seqlen = np.shape(x)[2]
  # n_batch, n_heads, seqlen, d_head --> n_batch, seqlen, d_feature
  return np.reshape(np.transpose(x, (0, 2, 1, 3)), (n_batch, seqlen, -1))


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

  def call(self, inputs, params=(), rng=None, **kwargs):
    del params
    q, k, v = inputs
    mask_size = q.shape[-2]
    mask = np.tril(np.ones((1, mask_size, mask_size), dtype=onp.bool_), k=0)
    res = tl.DotProductAttention(
        q, k, v, mask, dropout=self._dropout, mode=self._mode, rng=rng)
    return res

  def forward_and_vjp(self, inputs, ct, params=(), **kwargs):
    # Simultaneous forward pass and backprop through the attention mechanism.
    def do_call(x):
      return self.call(x, params, **kwargs)
    output, vjpfun = jax.vjp(do_call, inputs)
    return output, vjpfun(ct)[0]

  def new_parameters(self, input_shapes, input_dtype, rng):
    return ()

  def n_inputs(self):
    return 3

  def n_outputs(self):
    return 1


class MemoryEfficientDotProductAttention(DotProductAttention):
  """Memory-efficient dot product attention."""

  def __init__(self, loop_stride, dropout, mode):
    super(MemoryEfficientDotProductAttention, self).__init__(dropout, mode)
    self._loop_stride = loop_stride

  def call(self, inputs, params=(), **kwargs):
    output, _ = self.forward_and_vjp(inputs, None, params=params, **kwargs)
    return output

  def forward_and_vjp(self, inputs, ct, params=(), **kwargs):
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
      out_slice = np.matmul(dots, value)
      return out_slice

    def forward_and_vjp_slice(query_slice, q_loop_idx, key, value, ct_slice):
      output_slice, vjpfun = jax.vjp(
          forward_slice, query_slice, q_loop_idx, key, value)
      return output_slice, vjpfun(ct_slice)

    q_loop_idx = np.zeros((), dtype=np.int32)
    q_loop_max = query.shape[2]
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
          query, q_loop_idx, q_loop_stride, axis=2)

      if do_backprop:
        ct_slice = jax.lax.dynamic_slice_in_dim(
            ct, q_loop_idx, q_loop_stride, axis=2)
        out_slice, partial_ct = forward_and_vjp_slice(
            query_slice, q_loop_idx, key, value, ct_slice)
        query_ct_accum = jax.lax.dynamic_update_slice_in_dim(
            query_ct_accum, partial_ct[0], q_loop_idx, axis=2)
        # ignore partial_ct[1], which is wrt the loop idx
        key_ct_accum = key_ct_accum + partial_ct[2]
        value_ct_accum = value_ct_accum + partial_ct[3]
      else:
        out_slice = forward_slice(query_slice, q_loop_idx, key, value)

      out_accum = jax.lax.dynamic_update_slice_in_dim(
          out_accum, out_slice, q_loop_idx, axis=2)
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


class ReversibleAttentionHalfResidual(ReversibleLayerMixin, tl.Serial):
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

  def inverse_and_vjp(self, output, ct, params=(), **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = (None,) * self._n_layers
    if rng is not None:
      rngs = backend.random.split(rng, self._n_layers)

    if ct is None:
      reconstructed_x = output
      # Note that self.sublayers() aligns exactly with self.reverse_layers in
      # terms of parameter and rng usage, so no re-ordering is required.
      for layer, p, rng in zip(self.reverse_layers, params, rngs):
        reconstructed_x = layer(reconstructed_x, p, rng=rng, **kwargs)
      return reconstructed_x, None
    else:
      # Forward pass through self.pre_attention, while preparing for
      # later backprop.
      # Note: jax.vjp does not allow us to use **kwargs in the signature here.
      def call_pre_attention(x, params, kwargs):
        return self.pre_attention(x, params, **kwargs)
      pre_attention_kwargs = kwargs.copy()
      pre_attention_kwargs['rng'] = rngs[0]
      stack, pre_attention_vjpfun = jax.vjp(
          call_pre_attention, output, params[0], pre_attention_kwargs)

      # Backprop through adding the residual
      assert len(ct) == 2
      ct = saved_ct = (ct[0], ct[0], ct[1])

      # Backprop through self.post_attention with respect to the inputs only
      call_post_attention_kwargs = kwargs.copy()
      call_post_attention_kwargs['rng'] = rngs[2]
      def call_post_attention(x):
        return self.post_attention(x, params[2], **call_post_attention_kwargs)
      # Note: these are *not* the actual inputs to self.post_attention.
      # If self.post_attention is not linear, we will get incorrect gradients.
      dummy_inputs = (stack[-3], stack[-2], stack[-1])
      _, post_attention_vjpfun = jax.vjp(call_post_attention, dummy_inputs)
      (ct,) = post_attention_vjpfun(ct)

      # Simultaneous forward pass and backprop through the attention mechanism
      attention_kwargs = kwargs.copy()
      attention_kwargs['rng'] = rngs[1]
      stack, ct = self.attention.forward_and_vjp(
          stack, ct, **attention_kwargs)
      attention_params_ct = ()

      # Backprop through self.pre_attention
      (x_ct,
       pre_attention_params_ct,
       pre_attention_kwargs_ct) = pre_attention_vjpfun(ct)

      # Forward pass for self.post_attention, and backprop with respect to the
      # parameters only
      def call_post_attention2(params, kwargs):
        return self.post_attention(stack, params, **kwargs)
      stack, post_attention_vjpfun = jax.vjp(
          call_post_attention2, params[2], call_post_attention_kwargs)
      (post_attention_params_ct,
       post_attention_kwargs_ct) = post_attention_vjpfun(saved_ct)

      # Forward pass through subtracting the residual
      reconstructed_x = self.subtract_top(
          stack, params[-1], rng=rngs[-1], **kwargs)

      params_ct = (
          pre_attention_params_ct,
          attention_params_ct,
          post_attention_params_ct,
          (),
          )

      # We don't actually backprop through the kwargs, but the API requires that
      # we provide a value for kwargs_ct.
      kwargs_ct = pre_attention_kwargs_ct
      del post_attention_kwargs_ct

      return reconstructed_x, (x_ct, params_ct, kwargs_ct)


class ReversibleSwap(ReversibleLayerMixin, tl.Swap):
  """Swap the first two element on the stack."""

  def inverse_and_vjp(self, output, ct, params=(), **kwargs):
    if ct is None:
      # Swap is its own inverse
      return self.call(output, params, **kwargs), None
    else:
      return super(ReversibleSwap, self).inverse_and_vjp(
          output, ct, params, **kwargs)


class ReversibleSerial(ReversibleLayerMixin, tl.Serial):
  """A reversible version of tl.Serial (requires reversible sub-layers)."""

  def __init__(self, *layers):
    super(ReversibleSerial, self).__init__(*layers)

    # Note that sublayers has already been flattened to remove nested lists.
    for i, layer in enumerate(self.sublayers()):
      if not isinstance(layer, ReversibleLayerMixin):
        raise ValueError(
            'Sub-layer {} of ReversibleSerial is not reversible: {}'.format(
                i, layer))

  def inverse_and_vjp(self, output, ct, params=(), **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = (None,) * self._n_layers
    if rng is not None:
      rngs = backend.random.split(rng, self._n_layers)

    layer_val = output
    if ct is not None:
      layer_ct = ct
      params_ct = []
    for layer, p, rng in reversed(zip(self.sublayers(), params, rngs)):
      layer_val, layer_ct = layer.inverse_and_vjp(
          layer_val, layer_ct, p, rng=rng, **kwargs)
      if ct is not None:
        layer_ct, p_ct, kwargs_ct = layer_ct
        params_ct.insert(0, p_ct)

    # TODO(kitaev): Handle kwargs_ct properly. However, kwargs generally only
    # contains the rng, which is non-differentiable.
    for k in kwargs:
      if k != 'rng':
        raise NotImplementedError(
            'ReversibleSerial does not support differentiation wrt kwargs,'
            'and the key {} is not known to be non-differentiable.'.format(k))

    if ct is not None:
      return layer_val, (layer_ct, params_ct, kwargs_ct)
    else:
      return layer_val, None


def DecoderBlock(d_feature, d_feedforward, d_attention_key, d_attention_value,
                 n_heads, n_attention_chunks, attention_loop_stride,
                 dropout, mode):
  """Reversible transformer decoder layer.

  Args:
    d_feature: int:  depth of embedding
    d_feedforward: int: depth of feed-forward layer
    d_attention_key: int: depth of key vector for each attention head
    d_attention_value: int: depth of value vector for each attention head
    n_heads: int: number of attention heads
    n_attention_chunks: int: number of chunks for attention
    attention_loop_stride: int: number of query elements to compute attention
      for in parallel. Set to 0 to disable memory-efficient attention.
    dropout: float: dropout rate (how much to drop out)
    mode: str: 'train' or 'eval'

  Returns:
    the layer.
  """

  pre_attention = [
      Chunk(sections=n_attention_chunks),  # pylint: disable=no-value-for-parameter
      tl.LayerNorm(),
      tl.Dup(), tl.Dup(),
      tl.Parallel(
          [tl.Dense(d_attention_key * n_heads), SplitHeads(n_heads=n_heads)],  # pylint: disable=no-value-for-parameter
          [tl.Dense(d_attention_key * n_heads), SplitHeads(n_heads=n_heads)],  # pylint: disable=no-value-for-parameter
          [tl.Dense(d_attention_value * n_heads), SplitHeads(n_heads=n_heads)],  # pylint: disable=no-value-for-parameter
      ),
  ]

  # TODO(kitaev): add dropout
  if attention_loop_stride < 1:
    # Use the standard implementation if no loop_stride is provided.
    attention = DotProductAttention(dropout=None, mode=mode)
  else:
    attention = MemoryEfficientDotProductAttention(
        loop_stride=attention_loop_stride, dropout=None, mode=mode)

  # ReversibleAttentionHalfResidual requires that post_attention be linear in
  # its input (so the backward pass can be computed without knowing the input)
  post_attention = [
      JoinHeads(),  # pylint: disable=no-value-for-parameter
      tl.Dense(d_feature),
      Unchunk(sections=n_attention_chunks),  # pylint: disable=no-value-for-parameter
  ]

  feed_forward = [
      FeedForward(d_feature, d_feedforward, dropout, mode=mode),
  ]
  return [
      ReversibleAttentionHalfResidual(pre_attention, attention, post_attention),
      ReversibleSwap(),
      ReversibleHalfResidual(feed_forward),
      ReversibleSwap(),
  ]


def TransformerRevnetLM(vocab_size,
                        d_feature=512,
                        d_feedforward=2048,
                        d_attention_key=64,
                        d_attention_value=64,
                        n_layers=6,
                        n_heads=8,
                        dropout=0.1,
                        max_len=2048,
                        n_chunks=32,
                        n_attention_chunks=8,
                        attention_loop_stride=0,
                        mode='train'):
  """Reversible transformer language model (only uses a decoder, no encoder).

  Args:
    vocab_size: int: vocab size
    d_feature: int:  depth of *each half* of the two-part features
    d_feedforward: int: depth of feed-forward layer
    d_attention_key: int: depth of key vector for each attention head
    d_attention_value: int: depth of value vector for each attention head
    n_layers: int: number of decoder layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    max_len: int: maximum symbol length for positional encoding
    n_chunks: int: number of chunks (must match input pipeline)
    n_attention_chunks: int: number of chunks for attention
    attention_loop_stride: int: number of query elements to compute attention
      for in parallel. Set to 0 to disable memory-efficient attention.
    mode: str: 'train' or 'eval'

  Returns:
    the layer.
  """
  positional_embedder = [
      tl.Embedding(d_feature, vocab_size),
      # TODO(kitaev): add dropout
      tl.PositionalEncoding(max_len=max_len),
  ]
  return tl.Model(
      tl.Concatenate(n_items=n_chunks),
      tl.ShiftRight(),
      positional_embedder,
      tl.Dup(),
      ReversibleSerial([
          # pylint: disable=g-complex-comprehension
          DecoderBlock(d_feature, d_feedforward,
                       d_attention_key, d_attention_value, n_heads,
                       n_attention_chunks, attention_loop_stride,
                       dropout, mode)
          for _ in range(n_layers)
      ]),
      tl.Parallel(tl.LayerNorm(), tl.LayerNorm()),
      tl.Concatenate(),
      Split(sections=n_chunks, axis=-2),  # pylint: disable=no-value-for-parameter
      Map([
          tl.Dense(vocab_size),
          tl.LogSoftmax(),
      ], sections=n_chunks),
  )

