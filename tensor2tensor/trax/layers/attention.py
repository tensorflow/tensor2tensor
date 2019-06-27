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
          d_feature=d_feature, n_heads=n_heads, dropout=dropout, mode=mode),
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
    Multi-headed self-attention layer.
  """
  return [
      cb.Dup(), cb.Dup(),
      AttentionQKV(d_feature, n_heads=n_heads, dropout=dropout, mode=mode),
  ]


@base.layer()
def ShiftRight(x, **unused_kwargs):
  """Layer to shift the tensor to the right by padding on axis 1."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[1] = (1, 0)  # Padding on axis=1
  padded = np.pad(x, pad_widths, mode='constant',
                  constant_values=x.dtype.type(0))
  return padded[:, :-1]
