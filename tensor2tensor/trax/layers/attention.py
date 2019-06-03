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

import numpy as onp

from tensor2tensor.trax import backend
from tensor2tensor.trax.backend import numpy as np
from tensor2tensor.trax.layers import base
from tensor2tensor.trax.layers import combinators
from tensor2tensor.trax.layers import core


@base.layer(output_shape=lambda shape, axis=-1: (1, shape[axis], shape[axis]))
def CausalMask(x, params, axis=-1, **kwargs):
  del params, kwargs
  size = x.shape[axis]
  return onp.tril(onp.ones((1, size, size), dtype=onp.bool_), k=0)


@base.layer(output_shape=lambda shape, pad=0: (shape[0], 1, 1, shape[-1]))
def PaddingMask(x, params, pad=0, **kwargs):
  del params, kwargs
  return np.reshape(x != pad, (x.shape[0], 1, 1, x.shape[-1]))


def EncoderDecoderMaskShape(inputs):
  """Helper: shape for encoder-decoder mask."""
  (padding_mask_shape, decoder_input_shape) = inputs
  batch_size = padding_mask_shape[0]
  input_length = padding_mask_shape[-1]
  target_length = decoder_input_shape[1]
  return (batch_size, 1, target_length, input_length)


@base.layer(output_shape=EncoderDecoderMaskShape, stack_items_to_pass=0)
def EncoderDecoderMask(x, **unused_kwargs):
  """Make encoder-decoder mask from a padding mask and decoder input."""
  (padding_mask, decoder_input) = x
  padding_mask = np.reshape(
      padding_mask, (padding_mask.shape[0], 1, 1, padding_mask.shape[-1]))
  # Final mask shape is [batch, 1 for heads, decoder-len, encoder-len].
  return padding_mask + np.zeros((1, 1, decoder_input.shape[1], 1))


# Positional encoding.
def _positional_encoding_new_params(input_shape, rng, max_len=2048):  # pylint: disable=invalid-name
  """Helper: create positional encoding parameters."""
  del rng
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
    dots = np.where(mask, dots, -1e9)
  # Softmax.
  dots = np.exp(dots - backend.logsumexp(dots, axis=-1, keepdims=True))
  if dropout >= 1.0:
    raise ValueError('Dropout rates must be lower than 1.')
  if dropout is not None and dropout > 0.0 and mode == 'train':
    keep = backend.random.bernoulli(rng, 1.0 - dropout, dots.shape)
    dots = np.where(keep, dots / (1.0 - dropout), 0)
  out = np.matmul(dots, value)
  return out


# TODO(lukaszkaiser): make this a layer.
def PureDotProductAttention(dropout=0.0, mode='train'):
  """Pure single-headed self-attention.

  Args:
    dropout: float: dropout rate
    mode: str: 'train' or 'eval'

  Returns:
    Pure single-headed attention layer. (No Dense transforms on input.)
  """
  def init_fn(_, input_shapes):  # pylint: disable=invalid-name
    q_shape, _, v_shape, _ = input_shapes
    output_shape = q_shape[:-1] + (v_shape[-1],)
    return output_shape, ()
  def apply_fn(params, inputs, **kwargs):  # pylint: disable=invalid-name
    del params
    q, k, v, mask = inputs
    rng = kwargs.get('rng', None)
    return DotProductAttention(q, k, v, mask,
                               dropout=dropout, mode=mode, rng=rng)
  return init_fn, apply_fn


def _multihead_attention_output_shape(  # pylint: disable=invalid-name
    input_shapes, **unused_kwargs):
  """Helper: calculate multihead attention output shape."""
  q_shape = input_shapes[0]  # Inputs are (q, k, v, mask).
  v_shape = input_shapes[2]  # Inputs are (q, k, v, mask).
  mask_shape = input_shapes[3]
  res_shape = list(q_shape[:-1]) + [v_shape[-1]]
  return tuple(res_shape), mask_shape


@base.layer(output_shape=_multihead_attention_output_shape,
            stack_items_to_pass=4)
def PureMultiHeadedAttention(x, params, n_heads=8, dropout=0.0, mode='train',
                             **kwargs):
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


def MultiHeadedAttentionQKV(d_feature, n_heads=8, dropout=0.0, mode='train'):
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
      combinators.Parallel(
          core.Dense(d_feature),
          core.Dense(d_feature),
          core.Dense(d_feature),
      ),
      PureMultiHeadedAttention(  # pylint: disable=no-value-for-parameter
          d_feature=d_feature, n_heads=n_heads, dropout=dropout, mode=mode),
      core.Dense(d_feature),
  ]


def MultiHeadedAttention(
    d_feature, n_heads=8, dropout=0.0, mode='train'):
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
      combinators.Dup(),
      combinators.Dup(),
      MultiHeadedAttentionQKV(  # pylint: disable=no-value-for-parameter
          d_feature, n_heads=n_heads, dropout=dropout, mode=mode),
  ]


@base.layer()
def ShiftRight(x, **unused_kwargs):
  """Layer to shift the tensor to the right by padding on axis 1."""
  if not isinstance(x, (list, tuple)):  # non-chunked inputs
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[1] = (1, 0)  # Padding on axis=1
    padded = np.pad(x, pad_widths, mode='constant',
                    constant_values=x.dtype.type(0))
    return padded[:, :-1]
  # Handling chunked inputs. Recall that the list of chunks represents a big
  # sequence (the concatenation of the chunks). We want to shift that sequence,
  # so we put a 0 in the beginning of the first chunk and the last element of
  # that chunk is used as the new first element of the next chunk, and so on.
  padded = []
  last_value = np.zeros_like(x[0][:, -1])
  for chunk in x:
    padded_chunk = np.concatenate([last_value[:, np.newaxis], chunk], axis=1)
    last_value = chunk[:, -1]
    padded.append(padded_chunk[:, :-1])
  return padded
