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
  return onp.tril(onp.ones((1, size, size), dtype=x.dtype), k=0)


@base.layer(output_shape=lambda shape, pad=0: (shape[0], 1, 1, shape[-1]))
def PaddingMask(x, params, pad=0, **kwargs):
  del params, kwargs
  return np.reshape(x != pad, (x.shape[0], 1, 1, x.shape[-1]))


def MakeTargetMask(target, pad=0):
  """Create an attention mask to hide padding and future words."""
  target_mask = (target != pad)[ :, np.newaxis, :]
  target_dtype = target_mask.dtype
  causal_mask = onp.tril(onp.ones((1, target.shape[-1], target.shape[-1]),
                                  dtype=target_dtype), k=0)
  target_mask = target_mask & causal_mask
  return np.expand_dims(target_mask, axis=1)


def PreparePairedSequenceBatch(source, target_in, pad=0):
  """Build masks for this batch.

  Args:
    source: (batch, source_len) array of integer-coded symbols for inputs
    target_in: (batch, batch_len) array of integer-coded symbols for targets
    pad: int: the padding symbol used to pad the above

  Returns:
    Prepared batch of tuple of arrays: source, input-target, shifted-target,
    source mask, target mask, source-target "memory" mask, minibatch token count
  """
  target = target_in[:, :-1]
  target_y = target_in[:, 1:]
  source_mask = np.reshape(source != pad,
                           (source.shape[0], 1, 1, source.shape[-1]))
  target_mask = MakeTargetMask(target, pad)
  memory_mask = (
      np.reshape(np.arange(target.shape[-1]) < source.shape[-1], [-1, 1]))
  ntokens = np.sum(target_y != pad)
  return (source, target, target_y,
          source_mask, target_mask, memory_mask, ntokens)


# Layer normalization.
def _layer_norm_new_params(input_shape, rng, epsilon=1e-6):  # pylint: disable=invalid-name
  """Helper: create layer norm parameters."""
  del rng, epsilon
  features = input_shape[-1]
  scale = np.ones(features)
  bias = np.zeros(features)
  return (scale, bias)


@base.layer(new_parameters=_layer_norm_new_params)
def LayerNorm(x, params, epsilon=1e-6, **unused_kwargs):
  (scale, bias) = params
  mean = np.mean(x, axis=-1, keepdims=True)
  variance = np.mean((x - mean)**2, axis=-1, keepdims=True)
  norm_inputs = (x - mean) / np.sqrt(variance + epsilon)
  return norm_inputs * scale + bias


# Positional encoding.
def _positional_encoding_new_params(input_shape, rng, max_len=2048):  # pylint: disable=invalid-name
  """Helper: create positional encoding parameters."""
  del rng
  feature_depth = input_shape[-1]
  pe = onp.zeros((max_len, feature_depth), dtype=onp.float32)
  position = onp.arange(0, max_len)[:, onp.newaxis]
  div_term = onp.exp(
      onp.arange(0, feature_depth, 2) * -(onp.log(10000.0) / feature_depth))
  pe[:, 0::2] = onp.sin(position * div_term)
  pe[:, 1::2] = onp.cos(position * div_term)
  return np.array(pe[onp.newaxis, :])  # send to device


@base.layer(new_parameters=_positional_encoding_new_params)
def PositionalEncoding(x, params, **unused_kwargs):
  """Implements bare positional encoding."""
  symbol_size = np.shape(x)[1]
  return x + params[:, :symbol_size]


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
  def init_fun(_, input_shapes):  # pylint: disable=invalid-name
    q_shape, _, v_shape, _ = input_shapes
    output_shape = q_shape[:-1] + (v_shape[-1],)
    return output_shape, ()
  def apply_fun(params, inputs, **kwargs):  # pylint: disable=invalid-name
    del params
    q, k, v, mask = inputs
    rng = kwargs.get('rng', None)
    return DotProductAttention(q, k, v, mask,
                               dropout=dropout, mode=mode, rng=rng)
  return init_fun, apply_fun


def _multihead_attention_output_shape(  # pylint: disable=invalid-name
    input_shapes, feature_depth=None, **unused_kwargs):
  """Helper: calculate multihead attention output shape."""
  input_shape = input_shapes[0]  # Inputs are (q, k, v, mask).
  return input_shape[:-1] + (feature_depth,)


@base.layer(output_shape=_multihead_attention_output_shape)
def PureMultiHeadedAttention(x, params, feature_depth=None,
                             num_heads=8, dropout=0.0, mode='train', **kwargs):
  """Pure transformer-style multi-headed attention.

  Args:
    x: inputs (q, k, v, mask)
    params: parameters (none)
    feature_depth: int:  depth of embedding
    num_heads: int: number of attention heads
    dropout: float: dropout rate
    mode: str: 'train' or 'eval'
    **kwargs: other arguments including the rng

  Returns:
    Pure Multi-headed attention layer. (No Dense transforms on input.)
  """
  del params
  rng = kwargs.get('rng', None)
  (q, k, v), mask = x
  assert feature_depth % num_heads == 0
  head_depth = feature_depth // num_heads
  nbatch = np.shape(q)[0]
  # nbatch, seqlen, feature_depth --> nbatch, num_heads, seqlen, head_depth
  def SplitHeads(x):
    return np.transpose(
        np.reshape(x, (nbatch, -1, num_heads, head_depth)), (0, 2, 1, 3))
  # nbatch, num_heads, seqlen, head_depth --> nbatch, seqlen, feature_depth
  def JoinHeads(x):  # pylint: disable=invalid-name
    return np.reshape(
        np.transpose(x, (0, 2, 1, 3)), (nbatch, -1, num_heads*head_depth))
  # Split heads, dot-product attention, rejoin heads.
  return JoinHeads(
      DotProductAttention(
          SplitHeads(q), SplitHeads(k), SplitHeads(v), mask,
          dropout=dropout, mode=mode, rng=rng))


def MultiHeadedAttentionQKV(
    feature_depth, num_heads=8, dropout=0.0, mode='train'):
  """Transformer-style multi-headed attention.

  Accepts inputs of the form (q, k, v), mask.

  Args:
    feature_depth: int:  depth of embedding
    num_heads: int: number of attention heads
    dropout: float: dropout rate
    mode: str: 'train' or 'eval'

  Returns:
    Multi-headed self-attention layer.
  """
  return combinators.Serial(
      combinators.Parallel(
          combinators.Parallel(
              core.Dense(feature_depth,
                         kernel_initializer=core.XavierUniformInitializer()),
              core.Dense(feature_depth,
                         kernel_initializer=core.XavierUniformInitializer()),
              core.Dense(feature_depth,
                         kernel_initializer=core.XavierUniformInitializer()),
          ),
          combinators.Identity()
      ),
      PureMultiHeadedAttention(  # pylint: disable=no-value-for-parameter
          feature_depth=feature_depth, num_heads=num_heads,
          dropout=dropout, mode=mode),
      core.Dense(feature_depth,
                 kernel_initializer=core.XavierUniformInitializer()),
  )


def MultiHeadedAttention(
    feature_depth, num_heads=8, dropout=0.0, mode='train'):
  """Transformer-style multi-headed attention.

  Accepts inputs of the form (x, mask) and constructs (q, k, v) from x.

  Args:
    feature_depth: int:  depth of embedding
    num_heads: int: number of attention heads
    dropout: float: dropout rate
    mode: str: 'train' or 'eval'

  Returns:
    Multi-headed self-attention layer.
  """
  return combinators.Serial(
      combinators.Parallel(
          combinators.Branch(num_branches=3),  # q = k = v = first input
          combinators.Identity()  # pass the mask
      ),
      MultiHeadedAttentionQKV(  # pylint: disable=no-value-for-parameter
          feature_depth, num_heads=num_heads, dropout=dropout, mode=mode),
  )
