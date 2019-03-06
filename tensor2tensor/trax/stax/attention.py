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

from jax import random
import jax.experimental.stax as stax
import jax.numpy as np
import numpy as onp
import numpy.random as npr


def causal_mask(size, dtype=np.uint8):
  """Causal attention mask."""
  return onp.tril(onp.ones((1, size, size), dtype=dtype), k=0)


def make_target_mask(target, pad=0):
  """Create an attention mask to hide padding and future words."""
  target_mask = (target != pad)[ :, np.newaxis, :]
  target_dtype = target_mask.dtype
  target_mask = (
      (target_mask & stax.causal_mask(target.shape[-1])).astype(target_dtype))
  return np.expand_dims(target_mask, axis=1)


def prepare_paired_sequence_batch(source, target_in, pad=0):
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
  target_mask = make_target_mask(target, pad)
  memory_mask = (
      np.reshape(np.arange(target.shape[-1]) < source.shape[-1], [-1, 1]))
  ntokens = np.sum(target_y != pad)
  return (source, target, target_y,
          source_mask, target_mask, memory_mask, ntokens)


def xavier_uniform(out_dim=0, in_dim=1, rng=npr):
  """An initializer function for random uniform xavier-scaled coefficients."""
  def init(shape):
    fan_in, fan_out = shape[in_dim], shape[out_dim]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    a = onp.sqrt(3.0) * std
    return rng.uniform(low=-a, high=a, size=shape).astype('float32')
  return init


def LayerNorm(features, epsilon=1e-5):  # pylint: disable=invalid-name
  """Layer construction function for Layer Normalization layer.."""
  def init_fun(input_shape):
    a_2 = np.ones(features)
    b_2 = np.zeros(features)
    return input_shape, (a_2, b_2)

  def apply_fun(params, inputs, **kwargs):
    del kwargs
    (a_2, b_2) = params
    mean = np.mean(inputs, axis=-1, keepdims=True)
    std = np.std(inputs, axis=-1, keepdims=True)
    return a_2 * (inputs - mean) / (std + epsilon) + b_2

  return init_fun, apply_fun


def Embedding(feature_depth, vocab_size):  # pylint: disable=invalid-name
  """Layer constructor function for a dense embedding layer."""
  def init_fun(input_shape):
    output_shape = input_shape + (feature_depth,)
    dense_embedding = xavier_uniform()((vocab_size, feature_depth))
    return output_shape, dense_embedding
  def apply_fun(params, inputs, **kwargs):
    del kwargs
    dense_embedding = params
    return np.take(dense_embedding, inputs, axis=0)
  return init_fun, apply_fun


def PositionalEncoding(feature_depth, max_len):  # pylint: disable=invalid-name
  """Implements bare positional encoding."""
  def init_fun(input_shape):
    # Compute the positional encodings once in log space.
    pe = onp.zeros((max_len, feature_depth), dtype=onp.float32)
    position = onp.arange(0, max_len)[:, onp.newaxis]
    div_term = onp.exp(
        onp.arange(0, feature_depth, 2) * -(onp.log(10000.0) / feature_depth))
    pe[:, 0::2] = onp.sin(position * div_term)
    pe[:, 1::2] = onp.cos(position * div_term)
    pe = np.array(pe[onp.newaxis, :])  # send to device
    return input_shape, pe

  def apply_fun(params, inputs, **kwargs):
    del kwargs
    pe = params
    symbol_size = np.shape(inputs)[1]
    return inputs + pe[:, :symbol_size]

  return init_fun, apply_fun


def dot_product_attention(query, key, value, mask, dropout, mode, rng):
  """Core dot product self-attention.

  Args:
    query: array of representations
    key: array of representations
    value: array of representations
    mask: attention-mask, gates attention
    dropout: float: dropout rate - keep probability
    mode: 'eval' or 'train': whether to use dropout
    rng: JAX PRNGKey: subkey for disposable use

  Returns:
    Self attention for q, k, v arrays.
  """
  depth = np.shape(query)[-1]
  dots = np.matmul(query, np.swapaxes(key, -1, -2)) / np.sqrt(depth)
  if mask is not None:
    dots = np.where(mask, dots, -1e9)
  dots = stax.softmax(dots, axis=-1)
  if dropout is not None and mode == 'train':
    keep = random.bernoulli(rng, dropout, dots.shape)
    dots = np.where(keep, dots / dropout, 0)
  out = np.matmul(dots, value)
  return out


def PureDotProductAttention(dropout=1.0, mode='train'):  # pylint: disable=invalid-name
  """Pure single-headed self-attention.

  Args:
    dropout: float: dropout rate - keep probability
    mode: str: 'train' or 'eval'

  Returns:
    Pure single-headed attention layer. (No Dense transforms on input.)
  """
  def init_fun(input_shapes):
    q_shape, _, v_shape, _ = input_shapes
    output_shape = q_shape[:-1] + (v_shape[-1],)
    return output_shape, ()
  def apply_fun(params, inputs, **kwargs):
    del params
    q, k, v, mask = inputs
    rng = kwargs.get('rng', None)
    return dot_product_attention(q, k, v, mask,
                                 dropout=dropout, mode=mode, rng=rng)
  return init_fun, apply_fun


def PureMultiHeadedAttention(  # pylint: disable=invalid-name
    feature_depth, num_heads=8, dropout=1.0, mode='train'):
  """Pure transformer-style multi-headed attention.

  Args:
    feature_depth: int:  depth of embedding
    num_heads: int: number of attention heads
    dropout: float: dropout rate - keep probability
    mode: str: 'train' or 'eval'

  Returns:
    Pure Multi-headed attention layer. (No Dense transforms on input.)
  """
  def init_fun(input_shapes):
    input_shape = input_shapes[0]
    output_shape = input_shape[:-1] + (feature_depth,)
    return output_shape, ()
  def apply_fun(params, inputs, **kwargs):  # pylint: disable=missing-docstring
    del params
    rng = kwargs.get('rng', None)
    q, k, v, mask = inputs
    assert feature_depth % num_heads == 0
    head_depth = feature_depth // num_heads
    nbatch = np.shape(q)[0]
    # nbatch, seqlen, feature_depth --> nbatch, num_heads, seqlen, head_depth
    def split_heads(x):
      return np.transpose(
          np.reshape(x, (nbatch, -1, num_heads, head_depth)), (0, 2, 1, 3))
    # nbatch, num_heads, seqlen, head_depth --> nbatch, seqlen, feature_depth
    def join_heads(x):
      return np.reshape(
          np.transpose(x, (0, 2, 1, 3)), (nbatch, -1, num_heads*head_depth))
    # Split heads, dot-product attention, rejoin heads.
    return join_heads(
        dot_product_attention(
            split_heads(q), split_heads(k), split_heads(v), mask,
            dropout=dropout, mode=mode, rng=rng))
  return init_fun, apply_fun


def MultiHeadedAttention(  # pylint: disable=invalid-name
    feature_depth, num_heads=8, dropout=1.0, mode='train'):
  """Transformer-style multi-headed attention.

  Args:
    feature_depth: int:  depth of embedding
    num_heads: int: number of attention heads
    dropout: float: dropout rate - keep probability
    mode: str: 'train' or 'eval'

  Returns:
    Multi-headed self-attention layer.
  """
  return stax.serial(
      stax.parallel(
          stax.Dense(feature_depth, W_init=xavier_uniform()),
          stax.Dense(feature_depth, W_init=xavier_uniform()),
          stax.Dense(feature_depth, W_init=xavier_uniform()),
          stax.Identity
      ),
      PureMultiHeadedAttention(
          feature_depth, num_heads=num_heads, dropout=dropout, mode=mode),
      stax.Dense(feature_depth, W_init=xavier_uniform()),
  )
