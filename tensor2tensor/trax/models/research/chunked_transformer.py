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

"""Chunked Transformer Models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp

from tensor2tensor.trax import layers as tl
from tensor2tensor.trax.backend import numpy as np


# Chunked positional encoding.
def _chunked_positional_encoding_new_params(input_shape, rng, max_len=2048):  # pylint: disable=invalid-name
  """Helper: create positional encoding parameters."""
  del rng
  # Check if we are operating on chunked inputs by checking if the first
  # shape is a list/tuple of shapes (otherwise it's an int or numpy array).
  is_chunked = isinstance(input_shape[0], (list, tuple))
  d_feature = input_shape[0][-1] if is_chunked else input_shape[-1]
  pe = onp.zeros((max_len, d_feature), dtype=onp.float32)
  position = onp.arange(0, max_len)[:, onp.newaxis]
  div_term = onp.exp(
      onp.arange(0, d_feature, 2) * -(onp.log(10000.0) / d_feature))
  pe[:, 0::2] = onp.sin(position * div_term)
  pe[:, 1::2] = onp.cos(position * div_term)
  pe = pe[onp.newaxis, :, :]  # [1, max_len, d_feature]
  return np.array(pe)  # These are trainable parameters, initialized as above.


@tl.layer(new_parameters=_chunked_positional_encoding_new_params,
          stack_items_to_pass=0)
def ChunkedPositionalEncoding(x, params, **unused_kwargs):
  """Implements bare positional encoding."""
  if not isinstance(x, (list, tuple)):  # non-chunked inputs
    symbol_size = np.shape(x)[1]
    return x + params[:, :symbol_size, :]
  # Chunked case: apply to all chunks selecting as much as needed.
  offset = 0
  results = []
  for chunk in x:
    symbol_size = np.shape(chunk)[1]
    results.append(chunk + params[:, offset:offset + symbol_size, :])
    offset += symbol_size
  return results


# Chunked attention.
@tl.layer(stack_items_to_pass=0)
def ChunkedAttentionSelector(x, params, selector=None, **kwargs):
  """Select which chunks to attend to in chunked attention.

  Args:
    x: inputs, a list of elements of the form (q, k, v), mask for each chunk.
    params: parameters (unused).
    selector: a function from chunk_number -> list of chunk numbers that says
      which other chunks should be appended to the given one (previous if None).
    **kwargs: unused other arguments.

  Returns:
    a list of elements of the form (q, k', v', mask') where k', v' and mask' are
    concatenations of k, v and identity-extended masks from selected chunks.
  """
  del params, kwargs
  selector = selector or (lambda x: [] if x < 1 else [x-1])
  triples, masks = zip(*x)
  (queries, keys, values) = zip(*triples)
  result = []
  for i in range(len(x)):
    selected = selector(i)
    # Since keys and values are [batch, length, depth] we concatenate on axis=1.
    # We also always include the current key or value at the end.
    new_key_list = [keys[j] for j in selected]
    new_key = np.concatenate(new_key_list + [keys[i]], axis=1)
    new_value = np.concatenate(
        [values[j] for j in selected] + [values[i]], axis=1)
    # Masks are (1, query-len, key-len) so we concatenate on axis=2.
    new_mask_shapes = [(1, queries[i].shape[1], key.shape[1])
                       for key in new_key_list]
    cur_mask = masks[i]
    # Masks are all-1 for the added chunks (no masking).
    new_mask_list = [np.ones(s, dtype=cur_mask.dtype) for s in new_mask_shapes]
    # We still use the current (often causal) mask for the final chunk.
    new_mask = np.concatenate(new_mask_list + [cur_mask], axis=2)
    result.append((queries[i], new_key, new_value, new_mask))
  return tuple(result)


def ChunkedCausalMultiHeadedAttention(
    d_feature, n_heads=8, dropout=0.0, chunk_selector=None, mode='train'):
  """Transformer-style causal multi-headed attention operating on chunks.

  Accepts inputs that are a list of chunks and applies causal attention.

  Args:
    d_feature: int:  depth of embedding
    n_heads: int: number of attention heads
    dropout: float: dropout rate
    chunk_selector: a function from chunk number to list of chunks to attend.
    mode: str: 'train' or 'eval'

  Returns:
    Multi-headed self-attention layer.
  """
  prepare_attention_input = [
      tl.Branch(
          tl.Branch([], [], []),  # q = k = v = first input
          tl.CausalMask(axis=-2)
      ),
      tl.Parallel(
          tl.Parallel(
              tl.Dense(d_feature),
              tl.Dense(d_feature),
              tl.Dense(d_feature)
          ),
          []
      )
  ]
  return [
      tl.Map(prepare_attention_input),
      ChunkedAttentionSelector(selector=chunk_selector),  # pylint: disable=no-value-for-parameter
      tl.Map(tl.PureMultiHeadedAttention(d_feature=d_feature, n_heads=n_heads,
                                         dropout=dropout, mode=mode),
             check_shapes=False),
      tl.Map(tl.Select(0), check_shapes=False),  # drop masks
      tl.Map(tl.Dense(d_feature))
  ]


# Chunked residual.
def Residual(*layers, **unused_kwargs):
  """Constructs a residual version of layers, summing input to layers output."""
  return [
      tl.Branch(layers, []),
      tl.AddAll()
  ]


def ResidualFeedForward(d_feature, d_feedforward, dropout, mode):
  """Residual feed-forward layer with normalization at start."""
  return Residual(
      tl.LayerNorm(),
      tl.Dense(d_feedforward),
      tl.Relu(),
      tl.Dropout(rate=dropout, mode=mode),
      tl.Dense(d_feature),
      tl.Dropout(rate=dropout, mode=mode)
  )


def ChunkedDecoderLayer(d_feature,
                        d_feedforward,
                        n_heads,
                        dropout,
                        chunk_selector,
                        mode):
  """Transformer decoder layer operating on chunks.

  Args:
    d_feature: int:  depth of embedding
    d_feedforward: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    chunk_selector: a function from chunk number to list of chunks to attend.
    mode: str: 'train' or 'eval'

  Returns:
    The layers comprising a chunked decoder.
  """
  return [
      Residual(  # Self-attention block.
          tl.Map(tl.LayerNorm()),
          ChunkedCausalMultiHeadedAttention(
              d_feature, n_heads=n_heads, dropout=dropout,
              chunk_selector=chunk_selector, mode=mode),
          tl.Map(tl.Dropout(rate=dropout, mode=mode)),
      ),
      tl.Map(ResidualFeedForward(
          d_feature, d_feedforward, dropout, mode=mode))
  ]


def ChunkedTransformerLM(vocab_size,
                         d_feature=512,
                         d_feedforward=2048,
                         n_layers=6,
                         n_heads=8,
                         dropout=0.1,
                         chunk_selector=None,
                         max_len=2048,
                         mode='train'):
  """Transformer language model operating on chunks.

  The input to this  model is a sequence presented as a list or tuple of chunks:
    (chunk1, chunk2, chunks3, ..., chunkN).
  Each chunk should have the same shape (batch, chunk-length) and together they
  represent a long sequence that's a concatenation chunk1,chunk2,...,chunkN.

  Chunked Transformer emulates the operation of a Transformer on this long
  sequence except for the chunked attention layer, which may attend to only
  a subset of the chunks to reduce memory use.

  Args:
    vocab_size: int: vocab size
    d_feature: int:  depth of embedding
    d_feedforward: int: depth of feed-forward layer
    n_layers: int: number of encoder/decoder layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    chunk_selector: a function from chunk number to list of chunks to attend
      (if None, attends to the previous chunks which is equivalent to setting
       chunk_selector(x) = [] if x < 1 else [x-1] (TransformerXL); we attend
       to the current chunk with a causal mask too, selected chunks unmasked).
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train' or 'eval'

  Returns:
    the layer.
  """
  decoder_stack = [ChunkedDecoderLayer(d_feature, d_feedforward, n_heads,
                                       dropout, chunk_selector, mode)
                   for _ in range(n_layers)]
  # Below each Map(L) applies the layer L to each chunk independently.
  return tl.Model(
      tl.ShiftRight(),
      tl.Map(tl.Embedding(d_feature, vocab_size)),
      tl.Map(tl.Dropout(rate=dropout, mode=mode)),
      ChunkedPositionalEncoding(max_len=max_len),  # pylint: disable=no-value-for-parameter
      decoder_stack,
      tl.Map(tl.LayerNorm()),
      tl.Map(tl.Dense(vocab_size)),
      tl.Map(tl.LogSoftmax()),
  )
