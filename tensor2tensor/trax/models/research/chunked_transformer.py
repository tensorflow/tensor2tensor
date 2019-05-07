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

from tensor2tensor.trax import layers as tl
from tensor2tensor.trax.backend import numpy as np


# Chunked attention.
def _chunked_selector_output_shape(  # pylint: disable=invalid-name
    input_shapes, selector=None, **unused_kwargs):
  """Helper: calculate output shape for chunked key selector (see below)."""
  # Read the main function below first, the shape logic just follows the ops.
  selector = selector or (lambda x: [] if x < 1 else [x-1])
  triples, _ = zip(*input_shapes)
  (query_shapes, key_shapes, value_shapes) = zip(*triples)
  result = []
  for i in range(len(input_shapes)):
    selected = selector(i)
    cur_key_shape, cur_value_shape = key_shapes[i], value_shapes[i]
    # Since keys and values are [batch, length, depth] we concatenate on axis=1.
    new_key_len = sum([key_shapes[j][1] for j in selected]) + cur_key_shape[1]
    new_key_shape = (cur_key_shape[0], new_key_len, cur_key_shape[2])
    new_value_len = sum(
        [value_shapes[j][1] for j in selected]) + cur_value_shape[1]
    new_value_shape = (cur_value_shape[0], new_value_len, cur_value_shape[2])
    # Masks are (1, query-len, key-len).
    new_mask_shape = (1, query_shapes[i][1], new_key_len)
    new_shape = ((query_shapes[i], new_key_shape, new_value_shape),
                 new_mask_shape)
    result.append(new_shape)
  return tuple(result)


@tl.layer(output_shape=_chunked_selector_output_shape)
def ChunkedAttentionSelector(x, params, selector=None, **kwargs):
  """Select which chunks to attend to in chunked attention.

  Args:
    x: inputs, a list of elements of the form (q, k, v), mask for each chunk.
    params: parameters (unused).
    selector: a function from chunk_number -> list of chunk numbers that says
      which other chunks should be appended to the given one (previous if None).
    **kwargs: unused other arguments.

  Returns:
    a list of elements of the form (q, k', v'), mask' where k', v' and mask' are
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
    result.append(((queries[i], new_key, new_value), new_mask))
  return tuple(result)


def ChunkedCausalMultiHeadedAttention(
    feature_depth, num_heads=8, dropout=0.0, chunk_selector=None, mode='train'):
  """Transformer-style causal multi-headed attention operating on chunks.

  Accepts inputs that are a list of chunks and applies causal attention.

  Args:
    feature_depth: int:  depth of embedding
    num_heads: int: number of attention heads
    dropout: float: dropout rate
    chunk_selector: a function from chunk number to list of chunks to attend.
    mode: str: 'train' or 'eval'

  Returns:
    Multi-headed self-attention layer.
  """
  prepare_attention_input = tl.Serial(
      tl.Branch(
          tl.Branch(  # q = k = v = first input
              tl.Copy(), tl.Copy(), tl.Copy()),
          tl.CausalMask(axis=-2),
      ),
      tl.Parallel(
          tl.Parallel(
              tl.Dense(feature_depth),
              tl.Dense(feature_depth),
              tl.Dense(feature_depth),
          ),
          tl.Copy()
      )
  )
  return tl.Serial(
      tl.Map(prepare_attention_input),
      ChunkedAttentionSelector(selector=chunk_selector),  # pylint: disable=no-value-for-parameter
      tl.Map(tl.PureMultiHeadedAttention(
          feature_depth=feature_depth, num_heads=num_heads,
          dropout=dropout, mode=mode), check_shapes=False),
      tl.Map(tl.Select(0), check_shapes=False),  # drop masks
      tl.Map(tl.Dense(feature_depth))
  )


def ResidualFeedForward(feature_depth,
                        feedforward_depth,
                        dropout,
                        mode):
  """Residual feed-forward layer with normalization at start."""
  return tl.Residual(
      tl.LayerNorm(),
      tl.Dense(feedforward_depth),
      tl.Relu(),
      tl.Dropout(rate=dropout, mode=mode),
      tl.Dense(feature_depth),
      tl.Dropout(rate=dropout, mode=mode)
  )


def ChunkedDecoderLayer(feature_depth,
                        feedforward_depth,
                        num_heads,
                        dropout,
                        chunk_selector,
                        mode):
  """Transformer decoder layer operating on chunks.

  Args:
    feature_depth: int:  depth of embedding
    feedforward_depth: int: depth of feed-forward layer
    num_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    chunk_selector: a function from chunk number to list of chunks to attend.
    mode: str: 'train' or 'eval'

  Returns:
    the layer.
  """
  return tl.Serial(
      tl.Residual(  # Self-attention block.
          tl.Map(tl.LayerNorm()),
          ChunkedCausalMultiHeadedAttention(
              feature_depth, num_heads=num_heads, dropout=dropout,
              chunk_selector=chunk_selector, mode=mode),
          tl.Map(tl.Dropout(rate=dropout, mode=mode)),
      ),
      tl.Map(ResidualFeedForward(
          feature_depth, feedforward_depth, dropout, mode=mode))
  )


def ChunkedTransformerLM(vocab_size,
                         feature_depth=512,
                         feedforward_depth=2048,
                         num_layers=6,
                         num_heads=8,
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
    feature_depth: int:  depth of embedding
    feedforward_depth: int: depth of feed-forward layer
    num_layers: int: number of encoder/decoder layers
    num_heads: int: number of attention heads
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
  stack = [ChunkedDecoderLayer(feature_depth, feedforward_depth, num_heads,
                               dropout, chunk_selector, mode)
           for _ in range(num_layers)]
  # Below each Map(L) applies the layer L to each chunk independently.
  return tl.Serial(
      tl.ShiftRight(),
      tl.Map(tl.Embedding(feature_depth, vocab_size)),
      tl.Map(tl.Dropout(rate=dropout, mode=mode)),
      tl.PositionalEncoding(max_len=max_len),
      tl.Serial(*stack),
      tl.Map(tl.LayerNorm()),
      tl.Map(tl.Dense(vocab_size)),
      tl.Map(tl.LogSoftmax()),
  )
