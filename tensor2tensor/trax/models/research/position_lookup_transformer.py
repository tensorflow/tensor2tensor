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

"""Deep Lookups for Transformer Positions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp

from tensor2tensor.trax import layers as tl
from tensor2tensor.trax.backend import numpy as np


# pylint: disable=g-complex-comprehension
# pylint: disable=no-value-for-parameter

POS_VECTOR_SIZE = 32
_ABSOLUTE_MAX_LEN = 10000
_POSITIONS = onp.random.uniform(size=[_ABSOLUTE_MAX_LEN, POS_VECTOR_SIZE])


@tl.layer()
def NewPositionalEncoding(x, positions=None, **kwargs):
  """Implements new positional encoding."""
  del kwargs
  x_length = np.shape(x)[1]
  pos = np.array(positions)[np.newaxis, :x_length, :]
  pos += np.zeros((np.shape(x)[0], 1, 1))  # Broadcast on batch.
  res = np.concatenate([x, pos], axis=2)
  return res


# TODO(lukaszkaiser): This used to have stack_items_to_pass=0; fix as needed.
@tl.layer()
def CutPosition(xs, **unused_kwargs):
  """Splits x into a pair (x[:position], position)."""
  if not isinstance(xs, (list, tuple)):
    xs = [xs]
  x = xs[0]
  res = [x[:, :, :-POS_VECTOR_SIZE], x[:, :, -POS_VECTOR_SIZE:]]
  return tuple(res + list(xs[1:]))


@tl.layer()
def MixHeadsPos(x, h=8, **unused_kwargs):
  """Mix x = (x0, p) into x0_h1, p, x0_h2, p, ...."""
  head_size = (x.shape[2] - POS_VECTOR_SIZE) // h
  p = x[:, :, -POS_VECTOR_SIZE:]
  res, idx = [], 0
  for _ in range(h):
    res.append(x[:, :, idx:idx+head_size])
    res.append(p)
    idx += head_size
  return np.concatenate(res, axis=-1)


@tl.layer()
def CombineHeadsPos(x, h=8, **unused_kwargs):
  """Mix x = (x0, p0, ..., xH, pH) into x0, ...., xH, p_combined.

  The positions are added as vectors.

  Args:
    x: input vector, concatenated (x0, p0, ..., xH, pH).
    h: number of heads.

  Returns:
    the vector with combined positions.
  """
  head_size = int((x.shape[2] / h) - POS_VECTOR_SIZE)
  res, positions, idx = [], [], 0
  for _ in range(h):
    res.append(x[:, :, idx:idx+head_size])
    idx += head_size
    positions.append(x[:, :, idx:idx+POS_VECTOR_SIZE])
    idx += POS_VECTOR_SIZE
  combined_position = sum(positions)
  res.append(combined_position)
  return np.concatenate(res, axis=-1)


@tl.layer()
def CopyHeadsPos(x, h=8, **unused_kwargs):
  """Mix x = (x, p) into x_h1, p_h1, x_h2, p_h2, ...."""
  head_size = (x.shape[2] - h*POS_VECTOR_SIZE) // h
  p = x[:, :, -h*POS_VECTOR_SIZE:]
  res, idx = [], 0
  for i in range(h):
    res.append(x[:, :, idx:idx+head_size])
    res.append(p[:, :, i*POS_VECTOR_SIZE:(i+1)*POS_VECTOR_SIZE])
    idx += head_size
  return np.concatenate(res, axis=-1)


def DeepFlatten(xs):
  for x in xs:
    if isinstance(x, (list, tuple)):
      for y in DeepFlatten(x):
        yield y
    else:
      yield x


# TODO(lukaszkaiser): This used to have stack_items_to_pass=0; fix as needed.
@tl.layer()
def Unnest(xs, **unused_kwargs):
  return [x for x in DeepFlatten(xs)]


# TODO(lukaszkaiser): This used to have stack_items_to_pass=0; fix as needed.
@tl.layer()
def ConcatenateN(xs, params, n=2, axis=-1, **kwargs):
  """Concatenate first N inputs (and output remainder as is if non-empty)."""
  del params, kwargs
  res = np.concatenate(xs[:n], axis)
  rest = list(xs[n:])
  if rest:
    return tuple([res] + rest)
  return res


def PreservePosition(layer):
  """Execute layer without position but preserve it in parallel."""
  return tl.Serial(
      CutPosition(),
      layer,
      ConcatenateN()
  )


def ApplyAndQueryPositions(layer, pos):
  """Execute layer without position and pos-layers on positions.

  This takes an embedding including position x = (emb, p), and
  outputs layer(emb).pos1(x, p).....layer(emb).posn(x, p)
  where pos=[pos1...posn].

  Args:
    layer: layer to be executed without position information.
    pos: list of layers to be applied to positions.

  Returns:
    the result of this application.
  """
  n_heads = len(pos)
  return tl.Serial(
      tl.Dup(),
      CutPosition(),
      # TODO(lukaszkaiser): Rewrite without using Select.
      tl.Select(tuple([0] + [(2, 1)]*n_heads)),
      tl.Parallel(*([layer] + pos)),
      Unnest(),
      ConcatenateN(n=n_heads + 1)
  )


@tl.layer()
def QueryPositionKV(x, keys=None, values=None, binary=False, **unused_kwargs):
  """Query a table with a position vector."""
  if keys is None:
    return x
  k = np.array(keys)
  v = np.array(values)
  q = x
  if binary:
    q = np.concatenate([x, x], axis=-1)
  return tl.DotProductAttention(q, k, v, None, None, None, None)


def LearnedQP(keys=None, values=None, binary=False):
  """Get (query, pos), make learned weight of qeury and return with pos."""
  return tl.Parallel(
      tl.Dense(1),
      QueryPositionKV(keys=keys, values=values, binary=binary),
  )


# TODO(lukaszkaiser): This used to have stack_items_to_pass=0; fix as needed.
@tl.layer()
def SoftmaxBranches(x_list_in, n_branches=2, **unused_kwargs):
  """Softmax xs.

  The input xs is a list of embeddings and weights of the form
  w_1 e_1 .... w_n e_n (followed by optional rest that is preserved).

  Args:
    x_list_in: the input weights and embeddings.
    n_branches: what part of the list to use.

  Returns:
    softmax(w) * e for the joint weights w and embeddings e.
  """
  x_list, x_list_rest = x_list_in[:2*n_branches], x_list_in[2*n_branches:]
  softmax_activations = [x_list[2*i] for i in range(n_branches)]
  max_sa = softmax_activations[0]
  for x in softmax_activations:
    max_sa = np.maximum(max_sa, x)
  softmax_activations = [x - max_sa for x in softmax_activations]
  softmax_activations = [np.exp(x) for x in softmax_activations]
  sum_sa = sum(softmax_activations)
  softmax_activations = [x / sum_sa for x in softmax_activations]
  res = sum([x_list[2*i+1] * softmax_activations[i] for i in range(n_branches)])
  return tuple([res] + list(x_list_rest))


def SumLearnedPick(positions):
  """Get a pair (vec, pos) and pick new pos."""
  succ_keys = positions[:-1, :]
  succ_values = positions[1:, :]
  subtract_1_keys = positions[1:, :]
  subtract_1_values = positions[:-1, :]
  l = int(positions.shape[0]) // 2
  add_keys = np.array([np.concatenate([positions[i, :], positions[j, :]])
                       for i in range(l) for j in range(l)])
  add_values = np.array([positions[i + j, :]
                         for i in range(l) for j in range(l)])
  # TODO(lukaszkaiser): try this below: "for j in range(i) for i in range(2*l)"
  sub_keys = np.array([np.concatenate([positions[i, :], positions[j, :]])
                       for j in range(l) for i in range(l)])
  sub_values = np.array([positions[max(i - j, 0), :]
                         for j in range(l) for i in range(l)])
  return tl.Serial(
      tl.Dup(), tl.Dup(), tl.Dup(), tl.Dup(),
      tl.Parallel(
          LearnedQP(),
          LearnedQP(keys=succ_keys, values=succ_values),
          LearnedQP(keys=subtract_1_keys, values=subtract_1_values),
          LearnedQP(keys=add_keys, values=add_values, binary=True),
          LearnedQP(keys=sub_keys, values=sub_values, binary=True),
      ),
      Unnest(),
      SoftmaxBranches(n_branches=5)
  )


def MultiHeadedAttentionPosition(
    positions, d_feature, n_heads=8, dropout=0.0, mode='train'):
  """Transformer-style multi-headed attention."""
  return tl.Serial(
      tl.Dup(),
      tl.Dup(),
      tl.Parallel(
          ApplyAndQueryPositions(tl.Dense(d_feature),
                                 pos=[SumLearnedPick(positions)
                                      for _ in range(n_heads)]),
          PreservePosition(tl.Dense(d_feature)),
          PreservePosition(tl.Dense(d_feature)),
      ),
      tl.Parallel(
          CopyHeadsPos(h=n_heads),
          MixHeadsPos(h=n_heads),
          MixHeadsPos(h=n_heads),
      ),
      tl.PureMultiHeadedAttention(
          d_feature=d_feature, n_heads=n_heads,
          dropout=dropout, mode=mode),
      tl.Parallel([], tl.Drop()),  # Drop the mask.
      CombineHeadsPos(h=n_heads),
      PreservePosition(tl.Dense(d_feature)),
  )


def ResidualFeedForward(d_feature,
                        d_feedforward,
                        dropout,
                        mode):
  """Residual feed-forward layer with normalization at start."""
  stack = tl.Serial(
      tl.LayerNorm(),
      tl.Dense(d_feedforward),
      tl.Relu(),
      tl.Dropout(rate=dropout, mode=mode),
      tl.Dense(d_feature),
      tl.Dropout(rate=dropout, mode=mode)
  )
  return tl.Residual(PreservePosition(stack))


def DecoderLayer(positions,
                 d_feature,
                 d_feedforward,
                 n_heads,
                 dropout,
                 mode):
  """Transformer decoder layer.

  Args:
    positions: random vectors for positions
    d_feature: int:  depth of embedding
    d_feedforward: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    mode: str: 'train' or 'eval'

  Returns:
    the layer.
  """
  return [
      tl.Residual(  # Self-attention block.
          PreservePosition(tl.LayerNorm()),
          tl.Dup(),
          tl.Parallel([],  # activation for (q, k, v)
                      tl.CausalMask(axis=-2)),  # attention mask
          MultiHeadedAttentionPosition(positions,
                                       d_feature, n_heads=n_heads,
                                       dropout=dropout, mode=mode),
          PreservePosition(tl.Dropout(rate=dropout, mode=mode))
      ),
      ResidualFeedForward(d_feature, d_feedforward, dropout, mode=mode)
  ]


def PositionLookupTransformerLM(vocab_size=128,
                                d_feature=256,
                                d_feedforward=512,
                                n_layers=3,
                                n_heads=4,
                                dropout=0.1,
                                max_len=100,
                                mode='train'):
  """Transformer language model (only uses the decoder part of Transformer).

  Args:
    vocab_size: int: vocab size
    d_feature: int:  depth of embedding
    d_feedforward: int: depth of feed-forward layer
    n_layers: int: number of layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    max_len: maximal length
    mode: str: 'train' or 'eval'

  Returns:
    the layer.
  """
  positions = _POSITIONS[:max_len, :]
  return tl.Serial([
      tl.ShiftRight(),
      tl.Embedding(d_feature, vocab_size),
      tl.Dropout(rate=dropout, mode=mode),
      NewPositionalEncoding(positions=positions),
      [DecoderLayer(positions, d_feature, d_feedforward, n_heads, dropout, mode)
       for _ in range(n_layers)],
      PreservePosition(tl.LayerNorm()),
      tl.Dense(vocab_size),
      tl.LogSoftmax()
  ])
