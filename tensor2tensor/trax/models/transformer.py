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

from tensor2tensor.trax import layers as tl


def FeedForward(d_feature, d_feedforward, dropout, mode):
  """Feed-forward block with layer normalization at start."""
  return [
      tl.LayerNorm(),
      tl.Dense(d_feedforward),
      tl.Relu(),
      tl.Dropout(rate=dropout, mode=mode),
      tl.Dense(d_feature),
      tl.Dropout(rate=dropout, mode=mode),
  ]


def EncoderBlock(d_feature, d_feedforward, n_heads, dropout, mode):
  """Transformer encoder block.

  The input to the encoder is a pair (embedded source, mask) where
  the mask is created from the original source to prevent attending
  to the padding part of the input.

  Args:
    d_feature: int:  depth of embedding
    d_feedforward: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    mode: str: 'train' or 'eval'

  Returns:
    the layer, returning a pair (activations, mask).
  """
  attention = [
      tl.LayerNorm(),
      tl.MultiHeadedAttention(
          d_feature, n_heads=n_heads, dropout=dropout, mode=mode),
      tl.Dropout(rate=dropout, mode=mode),
  ]
  feed_forward = [
      FeedForward(d_feature, d_feedforward, dropout, mode=mode),
  ]
  return [
      tl.Residual(attention),
      tl.Residual(feed_forward),
  ]


def TransformerEncoder(vocab_size,
                       n_classes=10,
                       d_feature=512,
                       d_feedforward=2048,
                       n_layers=6,
                       n_heads=8,
                       dropout=0.1,
                       max_len=2048,
                       mode='train'):
  """Transformer encoder.

  Args:
    vocab_size: int: vocab size
    n_classes: how many classes on output
    d_feature: int:  depth of embedding
    d_feedforward: int: depth of feed-forward layer
    n_layers: int: number of encoder/decoder layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train' or 'eval'

  Returns:
    the Transformer encoder layer.
  """
  positional_embedder = [
      tl.Embedding(d_feature, vocab_size),
      tl.Dropout(rate=dropout, mode=mode),
      tl.PositionalEncoding(max_len=max_len),
  ]
  return tl.Model([
      tl.Branch(positional_embedder, tl.PaddingMask()),  # Create mask.
      [EncoderBlock(d_feature, d_feedforward, n_heads, dropout, mode)
       for _ in range(n_layers)],
      tl.Select(0),  # Drop mask.
      tl.LayerNorm(),
      tl.Mean(axis=1),  # Average on length.
      tl.Dense(n_classes),
      tl.LogSoftmax(),
  ])


def DecoderBlock(d_feature, d_feedforward, n_heads, dropout, mode):
  """Transformer decoder layer.

  Args:
    d_feature: int:  depth of embedding
    d_feedforward: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    mode: str: 'train' or 'eval'

  Returns:
    the layer.
  """
  self_attention = [
      tl.LayerNorm(),
      tl.Branch([], tl.CausalMask(axis=-2)),  # Create mask.
      tl.MultiHeadedAttention(
          d_feature, n_heads=n_heads, dropout=dropout, mode=mode),
      tl.Select(0),  # Drop mask.
      tl.Dropout(rate=dropout, mode=mode),
  ]
  feed_forward = [
      FeedForward(d_feature, d_feedforward, dropout, mode=mode),
  ]
  return [
      tl.Residual(self_attention),
      tl.Residual(feed_forward),
  ]


def TransformerLM(vocab_size,
                  d_feature=512,
                  d_feedforward=2048,
                  n_layers=6,
                  n_heads=8,
                  dropout=0.1,
                  max_len=2048,
                  mode='train'):
  """Transformer language model (only uses the decoder part of Transformer).

  Args:
    vocab_size: int: vocab size
    d_feature: int:  depth of embedding
    d_feedforward: int: depth of feed-forward layer
    n_layers: int: number of encoder/decoder layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train' or 'eval'

  Returns:
    the layer.
  """
  positional_embedder = [
      tl.Embedding(d_feature, vocab_size),
      tl.Dropout(rate=dropout, mode=mode),
      tl.PositionalEncoding(max_len=max_len),
  ]
  return tl.Model(
      tl.ShiftRight(),
      positional_embedder,
      [DecoderBlock(d_feature, d_feedforward, n_heads, dropout, mode)
       for _ in range(n_layers)],
      tl.LayerNorm(),
      tl.Dense(vocab_size),
      tl.LogSoftmax(),
  )


def EncoderDecoder(d_feature, d_feedforward, n_heads, dropout, mode):
  """Transformer encoder-decoder layer.

  The input is a triple pair (decoder_input, mask, encoder) where
  the mask is created from the original source to prevent attending
  to the padding part of the encoder.

  Args:
    d_feature: int:  depth of embedding
    d_feedforward: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    mode: str: 'train' or 'eval'

  Returns:
    the layer, returning a triple (decoder_activations, mask, encoder).
  """
  decoder_self_attention = [
      # TODO(jonni): Work on combinators so that this flow is cleaner/clearer.
      tl.LayerNorm(),
      tl.Dup(),
      tl.CausalMask(axis=-2),  # Create the self-attention mask.
      tl.Swap(),  # Put mask behind the activations.
      tl.MultiHeadedAttention(d_feature, n_heads=n_heads,
                              dropout=dropout, mode=mode),
      tl.Swap(),  # Put self-attention mask on top.
      tl.Drop(),   # Drop self-attention mask.
      tl.Dropout(rate=dropout, mode=mode),
  ]
  decoder_to_encoder_attention = [
      tl.Select((0, 2, 2, 1, 2)),  # (dec, enc, enc, mask, enc-copy)
      tl.MultiHeadedAttentionQKV(  # (q, k, v, mask, ...) --> (new, mask, ...)
          d_feature, n_heads=n_heads, dropout=dropout, mode=mode),
      tl.Dropout(rate=dropout, mode=mode),
  ]
  feed_forward = [
      FeedForward(d_feature, d_feedforward, dropout, mode=mode),
  ]
  return [
      tl.Residual(decoder_self_attention),
      tl.Residual(decoder_to_encoder_attention),
      tl.Residual(feed_forward),
  ]


# TODO(lukaszkaiser): allow different source and target vocabularies.
def Transformer(vocab_size,
                d_feature=512,
                d_feedforward=2048,
                n_layers=6,
                n_heads=8,
                dropout=0.1,
                max_len=2048,
                mode='train'):
  """Transformer.

  This model expects on input a pair (source, target).

  Args:
    vocab_size: int: vocab size (shared source and target).
    d_feature: int:  depth of embedding
    d_feedforward: int: depth of feed-forward layer
    n_layers: int: number of encoder/decoder layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train' or 'eval'

  Returns:
    the Transformer model.
  """
  positional_embedder = [
      tl.Embedding(d_feature, vocab_size),
      tl.Dropout(rate=dropout, mode=mode),
      tl.PositionalEncoding(max_len=max_len),
  ]
  encoder = [
      tl.Branch(positional_embedder, tl.PaddingMask()),
      [EncoderBlock(d_feature, d_feedforward, n_heads, dropout, mode)
       for _ in range(n_layers)],
      tl.LayerNorm(),
  ]
  return tl.Model(
      tl.Parallel([], tl.ShiftRight()),
      tl.Parallel(encoder, positional_embedder),
      tl.Select(inputs=(('encoder', 'mask'), 'decoder'),
                output=('decoder', ('mask', 'decoder'), 'encoder')),
      # (encoder_mask, decoder_input) -> encoder-decoder mask
      tl.Parallel([], tl.EncoderDecoderMask(), []),
      [EncoderDecoder(d_feature, d_feedforward, n_heads, dropout, mode)
       for _ in range(n_layers)],
      tl.Select(0),  # Drop mask and encoder.
      tl.LayerNorm(),
      tl.Dense(vocab_size),
      tl.LogSoftmax(),
  )
