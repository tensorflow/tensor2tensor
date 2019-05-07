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


def EncoderLayer(feature_depth,
                 feedforward_depth,
                 num_heads,
                 dropout,
                 mode):
  """Transformer encoder layer.

  The input to the encoder is a pair (embedded source, mask) where
  the mask is created from the original source to prevent attending
  to the padding part of the input.

  Args:
    feature_depth: int:  depth of embedding
    feedforward_depth: int: depth of feed-forward layer
    num_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    mode: str: 'train' or 'eval'

  Returns:
    the layer, returning a pair (actiavtions, mask).
  """
  return tl.Serial(
      tl.Residual(  # Attention block here.
          tl.Parallel(tl.LayerNorm(), tl.Copy()),
          tl.MultiHeadedAttention(feature_depth, num_heads=num_heads,
                                  dropout=dropout, mode=mode),
          tl.Parallel(tl.Dropout(rate=dropout, mode=mode), tl.Copy())
      ),
      tl.Parallel(
          ResidualFeedForward(
              feature_depth, feedforward_depth, dropout, mode=mode),
          tl.Div(divisor=2.0)  # Mask added to itself in the residual, divide.
      )
  )


def TransformerEncoder(vocab_size,
                       num_classes=10,
                       feature_depth=512,
                       feedforward_depth=2048,
                       num_layers=6,
                       num_heads=8,
                       dropout=0.1,
                       max_len=2048,
                       mode='train'):
  """Transformer encoder.

  Args:
    vocab_size: int: vocab size
    num_classes: how many classes on output
    feature_depth: int:  depth of embedding
    feedforward_depth: int: depth of feed-forward layer
    num_layers: int: number of encoder/decoder layers
    num_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train' or 'eval'

  Returns:
    the Transformer encoder layer.
  """
  input_embedding = tl.Serial(
      tl.Embedding(feature_depth, vocab_size),
      tl.Dropout(rate=dropout, mode=mode),
      tl.PositionalEncoding(max_len=max_len)
  )
  return tl.Serial(
      tl.Branch(input_embedding, tl.PaddingMask()),
      tl.Serial(*[EncoderLayer(feature_depth, feedforward_depth, num_heads,
                               dropout, mode)
                  for _ in range(num_layers)]),
      tl.Select(0),  # Drop the mask.
      tl.LayerNorm(),
      tl.Mean(axis=1),  # Average on length.
      tl.Dense(num_classes),
      tl.LogSoftmax()
  )


def DecoderLayer(feature_depth,
                 feedforward_depth,
                 num_heads,
                 dropout,
                 mode):
  """Transformer decoder layer.

  Args:
    feature_depth: int:  depth of embedding
    feedforward_depth: int: depth of feed-forward layer
    num_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    mode: str: 'train' or 'eval'

  Returns:
    the layer.
  """
  return tl.Serial(
      tl.Residual(  # Self-attention block.
          tl.LayerNorm(),
          tl.Branch(tl.Copy(), tl.CausalMask(axis=-2)),  # Create mask.
          tl.MultiHeadedAttention(feature_depth, num_heads=num_heads,
                                  dropout=dropout, mode=mode),
          tl.Select(0),  # Drop the mask.
          tl.Dropout(rate=dropout, mode=mode)
      ),
      ResidualFeedForward(feature_depth, feedforward_depth, dropout, mode=mode)
  )


def TransformerLM(vocab_size,
                  feature_depth=512,
                  feedforward_depth=2048,
                  num_layers=6,
                  num_heads=8,
                  dropout=0.1,
                  max_len=2048,
                  mode='train'):
  """Transformer language model (only uses the decoder part of Transformer).

  Args:
    vocab_size: int: vocab size
    feature_depth: int:  depth of embedding
    feedforward_depth: int: depth of feed-forward layer
    num_layers: int: number of encoder/decoder layers
    num_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train' or 'eval'

  Returns:
    the layer.
  """
  return tl.Serial(
      tl.ShiftRight(),
      tl.Embedding(feature_depth, vocab_size),
      tl.Dropout(rate=dropout, mode=mode),
      tl.PositionalEncoding(max_len=max_len),
      tl.Serial(*[DecoderLayer(feature_depth, feedforward_depth, num_heads,
                               dropout, mode)
                  for _ in range(num_layers)]),
      tl.LayerNorm(),
      tl.Dense(vocab_size),
      tl.LogSoftmax()
  )


def EncoderDecoderLayer(feature_depth,
                        feedforward_depth,
                        num_heads,
                        dropout,
                        mode):
  """Transformer encoder-decoder layer.

  The input is a triple pair (encoder, mask, decoder_input) where
  the mask is created from the original source to prevent attending
  to the padding part of the encoder.

  Args:
    feature_depth: int:  depth of embedding
    feedforward_depth: int: depth of feed-forward layer
    num_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    mode: str: 'train' or 'eval'

  Returns:
    the layer, returning a triple (encoder, mask, decoder_activations).
  """
  # Decoder self-attending to decoder.
  self_attention = tl.Residual(
      tl.LayerNorm(),
      tl.Branch(tl.Copy(), tl.CausalMask(axis=-2)),  # create mask
      tl.MultiHeadedAttention(feature_depth, num_heads=num_heads,
                              dropout=dropout, mode=mode),
      tl.Select(0),  # drop mask
      tl.Dropout(rate=dropout, mode=mode)
  )
  # Decoder attending to encoder.
  encoder_decoder_attention = tl.Serial(
      tl.Select(((2, 0, 0), 1)),  # ((dec, enc, enc), mask)
      tl.MultiHeadedAttentionQKV(  # ((q, k, v), mask) --> new, mask
          feature_depth, num_heads=num_heads, dropout=dropout, mode=mode),
      tl.Select(0),  # drop the mask
      tl.Dropout(rate=dropout, mode=mode),
  )
  return tl.Serial(
      tl.Parallel(tl.Copy(), tl.Copy(), self_attention),
      tl.Branch(tl.Copy(), encoder_decoder_attention),
      tl.UnnestBranches(),   # (encoder, mask, old_act, new_act)
      tl.Select((0, 1, (2, 3))),
      tl.Parallel(  # Residual after encoder-decoder attention.
          tl.Copy(), tl.Copy(), tl.Add()),
      tl.Parallel(  # Feed-forward on the third component (decoder).
          tl.Copy(), tl.Copy(), ResidualFeedForward(
              feature_depth, feedforward_depth, dropout, mode=mode)
      )
  )


# TODO(lukaszkaiser): allow different source and target vocabularies.
def Transformer(vocab_size,
                feature_depth=512,
                feedforward_depth=2048,
                num_layers=6,
                num_heads=8,
                dropout=0.1,
                max_len=2048,
                mode='train'):
  """Transformer.

  This model expects on input a pair (source, target).

  Args:
    vocab_size: int: vocab size (shared source and target).
    feature_depth: int:  depth of embedding
    feedforward_depth: int: depth of feed-forward layer
    num_layers: int: number of encoder/decoder layers
    num_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train' or 'eval'

  Returns:
    the Transformer model.
  """
  embedding = tl.Serial(
      tl.Embedding(feature_depth, vocab_size),
      tl.Dropout(rate=dropout, mode=mode),
      tl.PositionalEncoding(max_len=max_len)
  )
  encoder = tl.Serial(
      tl.Branch(embedding, tl.PaddingMask()),
      tl.Serial(*[EncoderLayer(feature_depth, feedforward_depth, num_heads,
                               dropout, mode)
                  for _ in range(num_layers)]),
      tl.Parallel(tl.LayerNorm(), tl.Copy())
  )
  stack = [EncoderDecoderLayer(feature_depth, feedforward_depth, num_heads,
                               dropout, mode)
           for _ in range(num_layers)]
  return tl.Serial(
      tl.Parallel(tl.Copy(), tl.ShiftRight()),
      tl.Parallel(encoder, embedding),
      tl.UnnestBranches(),  # (encoder, encoder_mask, decoder_input)
      tl.Select((0, (1, 2), 2)),
      tl.Parallel(  # (encoder_mask, decoder_input) -> encoder-decoder mask
          tl.Copy(), tl.EncoderDecoderMask(), tl.Copy()),
      tl.Serial(*stack),
      tl.Select(2),  # Drop encoder and mask.
      tl.LayerNorm(),
      tl.Dense(vocab_size),
      tl.LogSoftmax()
  )
