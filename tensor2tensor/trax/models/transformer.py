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

from tensor2tensor.trax import layers


def ResidualFeedForward(feature_depth,
                        feedforward_depth,
                        dropout,
                        mode):
  """Residual feed-forward layer with normalization at start."""
  return layers.Residual(
      layers.LayerNorm(),
      layers.Dense(feedforward_depth),
      layers.Relu(),
      layers.Dropout(rate=dropout, mode=mode),
      layers.Dense(feature_depth),
      layers.Dropout(rate=dropout, mode=mode)
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
  # The encoder block expects (activation, mask) as input and returns
  # the new activations only, we add the mask back to output next.
  encoder_block = layers.Serial(
      layers.Residual(  # Attention block here.
          layers.Parallel(layers.LayerNorm(), layers.Identity()),
          layers.MultiHeadedAttention(feature_depth, num_heads=num_heads,
                                      dropout=dropout, mode=mode),
          layers.Dropout(rate=dropout, mode=mode),
          shortcut=layers.FirstBranch()
      ),
      ResidualFeedForward(feature_depth, feedforward_depth, dropout, mode=mode)
  )
  # Now we add the mask back.
  return layers.Serial(
      layers.Reorder(output=((0, 1), 1)),  # (x, mask) --> ((x, mask), mask)
      layers.Parallel(encoder_block, layers.Identity())
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
  input_embedding = layers.Serial(
      layers.Embedding(feature_depth, vocab_size),
      layers.Dropout(rate=dropout, mode=mode),
      layers.PositionalEncoding(max_len=max_len)
  )
  return layers.Serial(
      layers.Branch(),  # Branch input to create embedding and mask.
      layers.Parallel(input_embedding, layers.PaddingMask()),
      layers.Serial(*[EncoderLayer(feature_depth, feedforward_depth, num_heads,
                                   dropout, mode)
                      for _ in range(num_layers)]),
      layers.FirstBranch(),  # Drop the mask.
      layers.LayerNorm(),
      layers.Mean(axis=1),  # Average on length.
      layers.Dense(num_classes),
      layers.LogSoftmax()
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
  return layers.Serial(
      layers.Residual(  # Self-attention block.
          layers.LayerNorm(),
          layers.Branch(),
          layers.Parallel(layers.Identity(),  # activation for (q, k, v)
                          layers.CausalMask(axis=-2)),  # attention mask
          layers.MultiHeadedAttention(feature_depth, num_heads=num_heads,
                                      dropout=dropout, mode=mode),
          layers.Dropout(rate=dropout, mode=mode)
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
  return layers.Serial(
      layers.ShiftRight(),
      layers.Embedding(feature_depth, vocab_size),
      layers.Dropout(rate=dropout, mode=mode),
      layers.PositionalEncoding(max_len=max_len),
      layers.Serial(*[DecoderLayer(feature_depth, feedforward_depth, num_heads,
                                   dropout, mode)
                      for _ in range(num_layers)]),
      layers.LayerNorm(),
      layers.Dense(vocab_size),
      layers.LogSoftmax()
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
  return layers.Serial(
      layers.Residual(  # Self-attention block.
          layers.Map(layers.LayerNorm()),
          layers.ChunkedCausalMultiHeadedAttention(
              feature_depth, num_heads=num_heads, dropout=dropout,
              chunk_selector=chunk_selector, mode=mode),
          layers.Map(layers.Dropout(rate=dropout, mode=mode)),
      ),
      layers.Map(ResidualFeedForward(
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
  return layers.Serial(
      layers.ShiftRight(),
      layers.Map(layers.Embedding(feature_depth, vocab_size)),
      layers.Map(layers.Dropout(rate=dropout, mode=mode)),
      layers.PositionalEncoding(max_len=max_len),
      layers.Serial(*stack),
      layers.Map(layers.LayerNorm()),
      layers.Map(layers.Dense(vocab_size)),
      layers.Map(layers.LogSoftmax()),
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
  self_attention = layers.Residual(
      layers.LayerNorm(),
      layers.Branch(),
      layers.Parallel(layers.Identity(),  # activation for (q, k, v)
                      layers.CausalMask(axis=-2)),  # attention mask
      layers.MultiHeadedAttention(feature_depth, num_heads=num_heads,
                                  dropout=dropout, mode=mode),
      layers.Dropout(rate=dropout, mode=mode)
  )
  # Decoder attending to encoder.
  encoder_decoder_attention = layers.Serial(
      layers.Reorder(output=((2, 0, 0), 1)),  # ((dec, enc, enc), mask)
      layers.MultiHeadedAttentionQKV(  # ((q, k, v), mask) --> new v
          feature_depth, num_heads=num_heads, dropout=dropout, mode=mode),
      layers.Dropout(rate=dropout, mode=mode),
  )
  return layers.Serial(
      layers.Parallel(layers.Identity(), layers.Identity(), self_attention),
      layers.Branch(),
      layers.Parallel(layers.Identity(), encoder_decoder_attention),
      layers.UnnestBranches(),   # (encoder, mask, old_act, new_act)
      layers.Reorder(output=(0, 1, (2, 3))),
      layers.Parallel(  # Residual after encoder-decoder attention.
          layers.Identity(), layers.Identity(), layers.SumBranches()),
      layers.Parallel(  # Feed-forward on the third component (decoder).
          layers.Identity(), layers.Identity(), ResidualFeedForward(
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
  embedding = layers.Serial(
      layers.Embedding(feature_depth, vocab_size),
      layers.Dropout(rate=dropout, mode=mode),
      layers.PositionalEncoding(max_len=max_len)
  )
  encoder = layers.Serial(
      layers.Branch(),  # Branch input to create embedding and mask.
      layers.Parallel(embedding, layers.PaddingMask()),
      layers.Serial(*[EncoderLayer(feature_depth, feedforward_depth, num_heads,
                                   dropout, mode)
                      for _ in range(num_layers)]),
      layers.Parallel(layers.LayerNorm(), layers.Identity())
  )
  stack = [EncoderDecoderLayer(feature_depth, feedforward_depth, num_heads,
                               dropout, mode)
           for _ in range(num_layers)]
  return layers.Serial(
      layers.Parallel(layers.Identity(), layers.ShiftRight()),
      layers.Parallel(encoder, embedding),
      layers.UnnestBranches(),  # (encoder, encoder_mask, decoder_input)
      layers.Reorder(output=(0, (1, 2), 2)),
      layers.Parallel(  # (encoder_mask, decoder_input) -> encoder-decoder mask
          layers.Identity(), layers.EncoderDecoderMask(), layers.Identity()),
      layers.Serial(*stack),
      layers.ThirdBranch(),
      layers.LayerNorm(),
      layers.Dense(vocab_size),
      layers.LogSoftmax()
  )
