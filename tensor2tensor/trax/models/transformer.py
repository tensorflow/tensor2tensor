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
  """Returns a layer sequence that implements a Transformer encoder block.

  The input to the layer sequence is a pair, (activations, mask), where the
  mask was created from the original source tokens to prevent attending to the
  padding part of the input.

  Args:
    d_feature: int:  depth of embedding
    d_feedforward: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    mode: str: 'train' or 'eval'

  Returns:
    A sequence of layers that maps an (activations, mask) pair to an
    (activations, mask) pair.
  """
  attention = [
      tl.LayerNorm(),
      tl.Attention(d_feature, n_heads=n_heads, dropout=dropout, mode=mode),
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
  """Returns a Transformer encoder model.

  The input to the model is a tensor of tokens.

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
    A Transformer model as a layer that maps from a tensor of tokens to
    activations over a set of output classes.
  """
  embedder = [
      tl.Embedding(d_feature, vocab_size),
      tl.Dropout(rate=dropout, mode=mode),
      tl.PositionalEncoding(max_len=max_len),
  ]
  return tl.Model([                             #      tokens
      tl.Dup(),                                 # toks toks
      tl.Parallel(embedder, tl.PaddingMask()),  # vecs mask
      [EncoderBlock(d_feature, d_feedforward, n_heads, dropout, mode)
       for _ in range(n_layers)],               # vecs mask
      tl.Parallel([], tl.Drop()),               # ____  0
      tl.LayerNorm(),                           # vecs
      tl.Mean(axis=1),  # Average on length.    # vecs
      tl.Dense(n_classes),                      # vecs
      tl.LogSoftmax(),                          # vecs
  ])


def DecoderBlock(d_feature, d_feedforward, n_heads, dropout, mode):
  """Returns a layer sequence that implements a Transformer decoder block.

  The input to the layer sequence is an activation tensor.

  Args:
    d_feature: int:  depth of embedding
    d_feedforward: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    mode: str: 'train' or 'eval'

  Returns:
    A sequence of layers that maps an activation tensor to an activation tensor.
  """
  self_attention = [
      tl.LayerNorm(),  # vec
      tl.Dup(),  # vec vec
      tl.Parallel([], tl.CausalMask(axis=-2)),  # vec mask
      tl.Attention(d_feature, n_heads=n_heads, dropout=dropout, mode=mode),
      tl.Parallel([], tl.Drop()),  # vec
      tl.Dropout(rate=dropout, mode=mode),  # vec
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
  """Returns a Transformer language model.

  The input to the model is a tensor of tokens. (This model uses only the
  decoder part of the overall Transformer.)

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
    A Transformer language model as a layer that maps from a tensor of tokens
    to activations over a vocab set.
  """
  embedder = [
      tl.Embedding(d_feature, vocab_size),
      tl.Dropout(rate=dropout, mode=mode),
      tl.PositionalEncoding(max_len=max_len),
  ]
  return tl.Model(                  # tokens
      tl.ShiftRight(),              # toks
      embedder,                     # vecs
      [DecoderBlock(d_feature, d_feedforward, n_heads, dropout, mode)
       for _ in range(n_layers)],   # vecs
      tl.LayerNorm(),               # vecs
      tl.Dense(vocab_size),         # vecs
      tl.LogSoftmax(),              # vecs
  )


def EncoderDecoder(d_feature, d_feedforward, n_heads, dropout, mode):
  """Transformer encoder-decoder layer.

  The input is a triple (decoder_input, mask, encoder) where the mask is
  created from the original source to prevent attending to the padding part
  of the encoder.

  Args:
    d_feature: int:  depth of embedding
    d_feedforward: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    mode: str: 'train' or 'eval'

  Returns:
    the layer, returning a triple (decoder_activations, mask, encoder).
  """
  decoder_self_attention = [                    #        vecs_d   pmask vecs_e
      tl.LayerNorm(),                           #        vecs_d   ..... ......
      tl.Dup(),                                 # vecs_d vecs_d   ..... ......
      tl.Parallel([], tl.CausalMask(axis=-2)),  # ______ masks    ..... ......
      tl.Attention(d_feature, n_heads=n_heads, dropout=dropout, mode=mode),
      tl.Parallel([], tl.Drop()),               # ______   0      ..... ......
      tl.Dropout(rate=dropout, mode=mode),      # vecs_d          ..... ......
  ]
  decoder_to_encoder_attention = [        # vecs_d        masks         vecs_e
      tl.LayerNorm(),                     # vecs_d        masks         vecs_e
      tl.Parallel([], [], tl.Dup()),      # ______        _____  vecs_e vecs_e
      tl.Parallel([], tl.Swap()),         # ______        vecs_e masks  ......
      tl.Parallel([], tl.Dup()),          # ______ vecs_e vecs_e .....  ......
      tl.AttentionQKV(  # (q k v masks ... --> vecs_d masks ...)
          d_feature, n_heads=n_heads, dropout=dropout, mode=mode),
      tl.Dropout(rate=dropout, mode=mode),  # vecs_d mask vecs_e
  ]
  feed_forward = [
      FeedForward(d_feature, d_feedforward, dropout, mode=mode),
  ]
  return [                                        # vecs_d masks vecs_e
      tl.Residual(decoder_self_attention),        # vecs_d masks vecs_e
      tl.Residual(decoder_to_encoder_attention),  # vecs_d masks vecs_e
      tl.Residual(feed_forward),                  # vecs_d masks vecs_e
  ]


def Transformer(input_vocab_size,
                output_vocab_size=None,
                d_feature=512,
                d_feedforward=2048,
                n_layers=6,
                n_heads=8,
                dropout=0.1,
                max_len=2048,
                mode='train'):
  """Returns a Transformer model.

  This model expects an input pair: target, source.

  Args:
    input_vocab_size: int: vocab size of the source.
    output_vocab_size: int (optional): vocab size of the target. If None, the
      source and target are assumed to have the same vocab.
    d_feature: int:  depth of embedding
    d_feedforward: int: depth of feed-forward layer
    n_layers: int: number of encoder/decoder layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train' or 'eval'

  Returns:
    A Transformer model as a layer that maps from a target, source pair to
    activations over a vocab set.
  """
  in_embed = [                                    # tokens
      tl.Embedding(d_feature, input_vocab_size),  # vecs
      tl.Dropout(rate=dropout, mode=mode),        # vecs
      tl.PositionalEncoding(max_len=max_len),     # vecs
  ]

  if output_vocab_size is None:
    output_vocab_size = input_vocab_size
    out_embed = in_embed
  else:
    out_embed = [                                    # tokens
        tl.Embedding(d_feature, output_vocab_size),  # vecs
        tl.Dropout(rate=dropout, mode=mode),         # vecs
        tl.PositionalEncoding(max_len=max_len),      # vecs
    ]

  encoder_stack = (  # masks vectors --> masks vectors
      [EncoderBlock(d_feature, d_feedforward, n_heads, dropout, mode)
       for _ in range(n_layers)])

  encoder_decoder_stack = (  # vecs_d masks vecs_e --> vecs_d masks vecs_e
      [EncoderDecoder(d_feature, d_feedforward, n_heads, dropout, mode)
       for _ in range(n_layers)])

  # Input: encoder_side_tokens, decoder_side_tokens
  return tl.Model(  # tokens_e tokens_d
      tl.Swap(),    # toks_d toks_e

      # Encode.
      tl.Parallel(                                       # toks_d        toks_e
          [], [tl.Dup(),                                 # ______ toks_e toks_e
               tl.Parallel(in_embed, tl.PaddingMask()),  # ______ vecs_e masks
               encoder_stack,                            # ______ vecs_e masks
               tl.LayerNorm(),                           # ______ vecs_e .....
               tl.Swap()]),                              # ______ masks  vecs_e

      # Decode.                                  #        toks_d masks vecs_e
      tl.ShiftRight(),                           #        toks_d ..... ......
      out_embed,                                 #        vecs_d ..... ......
      tl.Dup(),                                  # vecs_d vecs_d ..... ......
      tl.Parallel([], tl.EncoderDecoderMask()),  # ______    masks     ......
      encoder_decoder_stack,                     # vecs_d    masks     vecs_e
      tl.Parallel([], tl.Drop(), tl.Drop()),     # vecs_d
      tl.LayerNorm(),                            # vecs_d
      tl.Dense(output_vocab_size),               # vecs_d
      tl.LogSoftmax(),                           # vecs_d
  )
