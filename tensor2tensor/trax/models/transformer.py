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

"""Transformer Model."""
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
      layers.Dense(feedforward_depth,
                   kernel_initializer=layers.XavierUniformInitializer()),
      layers.Relu(),
      layers.Dropout(rate=dropout, mode=mode),
      layers.Dense(feature_depth,
                   kernel_initializer=layers.XavierUniformInitializer()),
      layers.Dropout(rate=dropout, mode=mode)
  )


def TransformerEncoder(mode='train',
                       num_layers=6,
                       feature_depth=512,
                       feedforward_depth=2048,
                       num_heads=8,
                       dropout=0.1):
  """Transformer Encoder Stack.

  Args:
    mode: str: 'train' or 'eval'
    num_layers: int: number of encoder/decoder layers
    feature_depth: int:  depth of embedding
    feedforward_depth: int: depth of feed-forward layer
    num_heads: int: number of attention heads
    dropout: float: dropout rate

  Returns:
    A layer for implementing a raw Transformer encoder stack.  No embedding
    or positional signals are added by this layer.
  """
  # Multi-headed Attention and Feed-forward layers
  multi_attention = layers.MultiHeadedAttention(
      feature_depth, num_heads=num_heads, dropout=dropout, mode=mode)

  @layers.Lambda
  def Encoder(embedded_source, source_mask):
    """Transformer encoder stack.

    Args:
      embedded_source: layer variable: embedded source sequences
      source_mask: layer variable: self-attention mask

    Returns:
      Layer variable that outputs encoded source.
    """
    encoder_layer = layers.Serial(
        # input attends to self
        layers.Residual(layers.LayerNorm(),
                        layers.FanOut(size=4),
                        layers.Parallel(layers.Identity(),  # query
                                        layers.Identity(),  # key
                                        layers.Identity(),  # value
                                        source_mask),  # attention mask
                        multi_attention,
                        layers.Dropout(rate=dropout, mode=mode)),
        # feed-forward
        ResidualFeedForward(
            feature_depth, feedforward_depth, dropout, mode=mode)
    )
    return layers.Serial(
        embedded_source,
        layers.repeat(encoder_layer, num_layers),
        layers.LayerNorm(),
    )

  return Encoder


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
    init and apply.
  """
  return layers.Serial(
      layers.Residual(  # Self-attention block.
          layers.LayerNorm(),
          layers.FanOut(size=4),
          layers.Parallel(layers.Identity(),  # query
                          layers.Identity(),  # key
                          layers.Identity(),  # value
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
    init and apply.
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
      layers.Dense(vocab_size,
                   kernel_initializer=layers.XavierUniformInitializer()),
      layers.LogSoftmax()
  )


# TODO(lukaszkaiser): rewrite the model below.


def Transformer(source_vocab_size,
                target_vocab_size,
                mode='train',
                num_layers=6,
                feature_depth=512,
                feedforward_depth=2048,
                num_heads=8,
                dropout=0.1,
                shared_embedding=True,
                max_len=200,
                return_evals=False):
  """Transformer model.

  Args:
    source_vocab_size: int: source vocab size
    target_vocab_size: int: target vocab size
    mode: str: 'train' or 'eval'
    num_layers: int: number of encoder/decoder layers
    feature_depth: int:  depth of embedding
    feedforward_depth: int: depth of feed-forward layer
    num_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    shared_embedding: bool: specify whether source/target embeddings are tied.
    max_len: int: maximum symbol length for positional encoding
    return_evals: bool: whether to generate decode-time evaluation functions

  Returns:
    A namedtuple containing model 'init' and 'apply' functions for training and
  the 'evals' functions that itself returns a namedtuple containing evaluation
  functions for the trained encoder, decoder, and generator substax.
  """
  # Input embedding and positional encoding
  inject_position = layers.Serial(
      layers.Dropout(dropout, mode=mode),
      layers.PositionalEncoding(feature_depth, max_len=max_len)
  )
  if shared_embedding:
    assert source_vocab_size == target_vocab_size
    # Weight-shared Embedding
    embedding = layers.Share(layers.Embedding(feature_depth, source_vocab_size))
    source_embedding_layer = layers.Serial(embedding, inject_position)
    target_embedding_layer = source_embedding_layer
  else:
    source_embedding = layers.Embedding(feature_depth, source_vocab_size)
    target_embedding = layers.Embedding(feature_depth, target_vocab_size)
    source_embedding_layer = layers.Serial(source_embedding, inject_position)
    target_embedding_layer = layers.Serial(target_embedding, inject_position)

  # Multi-headed Attention and Feed-forward layers
  multi_attention = layers.MultiHeadedAttention(
      feature_depth, num_heads=num_heads, dropout=dropout, mode=mode)

  # Encoder
  @layers.Lambda
  def Encoder(source, source_mask):
    """Transformer encoder stack.

    Args:
      source: layer variable: raw source sequences
      source_mask: layer variable: self-attention mask

    Returns:
      Layer variable that outputs encoded source.
    """
    encoder_layer = layers.Serial(
        # input attends to self
        layers.Residual(layers.LayerNorm(),
                        layers.FanOut(size=4),
                        layers.Parallel(layers.Identity(),  # query
                                        layers.Identity(),  # key
                                        layers.Identity(),  # value
                                        source_mask),  # attention mask
                        multi_attention,
                        layers.Dropout(dropout, mode=mode)),
        # feed-forward
        ResidualFeedForward(
            feature_depth, feedforward_depth, dropout, mode=mode),
    )
    return layers.Serial(
        source,
        source_embedding_layer,
        layers.repeat(encoder_layer, num_layers),
        layers.LayerNorm(),
    )

  # Decoder
  @layers.Lambda
  def Decoder(memory, target, target_mask, memory_mask):
    """Transformer decoder stack.

    Args:
      memory: layer variable: encoded source sequences
      target: layer variable: raw target sequences
      target_mask: layer variable: self-attention mask
      memory_mask: layer variable: memory attention mask

    Returns:
      Layer variable that outputs encoded source.
    """
    decoder_layer = layers.Serial(
        # target attends to self
        layers.Residual(layers.LayerNorm(),
                        layers.FanOut(size=4),
                        layers.Parallel(layers.Identity(),  # query
                                        layers.Identity(),  # key
                                        layers.Identity(),  # value
                                        target_mask),  # attention mask
                        multi_attention,
                        layers.Dropout(dropout, mode=mode)),
        # target attends to encoded source
        layers.Residual(layers.LayerNorm(),
                        layers.FanOut(size=4),
                        layers.Parallel(layers.Identity(),  # query
                                        memory,  # key
                                        memory,  # value
                                        memory_mask),  # attention mask
                        multi_attention,
                        layers.Dropout(dropout, mode=mode)),
        # feed-forward
        ResidualFeedForward(
            feature_depth, feedforward_depth, dropout, mode=mode)
    )
    return layers.Serial(
        target,
        target_embedding_layer,
        layers.repeat(decoder_layer, num_layers),
        layers.LayerNorm(),
    )

  # The Transformer
  @layers.Lambda
  def transformer(source, target, source_mask, target_mask, memory_mask):  # pylint: disable=invalid-name
    encoded_source = Encoder(source, source_mask)
    return Decoder(encoded_source, target, target_mask, memory_mask)

  # Finally, bind the generator transform to use later for inference.
  @layers.Lambda
  def Generator(encoded_target):
    return layers.Serial(
        encoded_target,
        layers.Dense(target_vocab_size,
                     kernel_initializer=layers.XavierUniformInitializer()),
        layers.LogSoftmax
    )

  # Model-Building and Evaluation Functions
  # Get entire model's init and apply pair
  top_init, top_apply = Generator(transformer)

  # By default act as a normal constructor and emit an (init, apply) pair.
  if not return_evals:
    return (top_init, top_apply)
  else:
    raise ValueError('inference in this model is still a work in progress')
