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

import collections
from jax import random
import jax.numpy as np
import tensor2tensor.trax.stax as stax


def TransformerEncoder(mode='train',  # pylint: disable=invalid-name
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
    dropout: float: dropout rate (how much to drop out; note that stax follows
      Tensorflow's keep_rate convention, so we use 1 - dropout in calls below)

  Returns:
    A staxlayer for implementing a raw Transformer encoder stack.  No embedding
    or positional signals are added by this layer.
  """
  keep_rate = 1.0 - dropout
  # Multi-headed Attention and Feed-forward layers
  multi_attention = stax.MultiHeadedAttention(
      feature_depth, num_heads=num_heads, dropout=keep_rate, mode=mode)

  feed_forward = stax.serial(
      stax.Dense(feedforward_depth, W_init=stax.xavier_uniform()),
      stax.Relu,
      stax.Dropout(keep_rate, mode=mode),
      stax.Dense(feature_depth, W_init=stax.xavier_uniform())
  )

  @stax.Lambda
  def encoder(embedded_source, source_mask):
    """Transformer encoder stack.

    Args:
      embedded_source: staxlayer variable: embedded source sequences
      source_mask: staxlayer variable: self-attention mask

    Returns:
      Staxlayer variable that outputs encoded source.
    """
    encoder_layer = stax.serial(
        # input attends to self
        stax.residual(stax.LayerNorm(feature_depth),
                      stax.multiplex(stax.Identity,  # query
                                     stax.Identity,  # key
                                     stax.Identity,  # value
                                     source_mask),  # attention mask
                      multi_attention,
                      stax.Dropout(keep_rate, mode=mode)),
        # feed-forward
        stax.residual(stax.LayerNorm(feature_depth),
                      feed_forward,
                      stax.Dropout(keep_rate, mode=mode))
    )
    return stax.serial(
        embedded_source,
        stax.repeat(encoder_layer, num_layers),
        stax.LayerNorm(feature_depth),
    )

  return encoder


def TransformerLM(vocab_size,  # pylint: disable=invalid-name
                  mode='train',
                  num_layers=6,
                  feature_depth=512,
                  feedforward_depth=2048,
                  num_heads=8,
                  dropout=0.1,
                  max_len=512):
  """Transformer language model (only uses the decoder part of Transformer).

  Args:
    vocab_size: int: vocab size
    mode: str: 'train' or 'eval'
    num_layers: int: number of encoder/decoder layers
    feature_depth: int:  depth of embedding
    feedforward_depth: int: depth of feed-forward layer
    num_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    max_len: int: maximum symbol length for positional encoding

  Returns:
    init and apply.
  """
  keep_rate = 1.0 - dropout
  # Multi-headed Attention and Feed-forward layers
  multi_attention = stax.MultiHeadedAttention(
      feature_depth, num_heads=num_heads, dropout=keep_rate, mode=mode)

  feed_forward = stax.serial(
      stax.Dense(feedforward_depth, W_init=stax.xavier_uniform()),
      stax.Relu,
      stax.Dropout(keep_rate, mode=mode),
      stax.Dense(feature_depth, W_init=stax.xavier_uniform())
  )

  # Single decoder layer
  decoder_layer = stax.serial(
      # target attends to self
      stax.residual(stax.LayerNorm(feature_depth),
                    stax.multiplex(stax.Identity,  # query
                                   stax.Identity,  # key
                                   stax.Identity,  # value
                                   stax.CausalMask(axis=-2)),  # attention mask
                    multi_attention,
                    stax.Dropout(keep_rate, mode=mode)),
      # feed-forward
      stax.residual(stax.LayerNorm(feature_depth),
                    feed_forward,
                    stax.Dropout(keep_rate, mode=mode))
  )

  return stax.serial(
      stax.ShiftRight(),
      stax.Embedding(feature_depth, vocab_size),
      stax.PositionalEncoding(feature_depth, max_len=max_len),
      stax.Dropout(keep_rate, mode=mode),
      stax.repeat(decoder_layer, num_layers),
      stax.LayerNorm(feature_depth),
      stax.Dense(vocab_size, W_init=stax.xavier_uniform()),
      stax.LogSoftmax
  )


def Transformer(source_vocab_size,  # pylint: disable=invalid-name
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
  keep_rate = 1.0 - dropout
  # Input embedding and positional encoding
  inject_position = stax.serial(
      stax.PositionalEncoding(feature_depth, max_len=max_len),
      stax.Dropout(keep_rate, mode=mode)
  )
  if shared_embedding:
    assert source_vocab_size == target_vocab_size
    # Weight-shared Embedding
    embedding = stax.Share(stax.Embedding(feature_depth, source_vocab_size))
    source_embedding_layer = stax.serial(embedding, inject_position)
    target_embedding_layer = source_embedding_layer
  else:
    source_embedding = stax.Embedding(feature_depth, source_vocab_size)
    target_embedding = stax.Embedding(feature_depth, target_vocab_size)
    source_embedding_layer = stax.serial(source_embedding, inject_position)
    target_embedding_layer = stax.serial(target_embedding, inject_position)

  # Multi-headed Attention and Feed-forward layers
  multi_attention = stax.MultiHeadedAttention(
      feature_depth, num_heads=num_heads, dropout=keep_rate, mode=mode)

  feed_forward = stax.serial(
      stax.Dense(feedforward_depth, W_init=stax.xavier_uniform()),
      stax.Relu,
      stax.Dropout(keep_rate, mode=mode),
      stax.Dense(feature_depth, W_init=stax.xavier_uniform())
  )

  # Encoder
  @stax.Lambda
  def encoder(source, source_mask):
    """Transformer encoder stack.

    Args:
      source: staxlayer variable: raw source sequences
      source_mask: staxlayer variable: self-attention mask

    Returns:
      Staxlayer variable that outputs encoded source.
    """
    encoder_layer = stax.serial(
        # input attends to self
        stax.residual(stax.LayerNorm(feature_depth),
                      stax.multiplex(stax.Identity,  # query
                                     stax.Identity,  # key
                                     stax.Identity,  # value
                                     source_mask),  # attention mask
                      multi_attention,
                      stax.Dropout(keep_rate, mode=mode)),
        # feed-forward
        stax.residual(stax.LayerNorm(feature_depth),
                      feed_forward,
                      stax.Dropout(keep_rate, mode=mode))
    )
    return stax.serial(
        source,
        source_embedding_layer,
        stax.repeat(encoder_layer, num_layers),
        stax.LayerNorm(feature_depth),
    )

  # Decoder
  @stax.Lambda
  def decoder(memory, target, target_mask, memory_mask):
    """Transformer decoder stack.

    Args:
      memory: staxlayer variable: encoded source sequences
      target: staxlayer variable: raw target sequences
      target_mask: staxlayer variable: self-attention mask
      memory_mask: staxlayer variable: memory attention mask

    Returns:
      Staxlayer variable that outputs encoded source.
    """
    decoder_layer = stax.serial(
        # target attends to self
        stax.residual(stax.LayerNorm(feature_depth),
                      stax.multiplex(stax.Identity,  # query
                                     stax.Identity,  # key
                                     stax.Identity,  # value
                                     target_mask),  # attention mask
                      multi_attention,
                      stax.Dropout(keep_rate, mode=mode)),
        # target attends to encoded source
        stax.residual(stax.LayerNorm(feature_depth),
                      stax.multiplex(stax.Identity,  # query
                                     memory,  # key
                                     memory,  # value
                                     memory_mask),  # attention mask
                      multi_attention,
                      stax.Dropout(keep_rate, mode=mode)),
        # feed-forward
        stax.residual(stax.LayerNorm(feature_depth),
                      feed_forward,
                      stax.Dropout(keep_rate, mode=mode))
    )
    return stax.serial(
        target,
        target_embedding_layer,
        stax.repeat(decoder_layer, num_layers),
        stax.LayerNorm(feature_depth),
    )

  # The Transformer
  @stax.Lambda
  def transformer(source, target, source_mask, target_mask, memory_mask):
    encoded_source = encoder(source, source_mask)
    return decoder(encoded_source, target, target_mask, memory_mask)

  # Finally, bind the generator transform to use later for inference.
  @stax.Lambda
  def generator(encoded_target):
    return stax.serial(
        encoded_target,
        stax.Dense(target_vocab_size, W_init=stax.xavier_uniform()),
        stax.LogSoftmax
    )

  # Model-Building and Evaluation Functions
  # Get entire model's init and apply pair
  top_init, top_apply = generator(transformer)

  # By default act as a normal Stax constructor and emit an (init, apply) pair.
  if not return_evals:
    return (top_init, top_apply)
  else:
    # Inference-time function for binding trained params to model and returning
    # the python-bound sub-expressions for evaluation and sequence generation.
    def make_namedtuple(**kwargs):
      return collections.namedtuple('Model', kwargs.keys())(**kwargs)

    def get_evals(params):
      # We need to feed _concrete_ trained parameters through the network once.
      # Otherwise the bound parameters point to abstract tracer values.
      # The inputs don't matter.
      fake_inputs = 5 * (np.ones((1), dtype=np.int32),)
      fake_key = random.PRNGKey(1)
      top_apply(params, fake_inputs, rng=fake_key)
      # We can now return eval functions from the bound pieces of the model.
      return make_namedtuple(
          encoder=stax.make_apply_fun(encoder),
          generator=stax.make_apply_fun(generator),
          decoder=stax.make_apply_fun(decoder),
      )

    # We return the functions needed to train and evaluate the Transformer.
    return make_namedtuple(
        init=top_init,
        apply=top_apply,
        evals=get_evals,
    )
