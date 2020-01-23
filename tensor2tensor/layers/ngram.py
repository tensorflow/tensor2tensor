# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""N-gram layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf


class NGram(tf.keras.layers.Layer):
  r"""N-gram layer.

  The layer takes as input an integer Tensor of shape [..., length], each
  element of which is a token index in [0, input_dim). It returns a real-valued
  Tensor of shape [..., num_ngrams], counting the number of times each n-gram
  appears in a batch element. The total number of n-grams is

  ```none
  num_ngrams = \sum_{minval <= n < maxval} input_dim^n.
  ```
  """

  def __init__(self, input_dim, minval, maxval, **kwargs):
    """Constructs layer.

    Args:
      input_dim: int > 0. Size of the vocabulary, i.e. maximum integer index +
        1.
      minval: Lowest inclusive value of n for computing n-grams. For example,
        setting it to 1 will compute starting from unigrams.
      maxval: Highest non-inclusive value of n for computing n-grams. For
        example, setting it to 3 will compute at most bigrams.
      **kwargs: kwargs of parent class.
    """
    super(NGram, self).__init__(**kwargs)
    self.input_dim = input_dim
    self.minval = minval
    self.maxval = maxval

  def call(self, inputs):
    batch_shape = tf.shape(inputs)[:-1]
    length = tf.shape(inputs)[-1]
    ngram_range_counts = []
    for n in range(self.minval, self.maxval):
      # Reshape inputs from [..., length] to [..., 1, length // n, n], dropping
      # remainder elements. Each n-vector is an ngram.
      reshaped_inputs = tf.reshape(
          inputs[..., :(n * (length // n))],
          tf.concat([batch_shape, [1], (length // n)[tf.newaxis], [n]], 0))
      # Count the number of times each ngram appears in the input. We do so by
      # checking whether each n-vector in the input is equal to each n-vector
      # in a Tensor of all possible ngrams. The comparison is batched between
      # the input Tensor of shape [..., 1, length // n, n] and the ngrams Tensor
      # of shape [..., input_dim**n, 1, n].
      ngrams = tf.reshape(
          list(np.ndindex((self.input_dim,) * n)),
          [1] * (len(inputs.shape)-1) + [self.input_dim**n, 1, n])
      is_ngram = tf.equal(
          tf.reduce_sum(tf.cast(tf.equal(reshaped_inputs, ngrams), tf.int32),
                        axis=-1),
          n)
      ngram_counts = tf.reduce_sum(tf.cast(is_ngram, tf.float32), axis=-1)
      ngram_range_counts.append(ngram_counts)
    return tf.concat(ngram_range_counts, axis=-1)

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    num_ngrams = sum([self.input_dim**n
                      for n in range(self.minval, self.maxval)])
    return input_shape[:-1].concatenate(num_ngrams)

  def get_config(self):
    config = {'minval': self.minval,
              'maxval': self.maxval}
    base_config = super(NGram, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
