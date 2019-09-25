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

"""Tests for Transformer models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as onp
from tensor2tensor.trax import backend
from tensor2tensor.trax import layers as tl
from tensor2tensor.trax.backend import numpy as np
from tensor2tensor.trax.models import transformer


class TransformerTest(parameterized.TestCase):

  def test_transformer_lm_forward_shape(self):
    """Run the Transformer LM forward and check output shape."""
    vocab_size = 16
    input_shape = [3, 5]
    model = transformer.TransformerLM(
        vocab_size, d_model=32, d_ff=64, n_layers=2, n_heads=2)
    final_shape = tl.check_shape_agreement(
        model, tuple(input_shape), integer_inputs=True)
    self.assertEqual(tuple(input_shape + [vocab_size]), final_shape)

  def _test_transformer_forward_shape(self, input_vocab_size,
                                      output_vocab_size):
    """Run the Transformer forward and check output shape."""
    single_input_shape = [3, 5]
    input_shape = (tuple(single_input_shape), tuple(single_input_shape))
    model = transformer.Transformer(
        input_vocab_size, output_vocab_size,
        d_model=32, d_ff=64, n_layers=2, n_heads=2)
    final_shape = tl.check_shape_agreement(
        model, input_shape, integer_inputs=True)
    expected_shape = (tuple(single_input_shape +
                            [output_vocab_size if output_vocab_size is not None
                             else input_vocab_size]))
    self.assertEqual(expected_shape, final_shape)

  def test_transformer_lm_fast_inference(self):
    with backend.use_backend('jax'):
      vocab_size = 16
      model_fn = functools.partial(
          transformer.TransformerLM,
          vocab_size=vocab_size, d_model=4, d_ff=8, n_layers=2, n_heads=2)
      model_slow = model_fn(mode='eval')
      model_fast = model_fn(mode='predict')
      rng = backend.random.get_prng(0)
      batch_size = 2
      (params, state_slow) = model_slow.initialize(
          input_shapes=(batch_size, 1), input_dtype=np.int32, rng=rng)
      (_, state_fast) = model_fast.initialize(
          input_shapes=(batch_size, 1), input_dtype=np.int32, rng=rng)

      max_length = 5
      buf = onp.zeros((batch_size, max_length), dtype=np.int32)
      next_sym = onp.zeros((batch_size, 1), dtype=onp.int32)

      for index in range(max_length):
        (logits_slow, state_slow) = model_slow(
            buf, params=params, state=state_slow, rng=rng)
        (logits_fast, state_fast) = model_fast(
            next_sym, params=params, state=state_fast, rng=rng)
        onp.testing.assert_array_almost_equal(
            logits_slow[:, index, :], logits_fast[:, 0, :])
        next_sym = onp.random.randint(vocab_size, size=(batch_size, 1))
        buf[:, index] = next_sym[:, 0]

  @parameterized.named_parameters(
      ('same_vocab', 16, None),
      ('same_size', 16, 16),
      ('different_size', 16, 50))
  def test_transformer_forward_shape(self, input_vocab_size, output_vocab_size):
    """Run the Transformer forward and check output shape."""
    self._test_transformer_forward_shape(input_vocab_size, output_vocab_size)


if __name__ == '__main__':
  absltest.main()
