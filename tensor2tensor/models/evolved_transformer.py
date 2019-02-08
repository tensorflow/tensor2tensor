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

"""Evolved Transformer model.

This implements the model described in arxiv.org/abs/1901.11117 .
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import transformer_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry


@registry.register_model
class EvolvedTransformer(transformer.Transformer):
  """The Evolved Transformer from arxiv.org/abs/1901.11117 ."""

  def __init__(self, *args, **kwargs):
    super(EvolvedTransformer, self).__init__(*args, **kwargs)
    self._encoder_function = transformer_layers.evolved_transformer_encoder
    self._decoder_function = transformer.evolved_transformer_decoder

  def _beam_decode(self, features, decode_length, beam_size, top_beams, alpha,
                   use_tpu):
    """Forced slow beam decode because cache is not supported.

    Args:
      features: an map of string to `Tensor`.
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
      use_tpu: Whether or not TPU is being used.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length].
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1).
      }
    """
    return self._beam_decode_slow(features, decode_length, beam_size, top_beams,
                                  alpha, use_tpu)


# TODO(davidso): Update optimizer, learning rate, and decay to match paper.
def add_evolved_transformer_hparams(hparams):
  """Add Evolved Transformer hparams.

  Note: These are for the Adam optimizer, not the Adafactor optimizer used in
  the paper.

  Args:
    hparams: Current hparams.

  Returns:
    hparams updated with Evolved Transformer values.
  """
  # Evolved Transformer "layers" are twice as deep as Transformer, so roughly
  # halve the number that we use. These numbers are taken from
  # arxiv.org/abs/1901.11117 .
  hparams.num_encoder_layers = 3
  hparams.num_decoder_layers = 4

  # Learning rate and decay scheme that mimics the transformer Adam config,
  # but with cosine decay instead of rsqrt.
  hparams.learning_rate_constant /= hparams.learning_rate_warmup_steps ** 0.5
  hparams.learning_rate_schedule = (
      "constant*linear_warmup*single_cycle_cos_decay*rsqrt_hidden_size")
  # The current infrastructure does not support exposing
  # `train_steps` to the decay functions, and so we are hard coding the decay
  # steps here to match the default number of train steps used in `t2t_trainer`.
  # TODO(davidso): Thread `train_steps` through to decay functions so we do not
  # have to worry about a `learning_rate_decay_steps` mismatch.
  hparams.learning_rate_decay_steps = 250000
  return hparams


@registry.register_hparams
def evolved_transformer_base():
  """Base parameters for Evolved Transformer model."""
  return add_evolved_transformer_hparams(transformer.transformer_base())


@registry.register_hparams
def evolved_transformer_big():
  """Big parameters for Evolved Transformer model on WMT."""
  return add_evolved_transformer_hparams(transformer.transformer_big())


@registry.register_hparams
def evolved_transformer_base_tpu():
  """Base parameters for Evolved Transformer model on TPU."""
  hparams = add_evolved_transformer_hparams(transformer.transformer_tpu())
  hparams.learning_rate_constant = 1 / hparams.learning_rate_warmup_steps ** 0.5
  hparams.learning_rate_schedule = (
      "constant*single_cycle_cos_decay")
  return hparams


@registry.register_hparams
def evolved_transformer_big_tpu():
  """Big parameters for Evolved Transformer model on TPU."""
  hparams = add_evolved_transformer_hparams(transformer.transformer_big_tpu())
  hparams.learning_rate_constant = 1 / hparams.learning_rate_warmup_steps ** 0.5
  hparams.learning_rate_schedule = (
      "constant*single_cycle_cos_decay")
  return hparams
