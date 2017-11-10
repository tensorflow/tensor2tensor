# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

"""Transformer Sketch for im2sketch problems.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.models import transformer_vae
from tensor2tensor.models.transformer import transformer_base
from tensor2tensor.models.transformer import transformer_n_da
from tensor2tensor.models.transformer import transformer_small
from tensor2tensor.utils import registry


@registry.register_model
class TransformerSketch(transformer.Transformer):
  """Transformer with strided convolutions."""

  def encode(self, inputs, target_space, hparams):
    """Add two layers strided convolutions ontop of encode."""
    inputs = common_layers.conv_block(
        inputs,
        hparams.hidden_size, [((1, 1), (3, 3))],
        first_relu=False,
        padding="SAME",
        force2d=True,
        name="small_image_conv")

    hparams.num_compress_steps = 2
    compressed_inputs = transformer_vae.compress(inputs, c=None, is_2d=True,
                                                 hparams=hparams,
                                                 name="convolutions")

    return super(TransformerSketch, self).encode(
        compressed_inputs, target_space, hparams)


@registry.register_hparams
def transformer_sketch():
  """Basic transformer_sketch hparams."""
  hparams = transformer_n_da()
  hparams.batch_size = 2048
  hparams.max_length = 784
  hparams.clip_grad_norm = 5.
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.2
  hparams.learning_rate_warmup_steps = 10000
  hparams.num_hidden_layers = 6
  hparams.initializer = "orthogonal"
  hparams.sampling_method = "random"
  return hparams


@registry.register_hparams
def transformer_base_sketch():
  """Parameters based on base."""
  hparams = transformer_base()
  hparams.batch_size = 2048
  hparams.max_length = 784
  hparams.clip_grad_norm = 5.
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate_warmup_steps = 8000
  hparams.learning_rate = 0.2
  hparams.num_hidden_layers = 6
  hparams.initializer = "orthogonal"
  hparams.sampling_method = "random"
  return hparams


@registry.register_hparams
def transformer_small_sketch():
  """Modified transformer_small."""
  hparams = transformer_small()
  hparams.batch_size = 2048
  hparams.max_length = 784
  hparams.clip_grad_norm = 5.
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.initializer = "orthogonal"
  hparams.sampling_method = "random"
  hparams.learning_rate_warmup_steps = 10000
  return hparams


@registry.register_hparams
def transformer_sketch_2layer():
  hparams = transformer_sketch()
  hparams.num_hidden_layers = 2
  return hparams


@registry.register_hparams
def transformer_sketch_4layer():
  hparams = transformer_sketch()
  hparams.num_hidden_layers = 4
  return hparams


@registry.register_hparams
def transformer_sketch_6layer():
  hparams = transformer_sketch()
  hparams.num_hidden_layers = 6
  return hparams


@registry.register_ranged_hparams("transformer_sketch_ranged")
def transformer_sketch_ranged(rhp):
  """Range of hparams for vizier."""

  hparams = transformer_sketch()
  common_hparams.fill_ranged_hparams_from_hparams(hparams, rhp)

  rhp.set_categorical("ffn_layer",
                      ["conv_hidden_relu_with_sepconv", "conv_hidden_relu"])
  rhp.set_discrete("batch_size", [1024, 2048, 4096])
  rhp.set_discrete("num_hidden_layers", [2, 3, 4, 5, 6])
  rhp.set_discrete("hidden_size", [32, 64, 128, 256, 512, 1024],
                   scale=rhp.LOG_SCALE)
  rhp.set_discrete("kernel_height", [1, 3, 5, 7])
  rhp.set_discrete("kernel_width", [1, 3, 5, 7])
  rhp.set_discrete("compress_steps", [0, 1, 2])
  rhp.set_float("dropout", 0.0, 0.5)
  rhp.set_float("weight_decay", 1e-4, .03, scale=rhp.LOG_SCALE)
  rhp.set_float("label_smoothing", 0.0, 0.2)
  rhp.set_float("clip_grad_norm", 0.01, 8.0, scale=rhp.LOG_SCALE)
  rhp.set_float("learning_rate", 0.1, 1.0, scale=rhp.LOG_SCALE)
  rhp.set_categorical("initializer",
                      ["uniform", "orthogonal", "uniform_unit_scaling"])
  rhp.set_float("initializer_gain", 0.5, 3.5)
  rhp.set_categorical("learning_rate_decay_scheme",
                      ["none", "sqrt", "noam", "exp10k"])
  rhp.set_float("optimizer_adam_epsilon", 1e-7, 1e-2, scale=rhp.LOG_SCALE)
  rhp.set_float("optimizer_adam_beta1", 0.8, 0.9)
  rhp.set_float("optimizer_adam_beta2", 0.995, 0.999)
  rhp.set_categorical("optimizer", [
      "Adam", "Adagrad", "Momentum", "RMSProp", "SGD", "YellowFin"])


@registry.register_hparams
def transformer_opt():
  """Parameters that work better."""
  hparams = transformer_sketch()
  hparams.batch_size = 1024
  hparams.learning_rate = 0.28
  hparams.num_hidden_layers = 3
  hparams.dropout = 0.35
  hparams.ffn_layer = "conv_hidden_relu_with_sepconv"
  hparams.hidden_size = 128
  hparams.initializer_gain = 2.6
  hparams.weight_decay = 0.
  return hparams
