# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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

"""Cycle GAN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import common_layers
from tensor2tensor.models.research import transformer_vae
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


def discriminator(x, compress, hparams, name, reuse=None):
  with tf.variable_scope(name, reuse=reuse):
    x = tf.stop_gradient(2 * x) - x  # Reverse gradient.
    if compress:
      x = transformer_vae.compress(x, None, False, hparams, "compress")
    else:
      x = transformer_vae.residual_conv(x, 1, 3, hparams, "compress_rc")
    y = tf.reduce_mean(x, axis=1)
    return tf.tanh(tf.layers.dense(y, 1, name="reduce"))


def generator(x, hparams, name, reuse=False):
  with tf.variable_scope(name, reuse=reuse):
    return transformer_vae.residual_conv(x, 1, 3, hparams, "generator")


def lossfn(real_input, fake_input, compress, hparams, lsgan, name):
  eps = 1e-12
  with tf.variable_scope(name):
    d1 = discriminator(real_input, compress, hparams, "discriminator")
    d2 = discriminator(fake_input, compress, hparams, "discriminator",
                       reuse=True)
    if lsgan:
      dloss = tf.reduce_mean(
          tf.squared_difference(d1, 0.9)) + tf.reduce_mean(tf.square(d2))
      gloss = tf.reduce_mean(tf.squared_difference(d2, 0.9))
      loss = (dloss + gloss)/2
    else:  # cross_entropy
      dloss = -tf.reduce_mean(
          tf.log(d1 + eps)) - tf.reduce_mean(tf.log(1 - d2 + eps))
      gloss = -tf.reduce_mean(tf.log(d2 + eps))
      loss = (dloss + gloss)/2
    return loss


def split_on_batch(x):
  batch_size = tf.shape(x)[0]
  i = batch_size // 2
  return x[:i, :, :, :], x[i:2*i, :, :, :]


def cycle_gan_internal(inputs, targets, _, hparams):
  """Cycle GAN, main step used for training."""
  with tf.variable_scope("cycle_gan"):
    # Embed inputs and targets.
    inputs_orig, targets_orig = tf.to_int32(inputs), tf.to_int32(targets)
    inputs = common_layers.embedding(
        inputs_orig, hparams.vocab_size, hparams.hidden_size, "embed")
    targets = common_layers.embedding(
        targets_orig, hparams.vocab_size, hparams.hidden_size,
        "embed", reuse=True)

    x, _ = split_on_batch(inputs)
    _, y = split_on_batch(targets)

    # Y --> X
    y_fake = generator(y, hparams, "Fy", reuse=False)
    y_to_x_loss = lossfn(y, y_fake, True, hparams, True, "YtoX")

    # X --> Y
    x_fake = generator(x, hparams, "Gx", reuse=False)
    x_to_y_loss = lossfn(y, x_fake, True, hparams, True, "XtoY")

    # Cycle-Consistency
    y_fake_ = generator(y_fake, hparams, "Gx", reuse=True)
    x_fake_ = generator(x_fake, hparams, "Fy", reuse=True)
    x_to_x_loss = hparams.cycle_loss_multiplier1 * tf.reduce_mean(
        tf.abs(x_fake_ - x))
    y_to_y_loss = hparams.cycle_loss_multiplier2 * tf.reduce_mean(
        tf.abs(y_fake_ - y))
    cycloss = x_to_x_loss + y_to_y_loss

    sample_generated = generator(inputs, hparams, "Gx", reuse=True)
    sample_generated = tf.layers.dense(
        sample_generated, hparams.vocab_size, name="softmax", reuse=None)
    sample_generated = tf.stop_gradient(
        tf.expand_dims(sample_generated, axis=2))

    losses = {"cycloss": cycloss,
              "y_to_x_loss": y_to_x_loss,
              "x_to_y_loss": x_to_y_loss}

    return sample_generated, losses


@registry.register_model
class CycleGAN(t2t_model.T2TModel):

  def body(self, features):
    return cycle_gan_internal(
        features["inputs"], features["targets"], features["target_space_id"],
        self._hparams)


@registry.register_hparams
def cycle_gan_small():
  """Set of hyperparameters."""
  hparams = transformer_vae.transformer_ae_small()
  hparams.batch_size = 2048
  hparams.input_modalities = "inputs:symbol:identity"
  hparams.target_modality = "symbol:identity"
  hparams.weight_decay = 3.0
  hparams.learning_rate = 0.05
  hparams.kl_warmup_steps = 5000
  hparams.learning_rate_warmup_steps = 3000
  hparams.add_hparam("vocab_size", 66)  # Vocabulary size, need to set here.
  hparams.add_hparam("cycle_loss_multiplier1", 10.0)
  hparams.add_hparam("cycle_loss_multiplier2", 10.0)
  return hparams
