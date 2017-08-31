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

"""Cycle GAN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer_vae
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


def reconstruct_loss(x, gt, hparams, reuse=None):
  pred = tf.layers.dense(x, hparams.vocab_size, name="softmax", reuse=reuse)
  xent, w = common_layers.padded_cross_entropy(pred, gt, 0.0)
  return xent / w


def discriminator(x, compress, hparams, name, reuse=None):
  with tf.variable_scope(name, reuse=reuse):
    x = tf.stop_gradient(2 * x) - x  # Reverse gradient.
    if compress:
      x = transformer_vae.compress(x, None, hparams, "compress")
    else:
      x = transformer_vae.residual_conv(x, 1, hparams, "compress_rc")
    y = tf.reduce_mean(x, axis=1)
    return tf.tanh(tf.layers.dense(y, 1, name="reduce"))


def discriminate_loss(x, y, compress, hparams, name):
  with tf.variable_scope(name):
    d1 = discriminator(x, compress, hparams, "discriminator")
    d2 = discriminator(y, compress, hparams, "discriminator", reuse=True)
    dloss = tf.reduce_mean(tf.abs(d1 - d2))
    return - dloss


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

    # Split the batch into input-input and target-target parts.
    inputs1, _ = split_on_batch(inputs)
    _, targets2 = split_on_batch(targets)

    # Define F and G, called inp2tgt and tgt2inp here.
    def inp2tgt(x, reuse=False):
      return transformer_vae.residual_conv(x, 1, hparams, "inp2tgt", reuse)
    def tgt2inp(x, reuse=False):
      return transformer_vae.residual_conv(x, 1, hparams, "tgt2inp", reuse)

    # Input-input part.
    inp1_tgt = inp2tgt(inputs1)
    inp1_back = tgt2inp(inp1_tgt)

    # Target-target part.
    tgt2_inp = tgt2inp(targets2, reuse=True)
    tgt2_back = inp2tgt(tgt2_inp, reuse=True)

    # Reconstruction losses.
    inp1_orig, _ = split_on_batch(inputs_orig)
    _, tgt2_orig = split_on_batch(targets_orig)
    inp1_loss = reconstruct_loss(
        inp1_back, tf.squeeze(inp1_orig, axis=3), hparams)
    tgt2_loss = reconstruct_loss(
        tgt2_back, tf.squeeze(tgt2_orig, axis=3), hparams, reuse=True)

    # Discriminator losses.
    dloss1 = discriminate_loss(inputs1, tgt2_inp, True, hparams, "inp_disc")
    dloss2 = discriminate_loss(targets2, inp1_tgt, True, hparams, "tgt_disc")

    # Reconstruct targets from inputs.
    tgt = inp2tgt(inputs, reuse=True)
    tgt = tf.layers.dense(tgt, hparams.vocab_size, name="softmax", reuse=True)

    # We use the reconstruction only for tracking progress, no gradients here!
    tgt = tf.stop_gradient(tf.expand_dims(tgt, axis=2))

    losses = {"input_input": hparams.cycle_loss_multiplier * inp1_loss,
              "target_target": hparams.cycle_loss_multiplier * tgt2_loss,
              "input_disc": dloss1,
              "target_disc": dloss2}
    return tgt, losses


@registry.register_model
class CycleGAN(t2t_model.T2TModel):

  def model_fn_body(self, features):
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
  hparams.add_hparam("vocab_size", 32)  # Vocabulary size, need to set here.
  hparams.add_hparam("cycle_loss_multiplier", 2.0)
  return hparams
