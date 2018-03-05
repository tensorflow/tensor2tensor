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

"""Simple Generative Adversarial Model with two linear layers.

Example of how to create a GAN in T2T.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


def generator(z, hparams, reuse=False):
  """Initalizes generator layers."""

  g_h1 = tf.layers.dense(
      z, hparams.hidden_dim, activation=tf.nn.relu, name="l1", reuse=reuse)
  g_log_prob = tf.layers.dense(
      g_h1, hparams.height * hparams.width, name="logp", reuse=reuse)
  g_prob = tf.nn.sigmoid(g_log_prob)

  return g_prob


def discriminator(x, hparams, reuse=False):
  """Initalizes discriminator layers."""
  d_h1 = tf.layers.dense(
      x, hparams.hidden_dim, activation=tf.nn.relu, name="d_h1", reuse=reuse)
  d_logit = tf.layers.dense(d_h1, 1, name="d_logit", reuse=reuse)
  d_prob = tf.nn.sigmoid(d_logit)

  return d_prob, d_logit


def reverse_grad(x):
  return tf.stop_gradient(2 * x) - x


def vanilla_gan_internal(inputs, hparams, train):
  with tf.variable_scope("vanilla_gan", reuse=tf.AUTO_REUSE):
    batch_size, height, width, _ = common_layers.shape_list(inputs)
    assert height == hparams.height
    assert width == hparams.width

    # Currently uses only one of RGB
    x = inputs
    x = x[:, :, :, 0]
    x = tf.reshape(x, [batch_size, height * width])

    # Generate a fake image
    z = tf.random_uniform(
        shape=[batch_size, hparams.random_sample_size],
        minval=-1,
        maxval=1,
        name="z")
    g_sample = generator(z, hparams)

    # Discriminate on the real image
    d_real, _ = discriminator(x, hparams)

    # Discriminate on the fake image
    d_fake, _ = discriminator(reverse_grad(g_sample), hparams, reuse=True)

    # GAN losses
    d_loss = -tf.reduce_mean(
        tf.log(d_real + hparams.epsilon) + tf.log(1. - d_fake))
    g_loss = -tf.reduce_mean(tf.log(d_fake + hparams.epsilon))

    losses = {}
    losses["discriminator"] = d_loss
    losses["generator"] = g_loss
    # Include a dummy training loss to skip self.top and self.loss
    losses["training"] = tf.constant(0., dtype=tf.float32)

    summary_g_image = tf.reshape(g_sample[0, :], [1, height, width, 1])
    tf.summary.image("generated", summary_g_image, max_outputs=1)

    if train:
      # Returns an dummy output and the losses dictionary.
      return tf.zeros([batch_size, 1]), losses
    else:
      return g_sample, losses


@registry.register_model
class VanillaGan(t2t_model.T2TModel):
  """Simple GAN for demonstration."""

  def body(self, features):
    """Computes the generator and discriminator loss.

    Args:
      features: A dictionary of key to Tensor. "inputs" should be an image.

    Returns:
      output: Tensor containing one zero. GANs do not make use of the modality
        loss.
      losses: a dictionary of losses containing the generator and discriminator
        losses.
    """
    train = self.hparams.mode == tf.estimator.ModeKeys.TRAIN
    return vanilla_gan_internal(features["inputs"], self.hparams, train)

  def infer(self,
            features=None,
            decode_length=50,
            beam_size=1,
            top_beams=1,
            last_position_only=False,
            alpha=0.0):
    with tf.variable_scope("body/vanilla_gan", reuse=tf.AUTO_REUSE):
      z = tf.random_uniform(
          shape=[1, self._hparams.random_sample_size],
          minval=-1,
          maxval=1,
          name="z")

      g_sample = generator(z, self._hparams)
      return g_sample


@registry.register_hparams
def vanilla_gan():
  """Basic parameters for a vanilla_gan."""
  hparams = common_hparams.basic_params1()

  hparams.batch_size = 32
  hparams.label_smoothing = 0.0
  hparams.add_hparam("hidden_dim", 128)
  hparams.add_hparam("random_sample_size", 100)
  hparams.add_hparam("height", 28)
  hparams.add_hparam("width", 28)
  hparams.add_hparam("epsilon", 1e-4)
  return hparams
