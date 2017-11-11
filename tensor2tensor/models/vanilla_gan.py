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

  g_h1 = tf.layers.dense(z, hparams.weight_size, activation=tf.nn.relu,
                         name="l1", reuse=reuse)
  g_log_prob = tf.layers.dense(g_h1, hparams.input_size, name="logp",
                               reuse=reuse)
  g_prob = tf.nn.sigmoid(g_log_prob)

  return g_prob


def discriminator(x, hparams, reuse=False):
  """Initalizes discriminator layers."""

  d_h1 = tf.layers.dense(x, hparams.weight_size, activation=tf.nn.relu,
                         name="d_h1", reuse=reuse)
  d_logit = tf.layers.dense(d_h1, 1, name="d_logit", reuse=reuse)
  d_prob = tf.nn.sigmoid(d_logit)

  return d_prob, d_logit


def reverse_grad(x):
  return tf.stop_gradient(2*x) - x


def vanilla_gan_internal(inputs, hparams, train):
  with tf.variable_scope("vanilla_gan", reuse=tf.AUTO_REUSE):
    x = common_layers.flatten4d3d(inputs)

    batch_size = tf.shape(inputs)[0]
    # Currently uses one of three color layers.
    x = x[:, :, 0]
    x.set_shape([None, hparams.input_size])

    if train:
      z = tf.random_uniform(shape=[batch_size,
                                   hparams.random_sample_size],
                            minval=-1, maxval=1, name="z")
    else:
      z = tf.random_uniform(shape=[1, hparams.random_sample_size],
                            minval=-1, maxval=1, name="z")

    g_sample = generator(z, hparams)

    d_real, _ = discriminator(x, hparams)

    d_fake, _ = discriminator(reverse_grad(g_sample), hparams,
                              reuse=True)
    d_loss = -tf.reduce_mean(tf.log(d_real+hparams.epsilon)
                             + tf.log(1. - d_fake))
    g_loss = -tf.reduce_mean(tf.log(d_fake+hparams.epsilon))

    losses = {}
    losses["discriminator"] = d_loss
    losses["generator"] = g_loss

    z_sampled = tf.random_uniform(shape=[1, hparams.random_sample_size],
                                  minval=-1, maxval=1, name="z")
    g_sample = generator(z_sampled, hparams, reuse=True)
    g_reshaped_sample = tf.reshape(g_sample,
                                   [1, hparams.height, hparams.width, 1])
    tf.summary.image("generated", g_reshaped_sample, max_outputs=1)

    if train:
      # Returns an empty output, and loss dictionary.
      return tf.zeros(shape=[1, 1]), losses
    else:
      return g_sample, losses


@registry.register_model
class VanillaGan(t2t_model.T2TModel):
  """Simple GAN.
  """

  def model_fn_body(self, features):
    """Computes the generator and discriminator loss.

    Args:
      features: A dictionary of key to Tensor. Each Tensor has shape
         [batch_size, ?, ?, hidden_size].

    Returns:
      output: Tensor containing one zero. GANs do not make use of the modality
        loss.
      losses: a dictionary of losses containing the generator and discriminator
        losses.
    """
    train = self._hparams.mode == tf.estimator.ModeKeys.TRAIN
    return vanilla_gan_internal(features["targets"], self._hparams, train)

  def infer(self,
            features=None,
            decode_length=50,
            beam_size=1,
            top_beams=1,
            last_position_only=False,
            alpha=0.0):
    with tf.variable_scope("body/vanilla_gan", reuse=tf.AUTO_REUSE):
      z = tf.random_uniform(shape=[1, self._hparams.random_sample_size],
                            minval=-1, maxval=1, name="z")

      g_sample = generator(z, self._hparams)
      return g_sample


@registry.register_hparams
def vanilla_gan():
  """Basic parameters for a vanilla_gan."""

  hparams = common_hparams.basic_params1()

  hparams.input_modalities = "image:no_loss"
  hparams.target_modality = "image:no_loss"

  hparams.batch_size = 2048  # 3136
  hparams.label_smoothing = 0.0
  hparams.add_hparam("startup_steps", 10000)

  hparams.train_steps = 100
  hparams.add_hparam("weight_size", 128)
  hparams.add_hparam("random_sample_size", 100)
  hparams.add_hparam("height", 28)
  hparams.add_hparam("width", 28)
  hparams.add_hparam("colors", 1)
  hparams.add_hparam("input_size", 784)
  hparams.add_hparam("epsilon", 1e-4)
  hparams.learning_rate_warmup_steps = 0
  hparams.learning_rate = 0.2
  hparams.learning_rate_decay_scheme = "none"
  return hparams


