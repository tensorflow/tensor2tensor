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
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


def lrelu(input_, leak=0.2, name="lrelu"):
  return tf.maximum(input_, leak * input_, name=name)


def deconv2d(
    input_, output_shape, k_h, k_w, d_h, d_w, stddev=0.02, name="deconv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable(
        "w", [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
        initializer=tf.random_normal_initializer(stddev=stddev))
    deconv = tf.nn.conv2d_transpose(
        input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
    biases = tf.get_variable(
        "biases", [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    return tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())


class AbstractGAN(t2t_model.T2TModel):
  """Base class for all GANs."""

  def discriminator(self, x, is_training, reuse=False):
    """Discriminator architecture based on InfoGAN.

    Args:
      x: input images, shape [bs, h, w, channels]
      is_training: boolean, are we in train or eval model.
      reuse: boolean, should params be re-used.

    Returns:
      out_logit: the output logits (before sigmoid).
    """
    hparams = self._hparams
    with tf.variable_scope(
        "discriminator", reuse=reuse,
        initializer=tf.random_normal_initializer(stddev=0.02)):
      batch_size = hparams.batch_size
      # Mapping x from [bs, h, w, c] to [bs, 1]
      net = tf.layers.conv2d(x, 64, (4, 4), strides=(2, 2),
                             padding="SAME", name="d_conv1")
      # [bs, h/2, w/2, 64]
      net = lrelu(net)
      net = tf.layers.conv2d(net, 128, (4, 4), strides=(2, 2),
                             padding="SAME", name="d_conv2")
      # [bs, h/4, w/4, 128]
      if hparams.discriminator_batchnorm:
        net = tf.layers.batch_normalization(net, training=is_training,
                                            momentum=0.999, name="d_bn2")
      net = lrelu(net)
      size = hparams.height * hparams.width
      net = tf.reshape(net, [batch_size, size * 8])  # [bs, h * w * 8]
      net = tf.layers.dense(net, 1024, name="d_fc3")  # [bs, 1024]
      if hparams.discriminator_batchnorm:
        net = tf.layers.batch_normalization(net, training=is_training,
                                            momentum=0.999, name="d_bn3")
      net = lrelu(net)
      out_logit = tf.layers.dense(net, 1, name="d_fc4")  # [bs, 1]
      return out_logit

  def generator(self, z, is_training, reuse=False):
    """Generator outputting image in [0, 1]."""
    hparams = self._hparams
    height = hparams.height
    width = hparams.width
    batch_size = hparams.batch_size
    with tf.variable_scope(
        "generator", reuse=reuse,
        initializer=tf.random_normal_initializer(stddev=0.02)):
      net = tf.layers.dense(z, 1024, name="g_fc1")
      net = tf.layers.batch_normalization(net, training=is_training,
                                          momentum=0.999, name="g_bn1")
      net = lrelu(net)
      net = tf.layers.dense(net, 128 * (height // 4) * (width // 4),
                            name="g_fc2")
      net = tf.layers.batch_normalization(net, training=is_training,
                                          momentum=0.999, name="g_bn2")
      net = lrelu(net)
      net = tf.reshape(net, [batch_size, height // 4, width // 4, 128])
      net = deconv2d(net, [batch_size, height // 2, width // 2, 64],
                     4, 4, 2, 2, name="g_dc3")
      net = tf.layers.batch_normalization(net, training=is_training,
                                          momentum=0.999, name="g_bn3")
      net = lrelu(net)
      net = deconv2d(net, [batch_size, height, width, hparams.c_dim],
                     4, 4, 2, 2, name="g_dc4")
      out = tf.nn.sigmoid(net)
      return out

  def body(self, features):
    """Body of the model.

    Args:
      features: a dictionary with the tensors.

    Returns:
      A pair (predictions, losses) where preditions is the generated image
      and losses is a dictionary of losses (that get added for the final loss).
    """
    features["targets"] = features["inputs"]
    is_training = self.hparams.mode == tf.estimator.ModeKeys.TRAIN

    # Input images.
    inputs = features["inputs"]

    # Noise vector.
    z = tf.random_uniform(
        shape=[self._hparams.batch_size, self._hparams.z_size],
        minval=-1,
        maxval=1,
        name="z")

    # Discriminator output for real images.
    d_real_logits = self.discriminator(
        inputs, is_training=is_training, reuse=False)

    # Discriminator output for fake images.
    g = self.generator(z, is_training=is_training, reuse=False)
    d_fake_logits_g = self.discriminator(
        g, is_training=is_training, reuse=True)
    # Discriminator doesn't backprop to generator.
    d_fake_logits_d = self.discriminator(
        tf.stop_gradient(g), is_training=is_training, reuse=True)

    # Loss on real and fake data.
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_real_logits, labels=tf.ones_like(d_real_logits)))
    d_loss_fake_g = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_fake_logits_g, labels=tf.zeros_like(d_fake_logits_g)))
    d_loss_fake_d = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_fake_logits_d, labels=tf.zeros_like(d_fake_logits_d)))
    d_loss = d_loss_real + d_loss_fake_d

    losses = {}  # All losses get added at the end.
    losses["discriminator"] = d_loss
    losses["generator"] = - d_loss_fake_g
    # Include a dummy training loss to skip self.loss.
    losses["training"] = tf.constant(0., dtype=tf.float32)

    hparams = self._hparams
    summary_g_image = tf.reshape(g[0, :], [1, hparams.height, hparams.width, 1])
    tf.summary.image("generated", summary_g_image, max_outputs=1)

    if is_training:
      # Returns an dummy output and the losses dictionary.
      return tf.zeros_like(inputs), losses
    return tf.reshape(g, tf.shape(inputs)), losses

  def top(self, body_output, features):
    """Override the top function to not do anything."""
    return body_output


@registry.register_model
class VanillaGan(AbstractGAN):
  """Simple GAN for demonstration."""

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

      g_sample = self.generator(z, self._hparams)
      return g_sample


@registry.register_hparams
def vanilla_gan():
  """Basic parameters for a vanilla_gan."""
  hparams = common_hparams.basic_params1()
  hparams.label_smoothing = 0.0
  hparams.hidden_size = 128
  hparams.batch_size = 64
  hparams.add_hparam("z_size", 64)
  hparams.add_hparam("c_dim", 1)
  hparams.add_hparam("height", 28)
  hparams.add_hparam("width", 28)
  hparams.add_hparam("discriminator_batchnorm", int(True))
  return hparams
