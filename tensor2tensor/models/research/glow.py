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
"""Glow generative model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.models.research import glow_ops
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
import tensorflow as tf

arg_scope = tf.contrib.framework.arg_scope
add_arg_scope = tf.contrib.framework.add_arg_scope

GLOW_DECODE_HPARAMS = ("identity_output=True,log_results=False,"
                       "decode_in_memory=True,display_decoded_images=True")


@registry.register_hparams
def glow_hparams():
  """Glow Hparams."""
  hparams = common_hparams.basic_params1()
  hparams.clip_grad_norm = None
  hparams.weight_decay = 0.0
  hparams.learning_rate_constant = 3e-4
  hparams.batch_size = 32
  # can be prev_level, prev_step or normal.
  # see: glow_ops.merge_level_and_latent_dist
  hparams.add_hparam("level_prior_scale", "prev_level")
  hparams.add_hparam("n_levels", 3)
  hparams.add_hparam("n_bits_x", 8)
  hparams.add_hparam("depth", 32)
  hparams.add_hparam("affine_coupling_width", 512)
  hparams.add_hparam("top_prior", "single_conv")
  return hparams


@registry.register_model
class Glow(t2t_model.T2TModel):
  """Glow generative model.

  Reference: https://arxiv.org/abs/1807.03039"""

  def preprocess(self, x):
    """Normalize x.

    Args:
      x: 4-D Tensor.

    Returns:
      x: Scaled such that x lies in-between -0.5 and 0.5
    """
    n_bits_x = self.hparams.n_bits_x
    n_bins = 2**n_bits_x
    x = tf.cast(x, dtype=tf.float32)
    if n_bits_x < 8:
      x = tf.floor(x / 2 ** (8 - n_bits_x))
    x = x / n_bins - 0.5
    return x

  def scale(self, x):
    """Scale x from -0.5 - 0.5 to 0 - 255."""
    x = tf.where(tf.is_nan(x), tf.ones_like(x), x)
    x = tf.where(tf.is_inf(x), tf.ones_like(x), x)
    x = tf.clip_by_value(x, -0.5, 0.5)
    x += 0.5
    x = x * 2**self.hparams.n_bits_x
    return tf.cast(tf.clip_by_value(x, 0, 255), dtype=tf.uint8)

  @property
  def is_training(self):
    return self.hparams.mode == tf.estimator.ModeKeys.TRAIN

  def infer(self, features, *args, **kwargs):  # pylint: disable=arguments-differ
    del args, kwargs
    x = features["inputs"]
    batch_size = common_layers.shape_list(x)[0]
    features["targets"] = tf.zeros(shape=(batch_size, 1, 1, 1))
    _, _ = self(features)  # pylint: disable=not-callable

    ops = [glow_ops.get_variable_ddi, glow_ops.actnorm]
    var_scope = tf.variable_scope("glow/body", reuse=True)
    # If eps=None, images are sampled from the prior.
    with arg_scope(ops, init=False), var_scope:
      predictions, _, _ = glow_ops.encoder_decoder(
          "codec", self.z_sample, self.hparams, eps=None, reverse=True)

    return self.scale(predictions)

  def top_prior(self, z):
    """Objective based on the prior over latent z.

    Args:
      z: 4-D Tensor, (batch_size, height, width, num_channels)
    Returns:
      objective: float, log-likelihood of z under the prior.
      dist: instance of tf.distributions.Normal, prior distribution.
    """
    return glow_ops.top_prior(
        "top_prior", z, learn_prior=self.hparams.top_prior)

  def body(self, features):
    x = features["inputs"]

    # Scale x such that the pixels lie in-between -0.5 and.0.5
    x = self.preprocess(x)
    x, objective = glow_ops.uniform_binning_correction(x)

    # The arg_scope call ensures that the actnorm parameters are set such that
    # the per-channel output activations have zero mean and unit variance
    # ONLY during the first step. After that the parameters are learned
    # through optimisation.
    global_step = tf.train.get_or_create_global_step()
    init_op = tf.logical_and(tf.equal(global_step, 0), self.is_training)
    ops = [glow_ops.get_variable_ddi, glow_ops.actnorm]
    with arg_scope(ops, init=init_op):
      self.z, encoder_objective, self.eps, _ = glow_ops.encoder_decoder(
          "codec", x, self.hparams, eps=None, reverse=False)
      objective += encoder_objective

      prior_objective, prior_dist = self.top_prior(self.z)
      tf.summary.scalar("top_prior", tf.reduce_mean(prior_objective))
      self.z_sample = prior_dist.sample()
      objective += prior_objective

    # bits per pixel
    _, h, w, c = common_layers.shape_list(x)
    objective = -objective / (np.log(2) * h * w * c)
    return tf.zeros_like(features["targets"]), {"training": objective}
