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

"""Glow generative model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.models.research import glow_init_hook
from tensor2tensor.models.research import glow_ops
from tensor2tensor.utils import contrib
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
import tensorflow.compat.v1 as tf

arg_scope = contrib.framework().arg_scope
add_arg_scope = contrib.framework().add_arg_scope

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
  hparams.add_hparam("level_scale", "prev_level")
  hparams.add_hparam("n_levels", 3)
  hparams.add_hparam("n_bits_x", 8)
  hparams.add_hparam("depth", 32)
  # Activation - Relu or Gatu
  hparams.add_hparam("activation", "relu")
  # Coupling layer, additive or affine.
  hparams.add_hparam("coupling", "affine")
  hparams.add_hparam("coupling_width", 512)
  hparams.add_hparam("coupling_dropout", 0.0)
  hparams.add_hparam("top_prior", "single_conv")
  # init_batch_size denotes the number of examples used for data-dependent
  # initialization. A higher init_batch_size is required for training
  # stability especially when hparams.batch_size is low.
  hparams.add_hparam("init_batch_size", 256)
  hparams.add_hparam("temperature", 1.0)

  return hparams


@registry.register_model
class Glow(t2t_model.T2TModel):
  """Glow generative model.

  Reference: https://arxiv.org/abs/1807.03039"""

  def init_preprocess(self, features):
    """Preprocessing as per the input modality."""
    return features

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

  @property
  def temperature(self):
    if self.is_predicting:
      return self.hparams.temperature
    return 1.0

  @property
  def is_training(self):
    return self.hparams.mode == tf.estimator.ModeKeys.TRAIN

  def infer(self, features, *args, **kwargs):  # pylint: disable=arguments-differ
    del args, kwargs
    x = features["inputs"]
    batch_size = common_layers.shape_list(x)[0]
    features["targets"] = tf.zeros(shape=(batch_size, 1, 1, 1))
    _, _ = self(features)  # pylint: disable=not-callable

    ops = [glow_ops.get_variable_ddi, glow_ops.actnorm, glow_ops.get_dropout]
    var_scope = tf.variable_scope("glow/body", reuse=True)
    # If eps=None, images are sampled from the prior.
    with arg_scope(ops, init=False), var_scope:
      predictions, _, _, _ = glow_ops.encoder_decoder(
          "codec", self.z_sample, self.hparams, eps=None, reverse=True,
          temperature=self.temperature)

    return glow_ops.postprocess(predictions, self.hparams.n_bits_x)

  def create_init_batch(self, features):
    """Returns a batch of size "hparams.init_batch_size" for initialization.

    Args:
      features: input features.
    Returns:
      init_features: initialization features.
    """
    train_dataset = self.hparams.problem.dataset(
        tf.estimator.ModeKeys.TRAIN, hparams=self.hparams)
    train_dataset = train_dataset.batch(self.hparams.init_batch_size)
    train_dataset = self.init_preprocess(train_dataset)
    return train_dataset.make_one_shot_iterator().get_next()

  @staticmethod
  def train_hooks(hook_context):
    del hook_context
    return [glow_init_hook.GlowInitHook()]

  def top_prior(self):
    """Objective based on the prior over latent z.

    Returns:
      dist: instance of tfp.distributions.Normal, prior distribution.
    """
    return glow_ops.top_prior(
        "top_prior", self.z_top_shape, learn_prior=self.hparams.top_prior,
        temperature=self.temperature)

  def body(self, features):
    exp_coupling = ["affine", "additive"]
    if self.hparams.coupling not in exp_coupling:
      raise ValueError("Expected hparams.coupling to be in %s, got %s" %
                       (exp_coupling, self.hparams.coupling))
    if self.is_training:
      init_features = self.create_init_batch(features)
      init_op = self.objective_tower(init_features, init=True)
      init_op = tf.Print(
          init_op, [init_op], message="Triggering data-dependent init.",
          first_n=20)
      tf.add_to_collection("glow_init_op", init_op)
    train_op = self.objective_tower(features, init=False)
    return tf.zeros_like(features["targets"]), {"training": train_op}

  def objective_tower(self, features, init=True):
    """Objective in terms of bits-per-pixel.

    Args:
      features: dict of tensors with "features" and "targets" keys.
      init: Whether or not to run data-dependent init.
    Returns:
      objective: float, bits-per-pixel.
    """
    x = features["inputs"]

    # Scale x such that the pixels lie in-between -0.5 and.0.5
    x = self.preprocess(x)
    x, objective = glow_ops.uniform_binning_correction(x)

    # The arg_scope call ensures that the actnorm parameters are set such that
    # the per-channel output activations have zero mean and unit variance
    # ONLY during the first step. After that the parameters are learned
    # through optimisation.
    ops = [glow_ops.get_variable_ddi, glow_ops.actnorm, glow_ops.get_dropout]
    with arg_scope(ops, init=init):
      encoder = glow_ops.encoder_decoder


      self.z, encoder_objective, self.eps, _, _ = encoder(
          "codec", x, self.hparams, eps=None, reverse=False)
      objective += encoder_objective

      self.z_top_shape = common_layers.shape_list(self.z)
      prior_dist = self.top_prior()
      prior_objective = tf.reduce_sum(
          prior_dist.log_prob(self.z), axis=[1, 2, 3])
      self.z_sample = prior_dist.sample()
      objective += prior_objective

    # bits per pixel
    _, h, w, c = common_layers.shape_list(x)
    objective = -objective / (np.log(2) * h * w * c)
    return objective
