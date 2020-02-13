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

"""Glow operations for text.

Adapted glow operations from tensor2tensor.models.research.glow_ops to be used
as a prior in Text VAEs (specifically for MT). Supports:
1. Log determinant Jacobian computation with variable length data and masking.
2. Transformer instead of convolution as a basic transformation.
3. Every transformation (affine, split) conditions on the source
  sentence.
4. Three different split functions in affine coupling.
5. Multi-head 1x1 convolution.
6. Actnorm with weight normalization.

Implementation based on Ma et al., 2019: https://arxiv.org/pdf/1909.02480.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import scipy
from tensor2tensor.layers import common_layers
import tensor2tensor.layers.transformer_glow_layers_ops as gops
import tensorflow.compat.v1 as tf


def actnorm(name, x, x_mask, inverse, init, logscale_factor=3.0):
  """Activation normalization, returns logabsdet of shape [B]."""
  eps = tf.keras.backend.epsilon()
  n_channels = common_layers.shape_list(x)[2]

  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    x_mean, x_var = gops.moments_over_bl(x, x_mask)
    b = gops.get_variable_ddi(
        "b", (n_channels), -x_mean, init, tf.zeros_initializer)
    log_w_init = -0.5 * tf.log(x_var + eps) / logscale_factor
    log_w = gops.get_variable_ddi(
        "log_w", (n_channels), log_w_init, init,
        tf.zeros_initializer) * logscale_factor

    if not inverse:
      x = (x + b) * tf.exp(log_w)
    else:
      x = x * tf.exp(-log_w) - b

    x_length = tf.reduce_sum(x_mask, -1)
    logabsdet = x_length * tf.reduce_sum(log_w)
    if inverse:
      logabsdet *= -1
    return x, logabsdet


def multihead_invertible_1x1_conv_np(
    name, x, x_mask, multihead_split, inverse, dtype):
  """Multi-head 1X1 convolution on x."""
  batch_size, length, n_channels_all = common_layers.shape_list(x)
  assert n_channels_all % 32 == 0
  n_channels = 32
  n_1x1_heads = n_channels_all // n_channels

  def get_init_np():
    """Initializer function for multihead 1x1 parameters using numpy."""
    results = []
    for _ in range(n_1x1_heads):
      random_matrix = np.random.rand(n_channels, n_channels)
      np_w = scipy.linalg.qr(random_matrix)[0].astype("float32")
      np_p, np_l, np_u = scipy.linalg.lu(np_w)
      np_s = np.diag(np_u)
      np_sign_s = np.sign(np_s)[np.newaxis, :]
      np_log_s = np.log(np.abs(np_s))[np.newaxis, :]
      np_u = np.triu(np_u, k=1)
      results.append(
          np.concatenate([np_p, np_l, np_u, np_sign_s, np_log_s], axis=0))
    return tf.convert_to_tensor(np.stack(results, axis=0))

  def get_mask_init():
    ones = tf.ones([n_1x1_heads, n_channels, n_channels], dtype=dtype)
    l_mask = tf.matrix_band_part(ones, -1, 0) - tf.matrix_band_part(ones, 0, 0)
    u_mask = tf.matrix_band_part(ones, 0, -1) - tf.matrix_band_part(ones, 0, 0)
    return tf.stack([l_mask, u_mask], axis=0)

  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    params = tf.get_variable("params", initializer=get_init_np, dtype=dtype)
    mask_params = tf.get_variable(
        "mask_params", initializer=get_mask_init, dtype=dtype, trainable=False)

    p = tf.stop_gradient(params[:, :n_channels, :])
    l = params[:, n_channels : 2*n_channels, :]
    u = params[:, 2*n_channels : 3*n_channels, :]
    sign_s = tf.stop_gradient(params[:, 3*n_channels, :])
    log_s = params[:, 3*n_channels+1, :]

    l_mask = mask_params[0]
    u_mask = mask_params[1]

    l_diag = l * l_mask + (
        tf.eye(n_channels, n_channels, [n_1x1_heads], dtype=dtype))
    u_diag = u * u_mask + (
        tf.matrix_diag(sign_s * tf.exp(log_s)))
    w = tf.matmul(p, tf.matmul(l_diag, u_diag))

    if multihead_split == "a":
      x = tf.reshape(x, [batch_size, length, n_channels, n_1x1_heads])
      x = tf.transpose(x, [3, 0, 1, 2])
    elif multihead_split == "c":
      x = tf.reshape(x, [batch_size, length, n_1x1_heads, n_channels])
      x = tf.transpose(x, [2, 0, 1, 3])
    else:
      raise ValueError("Multihead split not supported.")
    # [n_1x1_heads, batch_size, length, n_channels]

    if not inverse:
      # [n_1x1_heads, 1, n_channels, n_channels]
      x = tf.matmul(x, w[:, tf.newaxis, :, :])
    else:
      w_inv = tf.matrix_inverse(w)
      x = tf.matmul(x, w_inv[:, tf.newaxis, :, :])

    if multihead_split == "a":
      x = tf.transpose(x, [1, 2, 3, 0])
      x = tf.reshape(x, [batch_size, length, n_channels * n_1x1_heads])
    elif multihead_split == "c":
      x = tf.transpose(x, [1, 2, 0, 3])
      x = tf.reshape(x, [batch_size, length, n_1x1_heads * n_channels])
    else:
      raise ValueError("Multihead split not supported.")

    x_length = tf.reduce_sum(x_mask, -1)
    logabsdet = x_length * tf.reduce_sum(log_s)
    if inverse:
      logabsdet *= -1
  return x, logabsdet


def coupling(*args, **kwargs):
  """Coupling transform layer."""
  prior_type = kwargs["hparams"].prior_type
  posterior_type = kwargs["hparams"].posterior_type
  if prior_type == "affine" or posterior_type == "affine":
    return affine_coupling(*args, **kwargs)
  elif prior_type == "additive" or posterior_type == "additive":
    return additive_coupling(*args, **kwargs)


def additive_coupling(
    name, x, x_mask, inverse, split_dim, identity_first, init,
    decoder_self_attention_bias=None, **kwargs):
  """Additive coupling transform layer."""
  hparams = kwargs["hparams"]
  batch_size, length, n_channels = common_layers.shape_list(x)
  assert hparams.scale_width > 0.0 and hparams.scale_width < 1.0
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    x_id, x_tr, _, n_transform, bias, mask = gops.split_coupling(
        x, x_mask, split_dim, identity_first, decoder_self_attention_bias)
    z_id = x_id

    loc = gops.transformer_decoder_block(
        "theta_tr",
        n_layers=hparams.n_layers_transform_params,
        x=x_id,
        x_mask=mask,
        output_size=n_transform,
        init=init,
        decoder_self_attention_bias=bias,
        **kwargs)
    if not inverse:
      z_tr = x_tr + loc
    else:
      z_tr = x_tr - loc
    logabsdet = tf.constant(0.0, dtype=tf.float32)

    tf.summary.histogram("_loc", tf.boolean_mask(loc, mask))
    result = gops.join_coupling(z_id, z_tr, split_dim, identity_first)
    result = tf.reshape(result, [batch_size, length, n_channels])
    return result, logabsdet


def affine_coupling(
    name, x, x_mask, inverse, split_dim, identity_first, init,
    decoder_self_attention_bias=None, **kwargs):
  """Affine coupling transform layer.

  Args:
    name: variable scope.
    x: 3-D Tensor, shape=[B, L, C].
    x_mask : 2-D Tensor, shape=[B, L].
    inverse: Forward or inverse pass.
    split_dim: which dimension to split
      (time, channel_continuous, channel_alternate).
    identity_first: True means the first half remains constant. False for 2nd.
    init: init.
    decoder_self_attention_bias: bias.
    **kwargs: additional arguments. Contains hparams, encoder_output and
      encoder_decoder_attention_bias.

  Returns:
    z: data transformed by the affine coupling layer. shape=[B, L, C]
    logabsdets: Log absolute determinant Jacobian. shape=[B]
  """
  hparams = kwargs["hparams"]
  batch_size, length, n_channels = common_layers.shape_list(x)
  assert hparams.scale_width > 0.0 and hparams.scale_width < 1.0
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    x_id, x_tr, _, n_transform, bias, mask = gops.split_coupling(
        x, x_mask, split_dim, identity_first, decoder_self_attention_bias)
    z_id = x_id

    transform_params = gops.transformer_decoder_block(
        "theta_tr",
        n_layers=hparams.n_layers_transform_params,
        x=x_id,
        x_mask=mask,
        output_size=n_transform*2,
        init=init,
        decoder_self_attention_bias=bias,
        **kwargs)
    loc, unconstrained_scale = tf.split(transform_params, 2, axis=-1)
    scale = tf.sigmoid(unconstrained_scale + 2.0)
    if not inverse:
      z_tr = (x_tr + loc) * scale
    else:
      z_tr = x_tr / scale - loc

    logabsdet = gops.reduce_sum_over_lc(tf.log(scale), mask)  # [B]
    if inverse:
      logabsdet *= -1

    tf.summary.histogram("_loc", tf.boolean_mask(loc, mask))
    tf.summary.histogram("_scale", tf.boolean_mask(scale, mask))
    result = gops.join_coupling(z_id, z_tr, split_dim, identity_first)
    result = tf.reshape(result, [batch_size, length, n_channels])
    return result, logabsdet


def flow_step_glow(name, x, x_mask, split_dims, inverse, init, dtype, **kwargs):
  """One step of flow."""
  conv_fn = multihead_invertible_1x1_conv_np
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    reversible_ops = []
    for _, split_dim in enumerate(split_dims):
      identity_first = True
      reversible_ops += [functools.partial(actnorm, name="actnorm", init=init)]
      if split_dim in "ca":
        multihead_split = "a" if split_dim == "c" else "c"
        reversible_ops += [functools.partial(
            conv_fn, name="conv_{}".format(multihead_split),
            multihead_split=multihead_split, dtype=dtype)]
      reversible_ops += [functools.partial(
          coupling, name="coupling_{}".format(split_dim),
          split_dim=split_dim, identity_first=identity_first, init=init,
          **kwargs)]
    if inverse:
      reversible_ops = reversible_ops[::-1]

    logabsdets = tf.constant(0.0, dtype=dtype)
    for reversible_op in reversible_ops:
      x, logabsdet = reversible_op(x=x, x_mask=x_mask, inverse=inverse)
      logabsdets += logabsdet
    return x, logabsdets


def flow_level(
    name, x, x_mask, depth, split_dims, prior, inverse, init, dtype, **kwargs):
  """One level of flow."""
  flow_step_fn = flow_step_glow
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    reversible_ops = []
    for step in np.arange(depth):
      reversible_ops += [functools.partial(
          flow_step_fn, name="{}_step".format(step), split_dims=split_dims,
          init=init, dtype=dtype, **kwargs)]
    if prior:
      reversible_ops += [functools.partial(
          coupling, name="{}_prior".format(depth), split_dim="c",
          identity_first=True, init=init, **kwargs)]
    if inverse:
      reversible_ops = reversible_ops[::-1]

    logabsdets = tf.constant(0.0, dtype=dtype)
    for reversible_op in reversible_ops:
      x, logabsdet = reversible_op(x=x, x_mask=x_mask, inverse=inverse)
      logabsdets += logabsdet
    return x, logabsdets


def split(name, x, x_mask, inverse, temp=1.0, dtype=tf.float32, z=None):
  """Splits / concatenates x into x1 and x2 across number of channels.

  x2 is modelled with a standard gaussian distribution.
  Args:
    name: variable scope.
    x: 3-D Tensor, shape=[B, L, C].
    x_mask: 2-D Tensor, shape=[B, L].
    inverse: forward or inverse pass.
    temp: only used for inverse pass. temperature for sampling.
    dtype: dtype
    z: used in inverse pass to check invertibility.

  Returns:
    x: if forward, returns the 1st half of the channel dimensions.
      if inverse, return concat[input, N(0,1)]
    z: second half of the channel dimensions. modelled as standard normal.
    log_p: log p(x2; N(0,1)), shape=[B]
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    if not inverse:
      x1, x2 = tf.split(x, 2, axis=-1)
      log_p = gops.standard_normal_density(x2, x_mask)
      return x1, x2, log_p
    else:
      if z is None:
        x2 = tf.random.normal(
            common_layers.shape_list(x), stddev=temp, dtype=dtype)
      else:
        x2 = z
      log_p = gops.standard_normal_density(x2, x_mask)
      return tf.concat([x, x2], 2), None, log_p


def squeeze(name, x, factor, inverse):
  """Temporal squeezing of x to increase the number of channels."""
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    if factor == 1:
      return x
    batch_size, length, n_channels = common_layers.shape_list(x)
    if not inverse:
      x = tf.reshape(x, [batch_size, length//factor, factor, n_channels])
      # transposing groups neighbouring elements together.
      x = tf.transpose(x, [0, 1, 3, 2])
      x = tf.reshape(x, [batch_size, length//factor, n_channels*factor])
    else:
      x = tf.reshape(x, (batch_size, length, n_channels//factor, factor))
      x = tf.transpose(x, [0, 1, 3, 2])
      x = tf.reshape(x, (batch_size, length*factor, n_channels//factor))
    return x


def glow(
    name, x, max_x_mask, max_self_attn_bias, inverse, init, dtype=tf.float32,
    split_zs=None, temp=1.0, **kwargs):
  """Multi-scale glow model. Flow + (n_levels-1)*(Split + Squeeze + Flow).

  Note the original glow's ordering is Squeeze + Flow + Split.

  Args:
    name: variable scope.
    x: 3-D Tensor, shape=[B, L, C]. The length dimension is padded to the
      closest multiple of factor**n_levels.
    max_x_mask : 2-D Tensor, shape=[B, L]. Binary mask indicating padding.
    max_self_attn_bias : 4-D Tensor, shape=[B, 1, 1, L].
    inverse: forward or inverse pass.
    init: init.
    dtype: dtype.
    split_zs: intermediate latents modelled as a standard normal.
    temp: Only used in inverse. Temperature for sampling.
    **kwargs: additional arguments. Contains hparams, disable_dropout,
      encoder_output and encoder_decoder_attention_bias.

  Returns:
    x: if forward, data transformed to the base distribution.
      if inverse, base transformed to the data (latent) distribution.
    logabsdets: log absolute determinant Jacobian. [B]
    log_ps: log probability in the base distribution. [B]
    split_zs: all intermediate latents (only used to check invertibility.)
  """
  assert x.shape.rank == 3
  hparams = kwargs["hparams"]
  factor = hparams.factor
  if hparams.depths:
    depths = [int(depth_str) for depth_str in hparams.depths.split("/")]
  else:
    depths = []
  split_plans = hparams.split_plans.split("/")
  n_levels = len(depths)
  logabsdets = tf.constant(0.0, dtype=dtype)
  log_ps = tf.constant(0.0, dtype=dtype)
  with tf.variable_scope(name, use_resource=True, reuse=tf.AUTO_REUSE):
    if not inverse:  # z -> e (density estimation)
      x_mask, self_attn_bias = max_x_mask, max_self_attn_bias
      split_zs = []
      for level in range(n_levels):
        if level > 0:
          x, z, log_p_z = split(
              "{}_split".format(level), x, x_mask, inverse, dtype)
          log_ps += log_p_z
          split_zs.append(z)

          x = squeeze("{}_squeeze".format(level), x, factor, inverse)
          x_mask = max_x_mask[:, ::factor**level]
          self_attn_bias = max_self_attn_bias[..., ::factor**level]

        prior = level < n_levels - 1
        x, logabsdet = flow_level(
            "{}_level".format(level), x, x_mask, depths[level],
            split_plans[level], prior, inverse, init, dtype,
            decoder_self_attention_bias=self_attn_bias, **kwargs)
        logabsdets += logabsdet  # (B)

      log_p_base = gops.standard_normal_density(x, x_mask)
      log_ps += log_p_base
      return x, logabsdets, log_ps, split_zs

    else:  # e -> z (sampling)
      x_mask = max_x_mask[:, ::factor**(n_levels-1)]
      log_p_base = gops.standard_normal_density(x, x_mask)
      log_ps += log_p_base
      if split_zs is None:
        split_zs = [None] * (n_levels-1)

      for level in reversed(range(n_levels)):
        x_mask = max_x_mask[:, ::factor**level]
        self_attn_bias = max_self_attn_bias[..., ::factor**level]
        prior = level < n_levels - 1
        x, logabsdet = flow_level(
            "{}_level".format(level), x, x_mask, depths[level],
            split_plans[level], prior, inverse, init, dtype,
            decoder_self_attention_bias=self_attn_bias, **kwargs)
        logabsdets += logabsdet

        if level > 0:
          x = squeeze("{}_squeeze".format(level), x, factor, inverse)
          x_mask = max_x_mask[:, ::factor**(level-1)]
          x, _, log_p_z = split(
              "{}_split".format(level), x, x_mask, inverse, temp=temp,
              dtype=dtype, z=split_zs[level-1])
          log_ps += log_p_z

      return x, logabsdets, log_ps, None
