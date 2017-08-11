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

"""VAE Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


def residual_conv(x, repeat, hparams, name, reuse=None):
  """A stack of convolution blocks with residual connections."""
  with tf.variable_scope(name, reuse=reuse):
    k = (3, 1)
    dilations_and_kernels = [((1, 1), k) for _ in xrange(3)]
    for i in xrange(repeat):
      with tf.variable_scope("repeat_%d" % i):
        y = common_layers.conv_block(
            common_layers.layer_norm(x, hparams.hidden_size, name="lnorm"),
            hparams.hidden_size,
            dilations_and_kernels,
            padding="SAME",
            name="residual_conv")
        y = tf.nn.dropout(y, 1.0 - hparams.dropout)
        x += y
    return x


def attend(x, source, hparams, name):
  with tf.variable_scope(name):
    x = tf.squeeze(x, axis=2)
    if len(source.get_shape()) > 3:
      source = tf.squeeze(source, axis=2)
    source = common_attention.add_timing_signal_1d(source)
    y = common_attention.multihead_attention(
        common_layers.layer_preprocess(x, hparams), source, None,
        hparams.attention_key_channels or hparams.hidden_size,
        hparams.attention_value_channels or hparams.hidden_size,
        hparams.hidden_size, hparams.num_heads,
        hparams.attention_dropout)
    res = common_layers.layer_postprocess(x, y, hparams)
    return tf.expand_dims(res, axis=2)


def interleave(x, y, axis=1):
  x = tf.expand_dims(x, axis=axis+1)
  y = tf.expand_dims(y, axis=axis+1)
  return tf.concat([x, y], axis=axis+1)


def decompress_step(source, c, hparams, first_relu, name):
  """Decompression function."""
  with tf.variable_scope(name):
    shape = tf.shape(source)
    if c is not None:
      source = attend(source, c, hparams, "decompress_attend")
    first = common_layers.conv_block(
        source,
        hparams.hidden_size, [((1, 1), (3, 1)), ((1, 1), (3, 1))],
        first_relu=first_relu, padding="SAME", name="decompress_conv1")
    second = common_layers.conv_block(
        tf.concat([source, first], axis=3),
        hparams.hidden_size, [((1, 1), (3, 1)), ((1, 1), (3, 1))],
        first_relu=first_relu, padding="SAME", name="decompress_conv2")
    thicker = interleave(first, second)
    return tf.reshape(thicker, [shape[0], shape[1] * 2, 1, hparams.hidden_size])


def vae(x, hparams, name):
  with tf.variable_scope(name):
    mu = tf.layers.dense(x, hparams.z_size, name="mu")
    log_sigma = tf.layers.dense(x, hparams.z_size, name="log_sigma")
    shape = tf.shape(x)
    epsilon = tf.random_normal([shape[0], shape[1], 1, hparams.z_size])
    z = mu + tf.exp(log_sigma / 2) * epsilon
    kl = 0.5 * tf.reduce_mean(
        tf.exp(log_sigma) + tf.square(mu) - 1. - log_sigma, axis=-1)
    return z, tf.reduce_mean(kl), mu, log_sigma


def compress(x, c, hparams, name):
  """Compress."""
  with tf.variable_scope(name):
    # Run compression by strided convs.
    cur = x
    for i in xrange(hparams.num_compress_steps):
      if c is not None:
        cur = attend(cur, c, hparams, "compress_attend_%d" % i)
      cur = residual_conv(cur, 1, hparams, "compress_rc_%d" % i)
      cur = common_layers.conv_block(
          cur, hparams.hidden_size, [((1, 1), (2, 1))],
          strides=(2, 1), name="compress_%d" % i)
    return cur


def vae_compress(x, c, hparams, compress_name, decompress_name, reuse=None):
  """Compress, then VAE."""
  with tf.variable_scope(compress_name, reuse=reuse):
    cur = compress(x, c, hparams, "compress")
    # Convolve and ReLu to get state.
    cur = common_layers.conv_block(
        cur, hparams.hidden_size, [((1, 1), (1, 1))], name="mid_conv")
    z, kl_loss, mu, log_sigma = vae(cur, hparams, name="vae")

  with tf.variable_scope(decompress_name, reuse=reuse):
    # Decompress.
    z = tf.layers.dense(z, hparams.hidden_size, name="z_to_dense")

    for i in xrange(hparams.num_compress_steps):
      j = hparams.num_compress_steps - i - 1
      z = residual_conv(z, 1, hparams, "decompress_rc_%d" % j)
      z = decompress_step(z, c, hparams, i > 0, "decompress__step_%d" % j)
    return z, kl_loss, mu, log_sigma


def encode(x, x_space, hparams, name):
  """Transformer preparations and encoder."""
  with tf.variable_scope(name):
    (encoder_input, encoder_self_attention_bias,
     _) = transformer.transformer_prepare_encoder(x, x_space, hparams)
    encoder_input = tf.nn.dropout(encoder_input, 1.0 - hparams.dropout)
    return transformer.transformer_encoder(
        encoder_input, encoder_self_attention_bias, hparams)


def dropmask(targets, targets_dropout_max, is_training):
  if not is_training:
    return targets
  targets_drop_prob = tf.random_uniform([]) * targets_dropout_max
  drop_mask = tf.random_uniform(tf.shape(targets)[:-1])
  drop_mask = tf.to_float(tf.less(drop_mask, targets_drop_prob))
  keep_mask = tf.expand_dims(1.0 - drop_mask, axis=2)
  return targets * keep_mask


def ffn(x, hparams, name):
  with tf.variable_scope(name):
    y = transformer.transformer_ffn_layer(
        common_layers.layer_preprocess(x, hparams), hparams)
    return common_layers.layer_postprocess(x, y, hparams)


def vae_transformer_internal(inputs, targets, target_space, hparams):
  """VAE Transformer, main step used for training."""
  with tf.variable_scope("vae_transformer"):
    is_training = hparams.mode == tf.contrib.learn.ModeKeys.TRAIN
    # Prepare inputs, targets, and k.
    inputs = common_layers.flatten4d3d(inputs)
    input_len = tf.shape(inputs)[1]  # Double input size to cover targets.
    inputs = tf.pad(inputs, [[0, 0], [0, input_len], [0, 0]])
    inputs.set_shape([None, None, hparams.hidden_size])
    targets = common_layers.flatten4d3d(targets)
    k = 2**hparams.num_compress_steps
    inputs, targets = common_layers.pad_to_same_length(
        inputs, targets, final_length_divisible_by=k)
    inputs = encode(inputs, target_space, hparams, "input_enc")

    # Dropout targets or swap for zeros 5% of the time.
    targets_nodrop = targets
    max_prestep = hparams.kl_warmup_steps
    prob_targets = 0.95 if is_training else 1.0
    targets_dropout_max = common_layers.inverse_lin_decay(max_prestep) - 0.01
    targets = dropmask(targets, targets_dropout_max * 0.7, is_training)
    targets = tf.cond(tf.less(tf.random_uniform([]), prob_targets),
                      lambda: targets, lambda: tf.zeros_like(targets))
    targets = targets_nodrop

    # Compress and vae.
    z = tf.get_variable("z", [hparams.hidden_size])
    z = tf.reshape(z, [1, 1, 1, -1])
    z = tf.tile(z, [tf.shape(inputs)[0], 1, 1, 1])

    z = attend(z, inputs, hparams, "z_attendsi")
    z = ffn(z, hparams, "zff2")
    z = attend(z, targets, hparams, "z_attendst2")
    z = ffn(z, hparams, "zff3")
    z, kl_loss, _, _ = vae(z, hparams, name="vae")
    z = tf.layers.dense(z, hparams.hidden_size, name="z_to_dense")

    # z, kl_loss, _, _ = vae_compress(
    #     tf.expand_dims(targets, axis=2), tf.expand_dims(inputs, axis=2),
    #     hparams, "vae_compress", "vae_decompress")

    decoder_in = tf.squeeze(z, axis=2) + tf.zeros_like(targets)
    (decoder_input, decoder_self_attention_bias) = (
        transformer.transformer_prepare_decoder(decoder_in, hparams))
    ret = transformer.transformer_decoder(
        decoder_input, inputs, decoder_self_attention_bias, None, hparams)

    kl_loss *= common_layers.inverse_exp_decay(int(max_prestep * 1.5)) * 5.0
    losses = {"kl": kl_loss}
    return tf.expand_dims(ret, axis=2), losses


@registry.register_model
class TransformerVAE(t2t_model.T2TModel):

  def model_fn_body(self, features):
    return vae_transformer_internal(
        features["inputs"], features["targets"], features["target_space_id"],
        self._hparams)

  def infer(self, features=None, decode_length=50, beam_size=1, top_beams=1,
            last_position_only=False, alpha=0.0):
    """A inference method, see T2TModel."""
    if not features:
      features = {}
    inputs_old = None
    if "inputs" in features and len(features["inputs"].shape) < 4:
      inputs_old = features["inputs"]
      features["inputs"] = tf.expand_dims(features["inputs"], 2)

    # Create an initial targets tensor.
    if "partial_targets" in features:
      initial_output = tf.convert_to_tensor(features["partial_targets"])
    else:
      batch_size = tf.shape(features["inputs"])[0]
      initial_output = tf.zeros((batch_size, 1, 1, 1), dtype=tf.int64)

    features["targets"] = initial_output
    sharded_logits, _ = self.model_fn(
        features, False, last_position_only=last_position_only)
    sharded_samples = self._data_parallelism(tf.argmax, sharded_logits, 4)
    samples = tf.concat(sharded_samples, 0)

    # More steps.
    how_many_more_steps = 20
    for _ in xrange(how_many_more_steps):
      with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        features["targets"] = samples
        sharded_logits, _ = self.model_fn(
            features, False, last_position_only=last_position_only)
        sharded_samples = self._data_parallelism(tf.argmax, sharded_logits, 4)
        samples = tf.concat(sharded_samples, 0)

    if inputs_old is not None:  # Restore to not confuse Estimator.
      features["inputs"] = inputs_old
    return samples


@registry.register_hparams
def transformer_vae_small():
  """Set of hyperparameters."""
  hparams = transformer.transformer_small()
  hparams.batch_size = 2048
  hparams.learning_rate_warmup_steps = 16000
  hparams.add_hparam("z_size", 128)
  hparams.add_hparam("num_compress_steps", 4)
  hparams.add_hparam("kl_warmup_steps", 60000)
  return hparams


@registry.register_hparams
def transformer_vae_base():
  """Set of hyperparameters."""
  hparams = transformer_vae_small()
  hparams.hidden_size = 512
  hparams.filter_size = 2048
  hparams.attention_dropout = 0.0
  hparams.relu_dropout = 0.0
  hparams.dropout = 0.0
  hparams.num_hidden_layers = 3
  hparams.z_size = 256
  return hparams
