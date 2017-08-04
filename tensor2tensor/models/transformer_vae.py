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

from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


def decompress(source, hparams, name):
  """Decompression function."""
  with tf.variable_scope(name):
    shape = tf.shape(source)
    thicker = common_layers.conv_block(
        source, hparams.hidden_size * 2, [((1, 1), (1, 1))],
        name="decompress_conv")
    return tf.reshape(thicker, [shape[0], shape[1] * 2, 1, hparams.hidden_size])


def vae(x, hparams, name):
  with tf.variable_scope(name):
    mu = tf.layers.dense(x, hparams.z_size, name="mu")
    log_sigma = tf.layers.dense(x, hparams.z_size, name="log_sigma")
    shape = tf.shape(x)
    epsilon = tf.random_normal([shape[0], shape[1], 1, hparams.z_size])
    z = mu + tf.exp(log_sigma / 2) * epsilon
    dense = tf.layers.dense(z, hparams.hidden_size, name="z_to_dense")
    kl = 0.5 * tf.reduce_mean(
        tf.exp(log_sigma) + tf.square(mu) - 1. - log_sigma, axis=-1)
    return dense, tf.reduce_mean(kl)


def compress_vae(inputs, hparams, name):
  """Compress, then VAE."""
  with tf.variable_scope(name):
    # Run compression by strided convs.
    cur = tf.expand_dims(inputs, axis=2)
    for i in xrange(hparams.num_compress_steps):
      cur = common_layers.conv_block(
          cur, hparams.hidden_size, [((1, 1), (2, 1))],
          strides=(2, 1), name="compress_%d" % i)

    # Convolve and ReLu to get state.
    cur = common_layers.conv_block(
        cur, hparams.hidden_size, [((1, 1), (1, 1))], name="mid_conv")

    cur, kl_loss = vae(cur, hparams, name="vae")
    return cur, kl_loss


def vae_transformer_internal(inputs, targets, target_space, hparams):
  """VAE Transformer, main step used for training."""
  with tf.variable_scope("vae_transformer"):
    is_training = hparams.mode == tf.contrib.learn.ModeKeys.TRAIN
    # Prepare inputs, targets, and k.
    inputs = common_layers.flatten4d3d(inputs)
    targets = common_layers.flatten4d3d(targets)
    k = 2**hparams.num_compress_steps
    _, targets = common_layers.pad_to_same_length(
        inputs, targets, final_length_divisible_by=k)

    # Transformer preparations and encoder.
    (encoder_input, encoder_self_attention_bias,
     encoder_decoder_attention_bias) = transformer.transformer_prepare_encoder(
         inputs, target_space, hparams)
    residual_fn = transformer.get_residual_fn(hparams)
    encoder_input = tf.nn.dropout(encoder_input, 1.0 - hparams.residual_dropout)
    encoder_output = transformer.transformer_encoder(
        encoder_input, residual_fn, encoder_self_attention_bias, hparams)

    def get_decoder_autoregressive():
      """Decoder input for autoregressive computation."""
      (a, b) = transformer.transformer_prepare_decoder(targets, hparams)
      return (a, b, tf.constant(0.0))

    # 10% of the time we compress all-zeros, as will be at decoding start.
    prob_targets = 0.9 if is_training else 1.0
    to_compress = tf.cond(tf.less(tf.random_uniform([]), prob_targets),
                          lambda: targets, lambda: tf.zeros_like(targets))
    z, kl_loss = compress_vae(to_compress, hparams, "vae")
    # Decompress.
    for i in xrange(hparams.num_compress_steps):
      j = hparams.num_hidden_layers - i - 1
      z = decompress(z, hparams, "decompress_%d" % j)

    def get_decoder_from_vae():
      """Decoder input computed by VAE."""
      # Return decoder stuff.
      (a, b) = transformer.transformer_prepare_decoder(
          tf.squeeze(z, axis=2), hparams)
      return (a, b, kl_loss)

    # Randomize decoder inputs..
    prob_do_vae = common_layers.inverse_exp_decay(40000) * 0.7
    step = tf.to_float(tf.contrib.framework.get_global_step())
    if not is_training:
      prob_do_vae = tf.cond(tf.less(step, 40000.0), lambda: tf.constant(0.0),
                            lambda: tf.constant(1.0))
    (decoder_input, decoder_self_attention_bias, kl_loss2) = tf.cond(
        tf.less(tf.random_uniform([]), prob_do_vae),
        get_decoder_from_vae, get_decoder_autoregressive)

    # Transformer decoder.
    decoder_output = transformer.transformer_decoder(
        decoder_input, encoder_output, residual_fn, decoder_self_attention_bias,
        encoder_decoder_attention_bias, hparams)
    decoder_output = tf.expand_dims(decoder_output, 2)

    cond_self = tf.cond(tf.less(step, 30000.0), lambda: tf.constant(1.0),
                        lambda: tf.constant(0.0))
    prob_self = 0.4 if is_training else cond_self
    (ret, kl_loss) = tf.cond(tf.less(tf.random_uniform([]), prob_self),
                             lambda: (z, kl_loss),
                             lambda: (decoder_output, kl_loss2))

    kl_loss *= common_layers.inverse_exp_decay(50000) * 2.0
    return ret, kl_loss


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
    if inputs_old is not None:  # Restore to not confuse Estimator.
      features["inputs"] = inputs_old
    return samples


@registry.register_hparams
def transformer_vae_small():
  """Set of hyperparameters."""
  hparams = transformer.transformer_small()
  hparams.add_hparam("z_size", 128)
  hparams.add_hparam("num_compress_steps", 4)
  return hparams
