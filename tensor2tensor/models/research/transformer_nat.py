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
"""NAT Transformer from https://arxiv.org/abs/1805.11063."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
from six.moves import range
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
import tensorflow as tf
from tensorflow.python.training import moving_averages


def init_vq_bottleneck(bottleneck_size, hidden_size):
  """Get lookup table for VQ bottleneck."""
  means = tf.get_variable(
      name="means",
      shape=[bottleneck_size, hidden_size],
      initializer=tf.uniform_unit_scaling_initializer())
  ema_count = tf.get_variable(
      name="ema_count",
      shape=[bottleneck_size],
      initializer=tf.constant_initializer(0),
      trainable=False)
  with tf.colocate_with(means):
    ema_means = tf.get_variable(
        name="ema_means",
        initializer=means.initialized_value(),
        trainable=False)

  return means, ema_means, ema_count


def vq_nearest_neighbor(x, hparams):
  """Find the nearest element in means to elements in x."""
  bottleneck_size = 2**hparams.bottleneck_bits
  means = hparams.means
  x_norm_sq = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
  means_norm_sq = tf.reduce_sum(tf.square(means), axis=-1, keepdims=True)
  scalar_prod = tf.matmul(x, means, transpose_b=True)
  dist = x_norm_sq + tf.transpose(means_norm_sq) - 2 * scalar_prod
  if hparams.bottleneck_kind == "em":
    x_means_idx = tf.multinomial(-dist, num_samples=hparams.num_samples)
    x_means_hot = tf.one_hot(
        x_means_idx, depth=bottleneck_size)
    x_means_hot = tf.reduce_mean(x_means_hot, axis=1)
  else:
    x_means_idx = tf.argmax(-dist, axis=-1)
    x_means_hot = tf.one_hot(x_means_idx, depth=bottleneck_size)
  x_means = tf.matmul(x_means_hot, means)
  e_loss = tf.reduce_mean(tf.square(x - tf.stop_gradient(x_means)))
  return x_means_hot, e_loss


def vq_discrete_bottleneck(x, hparams):
  """Simple vector quantized discrete bottleneck."""
  tf.logging.info("Using EMA with beta = {}".format(hparams.beta))
  bottleneck_size = 2**hparams.bottleneck_bits
  x_shape = common_layers.shape_list(x)
  x = tf.reshape(x, [-1, hparams.hidden_size])
  x_means_hot, e_loss = vq_nearest_neighbor(
      x, hparams)
  means, ema_means, ema_count = (hparams.means, hparams.ema_means,
                                 hparams.ema_count)

  # Update the ema variables
  updated_ema_count = moving_averages.assign_moving_average(
      ema_count,
      tf.reduce_sum(x_means_hot, axis=0),
      hparams.decay,
      zero_debias=False)

  dw = tf.matmul(x_means_hot, x, transpose_a=True)
  updated_ema_means = moving_averages.assign_moving_average(
      ema_means, dw, hparams.decay, zero_debias=False)
  n = tf.reduce_sum(updated_ema_count, axis=-1, keepdims=True)
  updated_ema_count = (
      (updated_ema_count + hparams.epsilon) /
      (n + bottleneck_size * hparams.epsilon) * n)
  # pylint: disable=g-no-augmented-assignment
  updated_ema_means = updated_ema_means / tf.expand_dims(
      updated_ema_count, axis=-1)
  # pylint: enable=g-no-augmented-assignment
  with tf.control_dependencies([e_loss]):
    update_means = tf.assign(means, updated_ema_means)
    with tf.control_dependencies([update_means]):
      loss = hparams.beta * e_loss

  discrete = tf.reshape(x_means_hot, x_shape[:-1] + [bottleneck_size])
  return discrete, loss


def vq_discrete_unbottleneck(x, hparams):
  """Simple undiscretization from vector quantized representation."""
  x_shape = common_layers.shape_list(x)
  bottleneck_size = 2**hparams.bottleneck_bits
  means = hparams.means
  x_flat = tf.reshape(x, [-1, bottleneck_size])
  result = tf.matmul(x_flat, means)
  result = tf.reshape(result, x_shape[:-1] + [hparams.hidden_size])
  return result


def residual_conv(x, repeat, k, hparams, name, reuse=None):
  """A stack of convolution blocks with residual connections."""
  with tf.variable_scope(name, reuse=reuse):
    dilations_and_kernels = [((1, 1), k) for _ in range(3)]
    for i in range(repeat):
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


def decompress_step(source, hparams, first_relu, name):
  """Decompression function."""
  with tf.variable_scope(name):
    shape = common_layers.shape_list(source)
    multiplier = 2
    kernel = (1, 1)
    thicker = common_layers.conv_block(
        source,
        hparams.hidden_size * multiplier, [((1, 1), kernel)],
        first_relu=first_relu,
        name="decompress_conv")
    return tf.reshape(thicker, [shape[0], shape[1] * 2, 1, hparams.hidden_size])


def compress(x, hparams, name):
  """Compress."""
  with tf.variable_scope(name):
    # Run compression by strided convs.
    cur = x
    k1 = (3, 1)
    k2 = (2, 1)
    cur = residual_conv(cur, hparams.num_compress_steps, k1, hparams, "rc")
    for i in range(hparams.num_compress_steps):
      cur = common_layers.conv_block(
          cur,
          hparams.hidden_size, [((1, 1), k2)],
          strides=k2,
          name="compress_%d" % i)
    return cur


def encode(x, x_space, hparams, name):
  """Transformer preparations and encoder."""
  with tf.variable_scope(name):
    (encoder_input, encoder_self_attention_bias,
     ed) = transformer.transformer_prepare_encoder(x, x_space, hparams)
    encoder_input = tf.nn.dropout(encoder_input, 1.0 - hparams.dropout)
    return transformer.transformer_encoder(
        encoder_input, encoder_self_attention_bias, hparams), ed


def decode_transformer(encoder_output, encoder_decoder_attention_bias, targets,
                       hparams, name):
  """Original Transformer decoder."""
  with tf.variable_scope(name):
    targets = common_layers.flatten4d3d(targets)

    decoder_input, decoder_self_bias = (
        transformer.transformer_prepare_decoder(targets, hparams))

    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    decoder_output = transformer.transformer_decoder(
        decoder_input, encoder_output, decoder_self_bias,
        encoder_decoder_attention_bias, hparams)
    decoder_output = tf.expand_dims(decoder_output, axis=2)
    decoder_output_shape = common_layers.shape_list(decoder_output)
    decoder_output = tf.reshape(
        decoder_output, [decoder_output_shape[0], -1, 1, hparams.hidden_size])
    # Expand since t2t expects 4d tensors.
    return decoder_output


def get_latent_pred_loss(latents_pred, latents_discrete_hot, hparams):
  """Latent prediction and loss."""
  latents_logits = tf.layers.dense(
      latents_pred, 2**hparams.bottleneck_bits, name="extra_logits")
  loss = tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=tf.stop_gradient(latents_discrete_hot), logits=latents_logits)
  return loss


def ae_latent_sample_beam(latents_dense_in, inputs, ed, embed, hparams):
  """Sample from the latent space in the autoencoder."""
  def symbols_to_logits_fn(ids):
    """Go from ids to logits."""
    ids = tf.expand_dims(ids, axis=2)  # Ids start with added all-zeros.
    latents_discrete = tf.pad(ids[:, 1:], [[0, 0], [0, 1], [0, 0]])

    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
      latents_dense = embed(
          tf.one_hot(latents_discrete, depth=2**hparams.bottleneck_bits))
      latents_pred = decode_transformer(inputs, ed, latents_dense, hparams,
                                        "extra")
      logits = tf.layers.dense(
          latents_pred, 2**hparams.bottleneck_bits, name="extra_logits")
      current_output_position = common_layers.shape_list(ids)[1] - 1
      logits = logits[:, current_output_position, :, :]
    return tf.squeeze(logits, axis=[1])

  initial_ids = tf.zeros([tf.shape(latents_dense_in)[0]], dtype=tf.int32)
  length = tf.shape(latents_dense_in)[1]
  ids, _ = beam_search.beam_search(
      symbols_to_logits_fn,
      initial_ids,
      beam_size=1,
      decode_length=length,
      vocab_size=2**hparams.bottleneck_bits,
      alpha=0.0,
      eos_id=-1,
      stop_early=False)

  res = tf.expand_dims(ids[:, 0, :], axis=2)  # Pick first beam.
  return res[:, 1:]  # Remove the added all-zeros from ids.


def ae_transformer_internal(inputs, targets, target_space, hparams, cache=None):
  """Main step used for training."""
  # Encoder.
  inputs = common_layers.flatten4d3d(inputs)
  inputs, ed = encode(inputs, target_space, hparams, "input_enc")

  # Autoencoding.
  losses = {"extra": tf.constant(0.0), "latent_pred": tf.constant(0.0)}

  max_targets_len_from_inputs = tf.concat([inputs, inputs], axis=1)
  targets, _ = common_layers.pad_to_same_length(
      targets,
      max_targets_len_from_inputs,
      final_length_divisible_by=2**hparams.num_compress_steps)
  targets_c = compress(targets, hparams, "compress")
  if hparams.mode != tf.estimator.ModeKeys.PREDICT:
    # Compress and bottleneck.
    latents_discrete_hot, extra_loss = vq_discrete_bottleneck(
        x=targets_c, hparams=hparams)
    latents_dense = vq_discrete_unbottleneck(
        latents_discrete_hot, hparams=hparams)
    latents_dense = targets_c + tf.stop_gradient(latents_dense - targets_c)
    latents_discrete = tf.argmax(latents_discrete_hot, axis=-1)
    tf.summary.histogram("codes", tf.reshape(latents_discrete[:, 0, :], [-1]))
    losses["extra"] = extra_loss

    # Extra loss predicting latent code from input.
    latents_pred = decode_transformer(inputs, ed, latents_dense, hparams,
                                      "extra")
    latent_pred_loss = get_latent_pred_loss(latents_pred, latents_discrete_hot,
                                            hparams)
    losses["latent_pred"] = tf.reduce_mean(latent_pred_loss)
  else:
    latent_len = common_layers.shape_list(targets_c)[1]
    embed = functools.partial(vq_discrete_unbottleneck, hparams=hparams)
    latents_dense = tf.zeros_like(targets_c[:, :latent_len, :, :])
    if cache is None:
      cache = ae_latent_sample_beam(latents_dense, inputs, ed, embed,
                                    hparams)
    cache_hot = tf.one_hot(cache, depth=2**hparams.bottleneck_bits)
    latents_dense = embed(cache_hot)

  # Postprocess.
  d = latents_dense
  pos = tf.get_variable("pos", [1, 1000, 1, hparams.hidden_size])
  pos = pos[:, :common_layers.shape_list(latents_dense)[1] + 1, :, :]
  latents_dense = tf.pad(latents_dense, [[0, 0], [1, 0], [0, 0], [0, 0]]) + pos

  # Decompressing the dense latents
  for i in range(hparams.num_compress_steps):
    j = hparams.num_compress_steps - i - 1
    d = residual_conv(d, 1, (3, 1), hparams, "decompress_rc_%d" % j)
    d = decompress_step(d, hparams, i > 0, "decompress_%d" % j)

  masking = common_layers.inverse_lin_decay(hparams.mask_startup_steps)
  masking *= common_layers.inverse_exp_decay(
      hparams.mask_startup_steps // 4)  # Not much at start.
  masking = tf.minimum(tf.maximum(masking, 0.0), 1.0)
  if hparams.mode == tf.estimator.ModeKeys.PREDICT:
    masking = 1.0
  mask = tf.less(masking,
                 tf.random_uniform(common_layers.shape_list(targets)[:-1]))
  mask = tf.expand_dims(tf.to_float(mask), 3)

  # targets is always [batch, length, 1, depth]
  targets = mask * targets + (1.0 - mask) * d

  res = decode_transformer(inputs, ed, targets, hparams, "decoder")
  latent_time = tf.less(hparams.mask_startup_steps,
                        tf.to_int32(tf.train.get_global_step()))
  losses["latent_pred"] *= tf.to_float(latent_time)
  return res, losses, cache


@registry.register_model
class TransformerNAT(t2t_model.T2TModel):
  """Nonautoregressive Transformer from https://arxiv.org/abs/1805.11063."""

  def __init__(self, *args, **kwargs):
    super(TransformerNAT, self).__init__(*args, **kwargs)
    means, ema_means, ema_count = init_vq_bottleneck(
        2**self._hparams.bottleneck_bits, self._hparams.hidden_size)
    self._hparams.means = means
    self._hparams.ema_means = ema_means
    self._hparams.ema_count = ema_count

  @property
  def has_input(self):
    return self._problem_hparams.input_modality

  def body(self, features):
    inputs = features["inputs"] if "inputs" in features else None
    reuse = "cache_raw" in features
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
      res, loss, _ = ae_transformer_internal(
          inputs, features["targets"], features["target_space_id"],
          self._hparams, features.get("cache_raw", None))
      return res, loss

  def prepare_features_for_infer(self, features):
    batch_size = self._decode_hparams.batch_size
    inputs = tf.zeros([batch_size, 1, 1, self._hparams.hidden_size])
    inputs = inputs if "inputs" in features else None
    targets = tf.zeros([batch_size, 1, 1, self._hparams.hidden_size])
    with tf.variable_scope("transformer_nat/body"):
      _, _, cache = ae_transformer_internal(
          inputs, targets, features["target_space_id"], self._hparams)
    features["cache_raw"] = cache

  def infer(self,
            features=None,
            decode_length=50,
            beam_size=1,
            top_beams=1,
            alpha=0.0,
            use_tpu=False):
    """Produce predictions from the model."""
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
      batch_size = common_layers.shape_list(features["inputs"])[0]
      length = common_layers.shape_list(features["inputs"])[1]
      target_length = tf.to_int32(2.0 * tf.to_float(length))
      initial_output = tf.zeros((batch_size, target_length, 1, 1),
                                dtype=tf.int64)

    features["targets"] = initial_output
    logits, _ = self(features)  # pylint: disable=not-callable
    samples = tf.argmax(logits, axis=-1)
    if inputs_old is not None:  # Restore to not confuse Estimator.
      features["inputs"] = inputs_old
    return samples


@registry.register_hparams
def transformer_nat_small():
  """Set of hyperparameters."""
  hparams = transformer.transformer_small()
  hparams.batch_size = 2048
  hparams.learning_rate = 0.2
  hparams.learning_rate_warmup_steps = 4000
  hparams.num_hidden_layers = 3
  hparams.hidden_size = 384
  hparams.filter_size = 2048
  hparams.label_smoothing = 0.0
  hparams.force_full_predict = True
  hparams.optimizer = "Adam"
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.997
  hparams.add_hparam("bottleneck_kind", "vq")
  hparams.add_hparam("bottleneck_bits", 12)
  hparams.add_hparam("num_compress_steps", 3)
  hparams.add_hparam("beta", 0.25)
  hparams.add_hparam("epsilon", 1e-5)
  hparams.add_hparam("decay", 0.999)
  hparams.add_hparam("num_samples", 10)
  hparams.add_hparam("mask_startup_steps", 50000)
  return hparams


@registry.register_hparams
def transformer_nat_base():
  """Set of hyperparameters."""
  hparams = transformer_nat_small()
  hparams.batch_size = 2048
  hparams.hidden_size = 512
  hparams.filter_size = 4096
  hparams.num_hidden_layers = 6
  return hparams


@registry.register_hparams
def transformer_nat_big():
  """Set of hyperparameters."""
  hparams = transformer_nat_small()
  hparams.batch_size = 2048
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  hparams.num_hidden_layers = 6
  hparams.num_heads = 16
  hparams.layer_prepostprocess_dropout = 0.3
  return hparams
