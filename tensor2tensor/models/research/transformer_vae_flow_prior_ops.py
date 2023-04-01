# coding=utf-8
# Copyright 2023 The Tensor2Tensor Authors.
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

"""Various ops for TransformerVaeFlowPrior."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import transformer_glow_layers_ops as gops
from tensor2tensor.models.transformer import transformer_decoder_layer
from tensor2tensor.models.transformer import transformer_encoder
from tensor2tensor.models.transformer import transformer_prepare_encoder
from tensor2tensor.utils import learning_rate as lr
from tensor2tensor.utils import mlperf_log
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator


def _mixed_precision_is_enabled(hparams):
  """Should be the same as in common_attention, avoiding import."""
  activation_dtype = hparams.activation_dtype
  weight_dtype = hparams.weight_dtype
  return activation_dtype == tf.float16 and weight_dtype == tf.float32


def encoder(name, hparams, inputs, target_space):
  """Compute encoder outputs and attention bias."""
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    (encoder_input,
     encoder_self_attention_bias,
     encoder_decoder_attention_bias) = (
         transformer_prepare_encoder(inputs, target_space, hparams))
    encoder_input = tf.nn.dropout(encoder_input,
                                  rate=hparams.layer_prepostprocess_dropout)
    encoder_output = transformer_encoder(encoder_input,
                                         encoder_self_attention_bias,
                                         hparams)
    return encoder_output, encoder_decoder_attention_bias


def transformer_decoder_layers(name,
                               n_layers,
                               decoder_input,
                               **kwargs):
  """A transformation block composed of transformer decoder layers."""
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    hparams = kwargs["hparams"]
    outputs = decoder_input
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
      for layer_idx in range(n_layers):
        outputs = transformer_decoder_layer(
            decoder_input=outputs,
            layer_idx=layer_idx,
            **kwargs)
      outputs = common_layers.layer_preprocess(outputs, hparams)
    return outputs


def posterior(
    name, hparams, targets, targets_mask, decoder_self_attention_bias,
    **kwargs):
  """Compute mu and sigma for diagonal normal posterior q(z|x,y)."""
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    decoder_input = drop_2d(targets, hparams.mode, hparams.posterior_2d_dropout)
    decoder_input = common_attention.add_timing_signal_1d(decoder_input)
    decoder_input = tf.nn.dropout(decoder_input,
                                  rate=hparams.layer_prepostprocess_dropout)
    decoder_output = transformer_decoder_layers(
        "block",
        n_layers=hparams.n_posterior_layers,
        decoder_input=decoder_input,
        hparams=hparams,
        decoder_self_attention_bias=decoder_self_attention_bias,
        **kwargs)
    decoder_output = gops.dense_weightnorm(
        "h2o_out", decoder_output, hparams.latent_size * 2, targets_mask,
        init_scale=0.0, init=False)
    return decoder_output


def cond_prior(
    name, hparams, decoder_input, targets_mask, output_size,
    decoder_self_attention_bias, init_scale=0.0, **kwargs):
  """Compute hidden states for parameters for conditional prior."""
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    decoder_input = common_attention.add_timing_signal_1d(decoder_input)
    decoder_input = tf.nn.dropout(decoder_input,
                                  rate=hparams.layer_prepostprocess_dropout)
    decoder_output = transformer_decoder_layers(
        "block",
        n_layers=hparams.n_posterior_layers,
        decoder_input=decoder_input,
        hparams=hparams,
        decoder_self_attention_bias=decoder_self_attention_bias,
        **kwargs)
    decoder_output = gops.dense_weightnorm(
        "h2o_out", decoder_output, output_size, targets_mask,
        init_scale=init_scale, init=False)
    return decoder_output


def decoder(name, latents, hparams, decoder_self_attention_bias, **kwargs):
  """Compute final hidden states for p(y|z,x)."""
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    decoder_input = drop_2d(latents, hparams.mode, hparams.decoder_2d_dropout)
    if hparams.pos_attn:
      decoder_input = gops.positional_attention(
          "pos_attn", decoder_input, decoder_self_attention_bias, hparams)
    else:
      decoder_input = common_attention.add_timing_signal_1d(decoder_input)
    if common_layers.shape_list(latents)[-1] != hparams.hidden_size:
      decoder_input = gops.dense("lat2hid", latents, hparams.hidden_size)
    decoder_output = transformer_decoder_layers(
        "block",
        n_layers=hparams.n_decoder_layers,
        decoder_input=decoder_input,
        hparams=hparams,
        decoder_self_attention_bias=decoder_self_attention_bias,
        **kwargs)
    batch_size, targets_length = common_layers.shape_list(decoder_output)[:2]
    decoder_output = tf.reshape(
        decoder_output, [batch_size, targets_length, 1, hparams.hidden_size])
    # Expand since t2t expects 4d tensors.
    return decoder_output


def drop_2d(targets, mode, dropout_p):
  """Dropout in 2D."""
  if dropout_p > 0 and mode == tf_estimator.ModeKeys.TRAIN:
    batch_size, targets_length, hidden_size = common_layers.shape_list(targets)
    mask_prob = tf.random_uniform(
        shape=(batch_size, targets_length), minval=0.0, maxval=1.0)
    mask_prob = tf.tile(mask_prob[..., tf.newaxis], [1, 1, hidden_size])
    scale = 1 / (1 - dropout_p)
    targets_noisy = tf.where(
        mask_prob > dropout_p, targets * scale, tf.zeros_like(targets))
    return targets_noisy
  return targets


def sequence_mask(length, hparams):
  dtype = get_dtype(hparams)
  return tf.sequence_mask(length, dtype=dtype)


def get_padding(mask, hparams):
  dtype = get_dtype(hparams)
  return tf.cast(tf.equal(mask, 0.0), dtype=dtype)


def get_dtype(hparams):
  if hparams.activation_dtype == "float32":
    return tf.float32
  elif hparams.activation_dtype == "float64":
    return tf.float64
  elif hparams.activation_dtype == "bfloat16":
    return tf.bfloat16
  else:
    return None


def lenpred_mlp(name, logits, hidden_size, bound):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    logits = tf.layers.dense(logits, hidden_size)
    logits = tf.nn.elu(logits)
    logits = tf.layers.dense(logits, hidden_size)
    logits = tf.nn.elu(logits)
    logits = tf.layers.dense(logits, bound * 2 + 1)
  return logits


def predict_target_lengths(
    encoder_output, inputs_mask, hparams, length_diff=None):
  """Predict target lengths."""
  bound = hparams.lendiff_bound
  inputs_length = tf.cast(tf.reduce_sum(inputs_mask, 1), tf.int32)
  targets_length = inputs_length
  loss = None
  if hparams.predict_target_length:
    encoder_output = gops.reduce_mean_over_l(encoder_output, inputs_mask)
    logits = tf.stop_gradient(encoder_output)
    logits = lenpred_mlp("lenpred", logits, hparams.hidden_size, bound)
    if length_diff is not None:
      labels = tf.maximum(tf.minimum(length_diff, bound), -bound)
      labels = tf.cast(labels + bound, tf.int32)
      labels = tf.stop_gradient(labels)
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)
      loss = tf.reduce_mean(loss)
    diff_pred = tf.argmax(logits, 1)
    diff_pred = tf.cast(diff_pred - bound, tf.int32)
    targets_length = inputs_length + diff_pred
    targets_length = tf.maximum(targets_length, 1)
  divi = 4
  targets_length = tf.ceil(targets_length / divi) * divi
  targets_length = tf.cast(targets_length, tf.int32)
  return targets_length, loss


def lenpred_stats(targets_length_pred, targets_length):
  lenpred_diff = tf.abs(targets_length_pred - tf.cast(targets_length, tf.int32))
  lenpred_acc = tf.cast(tf.equal(lenpred_diff, 0), tf.float32)
  lenpred_acc = tf.reduce_mean(lenpred_acc)
  lenpred_acc5 = tf.cast(tf.less_equal(lenpred_diff, 5), tf.float32)
  lenpred_acc5 = tf.reduce_mean(lenpred_acc5)
  return lenpred_acc, lenpred_acc5


def save_log_loss(
    hparams, targets_mask, numerator, denominator, log_q_z, log_abs_det,
    log_p_z_base, z_q, lenpred_loss, targets_length_pred, targets_length):
  """Populate loss dictionary and summary."""
  anneal, kl_mask = get_anneal_mask(hparams)
  lenpred_acc, lenpred_acc5 = (
      lenpred_stats(targets_length_pred, targets_length))
  batch_length = tf.reduce_sum(targets_mask)

  z_q_norm = gops.reduce_mean_over_bl(
      tf.norm(z_q, axis=2, keepdims=True), targets_mask)[0]

  log_q_z = gops.reduce_mean_over_bl_sum_over_c(log_q_z, targets_mask)
  log_p_z_base = tf.reduce_sum(log_p_z_base, axis=0) / batch_length
  log_abs_det = tf.reduce_sum(log_abs_det, axis=0) / batch_length
  log_p_z_reg = gops.standard_normal_density(z_q, targets_mask, reduce_sum=True)

  log_p_x = -1 * numerator / denominator
  log_p_z = log_p_z_base + log_abs_det
  kl = log_q_z - log_p_z
  kl_reg = log_p_z - log_p_z_reg
  elbo = log_p_x - kl
  monitor = {
      "elbo": elbo,
      "kl": kl,
      "kl_reg": kl_reg,
      "log_p_x": log_p_x,
      "log_q_z": log_q_z,
      "log_p_z": log_p_z,
      "log_p_z_base": log_p_z_base,
      "log_abs_det": log_abs_det,
      "anneal": anneal,
      "z_q_norm": z_q_norm,
      "lenpred_acc": lenpred_acc,
      "lenpred_acc5": lenpred_acc5,
  }

  kl = kl * anneal
  kl_reg = hparams.kl_reg * kl_reg * anneal
  loss_dict = {
      "training": -1 * log_p_x,
      "kl": kl * kl_mask,
      "kl_reg": kl_reg * kl_mask,
  }
  if lenpred_loss is not None:
    monitor["lenpred_loss"] = lenpred_loss
    loss_dict["lenpred_loss"] = lenpred_loss
  return loss_dict, monitor


def get_anneal_mask(hparams):
  """Get anneal and kl mask."""
  startup = hparams.kl_startup_steps
  anneal = hparams.kl_anneal_steps
  global_step = tf.train.get_global_step()
  min_value = hparams.anneal_min_value
  step = tf.maximum(global_step - startup, 0)
  anneal = common_layers.inverse_lin_decay(
      anneal, min_value=min_value, step=step)
  kl_mask = tf.less(startup, tf.to_int32(global_step))
  kl_mask = tf.cast(kl_mask, tf.float32)
  return anneal, kl_mask


def embedding_to_non_padding(emb, dtype=tf.float32):
  """Calculates the padding mask based on which embeddings are not zero."""
  emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
  return tf.cast(tf.not_equal(emb_sum, 0.0), dtype=dtype)


def save_summary(monitor, name):
  with tf.name_scope(name):
    for key in list(monitor.keys()):
      tf.summary.scalar(key, monitor[key])


def _global_step(hparams):
  """Adjust global step if a multi-step optimizer is used."""
  step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)
  multiplier = hparams.optimizer_multistep_accumulate_steps
  if not multiplier:
    return step

  tf.logging.info("Dividing global step by %d for multi-step optimizer."
                  % multiplier)
  return step / tf.cast(multiplier, tf.float32)


def learning_rate_schedule(hparams):
  """Learning rate schedule based on hparams."""
  mlperf_log.transformer_print(key=mlperf_log.OPT_LR, deferred=True)
  mlperf_log.transformer_print(
      key=mlperf_log.OPT_LR_WARMUP_STEPS,
      value=hparams.learning_rate_warmup_steps)
  step_num = _global_step(hparams)
  # Simulate pretraining the encoder, decoder and posterior with the same
  # learning rate schedule, and then restoring the parameters.
  # using `warm_start_from` is not compatible with actnorm DDI on TPUs.
  step_num = tf.where(
      step_num < hparams.kl_startup_steps,
      step_num,
      step_num - hparams.kl_startup_steps)
  schedule_string = hparams.learning_rate_schedule
  names = schedule_string.split("*")
  names = [name.strip() for name in names if name.strip()]
  ret = tf.constant(1.0)
  for name in names:
    ret *= lr.learning_rate_factor(name, step_num, hparams)
  return ret


def prepare_for_iw(x, k):
  """Prepare feature for importance sampling."""
  batch_size = common_layers.shape_list(x)[0]
  remaining_shape = common_layers.shape_list(x)[1:]

  multiplier = [1] * x.shape.rank
  x = tf.tile(x[tf.newaxis, ...], [k] + multiplier)
  x = tf.reshape(x, [k * batch_size] + remaining_shape)
  return x


def unprepare_for_iw(x, k):
  """Unprepare feature for importance sampling."""
  batch_size_times_k = common_layers.shape_list(x)[0]
  remaining_shape = common_layers.shape_list(x)[1:]
  x = tf.reshape(x, [k, batch_size_times_k // k] + remaining_shape)
  return x


def generic_loss(top_out, targets, model_hparams, vocab_size, weights_fn):
  """Compute loss numerator and denominator for one shard of output."""
  del vocab_size  # unused arg
  logits = top_out
  logits = common_attention.maybe_upcast(logits, hparams=model_hparams)
  cutoff = getattr(model_hparams, "video_modality_loss_cutoff", 0.0)
  return common_layers.padded_cross_entropy(
      logits,
      targets,
      model_hparams.label_smoothing,
      cutoff=cutoff,
      weights_fn=weights_fn,
      reduce_sum=False)
