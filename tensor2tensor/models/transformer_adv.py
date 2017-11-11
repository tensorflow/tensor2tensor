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

"""Adversarial Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.models import transformer_vae
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


def encode(x, x_space, hparams, name):
  """Transformer preparations and encoder."""
  with tf.variable_scope(name):
    (encoder_input, encoder_self_attention_bias,
     ed) = transformer.transformer_prepare_encoder(x, x_space, hparams)
    encoder_input = tf.nn.dropout(encoder_input, 1.0 - hparams.dropout)
    return transformer.transformer_encoder(
        encoder_input, encoder_self_attention_bias, hparams), ed


def decode(encoder_output, encoder_decoder_attention_bias, targets,
           hparams, name, reuse=False):
  """Transformer decoder."""
  with tf.variable_scope(name, reuse=reuse):
    targets = common_layers.flatten4d3d(targets)

    decoder_input, decoder_self_bias = transformer.transformer_prepare_decoder(
        targets, hparams)

    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    decoder_output = transformer.transformer_decoder(
        decoder_input,
        encoder_output,
        decoder_self_bias,
        encoder_decoder_attention_bias,
        hparams)

    # Expand since t2t expects 4d tensors.
    return tf.expand_dims(decoder_output, axis=2)


def reverse_gradient(x, delta=1.0):
  return tf.stop_gradient((1.0 + delta) * x) - delta * x


def adversary(embedded, inputs, hparams, name, reuse=False):
  with tf.variable_scope(name, reuse=reuse):
    h0, i0 = common_layers.pad_to_same_length(
        embedded, inputs, final_length_divisible_by=16)
    h0 = tf.concat([h0, tf.expand_dims(i0, axis=2)], axis=-1)
    h0 = tf.layers.dense(h0, hparams.hidden_size, name="io")
    h1 = transformer_vae.compress(h0, None, False, hparams, "compress1")
    h2 = transformer_vae.compress(h1, None, False, hparams, "compress2")
    res_dense = tf.reduce_mean(h2, axis=[1, 2])
    res_single = tf.squeeze(tf.layers.dense(res_dense, 1), axis=-1)
    return tf.nn.sigmoid(res_single)


def softmax_embed(x, embedding, batch_size, hparams):
  """Softmax x and embed."""
  x = tf.reshape(tf.nn.softmax(x), [-1, 34*1024])
  x = tf.matmul(x, embedding)
  return tf.reshape(x, [batch_size, -1, 1, hparams.hidden_size])


def adv_transformer_internal(inputs, targets, target_space, hparams):
  """Adversarial Transformer, main step used for training."""
  with tf.variable_scope("adv_transformer"):
    batch_size = tf.shape(targets)[0]
    targets = tf.reshape(targets, [batch_size, -1, 1])
    intermediate = tf.constant(34*1024 - 1)
    intermediate += tf.zeros_like(targets)
    targets = tf.concat([targets, intermediate], axis=2)
    targets = tf.reshape(targets, [batch_size, -1, 1])
    embedding = tf.get_variable("embedding", [34*1024, hparams.hidden_size])
    targets_emb = tf.gather(embedding, targets)

    # Noisy embedded targets.
    targets_noisy = tf.one_hot(targets, 34*1024)
    noise_val = hparams.noise_val
    targets_noisy += tf.random_uniform(tf.shape(targets_noisy),
                                       minval=-noise_val, maxval=noise_val)
    targets_emb_noisy = softmax_embed(
        targets_noisy, embedding, batch_size, hparams)

    # Encoder.
    if inputs is not None:
      inputs_emb = common_layers.flatten4d3d(inputs)
      inputs, ed = encode(inputs_emb, target_space, hparams, "input_enc")
    else:
      ed = None

    # Masking.
    masking = common_layers.inverse_lin_decay(200000)
    masking *= common_layers.inverse_exp_decay(50000)  # Not much at start.
    masking -= tf.random_uniform([]) * 0.4
    masking = tf.minimum(tf.maximum(masking, 0.0), 1.0)
    mask = tf.less(masking, tf.random_uniform(tf.shape(targets)))
    mask = tf.expand_dims(tf.to_float(mask), 3)
    noise = tf.random_uniform(tf.shape(targets_emb))
    targets_emb = mask * targets_emb + (1.0 - mask) * noise

    # Decoder.
    res_dec = decode(inputs, ed, targets_emb, hparams, "decoder")
    res = tf.layers.dense(res_dec, 34*1024, name="res_sm")
    res_emb = softmax_embed(res, embedding, batch_size, hparams)

    # Extra steps.
    extra_step_prob = masking * 0.6 + 0.3
    if hparams.mode != tf.estimator.ModeKeys.TRAIN:
      extra_step_prob = 1.0
    for _ in xrange(hparams.extra_steps):
      def another_step(emb):
        res_dec = decode(inputs, ed, emb, hparams, "decoder", reuse=True)
        res = tf.layers.dense(res_dec, 34*1024, name="res_sm", reuse=True)
        return softmax_embed(res, embedding, batch_size, hparams), res
      res_emb, res = tf.cond(tf.less(tf.random_uniform([]), extra_step_prob),
                             lambda e=res_emb: another_step(e),
                             lambda: (res_emb, res))

    # Adversary.
    delta = masking * hparams.delta_max
    true_logit = adversary(tf.stop_gradient(targets_emb_noisy),
                           tf.stop_gradient(inputs + inputs_emb),
                           hparams, "adversary")
    gen_logit = adversary(reverse_gradient(res_emb, delta),
                          tf.stop_gradient(inputs + inputs_emb),
                          hparams, "adversary", reuse=True)
    losses = {"adv": gen_logit - true_logit}
    res = tf.stop_gradient(masking * res) + (1.0 - masking) * res
    return res, losses


@registry.register_model
class TransformerAdv(t2t_model.T2TModel):
  """Adversarial Transformer."""

  def model_fn_body(self, features):
    inputs = features.get("inputs", None)
    return adv_transformer_internal(
        inputs, features["targets_raw"],
        features["target_space_id"], self._hparams)

  def infer(self, features=None, decode_length=50, beam_size=1, top_beams=1,
            alpha=0.0):
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
      batch_size = tf.shape(features["inputs"])[0]
      length = tf.shape(features["inputs"])[1]
      initial_output = tf.zeros((batch_size, 2 * length, 1, 1), dtype=tf.int64)

    features["targets"] = initial_output
    sharded_logits, _ = self.model_fn(features, False)
    sharded_samples = self._data_parallelism(tf.argmax, sharded_logits, 4)
    samples = tf.concat(sharded_samples, 0)

    # More steps.
    how_many_more_steps = 5
    for _ in xrange(how_many_more_steps):
      with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        features["targets"] = samples
        sharded_logits, _ = self.model_fn(features, False)
        sharded_samples = self._data_parallelism(tf.argmax, sharded_logits, 4)
        samples = tf.concat(sharded_samples, 0)

    if inputs_old is not None:  # Restore to not confuse Estimator.
      features["inputs"] = inputs_old
    return samples


@registry.register_hparams
def transformer_adv_small():
  """Set of hyperparameters."""
  hparams = transformer.transformer_small()
  hparams.batch_size = 2048
  hparams.learning_rate_warmup_steps = 4000
  hparams.num_hidden_layers = 3
  hparams.hidden_size = 384
  hparams.filter_size = 2048
  hparams.label_smoothing = 0.0
  hparams.weight_decay = 0.1
  hparams.symbol_modality_skip_top = True
  hparams.target_modality = "symbol:ctc"
  hparams.add_hparam("num_compress_steps", 2)
  hparams.add_hparam("extra_steps", 0)
  hparams.add_hparam("noise_val", 0.3)
  hparams.add_hparam("delta_max", 2.0)
  return hparams


@registry.register_hparams
def transformer_adv_base():
  """Set of hyperparameters."""
  hparams = transformer_adv_small()
  hparams.batch_size = 1024
  hparams.hidden_size = 512
  hparams.filter_size = 4096
  hparams.num_hidden_layers = 6
  return hparams
