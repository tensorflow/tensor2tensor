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

"""Transformer VAE with Flow Priors for Non-Autoregressive MT."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import inspect
import math
import six

from tensor2tensor.data_generators import multi_problem
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
from tensor2tensor.layers import transformer_glow_layers as glow
from tensor2tensor.layers import transformer_glow_layers_ops as gops
from tensor2tensor.models import transformer
from tensor2tensor.research.models import transformer_vae_flow_prior_ops as ops
from tensor2tensor.utils import contrib
from tensor2tensor.utils import optimize
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
import tensorflow.compat.v1 as tf


@registry.register_model
class TransformerVaeFlowPrior(t2t_model.T2TModel):
  """Transformer VAE using flow priors."""

  def __init__(self, *args, **kwargs):
    super(TransformerVaeFlowPrior, self).__init__(*args, **kwargs)
    hparams = self._hparams
    if hparams.prior_type in ["affine", "additive", "rq"]:
      self._fparams = contrib.training.HParams(**hparams.values())
      for key, value in self._fparams.values().items():
        if key.startswith("flow_"):
          setattr(self._fparams, key[5:], value)

  @property
  def is_training(self):
    return self.hparams.mode == tf.estimator.ModeKeys.TRAIN

  @property
  def is_evaluating(self):
    return self._hparams.mode == tf.estimator.ModeKeys.EVAL

  @property
  def is_predicting(self):
    return self._hparams.mode == tf.estimator.ModeKeys.PREDICT

  def loss_iw(self, logits, features):
    if isinstance(logits, dict):
      losses = {}
      for k, v in six.iteritems(logits):
        losses[k] = self._loss_single_iw(
            v,
            k,
            features[k],
            weights=features.get(k + "_mask"))

        n, d = losses[k]
        if common_layers.should_generate_summaries():
          tf.summary.scalar(k + "_loss", n / d)
          tf.summary.scalar(k + "_loss_num", n)
          tf.summary.scalar(k + "_loss_den", d)
          if getattr(self.hparams, "visualize_logits_histogram", False):
            hist = tf.summary.histogram
            hist(k + "_predict", tf.argmax(tf.squeeze(v), axis=-1))
            hist(k + "_targets", features[k])

      return tf.add_n([n / d for n, d in losses.values()])
    else:
      return self._loss_single_iw(
          logits,
          "targets",
          features["targets"],
          weights=features.get("targets_mask"))

  def _loss_single_iw(self, logits, feature_name, feature, weights=None):
    # The current bfloat16 version still uses float32 for most parts of backward
    # propagation to keep model quality, so cast back before computing the loss
    # value.
    no_problem_err_str = (
        "The default implementation of %s requires that the "
        "model be used with a Problem. If using a Problem, augment the "
        "hparams object with trainer_lib.add_problem_hparams. If not, "
        "override %s.")
    no_problem_err = (
        lambda method_name: no_problem_err_str % (method_name, method_name))
    if not self._problem_hparams:
      t2t_model.log_warn(no_problem_err("loss"))
      return (tf.constant(0., dtype=tf.float32),
              tf.constant(1., dtype=tf.float32))

    # Calculate loss contribution.
    modality = self._problem_hparams.modality[feature_name]
    vocab_size = self._problem_hparams.vocab_size[feature_name]
    if vocab_size is not None and hasattr(self._hparams, "vocab_divisor"):
      vocab_size += (-vocab_size) % self._hparams.vocab_divisor
    # loss = self._hparams.loss.get(feature_name, modalities.get_loss(modality))
    loss = ops.generic_loss
    targets_weights_fn = self._hparams.weights_fn.get(
        "targets", modalities.get_weights_fn(modality))
    if weights is None:
      loss_num, loss_den = loss(logits, feature, self._hparams, vocab_size,
                                weights_fn=targets_weights_fn)
    else:

      def weights_fn(labels):
        """Per-token weights for loss."""
        # Use target_weights_fn() given by modality as well as explicitly given
        # weights.
        modality_weights = targets_weights_fn(labels)

        # Broadcast 'weights' along minor dimensions (TF's default is major).
        explicit_weights = weights
        if len(explicit_weights.shape) < len(modality_weights.shape):
          explicit_weights = common_layers.expand_squeeze_to_nd(
              weights, modality_weights.shape.ndims)

        return explicit_weights * modality_weights

      # Ensure that target.modality_loss() supports "weights_fn" keyword
      # argument. If it doesn't and "weights" is specified, raise an exception.
      argument_names = inspect.getargspec(loss).args
      if "weights_fn" not in argument_names:
        raise ValueError(
            "Explicit 'weights' given but default loss for modality doesn't "
            "support 'weights_fn' keyword argument: %s.loss(%s)." %
            (modality, ", ".join(argument_names)))

      loss_num, loss_den = loss(
          logits, feature, self._hparams, vocab_size, weights_fn=weights_fn)

    loss_num *= self._problem_hparams.loss_multiplier

    if hasattr(self.hparams, "problem") and hasattr(
        self.hparams.problem, "task_list"):
      if weights is not None:
        raise NotImplementedError("weights not yet implemented in "
                                  "multitask setting.")
      loss_num, loss_den, summaries = multi_problem.aggregate_task_losses(
          self.hparams,
          self._problem_hparams,
          logits,
          feature_name,
          feature
      )

      for key, val in summaries:
        tf.summary.scalar(key, val)

    return loss_num, loss_den

  def internal(self, features, real_features):
    """Main procedure for both training and inference."""
    inputs = common_layers.flatten4d3d(features["inputs"])
    targets = common_layers.flatten4d3d(features["targets"])
    target_space = features["target_space_id"]
    hparams = self._hparams
    inputs_mask = ops.embedding_to_non_padding(inputs)
    inputs_length = tf.reduce_sum(inputs_mask, axis=-1)

    encoder_output, encoder_decoder_attention_bias = (
        ops.encoder("encoder", hparams, inputs, target_space))
    kwargs = {"encoder_output": encoder_output,
              "encoder_decoder_attention_bias": encoder_decoder_attention_bias}
    losses, monitor = {}, {}
    log_abs_det = tf.constant(0.0)

    if not self.is_predicting:
      # Training
      targets_mask = ops.embedding_to_non_padding(targets)
      targets_length = tf.reduce_sum(targets_mask, axis=-1)
      length_diff = targets_length - inputs_length
      decoder_self_attention_bias = (
          common_attention.attention_bias_ignore_padding(1.0 - targets_mask))
      z_q, log_q_z, q_dist = self.sample_q(
          targets, targets_mask, decoder_self_attention_bias, n_samples=1,
          temp=1.0, **kwargs)

      body_output = ops.decoder(
          "decoder", z_q, hparams, decoder_self_attention_bias, **kwargs)
      logits = self.top(body_output, real_features)
      numerator, denominator = self.loss(logits, real_features)

      if not (self.is_evaluating and (
          hparams.compute_kl_refinement or hparams.compute_iw_marginal)):
        targets_length_pred, lenpred_loss = ops.predict_target_lengths(
            encoder_output, inputs_mask, hparams, length_diff)
        log_p_z_base, log_abs_det = self.compute_prior_log_prob(
            z_q, targets_mask, decoder_self_attention_bias,
            check_invertibility=False, **kwargs)
        losses, monitor = ops.save_log_loss(
            hparams, targets_mask, numerator, denominator, log_q_z, log_abs_det,
            log_p_z_base, z_q, lenpred_loss, targets_length_pred,
            targets_length)

      if self.is_evaluating:
        if hparams.compute_kl_refinement:
          z_p, _ = self.sample_p(
              targets_length, temp=self._decode_hparams.temp,
              check_invertibility=False, targets_mask=targets_mask, **kwargs)
          z_dq = self.delta_posterior(
              z_p, targets_mask, decoder_self_attention_bias,
              self._decode_hparams.n_gibbs_steps, **kwargs)
          log_q_z_ = q_dist.log_prob(z_dq)
          log_q_z_ = gops.reduce_mean_over_bl_sum_over_c(log_q_z_, targets_mask)
          losses = {"training": log_q_z_}

        if hparams.compute_iw_marginal:
        # if True:
          log_p_y_x = self.compute_iw_marginal(
              targets, targets_mask, decoder_self_attention_bias,
              real_features, self._decode_hparams.n_samples, **kwargs)
              # real_features, 1, **kwargs)
          losses = {"training": log_p_y_x}

      return logits, losses, monitor, targets_mask

    else:
      # Inference
      targets_length, _ = ops.predict_target_lengths(
          encoder_output, inputs_mask, hparams)
      targets_mask = ops.sequence_mask(targets_length, hparams)
      decoder_self_attention_bias = (
          common_attention.attention_bias_ignore_padding(1.0 - targets_mask))
      z_p, _ = self.sample_p(
          targets_length, temp=self._decode_hparams.temp,
          check_invertibility=False, **kwargs)
      z_q = self.delta_posterior(
          z_p, targets_mask, decoder_self_attention_bias,
          self._decode_hparams.n_gibbs_steps, **kwargs)
          # 0, **kwargs)

      body_output = ops.decoder(
          "decoder", z_q, hparams, decoder_self_attention_bias, **kwargs)
      return body_output, losses, monitor, targets_mask

  def sample_q(
      self, targets, targets_mask, decoder_self_attention_bias, n_samples,
      temp, **kwargs):
    hparams = self._hparams
    batch_size, targets_max_length = common_layers.shape_list(targets_mask)[:2]
    q_params = ops.posterior("posterior", hparams, targets, targets_mask,
                             decoder_self_attention_bias, **kwargs)
    q_dist = gops.diagonal_normal(q_params, "posterior")
    loc, scale = q_dist.loc, q_dist.scale
    z_shape = [batch_size, targets_max_length, hparams.latent_size]
    iw_z_shape = [n_samples*batch_size, targets_max_length, hparams.latent_size]
    if n_samples == 1:
      noise = tf.random_normal(z_shape, stddev=temp)
      z_q = loc + scale * noise
      log_q_z = q_dist.log_prob(z_q)  # [B, L, C]
    else:
      noise = tf.random_normal([n_samples] + z_shape, stddev=temp)
      z_q = loc[tf.newaxis, ...] + scale[tf.newaxis, ...] * noise
      log_q_z = q_dist.log_prob(z_q)  # [K, B, L, C]
      z_q = tf.reshape(z_q, iw_z_shape)
      log_q_z = tf.reshape(log_q_z, iw_z_shape)
    return z_q, log_q_z, q_dist

  def compute_iw_marginal(
      self, targets, targets_mask, decoder_self_attention_bias, features,
      n_samples, reduce_mean=True, **kwargs):
    hparams = self._hparams
    z_q, log_q_z, _ = self.sample_q(
        targets, targets_mask, decoder_self_attention_bias,
        n_samples=n_samples, temp=1.0, **kwargs)  # [K*B, L, C]
    iw_kwargs = {key: ops.prepare_for_iw(value, n_samples) for (
        key, value) in kwargs.items()}
    iw_targets_mask = ops.prepare_for_iw(targets_mask, n_samples)
    iw_decoder_self_attention_bias = (
        common_attention.attention_bias_ignore_padding(1.0 - iw_targets_mask))
    iw_features = copy.copy(features)
    iw_features["targets"] = ops.prepare_for_iw(
        features["targets"], n_samples)

    log_p_z_base, log_abs_det = self.compute_prior_log_prob(
        z_q, iw_targets_mask, iw_decoder_self_attention_bias,
        check_invertibility=False, **iw_kwargs)
    log_p_z = log_p_z_base + log_abs_det

    body_output = ops.decoder(
        "decoder", z_q, hparams, iw_decoder_self_attention_bias, **iw_kwargs)
    logits = self.top(body_output, iw_features)
    numerator, denominator = self.loss_iw(logits, iw_features)
    numerator = tf.reduce_sum(numerator[..., 0, 0], 1)  # [K*B]
    denominator = tf.reduce_sum(denominator[..., 0, 0], 1)  # [K*B]
    log_p_x = -1 * numerator / denominator
    log_q_z = gops.reduce_mean_over_l_sum_over_c(log_q_z, iw_targets_mask)
    log_p_z = log_p_z / tf.reduce_sum(iw_targets_mask, 1)

    log_p_x, log_q_z, log_p_z = [ops.unprepare_for_iw(ii, n_samples) for ii in [
        log_p_x, log_q_z, log_p_z]]

    log_w_n = log_p_z - log_q_z
    log_w_n = tf.nn.log_softmax(log_w_n, axis=0)  # [K, B]

    iw_marginal = log_p_x + log_w_n
    iw_marginal = tf.reduce_logsumexp(iw_marginal, 0)  # [B]

    if reduce_mean:
      iw_marginal = tf.cast(tf.reduce_mean(iw_marginal, 0), tf.float32)  # [1]
    else:
      iw_marginal = tf.cast(iw_marginal, tf.float32)  # [1]
    return iw_marginal

  def argmax_decode(self, z, decoder_self_attention_bias, **kwargs):
    hparams = self._hparams
    body_output = ops.decoder(
        "decoder", z, hparams, decoder_self_attention_bias, **kwargs)
    logits = self.top(body_output, {"targets": None})
    targets = tf.argmax(logits, axis=-1)
    targets_emb = self.bottom({"targets": targets})["targets"][..., 0, :]
    return targets, targets_emb

  def delta_posterior(
      self, z, targets_mask, decoder_self_attention_bias, n_gibbs_steps,
      **kwargs):
    hparams = self._hparams
    for _ in range(n_gibbs_steps):
      _, targets_emb = self.argmax_decode(
          z, decoder_self_attention_bias, **kwargs)
      q_params = ops.posterior(
          "posterior", hparams, targets_emb, targets_mask,
          decoder_self_attention_bias, **kwargs)
      q_dist = gops.diagonal_normal(q_params, "posterior")
      z = q_dist.loc  # [B, L, C]
    return z

  def compute_prior_log_prob(
      self, z_q, targets_mask, decoder_self_attention_bias,
      check_invertibility=False, **kwargs):
    hparams = self._hparams
    batch_size, targets_max_length = (
        common_layers.shape_list(targets_mask)[:2])
    prior_shape = [batch_size, targets_max_length, hparams.latent_size]
    log_abs_det = tf.zeros([batch_size])

    if hparams.prior_type == "standard_normal":
      log_p_z_base = gops.standard_normal_density(z_q, targets_mask)
    elif hparams.prior_type == "diagonal_normal":
      diag_prior_params = ops.cond_prior(
          "diag_prior", hparams, tf.zeros(prior_shape), targets_mask,
          hparams.latent_size*2, decoder_self_attention_bias, **kwargs)
      p_dist = gops.diagonal_normal(diag_prior_params, "diag_prior")
      log_p_z_base = p_dist.log_prob(z_q)  # [B, L, C]
      log_p_z_base = gops.reduce_sum_over_lc(log_p_z_base, targets_mask)  # [B]
    elif hparams.prior_type in ["affine", "additive", "rq"]:
      if self.is_evaluating:
        disable_dropout = True
        init = False
      elif self.is_training:
        disable_dropout = False
        init = tf.equal(hparams.kl_startup_steps,
                        tf.cast(tf.train.get_global_step(), tf.int32))
      else:
        raise ValueError("compute_prior shouldn't be used in decoding.")

      z_inv, log_abs_det, log_p_z_base, zs = glow.glow(
          "glow", z_q, targets_mask, decoder_self_attention_bias,
          inverse=False, init=init, hparams=self._fparams,
          disable_dropout=disable_dropout, **kwargs)
      if self.is_evaluating and check_invertibility:
        z_inv_inv, _, _, _ = glow.glow(
            "glow", z_inv, targets_mask, decoder_self_attention_bias,
            inverse=True, split_zs=zs, init=False, hparams=self._fparams,
            disable_dropout=True, **kwargs)
        z_diff = z_q - z_inv_inv
        tf.summary.scalar("flow_recon_forward", tf.reduce_max(tf.abs(z_diff)))
    return log_p_z_base, log_abs_det

  def sample_p(
      self, targets_length, temp, check_invertibility=False, targets_mask=None,
      **kwargs):
    hparams = self._hparams
    if targets_mask is None:
      targets_mask = ops.sequence_mask(targets_length, hparams)
    decoder_self_attention_bias = (
        common_attention.attention_bias_ignore_padding(1.0 - targets_mask))
    batch_size, targets_max_length = (
        common_layers.shape_list(targets_mask)[:2])
    prior_shape = [batch_size, targets_max_length, hparams.latent_size]
    noise = tf.random.normal(prior_shape, stddev=temp)
    p_dist = None

    if hparams.prior_type == "standard_normal":
      z_p = noise
    elif hparams.prior_type == "diagonal_normal":
      diag_prior_params = ops.cond_prior(
          "diag_prior", hparams, tf.zeros(prior_shape), targets_mask,
          hparams.latent_size*2, decoder_self_attention_bias, **kwargs)
      p_dist = gops.diagonal_normal(diag_prior_params, "diag_prior")
      z_p = p_dist.loc + p_dist.scale * noise
    elif hparams.prior_type in ["affine", "additive", "rq"]:
      n_levels = len(hparams.depths.split("/"))
      divi = max(1, hparams.factor**(n_levels-1))
      flow_prior_shape = [
          batch_size, targets_max_length//divi, hparams.latent_size]
      noise = tf.random_normal(flow_prior_shape, stddev=temp)
      z_p, _, _, _ = glow.glow(
          "glow", noise, targets_mask, decoder_self_attention_bias,
          inverse=True, init=False, hparams=self._fparams,
          disable_dropout=True, temp=temp, **kwargs)
      if self.is_evaluating and check_invertibility:
        noise_inv, _, _, _ = glow.glow(
            "glow", z_p, targets_mask, decoder_self_attention_bias,
            inverse=False, init=False, hparams=self._fparams,
            disable_dropout=True, **kwargs)
        z_diff = noise - noise_inv
        tf.summary.scalar("flow_recon_inverse", tf.reduce_max(tf.abs(z_diff)))
    return z_p, p_dist

  def optimize(self, loss, num_async_replicas=1, use_tpu=False, variables=None):
    """Return a training op minimizing loss."""
    lr = ops.learning_rate_schedule(self.hparams)
    if num_async_replicas > 1:
      t2t_model.log_info("Dividing learning rate by num_async_replicas: %d",
                         num_async_replicas)
    lr /= math.sqrt(float(num_async_replicas))
    train_op = optimize.optimize(
        loss, lr, self.hparams, use_tpu=use_tpu, variables=variables)
    return train_op

  def body(self, features, real_features):
    return self.internal(features, real_features)

  def infer(self,
            features,
            *args,
            **kwargs):
    """Produce predictions from the model."""
    del args, kwargs
    inputs_old = None
    if "inputs" in features and len(features["inputs"].shape) < 4:
      inputs_old = features["inputs"]
      features["inputs"] = tf.expand_dims(features["inputs"], 2)
    features["targets"] = tf.identity(features["inputs"])

    # logits, _ = self(features)
    t2t_model.set_custom_getter_compose(self._custom_getter)
    tf.get_variable_scope().set_initializer(
        optimize.get_variable_initializer(self.hparams))
    with self._eager_var_store.as_default():
      self._fill_problem_hparams_features(features)
      # intentionally disable sharding during inference (in multi GPU)
      with tf.variable_scope(self.name):
        logits, _, _, targets_mask = self.model_fn(features)

    samples = tf.argmax(logits, axis=-1)
    samples = tf.where(
        tf.cast(targets_mask[..., tf.newaxis, tf.newaxis], tf.bool),
        samples, tf.ones_like(samples))
    if inputs_old is not None:  # Restore to not confuse Estimator.
      features["inputs"] = inputs_old
    return samples

  def model_fn(self, features):
    with tf.variable_scope(
        tf.get_variable_scope(), use_resource=True, reuse=tf.AUTO_REUSE):
      transformed_features = self.bottom(features)

      if self.hparams.activation_dtype == "bfloat16":
        for k, v in sorted(six.iteritems(transformed_features)):
          if v.dtype == tf.float32:
            transformed_features[k] = tf.cast(v, tf.bfloat16)

      t2t_model.log_info("Building model body")
      output, losses, monitor, targets_mask = self.body(
          transformed_features, features)
      output, losses = self._normalize_body_output((output, losses))

      if "training" in losses:
        t2t_model.log_info(
            "Skipping T2TModel top and loss because training loss "
            "returned from body")
        logits = output
      else:
        logits = self.top(output, features)
        losses["training"] = 0.0
        if (self._hparams.mode != tf.estimator.ModeKeys.PREDICT and
            self._hparams.mode != "attack"):
          losses["training"] = self.loss(logits, features)

    return logits, losses, monitor, targets_mask

  def model_fn_sharded(self, sharded_features):
    """Estimator model_fn sharded along batch dimension.

    Args:
      sharded_features: {str: [Tensor]}. Features sharded along batch dimension.
        Each list is the same length (== number of shards).

    Returns:
      sharded_logits: [Tensor]. Logits for each shard of examples.
      losses: {str: 0-D Tensor}. Loss averaged across shards.
    """
    dp = self._data_parallelism

    # [{str: Tensor}]. Transpose of 'sharded_features'.
    datashard_to_features = self._to_features_per_datashard(sharded_features)
    sharded_logits, sharded_losses, sharded_monitors, _ = (
        dp(self.model_fn, datashard_to_features))
    sharded_logits, sharded_losses = dp(
        self.maybe_scheduled_sampling,
        datashard_to_features, sharded_logits, sharded_losses)
    if isinstance(sharded_logits[0], dict):
      temp_dict = {k: [] for k, _ in six.iteritems(sharded_logits[0])}
      for k, _ in six.iteritems(sharded_logits[0]):
        for l in sharded_logits:
          temp_dict[k].append(l[k])
      sharded_logits = temp_dict
    losses = t2t_model.average_sharded_losses(sharded_losses)
    monitor = {}
    for key in list(sharded_monitors[0].keys()):
      monitor[key] = (
          tf.add_n([m[key] for m in sharded_monitors]) / len(sharded_monitors))
    ops.save_summary(monitor, "monitor")

    return sharded_logits, losses


@registry.register_hparams
def wmt_enro_tpu():
  """HParams for Transformer model on TPU."""
  hparams = transformer.transformer_base()
  hparams = transformer.update_hparams_for_tpu(hparams)
  hparams.batch_size = 512
  return hparams


@registry.register_hparams
def iwslt_baseline_gpu():
  """HParams for Transformer model on TPU."""
  hparams = transformer.transformer_base()
  hparams.hidden_size = 256
  hparams.filter_size = 1024
  hparams.num_hidden_layers = 5
  hparams.num_heads = 2
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1
  hparams.dropout = 0.1
  return hparams


@registry.register_hparams
def iwslt_baseline_single_gpu():
  """HParams for Transformer model on TPU."""
  hparams = iwslt_baseline_gpu()
  hparams.batch_size = 1024
  hparams.learning_rate_schedule = "constant*linear_warmup*rsqrt_decay"
  hparams.learning_rate_constant = 0.1
  hparams.learning_rate_warmup_steps = 16000
  return hparams


@registry.register_hparams
def iwslt_baseline_tpu():
  """HParams for Transformer model on TPU."""
  hparams = transformer.transformer_base()
  transformer.update_hparams_for_tpu(hparams)
  hparams.hidden_size = 256
  hparams.filter_size = 1024
  hparams.num_hidden_layers = 5
  hparams.num_heads = 2
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1
  hparams.dropout = 0.1
  hparams.add_hparam("pos_attn", False)
  return hparams


@registry.register_hparams
def iwslt_base():
  """Set of hyperparameters."""
  # Model architecture flags.
  hparams = transformer.transformer_base()
  hparams.num_hidden_layers = 5
  hparams.hidden_size = 256
  hparams.filter_size = 1024
  hparams.num_heads = 4
  # Other flags.
  hparams.summarize_grads = False
  hparams.summarize_vars = False
  # Optimization-related flags.
  hparams.clip_grad_norm = 1.0
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate_warmup_steps = 8000
  hparams.learning_rate = 0.2
  hparams.learning_rate_schedule = (
      "constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size")
  hparams.learning_rate_constant = 2.0
  hparams.add_hparam("predict_target_length", True)
  hparams.add_hparam("lendiff_bound", 30)
  hparams = update_hparams_for_tpu(hparams)
  hparams.add_hparam("pos_attn", False)
  return hparams


@registry.register_hparams
def iwslt_diag():
  """Set of hyperparameters."""
  hparams = iwslt_base()
  hparams.batch_size = 4096
  # Other flags.
  hparams.force_full_predict = True
  hparams.causal_decoder_self_attention = False
  # VAE-related flags.
  hparams.add_hparam("latent_size", 256)
  hparams.add_hparam("anneal_min_value", 0.0)
  hparams.add_hparam("kl_startup_steps", 5000)
  hparams.add_hparam("kl_anneal_steps", 20000)
  hparams.add_hparam("n_posterior_layers", 3)
  hparams.add_hparam("n_decoder_layers", 3)
  hparams.add_hparam("posterior_2d_dropout", 0.20)
  # diagonal_normal / affine / additive / rq
  hparams.add_hparam("posterior_type", "diagonal_normal")
  # standard_normal / diagonal_normal
  hparams.add_hparam("prior_type", "diagonal_normal")
  hparams.add_hparam("decoder_2d_dropout", 0.00)
  # Optimization-related flags.
  hparams.learning_rate_warmup_steps = 8000
  hparams.learning_rate_constant = 2.0
  hparams.layer_prepostprocess_dropout = 0.2
  hparams.attention_dropout = 0.2
  hparams.relu_dropout = 0.2
  hparams.dropout = 0.2
  # Optimization-related flags.
  hparams.add_hparam("kl_reg", 0.0)
  hparams.add_hparam("n_gibbs_steps", 0)
  hparams.add_hparam("compute_kl_refinement", False)
  hparams.add_hparam("compute_iw_marginal", False)
  hparams.add_hparam("n_samples", 1)
  return hparams


@registry.register_hparams
def wmt_diag_base():
  """Set of hyperparameters."""
  hparams = iwslt_diag()
  hparams.batch_size = 4096
  hparams.num_hidden_layers = 6
  hparams.hidden_size = 512
  hparams.filter_size = 2048
  hparams.num_heads = 8
  # VAE-related flags.
  hparams.latent_size = 512
  hparams.n_posterior_layers = 4
  hparams.n_decoder_layers = 6
  hparams.dropout = 0.1
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1
  return hparams


@registry.register_hparams
def wmt_diag_small():
  """Set of hyperparameters."""
  hparams = wmt_diag_base()
  hparams.n_posterior_layers = 3
  hparams.n_decoder_layers = 3
  hparams.kl_reg = 1e-4
  return hparams


@registry.register_hparams
def wmt_diag_small_trueadam():
  """Set of hyperparameters."""
  hparams = wmt_diag_small()
  hparams.optimizer = "true_adam"
  return hparams


@registry.register_hparams
def wmt_diag_small_trueadam_longer():
  """Set of hyperparameters."""
  hparams = wmt_diag_small_trueadam()
  hparams.learning_rate_constant = 4.0
  hparams.learning_rate_warmup_steps = 20000
  return hparams


@registry.register_hparams
def wmt_diag_small_trueadam_shorter():
  """Set of hyperparameters."""
  hparams = wmt_diag_small_trueadam()
  hparams.learning_rate_constant = 2.0
  hparams.learning_rate_warmup_steps = 4000
  return hparams


@registry.register_hparams
def wmt_diag_base_trueadam_1e4():
  """Set of hyperparameters."""
  hparams = wmt_diag_base()
  hparams.kl_reg = 1e-4
  hparams.optimizer = "true_adam"
  hparams.learning_rate_constant = 2.0
  hparams.learning_rate_warmup_steps = 8000
  return hparams


@registry.register_hparams
def wmt_diag_base_trueadam_longer_1e4():
  """Set of hyperparameters."""
  hparams = wmt_diag_base_trueadam_1e4()
  hparams.learning_rate_constant = 4.0
  hparams.learning_rate_warmup_steps = 20000
  return hparams


@registry.register_hparams
def wmt_diag_base_trueadam_shorter_1e4():
  """Set of hyperparameters."""
  hparams = wmt_diag_base_trueadam_1e4()
  hparams.learning_rate_constant = 2.0
  hparams.learning_rate_warmup_steps = 4000
  return hparams


@registry.register_hparams
def wmt_diag_base_1e4_trueadam():
  """Set of hyperparameters."""
  hparams = wmt_diag_base()
  hparams.kl_reg = 1e-4
  hparams.optimizer = "true_adam"
  return hparams


@registry.register_hparams
def wmt_diag_base_1e4_trueadam_longer():
  """Set of hyperparameters."""
  hparams = wmt_diag_base_1e4_trueadam()
  hparams.learning_rate_constant = 4.0
  hparams.learning_rate_warmup_steps = 20000
  return hparams


@registry.register_hparams
def wmt_diag_base_1e4_trueadam_shorter():
  """Set of hyperparameters."""
  hparams = wmt_diag_base_1e4_trueadam()
  hparams.learning_rate_constant = 2.0
  hparams.learning_rate_warmup_steps = 4000
  return hparams


@registry.register_hparams
def wmt_diag_base_1e4():
  """Set of hyperparameters."""
  hparams = wmt_diag_base()
  hparams.kl_reg = 1e-4
  return hparams


@registry.register_hparams
def wmt_diag_base_longer_1e4():
  """Set of hyperparameters."""
  hparams = wmt_diag_base_1e4()
  hparams.learning_rate_constant = 4.0
  hparams.learning_rate_warmup_steps = 20000
  return hparams


@registry.register_hparams
def wmt_diag_base_shorter_1e4():
  """Set of hyperparameters."""
  hparams = wmt_diag_base_1e4()
  hparams.learning_rate_constant = 2.0
  hparams.learning_rate_warmup_steps = 4000
  return hparams


@registry.register_hparams
def iwslt_diag_1e5():
  """Set of hyperparameters."""
  hparams = iwslt_diag()
  hparams.kl_reg = 1e-5
  return hparams


@registry.register_hparams
def iwslt_diag_1e4():
  """Set of hyperparameters."""
  hparams = iwslt_diag()
  hparams.kl_reg = 1e-4
  return hparams


@registry.register_hparams
def iwslt_affine():
  """Set of hyperparameters."""
  hparams = iwslt_diag()
  hparams.prior_type = "affine"
  hparams.batch_size = 2048
  hparams.latent_size = 256
  # Glow-related flags.
  hparams.add_hparam("depths", "4/8/8")  # infer n_levels from depths
  hparams.add_hparam("step_fn", "glow")  # glow / chunting
  hparams.add_hparam("affine_scale", "glow")  # glow / jason
  hparams.add_hparam("conv_fn", "np")  # np / tf
  hparams.add_hparam("split_plans", "cat/cat/ca")
  hparams.add_hparam("factor", 2)  # squeezing factor
  hparams.add_hparam("n_layers_transform_params", 1)
  hparams.add_hparam("n_1x1_heads", 4)
  hparams.add_hparam("flow_num_heads", 4)
  hparams.add_hparam("flow_hidden_size", 256)
  hparams.add_hparam("flow_filter_size", 512)
  # Control max scale change.
  hparams.add_hparam("scale_width", 0.999)
  # Optimization-related flags.
  # hparams.learning_rate_warmup_steps = 20000
  hparams.add_hparam("flow_layer_prepostprocess_dropout", 0.0)
  hparams.add_hparam("flow_attention_dropout", 0.0)
  hparams.add_hparam("flow_relu_dropout", 0.0)
  # hparams.optimizer_adam_beta1 = 0.9
  # hparams.optimizer_adam_beta2 = 0.999
  # hparams.optimizer_adam_epsilon = 1e-8
  # Precision-related flags.
  hparams.activation_dtype = "float32"
  hparams.weight_dtype = "float32"

  return hparams


@registry.register_hparams
def wmt_affine():
  """Set of hyperparameters."""
  hparams = iwslt_affine()
  hparams.batch_size = 2048  # TODO(jason) : address this later.
  hparams.num_hidden_layers = 6
  hparams.hidden_size = 256
  hparams.filter_size = 1024
  hparams.num_heads = 8
  # VAE-related flags.
  hparams.latent_size = 256
  hparams.n_posterior_layers = 4
  hparams.n_decoder_layers = 4
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1
  # Glow-related flags.
  hparams.flow_num_heads = 8
  hparams.flow_filter_size = 512
  return hparams


@registry.register_hparams
def wmt_affine_base():
  """Set of hyperparameters."""
  hparams = wmt_affine()
  hparams.batch_size = 2048
  hparams.hidden_size = 320
  hparams.latent_size = 320
  hparams.flow_filter_size = 640
  return hparams


@registry.register_hparams
def wmt_affine_base_small():
  """Set of hyperparameters."""
  hparams = wmt_affine_base()
  hparams.depths = "4/4/4"
  hparams.kl_reg = 1e-4
  hparams.learning_rate_constant = 2.0
  hparams.learning_rate_warmup_steps = 8000
  return hparams


@registry.register_hparams
def wmt_affine_base_trueadam_small():
  """Set of hyperparameters."""
  hparams = wmt_affine_base_small()
  hparams.optimizer = "true_adam"
  return hparams


@registry.register_hparams
def wmt_affine_base_trueadam_longer_small():
  """Set of hyperparameters."""
  hparams = wmt_affine_base_trueadam_small()
  hparams.learning_rate_constant = 4.0
  hparams.learning_rate_warmup_steps = 20000
  return hparams


@registry.register_hparams
def wmt_affine_base_trueadam_shorter_small():
  """Set of hyperparameters."""
  hparams = wmt_affine_base_trueadam_small()
  hparams.learning_rate_constant = 2.0
  hparams.learning_rate_warmup_steps = 4000
  return hparams


@registry.register_hparams
def wmt_affine_base_trueadam():
  """Set of hyperparameters."""
  hparams = wmt_affine_base()
  hparams.optimizer = "true_adam"
  # hparams.optimizer_adam_beta1 = 0.9
  # hparams.optimizer_adam_beta2 = 0.999
  # hparams.optimizer_adam_epsilon = 1e-8
  hparams.kl_reg = 1e-4
  hparams.learning_rate_constant = 2.0
  hparams.learning_rate_warmup_steps = 8000
  return hparams


@registry.register_hparams
def wmt_affine_base_trueadam_longer():
  """Set of hyperparameters."""
  hparams = wmt_affine_base_trueadam()
  hparams.learning_rate_constant = 4.0
  hparams.learning_rate_warmup_steps = 20000
  return hparams


@registry.register_hparams
def wmt_affine_base_trueadam_shorter():
  """Set of hyperparameters."""
  hparams = wmt_affine_base_trueadam()
  hparams.learning_rate_constant = 2.0
  hparams.learning_rate_warmup_steps = 4000
  return hparams


@registry.register_hparams
def wmt_affine_base_1e4():
  """Set of hyperparameters."""
  hparams = wmt_affine_base()
  hparams.kl_reg = 1e-4
  hparams.learning_rate_constant = 2.0
  hparams.learning_rate_warmup_steps = 8000
  return hparams


@registry.register_hparams
def wmt_affine_base_longer_1e4():
  """Set of hyperparameters."""
  hparams = wmt_affine_base_1e4()
  hparams.learning_rate_constant = 4.0
  hparams.learning_rate_warmup_steps = 20000
  return hparams


@registry.register_hparams
def wmt_affine_base_shorter_1e4():
  """Set of hyperparameters."""
  hparams = wmt_affine_base_1e4()
  hparams.learning_rate_constant = 2.0
  hparams.learning_rate_warmup_steps = 4000
  return hparams


@registry.register_hparams
def wmt_affine_1e4():
  """Set of hyperparameters."""
  hparams = wmt_affine()
  hparams.kl_reg = 1e-4
  return hparams


@registry.register_hparams
def wmt_affine_large():
  """Set of hyperparameters."""
  hparams = iwslt_affine()
  hparams.batch_size = 2048
  hparams.num_hidden_layers = 6
  hparams.hidden_size = 512
  hparams.filter_size = 1024
  hparams.num_heads = 8
  # VAE-related flags.
  hparams.latent_size = 512
  hparams.n_posterior_layers = 4
  hparams.n_decoder_layers = 4
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1
  # Glow-related flags.
  hparams.flow_num_heads = 8
  hparams.flow_filter_size = 1024
  return hparams


@registry.register_hparams
def wmt_affine_large_1e4():
  """Set of hyperparameters."""
  hparams = wmt_affine_large()
  hparams.kl_reg = 1e-4
  return hparams


@registry.register_hparams
def iwslt_affine_tiny():
  """Set of hyperparameters."""
  hparams = iwslt_affine()
  hparams.depths = "1"
  hparams.split_plans = "c"
  return hparams


@registry.register_hparams
def iwslt_affine_small():
  """Set of hyperparameters."""
  hparams = iwslt_affine()
  hparams.depths = "4/4/4"
  return hparams


@registry.register_hparams
def iwslt_affine_small_1e4_trueadam():
  """Set of hyperparameters."""
  hparams = iwslt_affine_small_1e4()
  hparams.optimizer = "true_adam"
  return hparams


@registry.register_hparams
def iwslt_affine_small_1e4_trueadam_longer():
  """Set of hyperparameters."""
  hparams = iwslt_affine_small_1e4_trueadam()
  hparams.learning_rate_constant = 4.0
  hparams.learning_rate_warmup_steps = 20000
  return hparams


@registry.register_hparams
def iwslt_affine_small_1e4_trueadam_shorter():
  """Set of hyperparameters."""
  hparams = iwslt_affine_small_1e4_trueadam()
  hparams.learning_rate_constant = 2.0
  hparams.learning_rate_warmup_steps = 4000
  return hparams


@registry.register_hparams
def iwslt_affine_small_1e4():
  """Set of hyperparameters."""
  hparams = iwslt_affine_small()
  hparams.kl_reg = 1e-4
  return hparams


@registry.register_hparams
def iwslt_affine_tpu_glow_glow_np_1e4_trueadam():
  """Set of hyperparameters."""
  hparams = iwslt_affine_tpu_glow_glow_np_1e4()
  hparams.optimizer = "true_adam"
  return hparams


@registry.register_hparams
def iwslt_affine_tpu_glow_glow_np_1e4_trueadam_longer():
  """Set of hyperparameters."""
  hparams = iwslt_affine_tpu_glow_glow_np_1e4_trueadam()
  hparams.learning_rate_constant = 4.0
  hparams.learning_rate_warmup_steps = 20000
  return hparams


@registry.register_hparams
def iwslt_affine_tpu_glow_glow_np_1e4_trueadam_shorter():
  """Set of hyperparameters."""
  hparams = iwslt_affine_tpu_glow_glow_np_1e4_trueadam()
  hparams.learning_rate_constant = 2.0
  hparams.learning_rate_warmup_steps = 4000
  return hparams


@registry.register_hparams
def iwslt_affine_tpu_glow_glow_np_1e4():
  """Set of hyperparameters."""
  hparams = iwslt_affine()
  hparams.conv_fn = "np"
  hparams.kl_reg = 1e-4
  return hparams


def update_hparams_for_tpu(hparams):
  """Change hparams to be compatible with TPU training."""

  # Adafactor uses less memory than Adam.
  # switch to Adafactor with its recommended learning rate scheme.
  # hparams.optimizer = "Adafactor"
  # hparams.learning_rate_schedule = "rsqrt_decay"
  # hparams.learning_rate_warmup_steps = 10000

  # Avoid an expensive concat on TPU.
  # >1 shards helps with faster parameter distribution on multi-GPU machines
  hparams.symbol_modality_num_shards = 1

  # Adaptive batch sizes and sequence lengths are not supported on TPU.
  # Instead, every batch has the same sequence length and the same batch size.
  # Longer sequences are dropped and shorter ones are padded.
  #
  # It is therefore suggested to use a problem where examples have been combined
  # to a longer length, e.g. the "_packed" problems.
  #
  # For problems with variable sequence lengths, this parameter controls the
  # maximum sequence length.  Shorter sequences are dropped and longer ones
  # are padded.
  #
  # For problems with fixed sequence lengths - e.g. the "_packed" problems,
  # this hyperparameter is ignored.
  hparams.max_length = 64

  # TPUs have less memory than GPUs, so decrease the batch size if it's too high
  if hparams.batch_size > 2048:
    hparams.batch_size = 2048

  # Using noise broadcast in the dropout layers saves memory during training.
  hparams.attention_dropout_broadcast_dims = "0,1"  # batch, heads
  hparams.relu_dropout_broadcast_dims = "1"  # length
  hparams.layer_prepostprocess_dropout_broadcast_dims = "1"  # length
  return hparams
