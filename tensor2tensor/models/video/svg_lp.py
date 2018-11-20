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

"""Model architecture for video prediction model.

   based on following paper:
   "Stochastic Video Generation with a Learned Prior"
   https://arxiv.org/pdf/1802.07687.pdf
   by Emily Denton and Rob Fergus.

   This code is a translation of the original code from PyTorch:
   https://github.com/edenton/svg
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_video
from tensor2tensor.models.video import emily
from tensor2tensor.models.video import sv2p_params
from tensor2tensor.utils import registry

import tensorflow as tf

tfl = tf.layers
tfcl = tf.contrib.layers


@registry.register_model
class NextFrameSVGLP(emily.NextFrameEmily):
  """Stochastic Variational Video Prediction With Learned Prior."""

  def rnn_model(self, hidden_size, nlayers, rnn_type, name):
    """Stacked RNN cell constructor.

    Args:
      hidden_size: number of lstm units
      nlayers: number of lstm layers
      rnn_type: type of RNN cell to use
      name: RNN name
    Returns:
      stacked_rnn: stacked RNN cell
    """
    layers_units = [hidden_size] * nlayers
    if rnn_type == "lstm":
      rnn_cell = tf.contrib.rnn.LSTMCell
    elif rnn_type == "gru":
      rnn_cell = tf.contrib.rnn.GRUCell
    else:
      rnn_cell = tf.contrib.rnn.RNNCell
    cells = [rnn_cell(units, name=name) for units in layers_units]
    stacked_rnn = tf.contrib.rnn.MultiRNNCell(cells)
    return stacked_rnn

  def deterministic_rnn(self, cell, inputs, states, output_size, scope):
    """Deterministic RNN step function.

    Args:
      cell: RNN cell to forward through
      inputs: input to RNN cell
      states: previous RNN state
      output_size: size of the output
      scope: scope of the current RNN forward computation parameters
    Returns:
      outputs: deterministic RNN output vector
      states: updated RNN states
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      embedded = tfl.dense(
          inputs, cell.output_size, activation=tf.nn.relu, name="embed")
      hidden, states = cell(embedded, states)
      outputs = tfl.dense(
          hidden, output_size, activation=tf.nn.relu, name="output")

    return outputs, states

  def gaussian_rnn(self, cell, inputs, states, output_size, scope):
    """Deterministic RNN step function.

    Args:
      cell: RNN cell to forward through
      inputs: input to RNN cell
      states: previous RNN state
      output_size: size of the output
      scope: scope of the current RNN forward computation parameters
    Returns:
      mu: mean of the predicted gaussian
      logvar: log(var) of the predicted gaussian
      states: updated RNN states
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      embedded = tfl.dense(
          inputs, cell.output_size, activation=tf.nn.relu, name="embed")
      hidden, states = cell(embedded, states)
      mu = tfl.dense(
          hidden, output_size, activation=None, name="mu")
      logvar = tfl.dense(
          hidden, output_size, activation=None, name="logvar")

    return mu, logvar, states

  def sample(self, mu, logvar):
    eps = tf.random_normal([self.hparams.batch_size, self.hparams.z_dim], 0, 1)
    sigma = tf.exp(tf.multiply(0.5, logvar))
    z = tf.add(mu, tf.multiply(sigma, eps))

    return z

  def construct_model(self, images, actions, rewards):
    """Builds the stochastic model.

    The model first encodes all the images (x_t) in the sequence
    using the encoder. Let"s call the output e_t. Then it predicts the
    latent state of the next frame using a recurrent posterior network
    z ~ q(z|e_{0:t}) = N(mu(e_{0:t}), sigma(e_{0:t})).
    Another recurrent network predicts the embedding of the next frame
    using the approximated posterior e_{t+1} = p(e_{t+1}|e_{0:t}, z)
    Finally, the decoder decodes e_{t+1} into x_{t+1}.
    Skip connections from encoder to decoder help with reconstruction.

    Args:
      images: tensor of ground truth image sequences
      actions: NOT used list of action tensors
      rewards: NOT used list of reward tensors

    Returns:
      gen_images: generated images
      fakr_rewards: input rewards as reward prediction!
      pred_mu: predited means of posterior
      pred_logvar: predicted log(var) of posterior
    """
    # model does not support action conditioned and reward prediction
    fake_reward_prediction = rewards
    del actions, rewards

    mode = self.hparams.mode
    z_dim = self.hparams.z_dim
    g_dim = self.hparams.g_dim
    rnn_size = self.hparams.rnn_size
    rnn_type = self.hparams.rnn_type
    prior_rnn_layers = self.hparams.prior_rnn_layers
    posterior_rnn_layers = self.hparams.posterior_rnn_layers
    predictor_rnn_layers = self.hparams.predictor_rnn_layers
    context_frames = self.hparams.video_num_input_frames
    has_batchnorm = self.hparams.has_batchnorm

    # Create RNN cells
    predictor_cell = self.rnn_model(
        rnn_size, predictor_rnn_layers, rnn_type, "frame_predictor")
    prior_cell = self.rnn_model(
        rnn_size, prior_rnn_layers, rnn_type, "prior")
    posterior_cell = self.rnn_model(
        rnn_size, posterior_rnn_layers, rnn_type, "posterior")

    seq_len, batch_size, _, _, color_channels = common_layers.shape_list(images)

    # RNN initialize states.
    prior_states = prior_cell.zero_state(batch_size, tf.float32)
    predictor_states = predictor_cell.zero_state(batch_size, tf.float32)
    posterior_states = posterior_cell.zero_state(batch_size, tf.float32)

    tf.logging.info(">>>> Encoding")
    # Encoding:
    enc_images, enc_skips = [], []
    images = tf.unstack(images, axis=0)
    for i, image in enumerate(images):
      with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        enc, skips = self.encoder(image, g_dim, has_batchnorm=has_batchnorm)
        enc = tfcl.flatten(enc)
        enc_images.append(enc)
        enc_skips.append(skips)

    tf.logging.info(">>>> Prediction")
    # Prediction
    pred_mu = []
    pred_logvar = []
    pred_mu_p = []
    pred_logvar_p = []
    gen_images = []
    for i in range(1, seq_len):
      with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        # current encoding
        if (mode == tf.estimator.ModeKeys.TRAIN or
            len(gen_images) < context_frames):
          h_current = enc_images[i-1]
        else:
          h_current, _ = self.encoder(gen_images[-1], g_dim)
          h_current = tfcl.flatten(h_current)

        # target encoding
        h_target = enc_images[i]

      with tf.variable_scope("prediction", reuse=tf.AUTO_REUSE):
        # Prior parameters
        mu_p, logvar_p, prior_states = self.gaussian_rnn(
            prior_cell, h_current, prior_states, z_dim, "prior")

        # Only use Posterior if it's training time
        if mode == tf.estimator.ModeKeys.TRAIN:
          mu, logvar, posterior_states = self.gaussian_rnn(
              posterior_cell, h_target, posterior_states, z_dim, "posterior")
          z = self.sample(mu, logvar)
        else:
          mu = tf.zeros_like(mu_p)
          logvar = tf.zeros_like(logvar_p)
          z = self.sample(mu_p, logvar_p)

        # Predict output images
        h_pred, predictor_states = self.deterministic_rnn(
            predictor_cell, tf.concat([h_current, z], axis=1),
            predictor_states, g_dim, "predictor")

        pred_mu.append(tf.identity(mu, "mu"))
        pred_logvar.append(tf.identity(logvar, "logvar"))
        pred_mu_p.append(tf.identity(mu_p, "mu_p"))
        pred_logvar_p.append(tf.identity(logvar_p, "log_var_p"))

      with tf.variable_scope("decoding", reuse=tf.AUTO_REUSE):
        skip_index = min(context_frames-1, i-1)
        h_pred = tf.reshape(h_pred, [batch_size, 1, 1, g_dim])
        if self.hparams.has_skips:
          x_pred = self.decoder(
              h_pred, color_channels,
              skips=enc_skips[skip_index], has_batchnorm=has_batchnorm)
        else:
          x_pred = self.decoder(
              h_pred, color_channels, has_batchnorm=has_batchnorm)
        gen_images.append(x_pred)

    tf.logging.info(">>>> Done")
    gen_images = tf.stack(gen_images, axis=0)
    return (gen_images, fake_reward_prediction,
            pred_mu, pred_logvar, pred_mu_p, pred_logvar_p)

  def get_extra_loss(self,
                     latent_means, latent_logvars,
                     latent_means_p, latent_logvars_p):
    """Losses in addition to the default modality losses."""
    return self.get_kl_loss(
        latent_means, latent_logvars, latent_means_p, latent_logvars_p)

  def body(self, features):
    hparams = self.hparams
    batch_size = common_layers.shape_list(features["inputs"])[0]

    # Swap time and batch axes.
    input_frames = common_video.swap_time_and_batch_axes(features["inputs"])
    target_frames = common_video.swap_time_and_batch_axes(features["targets"])

    # Get actions if exist otherwise use zeros
    input_actions = self.get_input_if_exists(
        features, "input_action", batch_size, hparams.video_num_input_frames)
    target_actions = self.get_input_if_exists(
        features, "target_action", batch_size, hparams.video_num_target_frames)

    # Get rewards if exist otherwise use zeros
    input_rewards = self.get_input_if_exists(
        features, "input_reward", batch_size, hparams.video_num_input_frames)
    target_rewards = self.get_input_if_exists(
        features, "target_reward", batch_size, hparams.video_num_target_frames)

    all_actions = tf.concat([input_actions, target_actions], axis=0)
    all_rewards = tf.concat([input_rewards, target_rewards], axis=0)
    all_frames = tf.concat([input_frames, target_frames], axis=0)

    # Each image is being used twice, in latent tower and main tower.
    # This is to make sure we are using the *same* image for both, ...
    # ... given how TF queues work.
    # NOT sure if this is required at all. Doesn"t hurt though! :)
    all_frames = tf.identity(all_frames)

    retvals = self.construct_model(
        images=all_frames, actions=all_actions, rewards=all_rewards)

    # retrieve tensors returned by the model contructor
    gen_images = retvals[0]
    gen_rewards = retvals[1]
    latent_means = retvals[2]
    latent_logvars = retvals[3]
    latent_means_p = retvals[4]
    latent_logvars_p = retvals[5]

    extra_loss = self.get_extra_loss(
        latent_means=latent_means,
        latent_logvars=latent_logvars,
        latent_means_p=latent_means_p,
        latent_logvars_p=latent_logvars_p)

    # Visualize predictions in Tensorboard
    if self.is_training:
      self.visualize_predictions(all_frames[1:], gen_images)

    # Ignore the predictions from the input frames.
    # This is NOT the same as original paper/implementation.
    predictions = gen_images[hparams.video_num_input_frames-1:]
    reward_pred = gen_rewards[hparams.video_num_input_frames-1:]
    reward_pred = tf.squeeze(reward_pred, axis=2)  # Remove extra dimension.

    # Swap back time and batch axes.
    predictions = common_video.swap_time_and_batch_axes(predictions)
    reward_pred = common_video.swap_time_and_batch_axes(reward_pred)

    if self.is_training and hparams.internal_loss:
      # add the loss for input frames as well.
      extra_gts = all_frames[1:hparams.video_num_input_frames]
      extra_gts = common_video.swap_time_and_batch_axes(extra_gts)
      extra_pds = gen_images[:hparams.video_num_input_frames-1]
      extra_pds = common_video.swap_time_and_batch_axes(extra_pds)
      extra_raw_gts = features["inputs_raw"][:, 1:]
      recon_loss = self.get_extra_internal_loss(
          extra_raw_gts, extra_gts, extra_pds)
      extra_loss += recon_loss

    return_targets = predictions
    if hparams.reward_prediction:
      return_targets = {"targets": predictions, "target_reward": reward_pred}

    return return_targets, extra_loss


@registry.register_hparams
def next_frame_svglp():
  """SVG with learned prior model hparams."""
  hparams = sv2p_params.next_frame_sv2p()
  hparams.video_num_input_frames = 2
  hparams.video_num_target_frames = 10
  hparams.learning_rate_constant = 1e-4
  seq_length = hparams.video_num_input_frames + hparams.video_num_target_frames
  # The latent_loss_multiplier is divided by the number of frames because
  # the image sequence loss in t2t is averaged instead of added through
  # time as they do in the SVG-LP paper
  hparams.latent_loss_multiplier = 1e-4 / seq_length
  hparams.reward_prediction = False
  hparams.num_iterations_1st_stage = -1
  hparams.num_iterations_2nd_stage = -1
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.999
  hparams.optimizer_adam_epsilon = 1e-08
  hparams.anneal_end = -1
  hparams.clip_grad_norm = 5.0
  hparams.add_hparam("learned_prior", True)
  hparams.add_hparam("z_dim", 64)
  hparams.add_hparam("g_dim", 128)
  hparams.add_hparam("rnn_size", 256)
  hparams.add_hparam("rnn_type", "lstm")
  hparams.add_hparam("prior_rnn_layers", 1)
  hparams.add_hparam("posterior_rnn_layers", 1)
  hparams.add_hparam("predictor_rnn_layers", 2)
  hparams.add_hparam("has_skips", True)
  hparams.add_hparam("has_batchnorm", True)
  return hparams

