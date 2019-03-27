# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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
from tensor2tensor.models.video import sv2p
from tensor2tensor.models.video import sv2p_params
from tensor2tensor.utils import registry

import tensorflow as tf

tfl = tf.layers
tfcl = tf.contrib.layers


@registry.register_model
class NextFrameEmily(sv2p.NextFrameSv2pLegacy):
  """Stochastic Variational Video Prediction Without Learned Prior."""

  def encoder(self, inputs, nout, has_batchnorm=True):
    """VGG based image encoder.

    Args:
      inputs: image tensor with size BSx64x64xC
      nout: number of output channels
      has_batchnorm: variable to use or not use batch normalization
    Returns:
      net: encoded image with size BSxNout
      skips: skip connection after each layer
    """
    vgg_layer = common_video.vgg_layer
    net01 = inputs

    skips = []

    # The original model only supports 64x64. We can support higher resolutions
    # as long as they are square and the side-length is a power of two
    # by inserting more downscaling layers. Corresponding upscaling can be found
    # in the decoder, as well.
    # (This procedure is ad-hoc, i.e., not from the SVP-FP paper)
    _, res_y, res_x, _ = inputs.shape.as_list()
    assert res_x == res_y, "Model only supports square inputs"
    is_power_of_two = lambda x: ((x & (x - 1)) == 0) and x != 0
    assert is_power_of_two(res_x), "Input resolution must be power of 2"
    assert res_x >= 64, "Input resolution must be >= 64"
    ds_idx = 0
    while res_x > 64:
      h = tfcl.repeat(net01, 2, vgg_layer, 64, scope="downscale%d" % ds_idx,
                      is_training=self.is_training, activation=tf.nn.relu,
                      has_batchnorm=has_batchnorm)
      net01 = tfl.max_pooling2d(h, [2, 2], strides=(2, 2),
                                name="downscale%d_pool" % ds_idx)
      skips.append(h)
      ds_idx += 1
      res_x /= 2

    # h1
    net11 = tfcl.repeat(net01, 2, vgg_layer, 64,
                        scope="h1", is_training=self.is_training,
                        activation=tf.nn.relu, has_batchnorm=has_batchnorm)
    net12 = tfl.max_pooling2d(net11, [2, 2], strides=(2, 2), name="h1_pool")
    # h2
    net21 = tfcl.repeat(net12, 2, vgg_layer, 128,
                        scope="h2", is_training=self.is_training,
                        activation=tf.nn.relu, has_batchnorm=has_batchnorm)
    net22 = tfl.max_pooling2d(net21, [2, 2], strides=(2, 2), name="h2_pool")
    # h3
    net31 = tfcl.repeat(net22, 3, vgg_layer, 256,
                        scope="h3", is_training=self.is_training,
                        activation=tf.nn.relu, has_batchnorm=has_batchnorm)
    net32 = tfl.max_pooling2d(net31, [2, 2], strides=(2, 2), name="h3_pool")
    # h4
    net41 = tfcl.repeat(net32, 3, vgg_layer, 512,
                        scope="h4", is_training=self.is_training,
                        activation=tf.nn.relu, has_batchnorm=has_batchnorm)
    net42 = tfl.max_pooling2d(net41, [2, 2], strides=(2, 2), name="h4_pool")
    # h5
    net51 = tfcl.repeat(net42, 1, vgg_layer, nout,
                        kernel_size=4, padding="VALID", activation=tf.nn.relu,
                        scope="h5", is_training=self.is_training,
                        has_batchnorm=has_batchnorm)

    skips += [net11, net21, net31, net41]
    return net51, skips

  def decoder(self, inputs, nout, skips=None, has_batchnorm=True):
    """VGG based image decoder.

    Args:
      inputs: image tensor with size BSxX
      nout: number of output channels
      skips: optional skip connections from encoder
      has_batchnorm: variable to use or not use batch normalization
    Returns:
      net: decoded image with size BSx64x64xNout
      skips: skip connection after each layer
    """
    vgg_layer = common_video.vgg_layer
    net = inputs
    # d1
    net = tfl.conv2d_transpose(net, 512, kernel_size=4, padding="VALID",
                               name="d1_deconv", activation=tf.nn.relu)
    if has_batchnorm:
      net = tfl.batch_normalization(
          net, training=self.is_training, name="d1_bn")
    net = tf.nn.relu(net)
    net = common_layers.upscale(net, 2)
    # d2
    if skips is not None:
      net = tf.concat([net, skips[-1]], axis=3)
    net = tfcl.repeat(net, 2, vgg_layer, 512, scope="d2a",
                      is_training=self.is_training,
                      activation=tf.nn.relu, has_batchnorm=has_batchnorm)
    net = tfcl.repeat(net, 1, vgg_layer, 256, scope="d2b",
                      is_training=self.is_training,
                      activation=tf.nn.relu, has_batchnorm=has_batchnorm)
    net = common_layers.upscale(net, 2)
    # d3
    if skips is not None:
      net = tf.concat([net, skips[-2]], axis=3)
    net = tfcl.repeat(net, 2, vgg_layer, 256, scope="d3a",
                      is_training=self.is_training,
                      activation=tf.nn.relu, has_batchnorm=has_batchnorm)
    net = tfcl.repeat(net, 1, vgg_layer, 128, scope="d3b",
                      is_training=self.is_training,
                      activation=tf.nn.relu, has_batchnorm=has_batchnorm)
    net = common_layers.upscale(net, 2)
    # d4
    if skips is not None:
      net = tf.concat([net, skips[-3]], axis=3)
    net = tfcl.repeat(net, 1, vgg_layer, 128, scope="d4a",
                      is_training=self.is_training,
                      activation=tf.nn.relu, has_batchnorm=has_batchnorm)
    net = tfcl.repeat(net, 1, vgg_layer, 64, scope="d4b",
                      is_training=self.is_training,
                      activation=tf.nn.relu, has_batchnorm=has_batchnorm)
    net = common_layers.upscale(net, 2)
    # d5
    if skips is not None:
      net = tf.concat([net, skips[-4]], axis=3)
    net = tfcl.repeat(net, 1, vgg_layer, 64, scope="d5",
                      is_training=self.is_training,
                      activation=tf.nn.relu, has_batchnorm=has_batchnorm)

    # if there are still skip connections left, we have more upscaling to do
    if skips is not None:
      for i, s in enumerate(skips[-5::-1]):
        net = common_layers.upscale(net, 2)
        net = tf.concat([net, s], axis=3)
        net = tfcl.repeat(net, 1, vgg_layer, 64, scope="upscale%d" % i,
                          is_training=self.is_training,
                          activation=tf.nn.relu, has_batchnorm=has_batchnorm)

    net = tfl.conv2d_transpose(net, nout, kernel_size=3, padding="SAME",
                               name="d6_deconv", activation=None)
    return net

  def stacked_lstm(self, inputs, states, hidden_size, output_size, nlayers):
    """Stacked LSTM layers with FC layers as input and output embeddings.

    Args:
      inputs: input tensor
      states: a list of internal lstm states for each layer
      hidden_size: number of lstm units
      output_size: size of the output
      nlayers: number of lstm layers
    Returns:
      net: output of the network
      skips: a list of updated lstm states for each layer
    """
    net = inputs
    net = tfl.dense(
        net, hidden_size, activation=None, name="af1")
    for i in range(nlayers):
      net, states[i] = common_video.basic_lstm(
          net, states[i], hidden_size, name="alstm%d"%i)
    net = tfl.dense(
        net, output_size, activation=tf.nn.tanh, name="af2")
    return net, states

  def lstm_gaussian(self, inputs, states, hidden_size, output_size, nlayers,
                    name):
    """Stacked LSTM layers with FC layer as input and gaussian as output.

    Args:
      inputs: input tensor
      states: a list of internal lstm states for each layer
      hidden_size: number of lstm units
      output_size: size of the output
      nlayers: number of lstm layers
      name: the lstm name for scope definition
    Returns:
      mu: mean of the predicted gaussian
      logvar: log(var) of the predicted gaussian
      skips: a list of updated lstm states for each layer
    """
    net = inputs
    net = tfl.dense(net, hidden_size, activation=None, name="%sf1"%name)
    for i in range(nlayers):
      net, states[i] = common_video.basic_lstm(
          net, states[i], hidden_size, name="%slstm%d"%(name, i))
    mu = tfl.dense(net, output_size, activation=None, name="%sf2mu"%name)
    logvar = tfl.dense(net, output_size, activation=None, name="%sf2log"%name)
    return mu, logvar, states

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

    z_dim = self.hparams.z_dim
    g_dim = self.hparams.g_dim
    rnn_size = self.hparams.rnn_size
    prior_rnn_layers = self.hparams.prior_rnn_layers
    posterior_rnn_layers = self.hparams.posterior_rnn_layers
    predictor_rnn_layers = self.hparams.predictor_rnn_layers
    context_frames = self.hparams.video_num_input_frames
    has_batchnorm = self.hparams.has_batchnorm

    seq_len, batch_size, _, _, color_channels = common_layers.shape_list(images)

    # LSTM initial sizesstates.
    prior_states = [None] * prior_rnn_layers
    posterior_states = [None] * posterior_rnn_layers
    predictor_states = [None] * predictor_rnn_layers

    tf.logging.info(">>>> Encoding")
    # Encoding:
    enc_images, enc_skips = [], []
    images = tf.unstack(images, axis=0)
    for i, image in enumerate(images):
      with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        enc, skips = self.encoder(image, g_dim, has_batchnorm=has_batchnorm)
        enc = tfl.flatten(enc)
        enc_images.append(enc)
        enc_skips.append(skips)

    tf.logging.info(">>>> Prediction")
    # Prediction
    pred_mu_pos = []
    pred_logvar_pos = []
    pred_mu_prior = []
    pred_logvar_prior = []
    gen_images = []
    for i in range(1, seq_len):
      with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        # current encoding
        if self.is_training or len(gen_images) < context_frames:
          h_current = enc_images[i - 1]
        else:
          h_current, _ = self.encoder(gen_images[-1], g_dim)
          h_current = tfl.flatten(h_current)

        # target encoding
        h_target = enc_images[i]

      with tf.variable_scope("prediction", reuse=tf.AUTO_REUSE):
        # Prior parameters
        if self.hparams.learned_prior:
          mu_prior, logvar_prior, prior_states = self.lstm_gaussian(
              h_current, prior_states, rnn_size, z_dim, prior_rnn_layers,
              "prior")
        else:
          mu_prior = tf.zeros((batch_size, z_dim))
          logvar_prior = tf.zeros((batch_size, z_dim))

        # Only use Posterior if it's training time
        if self.is_training or len(gen_images) < context_frames:
          mu_pos, logvar_pos, posterior_states = self.lstm_gaussian(
              h_target, posterior_states, rnn_size, z_dim, posterior_rnn_layers,
              "posterior")
          # Sample z from posterior distribution
          z = common_video.get_gaussian_tensor(mu_pos, logvar_pos)
        else:
          mu_pos = tf.zeros_like(mu_prior)
          logvar_pos = tf.zeros_like(logvar_prior)
          z = common_video.get_gaussian_tensor(mu_prior, logvar_prior)

        # Predict output encoding
        h_pred, predictor_states = self.stacked_lstm(
            tf.concat([h_current, z], axis=1),
            predictor_states, rnn_size, g_dim, predictor_rnn_layers)

        pred_mu_pos.append(tf.identity(mu_pos, "mu_pos"))
        pred_logvar_pos.append(tf.identity(logvar_pos, "logvar_pos"))
        pred_mu_prior.append(tf.identity(mu_prior, "mu_prior"))
        pred_logvar_prior.append(tf.identity(logvar_prior, "logvar_prior"))

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
    return {"gen_images": gen_images,
            "fake_reward_prediction": fake_reward_prediction,
            "pred_mu_pos": pred_mu_pos,
            "pred_logvar_pos": pred_logvar_pos,
            "pred_mu_prior": pred_mu_prior,
            "pred_logvar_prior": pred_logvar_prior}

  def get_extra_loss(self,
                     latent_means_pos, latent_logvars_pos,
                     latent_means_prior, latent_logvars_prior):
    """Losses in addition to the default modality losses."""
    return self.get_kl_loss(
        latent_means_pos, latent_logvars_pos,
        latent_means_prior, latent_logvars_prior)

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
    gen_images = retvals["gen_images"]
    gen_rewards = retvals["fake_reward_prediction"]
    latent_means_pos = retvals["pred_mu_pos"]
    latent_logvars_pos = retvals["pred_logvar_pos"]
    latent_means_prior = retvals["pred_mu_prior"]
    latent_logvars_prior = retvals["pred_logvar_prior"]

    extra_loss = self.get_extra_loss(
        latent_means_pos=latent_means_pos,
        latent_logvars_pos=latent_logvars_pos,
        latent_means_prior=latent_means_prior,
        latent_logvars_prior=latent_logvars_prior)

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
def next_frame_emily():
  """Emily's model hparams."""
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
  hparams.add_hparam("prior_rnn_layers", 1)
  hparams.add_hparam("posterior_rnn_layers", 1)
  hparams.add_hparam("predictor_rnn_layers", 2)
  hparams.add_hparam("has_skips", True)
  hparams.add_hparam("has_batchnorm", True)
  return hparams
