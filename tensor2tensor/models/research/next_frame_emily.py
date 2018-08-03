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
from tensor2tensor.models.research import next_frame_params  # pylint: disable=unused-import
from tensor2tensor.models.research import next_frame_sv2p
from tensor2tensor.utils import registry
import tensorflow as tf

tfl = tf.layers
tfcl = tf.contrib.layers


@registry.register_model
class NextFrameStochasticEmily(next_frame_sv2p.NextFrameStochastic):
  """Stochastic Variational Video Prediction Without Learned Prior."""

  def encoder(self, inputs, nout):
    """VGG based image encoder.

    Args:
      inputs: image tensor with size BSx64x64xC
      nout: number of output channels
    Returns:
      net: encoded image with size BSxNout
      skips: skip connection after each layer
    """
    vgg_layer = common_video.vgg_layer
    net01 = inputs
    # h1
    net11 = tfcl.repeat(net01, 2, vgg_layer, 64,
                        scope="h1", is_training=self.is_training)
    net12 = tfl.max_pooling2d(net11, [2, 2], strides=(2, 2), name="h1_pool")
    # h2
    net21 = tfcl.repeat(net12, 2, vgg_layer, 128,
                        scope="h2", is_training=self.is_training)
    net22 = tfl.max_pooling2d(net21, [2, 2], strides=(2, 2), name="h2_pool")
    # h3
    net31 = tfcl.repeat(net22, 3, vgg_layer, 256,
                        scope="h3", is_training=self.is_training)
    net32 = tfl.max_pooling2d(net31, [2, 2], strides=(2, 2), name="h3_pool")
    # h4
    net41 = tfcl.repeat(net32, 3, vgg_layer, 512,
                        scope="h4", is_training=self.is_training)
    net42 = tfl.max_pooling2d(net41, [2, 2], strides=(2, 2), name="h4_pool")
    # h5
    net51 = tfcl.repeat(net42, 1, vgg_layer, nout,
                        kernel_size=4, padding="VALID", activation=tf.tanh,
                        scope="h5", is_training=self.is_training)
    skips = [net11, net21, net31, net41]
    return net51, skips

  def decoder(self, inputs, skips, nout):
    """VGG based image decoder.

    Args:
      inputs: image tensor with size BSxX
      skips: skip connections from encoder
      nout: number of output channels
    Returns:
      net: decoded image with size BSx64x64xNout
      skips: skip connection after each layer
    """
    vgg_layer = common_video.vgg_layer
    net = inputs
    # d1
    net = tfl.conv2d_transpose(net, 512, kernel_size=4, padding="VALID",
                               name="d1_deconv", activation=None)
    net = tfl.batch_normalization(net, training=self.is_training, name="d1_bn")
    net = tf.nn.leaky_relu(net)
    net = common_layers.upscale(net, 2)
    # d2
    net = tf.concat([net, skips[3]], axis=3)
    net = tfcl.repeat(net, 2, vgg_layer, 512, scope="d2a")
    net = tfcl.repeat(net, 1, vgg_layer, 256, scope="d2b")
    net = common_layers.upscale(net, 2)
    # d3
    net = tf.concat([net, skips[2]], axis=3)
    net = tfcl.repeat(net, 2, vgg_layer, 256, scope="d3a")
    net = tfcl.repeat(net, 1, vgg_layer, 128, scope="d3b")
    net = common_layers.upscale(net, 2)
    # d4
    net = tf.concat([net, skips[1]], axis=3)
    net = tfcl.repeat(net, 1, vgg_layer, 128, scope="d4a")
    net = tfcl.repeat(net, 1, vgg_layer, 64, scope="d4b")
    net = common_layers.upscale(net, 2)
    # d5
    net = tf.concat([net, skips[0]], axis=3)
    net = tfcl.repeat(net, 1, vgg_layer, 64, scope="d5")
    net = tfl.conv2d_transpose(net, nout, kernel_size=3, padding="SAME",
                               name="d6_deconv", activation=tf.sigmoid)
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

  def lstm_gaussian(self, inputs, states, hidden_size, output_size, nlayers):
    """Stacked LSTM layers with FC layer as input and gaussian as output.

    Args:
      inputs: input tensor
      states: a list of internal lstm states for each layer
      hidden_size: number of lstm units
      output_size: size of the output
      nlayers: number of lstm layers
    Returns:
      mu: mean of the predicted gaussian
      logvar: log(var) of the predicted gaussian
      skips: a list of updated lstm states for each layer
    """
    net = inputs
    net = tfl.dense(net, hidden_size, activation=None, name="bf1")
    for i in range(nlayers):
      net, states[i] = common_video.basic_lstm(
          net, states[i], hidden_size, name="blstm%d"%i)
    mu = tfl.dense(net, output_size, activation=None, name="bf2mu")
    logvar = tfl.dense(net, output_size, activation=None, name="bf2log")
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
    posterior_rnn_layers = self.hparams.posterior_rnn_layers
    predictor_rnn_layers = self.hparams.predictor_rnn_layers
    context_frames = self.hparams.video_num_input_frames

    seq_len, batch_size, _, _, color_channels = common_layers.shape_list(images)

    # LSTM initial sizesstates.
    predictor_states = [None] * predictor_rnn_layers
    posterior_states = [None] * posterior_rnn_layers

    tf.logging.info(">>>> Encoding")
    # Encoding:
    enc_images, enc_skips = [], []
    images = tf.unstack(images, axis=0)
    for i, image in enumerate(images):
      with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        enc, skips = self.encoder(image, rnn_size)
        enc = tfcl.flatten(enc)
        enc_images.append(enc)
        enc_skips.append(skips)

    tf.logging.info(">>>> Prediction")
    # Prediction
    pred_enc, pred_mu, pred_logvar = [], [], []
    for i in range(1, seq_len):
      with tf.variable_scope("prediction", reuse=tf.AUTO_REUSE):
        # current encoding
        h_current = enc_images[i-1]
        # target encoding
        h_target = enc_images[i]

        z = tf.random_normal([batch_size, z_dim], 0, 1, dtype=tf.float32)
        mu, logvar = tf.zeros_like(z), tf.zeros_like(z)

        # Only use Posterior if it's training time
        if self.hparams.mode == tf.estimator.ModeKeys.TRAIN:
          mu, logvar, posterior_states = self.lstm_gaussian(
              h_target, posterior_states, rnn_size, z_dim, posterior_rnn_layers)

          # The original implementation has a multiplier of 0.5
          # Removed here for simplicity i.e. replacing var with std
          z = z * tf.exp(logvar) + mu

        # Predict output encoding
        h_pred, predictor_states = self.stacked_lstm(
            tf.concat([h_current, z], axis=1),
            predictor_states, rnn_size, g_dim, predictor_rnn_layers)

        pred_enc.append(h_pred)
        pred_mu.append(mu)
        pred_logvar.append(logvar)

    tf.logging.info(">>>> Decoding")
    # Decoding
    gen_images = []
    for i in range(seq_len-1):
      with tf.variable_scope("decoding", reuse=tf.AUTO_REUSE):
        # use skip values of last available frame
        skip_index = min(context_frames-1, i)

        h_pred = tf.reshape(pred_enc[i], [batch_size, 1, 1, g_dim])
        x_pred = self.decoder(h_pred, enc_skips[skip_index], color_channels)
        gen_images.append(x_pred)

    tf.logging.info(">>>> Done")
    gen_images = tf.stack(gen_images, axis=0)
    return gen_images, fake_reward_prediction, pred_mu, pred_logvar
