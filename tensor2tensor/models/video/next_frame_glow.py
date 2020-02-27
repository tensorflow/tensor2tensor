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

"""Experimental testbed for nfg."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_video
from tensor2tensor.layers import modalities
from tensor2tensor.models.research import glow
from tensor2tensor.models.research import glow_ops
from tensor2tensor.utils import contrib
from tensor2tensor.utils import registry
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

arg_scope = contrib.framework().arg_scope


@registry.register_hparams
def next_frame_glow_hparams():
  """Hparams for next_frame_glow."""
  hparams = glow.glow_hparams()
  # Possible modes are conditional and unconditional
  hparams.add_hparam("gen_mode", "conditional")
  hparams.add_hparam("learn_top_scale", False)
  hparams.add_hparam("condition_all_levels", True)
  # For each video, substitutes "num_input_frames + num_output_frames" with a
  # randomly sampled patch of length "num_train_frames" during training.
  # -1 indicates that the entire video is used for training.
  hparams.add_hparam("num_train_frames", -1)
  # The following are hparams that model the latent transitions.
  # Encoder that maps the latents to a Gaussian distribution.
  # This function is used to model the prior over z_{t}. Can be,
  # Pointwise -> point-wise multiplication of z_{t-1}.
  # conv_net -> one-layer convolution over z_{t-1} .. z_{t - num_cond_latents}
  # conv3d_net or conv_lstm
  hparams.add_hparam("latent_dist_encoder", "conv_net")
  # Number of latents used in the encoder above.
  hparams.add_hparam("num_cond_latents", 1)
  hparams.add_hparam("latent_architecture", "glow_resnet")
  hparams.add_hparam("latent_apply_dilations", False)
  hparams.add_hparam("latent_dilation_rates", [1, 3])
  # Use latent skip connections
  hparams.add_hparam("model_input", False)
  hparams.add_hparam("cond_first_frame", False)
  hparams.add_hparam("latent_skip", True)
  hparams.add_hparam("latent_encoder_depth", 2)
  hparams.add_hparam("latent_encoder_width", 512)
  hparams.add_hparam("latent_dropout", 0.0)
  hparams.add_hparam("latent_pre_output_channels", 512)
  hparams.add_hparam("latent_activation", "relu")
  hparams.add_hparam("latent_noise", 0.0)
  # Pretrains the glow encoder for "pretrain_steps" number of steps.
  # By default, don't pretrain and learn end-to-end
  hparams.add_hparam("pretrain_steps", -1)
  hparams.bottom = {
      "inputs": modalities.video_raw_bottom,
      "targets": modalities.video_raw_targets_bottom,
  }
  hparams.loss = {
      "targets": modalities.video_l1_raw_loss,
  }
  hparams.top = {
      "targets": modalities.video_raw_top,
  }
  hparams.init_batch_size = 256
  hparams.batch_size = 32
  # Possible options: are prev_frame, single_conv and normal
  hparams.top_prior = "single_conv"
  return hparams


@registry.register_hparams
def next_frame_glow_bair_quant():
  """Hparams to reproduce bits-per-pixel results on BAIR action-free dataset."""
  hparams = next_frame_glow_hparams()
  hparams.video_num_input_frames = 3
  hparams.video_num_target_frames = 10
  hparams.num_train_frames = 4
  hparams.num_cond_latents = 3
  hparams.depth = 24
  hparams.latent_dist_encoder = "conv3d_net"
  hparams.latent_encoder_width = 256
  hparams.latent_architecture = "glow_resnet"
  hparams.latent_encoder_depth = 5
  hparams.latent_apply_dilations = True
  hparams.latent_activation = "gatu"
  hparams.activation = "gatu"
  hparams.learning_rate_constant = 3e-4
  hparams.learning_rate_schedule = "constant*linear_warmup"
  hparams.learning_rate_warmup_steps = 10000
  hparams.init_batch_size = 128
  hparams.batch_size = 5
  return hparams


@registry.register_hparams
def next_frame_glow_bair_qual():
  """Hparams for qualitative video generation results."""
  hparams = next_frame_glow_bair_quant()
  hparams.coupling = "additive"
  hparams.temperature = 0.5
  hparams.coupling_width = 392
  return hparams


@registry.register_hparams
def next_frame_glow_shapes():
  """Hparams for qualitative and quantitative results on shapes dataset."""
  hparams = next_frame_glow_bair_quant()
  hparams.video_num_input_frames = 1
  hparams.video_num_target_frames = 2
  hparams.num_train_frames = 2
  hparams.num_cond_latents = 1
  hparams.coupling = "additive"
  hparams.coupling_width = 512
  hparams.latent_encoder_depth = 10
  hparams.latent_skip = False
  hparams.learning_rate_constant = 1e-4
  hparams.batch_size = 10
  return hparams


@registry.register_hparams
def frame_glow_hparams():
  """Unconditional generation on video-frames."""
  hparams = next_frame_glow_hparams()
  hparams.gen_mode = "unconditional"
  hparams.num_train_frames = 1
  return hparams


def get_cond_latents(all_latents=None, hparams=None):
  """Get z^{cond}_{t} given z^{1..t-1}.

  Args:
    all_latents: list of list of tensors,
                 outer-size equals no.of time_steps-1
                 inner-size equals hparams.n_levels.
    hparams: See next_frame_glow_hparams.
  Returns:
    cond_latents: conditional latents at time-step t.
  """
  cond_latents = None
  if hparams.gen_mode == "conditional":
    if hparams.latent_dist_encoder in ["conv_net", "conv3d_net"]:
      num_cond_latents = (hparams.num_cond_latents +
                          int(hparams.cond_first_frame))
      if len(all_latents) >= num_cond_latents:
        cond_latents = all_latents[-hparams.num_cond_latents:]
        if hparams.cond_first_frame:
          cond_latents = [all_latents[0]] + cond_latents
    elif hparams.latent_dist_encoder in ["pointwise", "conv_lstm"]:
      if all_latents:
        cond_latents = all_latents[-1]

  if hparams.gen_mode == "conditional":
    global_step = tf.train.get_or_create_global_step()
    condition = tf.greater(global_step, hparams.pretrain_steps)
  else:
    condition = tf.constant(False, dtype=tf.bool)
  return condition, cond_latents


@registry.register_model
class NextFrameGlow(glow.Glow):
  """Extend Glow for video."""

  def init_preprocess_single(self, features):
    for label in ["inputs", "targets"]:
      features[label] = common_layers.convert_rgb_to_real(features[label])
    return features

  def init_preprocess(self, features):
    """Preprocessing as per the input modality.

    Equivalent to calling self.bottom(features).

    Args:
      features: dict of strings to tensors.
    Returns:
      features: dict of strings to tensors.
    """
    return features.map(self.init_preprocess_single)

  def preprocess(self, x):
    """Converts x from [0, 1] to [-0.5, 0.5].

    All inputs are already normalized to be in the range [0, 1] through the
    VideoModalityL1Raw modality.

    Args:
      x: 4-D Tensor.

    Returns:
      x: Scaled such that x lies in-between -0.5 and 0.5
    """
    return x - 0.5

  def infer(self, features, *args, **kwargs):  # pylint: disable=arguments-differ
    del args, kwargs

    # Make a copy of features that can be used in the call to self
    # that builds the graph.
    new_features = {}
    new_features["inputs"] = features["inputs"]
    new_features["targets"] = features["infer_targets"]
    _, _ = self(new_features)  # pylint: disable=not-callable

    if self.hparams.gen_mode == "unconditional":
      num_target_frames = 1
    else:
      num_target_frames = self.hparams.video_num_target_frames

    ops = [glow_ops.get_variable_ddi, glow_ops.actnorm, glow_ops.get_dropout]
    var_scope = tf.variable_scope("next_frame_glow/body", reuse=True)
    all_frames = []

    # If eps=None, images are sampled from the prior.
    with arg_scope(ops, init=False), var_scope:
      for target_frame in range(1, num_target_frames + 1):

        # subscript -> timestep, superscript -> level.
        # self.z_sample equals z^0_{t} (top-level latent)
        # (X_{t}, z^{1..l}_{t}) = Glow(z^0_{t}, z^{1..l}_{t-1})
        # Get current set of cond_latents.
        cond_level, cond_level_latents = get_cond_latents(
            self.all_level_latents, self.hparams)

        glow_vals = glow_ops.encoder_decoder(
            "codec", self.z_sample, self.hparams, eps=None, reverse=True,
            cond_latents=cond_level_latents, states=self.level_states,
            condition=cond_level, temperature=self.temperature)
        predicted_frame, _, curr_latents, self.level_states = glow_vals
        all_frames.append(predicted_frame)
        self.all_level_latents.append(curr_latents)

        # Compute z^0_{t+1} = f(z^0_{t})
        if target_frame < num_target_frames:
          cond_top, cond_top_latents = get_cond_latents(
              self.all_top_latents, self.hparams)
          prior_dist = self.top_prior(
              condition=cond_top, cond_latents=cond_top_latents)
          self.z_sample = prior_dist.sample()
          self.all_top_latents.append(self.z_sample)

    all_frames = tf.stack(all_frames)
    predicted_video = common_video.swap_time_and_batch_axes(all_frames)

    # The video-decode API requires the predicted video to be the same shape
    # as the target-video. Hence, for unconditional generation,
    # tile across time to ensure same shape.
    if self.hparams.gen_mode == "unconditional":
      predicted_video = tf.tile(
          predicted_video, [1, self.hparams.video_num_target_frames, 1, 1, 1])
    predicted_video = glow_ops.postprocess(predicted_video)

    # Output of a single decode / sample.
    output_features = {}
    output_features["targets"] = tf.zeros_like(predicted_video)
    output_features["outputs"] = predicted_video
    output_features["scores"] = tf.zeros_like(predicted_video)
    return output_features

  def get_squeeze_prior(self):
    """Model the prior over z_{t} as a function of X_{t-1}.

    Returns:
      objective: float, log-likelihood.
      dist: instance of tfp.distributions.Normal.

    Raises:
      ValueError: If input_height is not equal to input_width, not even
                   or if the image width is smaller than the latent width.
    """
    _, prior_height, _, prior_channels = self.z_top_shape
    _, input_height, input_width, _ = common_layers.shape_list(self.input_frame)

    if input_height != input_width:
      raise ValueError("input height should be equal to input width")
    if input_height % 2 != 0:
      raise ValueError("input height should be even")
    if input_height < prior_height:
      raise ValueError("input should be larger than the prior.")

    # mean, log_std = NN(X_0)
    # Reduce the spatial dimension by a factor of "squeeze_factor".
    # and convolve with a stride of 2
    squeeze_factor = input_height // (2 * prior_height)
    x = glow_ops.squeeze(
        "prior_squeeze", self.input_frame, factor=squeeze_factor, reverse=False)
    mean_and_log_std = glow_ops.conv(
        "prior_conv", x, 2*prior_channels, stride=[2, 2], apply_actnorm=False,
        conv_init="zeros")
    mean, log_scale = tf.split(mean_and_log_std, num_or_size_splits=2, axis=-1)
    return tfp.distributions.Normal(mean, tf.exp(log_scale))

  def top_cond_prior(self, name, cond_top_latents):
    """Maps the conditional top latents to a distribution.

    Args:
      name: variable scope.
      cond_top_latents: Tensor or a list of tensors.
                        Latent variables at the previous time-step.
                        If "pointwise", this is a single tensor.
                        If "conv_net", this is a list of tensors with length
                        equal to hparams.num_cond_latents.
    Returns:
      cond_dist: tfp.distributions.Normal
    Raises:
      ValueError: If cond_top_latents are not of the expected length.
    """
    with tf.variable_scope("top", reuse=tf.AUTO_REUSE):
      if self.hparams.latent_dist_encoder == "pointwise":
        last_latent = cond_top_latents
        top = glow_ops.scale_gaussian_prior(
            name, cond_top_latents, trainable=self.hparams.learn_top_scale)
      elif self.hparams.latent_dist_encoder == "conv_net":
        num_cond_latents = (self.hparams.num_cond_latents +
                            int(self.hparams.cond_first_frame))
        if len(cond_top_latents) != num_cond_latents:
          raise ValueError(
              "Expected length of cond_top_latents %d, got %d"
              % (num_cond_latents, len(cond_top_latents)))
        last_latent = cond_top_latents[-1]
        output_channels = common_layers.shape_list(last_latent)[-1]
        cond_top_latents = tf.concat(cond_top_latents, axis=-1)

        # Maps the latent-stack to a distribution.
        cond_top_latents = glow_ops.noise_op(cond_top_latents, self.hparams)
        top = glow_ops.latent_to_dist(
            name, cond_top_latents, hparams=self.hparams,
            output_channels=output_channels)
      elif self.hparams.latent_dist_encoder == "conv_lstm":
        last_latent = cond_top_latents
        output_channels = common_layers.shape_list(cond_top_latents)[-1]
        # (h_t, c_t) = LSTM(z_{t-1}; (h_{t-1}, c_{t-1}))
        # (mu_t, sigma_t) = conv(h_t)
        cond_top_latents = glow_ops.noise_op(cond_top_latents, self.hparams)
        _, self.top_state = common_video.conv_lstm_2d(
            cond_top_latents, self.top_state, self.hparams.latent_encoder_width,
            kernel_size=3, name="conv_lstm")
        top = glow_ops.single_conv_dist(
            name, self.top_state.h, output_channels=output_channels)
      elif self.hparams.latent_dist_encoder == "conv3d_net":
        last_latent = cond_top_latents[-1]
        cond_top_latents = tf.stack(cond_top_latents, axis=1)
        cond_top_latents = glow_ops.noise_op(cond_top_latents, self.hparams)
        top = glow_ops.temporal_latent_to_dist(
            "conv3d", cond_top_latents, self.hparams)

      # mu(z_{t}) = z_{t-1} + latent_encoder(z_{cond})
      if self.hparams.latent_skip:
        top = tfp.distributions.Normal(last_latent + top.loc, top.scale)
    return top

  def uncond_top_dist(self):
    """Get an unconditional prior distribution on the top latent."""
    prior_dist = glow_ops.top_prior(
        "unconditional", self.z_top_shape, learn_prior="single_conv")
    return prior_dist.loc, prior_dist.scale

  def cond_top_dist(self, cond_latents):
    """Get a conditional prior distribution on the top latent."""
    prior_dist = self.top_cond_prior("conditional", cond_latents)
    return prior_dist.loc, prior_dist.scale

  def top_prior(self, condition=False, cond_latents=None):
    """Objective based on the prior over latent z.

    Args:
      condition: Whether or not to condition on cond_latents.
      cond_latents: tensor or list of tensors depending on
                    hparams.latent_dist_encoder
    Returns:
      objective: float, log-likelihood of z under the prior.
      dist: instance of tfp.distributions.Normal, prior distribution.
    Raises:
      ValueError: If input is smaller than the prior, uneven height
                  or rectangular.
    """
    if isinstance(condition, bool):
      condition = tf.constant(condition, dtype=tf.bool)
    self._all_conds.append(condition)

    if self.hparams.gen_mode == "conditional":
      # cond_top_latents is None when
      # latent_dist_encoder is a lstm and frame_ind == 0.
      # latent_dist_encoder is conv_net and frame_ind < num_cond_frames.
      marginal_mean, marginal_scale = self.uncond_top_dist()
      if cond_latents is None:
        mean, scale = marginal_mean, marginal_scale
      else:
        cond_mean, cond_scale = self.cond_top_dist(cond_latents)
        mean, scale = tf.cond(
            condition, lambda: (cond_mean, cond_scale),
            lambda: (marginal_mean, marginal_scale))
      return glow_ops.TemperedNormal(mean, scale, self.temperature)
    if self.hparams.top_prior == "prev_frame":
      return self.get_squeeze_prior()
    else:
      return super(NextFrameGlow, self).top_prior()

  def get_z_top_shape(self, init=False):
    """Get latent shape at level."""
    if init:
      batch_size = self.hparams.init_batch_size
    else:
      batch_size = self.hparams.batch_size
    height, _, channels = self.hparams.problem.frame_shape
    n_levels = self.hparams.n_levels
    z_width = height // 2**n_levels
    z_channels = channels * 2**n_levels * 2
    return [batch_size, z_width, z_width, z_channels]

  def squeeze_video(self, video, init=False):
    """Squeeze a 5-D Tensor video with one timestep to a 4-D frame."""
    if init:
      batch_size = self.hparams.init_batch_size
    else:
      batch_size = self.hparams.batch_size
    frame_shape = [batch_size] + self.hparams.problem.frame_shape
    return tf.reshape(video, frame_shape)

  def glow_encoder(self, frame, condition=False, cond_latents=None, init=False):
    """Glow network that encodes frame to a hierarchy of latents.

    Args:
      frame: 5-D Tensor of shape (batch_size, 1, height, width, channels).
      condition: Whether or not to condition on cond_latents.
      cond_latents: optional, list of tensors with length equal to
                    hparams.n_levels - 1. If provided, the latent at level l is
                    conditioned on the cond_latent at level l.
      init: Whether the given batch is an "init" batch or a "train" batch.
    Returns:
      objective: log-likelihood of the frame per the model.
      z_top: top-level latent.
      z_levels: a list of tensors with latents at all levels.
    """
    frame = self.squeeze_video(frame, init=init)
    frame = self.preprocess(frame)
    frame, objective = glow_ops.uniform_binning_correction(frame)

    glow_vals = glow_ops.encoder_decoder(
        "codec", frame, self.hparams, eps=None, reverse=False,
        cond_latents=cond_latents, states=self.level_states,
        condition=condition)
    z_top, encoder_objective, self.eps, z_levels, self.level_states = glow_vals
    objective += encoder_objective
    return objective, z_top, z_levels

  def get_num_train_frames(self):
    """Returns the number of frames as a normalizing factor."""
    num_target = self.hparams.video_num_target_frames
    num_input = self.hparams.video_num_input_frames

    # For unconditional generation, this picks a random frame during training
    # and evaluates the marginal likelihood over "num_input" + "num_target"
    # frames during eval.
    if self.hparams.gen_mode == "unconditional":
      if self.is_training:
        return 1
      return num_input + num_target

    # During eval we measure the true objective.
    if not self.is_training or self.hparams.num_train_frames == -1:
      total_frames = num_target
    # if hparams.num_train_frames=-1, we use an approxination to the true
    # objective.
    else:
      total_frames = self.hparams.num_train_frames - num_input
    if self.hparams.model_input:
      total_frames += num_input
    return total_frames

  def get_all_frames(self, input_frames, target_frames):
    """Get the frames used as input to the model.

    Args:
      input_frames: 5-D Tensor, (NTHWC)
      target_frames: 5-D Tensor, (NTHWC)
    Returns:
      frames: 5-D Tensor used as input to the model.
    """
    if self.is_predicting:
      all_frames = input_frames
    elif self.is_training:
      all_frames = tf.concat((input_frames, target_frames), axis=1)
      all_frames = common_video.extract_random_video_patch(
          all_frames, self.hparams.num_train_frames)
    # Measure the mean bit-per-pixel of the target_frames during eval.
    else:
      all_frames = tf.concat((input_frames, target_frames), axis=1)
    if self.hparams.cond_first_frame:
      first_frame = all_frames[:, 0:1, :, :, :]
      all_frames = tf.concat((first_frame, all_frames), axis=1)
    return all_frames

  def video_objective_tower(self, input_frames, target_frames, init=False):
    """Returns the bits-per-pixel of the video.

    Args:
      input_frames: 5-D Tensor of shape (N, 1, H, W, C)
      target_frames: 5-D Tensor of shape (N, T, H, W, C)
      init: Whether or not to run data-dependent initialization.
    Returns:
      objective: bits-per-pixel.
    """
    # The arg_scope call ensures that the actnorm parameters are set such that
    # the per-channel output activations have zero mean and unit variance
    # ONLY during the first step. After that the parameters are learned
    # through optimisation.
    num_input_frames = (self.hparams.video_num_input_frames +
                        int(self.hparams.cond_first_frame))

    # Set num total frames to average the objective.
    total_frames = self.get_num_train_frames()

    # Compute the log-likelihood of target_frames at both train and predict
    # time.
    all_frames = self.get_all_frames(input_frames, target_frames)
    all_frames = tf.unstack(all_frames, axis=1)

    cond_level_latents, cond_top_latents = None, None
    total_objective = 0.0
    ops = [glow_ops.get_variable_ddi, glow_ops.actnorm, glow_ops.get_dropout]

    with arg_scope(ops, init=init):
      for frame_ind, frame in enumerate(all_frames):

        # Get current set of cond latents of non-top levels.
        cond_level, cond_level_latents = get_cond_latents(
            self.all_level_latents, self.hparams)

        # Get current set of cond latents of the top-level
        cond_top, cond_top_latents = get_cond_latents(
            self.all_top_latents, self.hparams)

        # Superscript -> level, Subscript -> Time.
        # (z^{0}_t, z^{1..l}_t) = Glow(X_{t}, z^{1..l}_{cond_t})
        frame_obj, curr_top_latent, curr_level_latents = self.glow_encoder(
            frame, condition=cond_level, cond_latents=cond_level_latents,
            init=init)

        # z^0_t ~ N(f(z^0_{t-1}))
        # cond_top_latents is None when
        # latent_dist_encoder is conv_net and frame_ind < num_cond_frames.
        prior_dist = self.top_prior(
            condition=cond_top, cond_latents=cond_top_latents)
        prior_objective = tf.reduce_sum(
            prior_dist.log_prob(curr_top_latent), axis=[1, 2, 3])
        frame_obj += prior_objective

        # Loss computation.
        # Do not model the probabililty of the input frames by default.
        # Consistent with other video models.
        if (frame_ind > num_input_frames - 1 or self.hparams.model_input or
            self.hparams.gen_mode == "unconditional"):
          total_objective += frame_obj
        self.all_level_latents.append(curr_level_latents)
        self.all_top_latents.append(curr_top_latent)

      # During prediction time, store z_sample ~ N(f(z_{num_input_frames}))
      # to generate the first target frame.
      if self.is_predicting:
        # Get current set of cond_top_latents
        cond_top, cond_top_latents = get_cond_latents(
            self.all_top_latents, self.hparams)
        prior_dist = self.top_prior(
            condition=cond_top, cond_latents=cond_top_latents)
        self.z_sample = prior_dist.sample()
        self.all_top_latents.append(self.z_sample)

      # Converts log-probability to bits-per-pixel.
      hwc = np.prod(self.hparams.problem.frame_shape)
      total_objective = -total_objective / (np.log(2) * hwc * total_frames)
    return total_objective

  def objective_tower(self, features, init=False):
    input_frames, target_frames = features["inputs"], features["targets"]
    self.cond_latents, self.top_state = None, None
    self.all_level_latents, self.all_top_latents = [], []
    self._all_conds = []
    self.level_states = [None] * (self.hparams.n_levels - 1)
    self.z_top_shape = self.get_z_top_shape(init=init)
    num_input_frames = self.hparams.video_num_input_frames
    latent_dist_encoder = self.hparams.latent_dist_encoder
    num_cond_latents = self.hparams.num_cond_latents

    exp_modes = ["conditional", "unconditional"]
    if self.hparams.gen_mode not in exp_modes:
      raise ValueError("Expected mode to be in %s, got %s" %
                       (exp_modes, self.hparams.gen_mode))

    # Error checks for conditional video generation.
    if self.hparams.gen_mode == "conditional":
      exp_latent_encoders = ["pointwise", "conv_net", "conv_lstm", "conv3d_net"]
      if latent_dist_encoder not in exp_latent_encoders:
        raise ValueError("Expected latent_dist_encoder is %s, got %s" %
                         (exp_latent_encoders, latent_dist_encoder))
      if (latent_dist_encoder == "pointwise" and num_cond_latents != 1):
        raise ValueError("Expected num_cond_latents: 1, with 'pointwise' "
                         "latent_dist_encoder, got %d" % num_cond_latents)
      if (latent_dist_encoder == "conv_net" and
          num_cond_latents > num_input_frames):
        raise ValueError("Expected num_cond_latents <= %d, got %d" %
                         (num_input_frames, num_cond_latents))
      if (latent_dist_encoder == "pointwise" and
          self.hparams.init_batch_size != self.hparams.batch_size):
        raise ValueError("init_batch_size different from batch_size not "
                         "supported for latent_dist_encoder=pointwise")
    if self.hparams.gen_mode == "unconditional":
      if self.hparams.num_train_frames != 1:
        raise ValueError("Expected num_train_frames to be 1 when "
                         "hparams.gen_mode is unconditional, got %d" %
                         self.hparams.num_train_frames)
      if self.hparams.video_num_input_frames != 1:
        raise ValueError("Expected num_input_frames to be 1 when "
                         "hparams.gen_mode is unconditional, got %d" %
                         self.hparams.video_num_input_frames)
    return self.video_objective_tower(input_frames, target_frames, init=init)
