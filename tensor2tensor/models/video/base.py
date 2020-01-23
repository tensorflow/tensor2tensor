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

"""Basic models for testing simple tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import six

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_video
from tensor2tensor.layers import discretization
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow.compat.v1 as tf


def flat_lists(list_of_lists):
  return [x for l in list_of_lists for x in l]  # pylint: disable=g-complex-comprehension


def pixels_from_softmax(frame_logits, pure_sampling=False,
                        temperature=1.0, gumbel_noise_factor=0.2):
  """Given frame_logits from a per-pixel softmax, generate colors."""
  # If we're purely sampling, just sample each pixel.
  if pure_sampling or temperature == 0.0:
    return common_layers.sample_with_temperature(frame_logits, temperature)

  # Gumbel-sample from the pixel sofmax and average by pixel values.
  pixel_range = tf.to_float(tf.range(256))
  for _ in range(len(frame_logits.get_shape().as_list()) - 1):
    pixel_range = tf.expand_dims(pixel_range, axis=0)

  frame_logits = tf.nn.log_softmax(frame_logits)
  gumbel_samples = discretization.gumbel_sample(
      common_layers.shape_list(frame_logits)) * gumbel_noise_factor

  frame = tf.nn.softmax((frame_logits + gumbel_samples) / temperature, axis=-1)
  result = tf.reduce_sum(frame * pixel_range, axis=-1)
  # Round on the forward pass, not on the backward one.
  return result + tf.stop_gradient(tf.round(result) - result)


@registry.register_model
class NextFrameBase(t2t_model.T2TModel):
  """Base class for next_frame models.

    This is the base class for the models that given the previous frames
    can predict the next frame. They may also support reward prediction
    and action condition prediction which enables them to be run as
    a world model in model-based RL pipeline.

    The API supports both recurrent and stacked frames models. Please look
    at the documents for next_frame function for the API.

    If you are implementing a next frame prediction model consider
    following the API presented in this class. But if the API
    is too limiting for your models, feel free to override lower level
    functions and/or inheret from T2TModel directly.

  """

  # ============================================================================
  # BEGIN SUBCLASS INTERFACE
  # ============================================================================
  def next_frame(self,
                 frames, actions, rewards,
                 target_frame, internal_states, video_features):
    """The main prediction function of next frame models.

      This is the main function that should be overridden to implement models.

    Args:
      frames: The list of input frames.
              Only previous frame in case of recurrent models.
      actions: The list of input actions.
              Only previous action in case of recurrent models.
      rewards: The list of input rewards.
              Only previous reward in case of recurrent models.
      target_frame: The target frame.
              Usually required for approximating the posterior.
      internal_states: Internal model states. Only useful for recurrent models
              to keep the state from the previous time index.
              internal_states is None at the first frame and should be
              initialized properly.
      video_features: video wide features. None by default.
              Please refer to video_features function for description.

    Returns:
      pred_frame: predicted frame BSxWxHxC
              where C is 3 for L1/L2 modality and 3*256 for Softmax.
      pred_reward: the same size as input reward.
              None if the model does not detect rewards.
      pred_action: predicted action logits
      pred_value: predicted value
      extra_loss: any extra loss other than predicted frame and reward.
              e.g. KL loss in case of VAE models.
      internal_states: updated internal models states.
    """
    raise NotImplementedError("Base video model.")

  def video_features(
      self, all_frames, all_actions, all_rewards, all_raw_frames):
    """Optional video wide features.

      If the model requires access to all of the video frames
      (e.g. in case of approximating one latent for the whole video)
      override this function to add them. They will be accessible
      as video_features in next_frame function.

    Args:
      all_frames: list of all frames including input and target frames.
      all_actions: list of all actions including input and target actions.
      all_rewards: list of all rewards including input and target rewards.
      all_raw_frames: list of all raw frames (before modalities).

    Returns:
      video_features: a dictionary containing video-wide features.
    """
    del all_frames, all_actions, all_rewards, all_raw_frames
    return None

  def video_extra_loss(self, frames_predicted, frames_target,
                       internal_states, video_features):
    """Optional video wide extra loss.

      If the model needs to calculate some extra loss across all predicted
      frames (e.g. in case of video GANS loss) override this function.

    Args:
      frames_predicted: list of all predicted frames.
      frames_target: list of all target frames.
      internal_states: internal states of the video.
      video_features: video wide features coming from video_features function.

    Returns:
      extra_loss: extra video side loss.
    """
    del frames_predicted, frames_target, internal_states, video_features
    return 0.0

  @property
  def is_recurrent_model(self):
    """Set to true if your model is recurrent. False otherwise.

    This mainly affects how the inputs will be fed into next_frame function.
    """
    raise NotImplementedError("Base video model.")

  def init_internal_states(self):
    """Allows a model to preserve its internal model across multiple runs.

    This optional function is only useful for any model with internal states
    (usually recurrent models) which need to preserve states after any call.
    """
    return None

  def reset_internal_states_ops(self):
    """Resets internal states to initial values."""
    return [[tf.no_op()]]

  def load_internal_states_ops(self):
    """Loade internal states from class variables."""
    return [[tf.no_op()]]

  def save_internal_states_ops(self, internal_states):
    """Saves internal states into class variables."""
    return [[tf.no_op()]]

  # ============================================================================
  # END SUBCLASS INTERFACE
  # ============================================================================

  def __init__(self, *args, **kwargs):
    super(NextFrameBase, self).__init__(*args, **kwargs)
    self.internal_states = self.init_internal_states()

  @property
  def _target_modality(self):
    return self.problem_hparams.modality["targets"]

  @property
  def is_per_pixel_softmax(self):
    # TODO(trandustin): This is a hack.
    return "targets" not in self.hparams.get("loss")

  def get_iteration_num(self):
    step_num = tf.train.get_global_step()
    # TODO(lukaszkaiser): what should it be if it's undefined?
    if step_num is None:
      step_num = 10000000
    return step_num

  def visualize_predictions(self, predics, targets):
    predics = tf.concat(predics, axis=1)
    targets = tf.concat(targets, axis=1)
    side_by_side_video = tf.concat([predics, targets], axis=2)
    tf.summary.image("full_video", side_by_side_video)

  def get_scheduled_sample_func(self, batch_size):
    """Creates a function for scheduled sampling based on given hparams."""
    with tf.variable_scope("scheduled_sampling_func", reuse=tf.AUTO_REUSE):
      iter_num = self.get_iteration_num()

      # Simple function to bypass scheduled sampling in gt or pred only modes.
      def scheduled_sampling_simple(ground_truth_x, generated_x,
                                    batch_size, scheduled_sample_var):
        del batch_size
        if scheduled_sample_var:
          return ground_truth_x
        return generated_x

      mode = self.hparams.scheduled_sampling_mode
      if mode == "ground_truth_only":
        scheduled_sampling_func = scheduled_sampling_simple
        scheduled_sampling_func_var = True
      elif mode == "prediction_only":
        scheduled_sampling_func = scheduled_sampling_simple
        scheduled_sampling_func_var = False
      elif mode == "prob":
        decay_steps = self.hparams.scheduled_sampling_decay_steps
        probability = tf.train.polynomial_decay(
            1.0, iter_num, decay_steps, 0.0)
        scheduled_sampling_func = common_video.scheduled_sample_prob
        scheduled_sampling_func_var = probability
      elif mode == "prob_inverse_exp":
        decay_steps = self.hparams.scheduled_sampling_decay_steps
        probability = common_layers.inverse_exp_decay(
            decay_steps, step=iter_num)
        probability *= self.hparams.scheduled_sampling_max_prob
        probability = 1.0 - probability
        scheduled_sampling_func = common_video.scheduled_sample_prob
        scheduled_sampling_func_var = probability
      elif mode == "prob_inverse_lin":
        decay_steps = self.hparams.scheduled_sampling_decay_steps
        probability = common_layers.inverse_exp_decay(
            decay_steps // 4, step=iter_num)  # Very low at start.
        probability *= common_layers.inverse_lin_decay(
            decay_steps, step=iter_num)
        probability *= self.hparams.scheduled_sampling_max_prob
        probability = 1.0 - probability
        scheduled_sampling_func = common_video.scheduled_sample_prob
        scheduled_sampling_func_var = probability
      elif mode == "count":
        # Calculate number of ground-truth frames to pass in.
        k = self.hparams.scheduled_sampling_k
        num_ground_truth = tf.to_int32(
            tf.round(
                tf.to_float(batch_size) *
                (k / (k + tf.exp(tf.to_float(iter_num) / tf.to_float(k))))))
        scheduled_sampling_func = common_video.scheduled_sample_count
        scheduled_sampling_func_var = num_ground_truth
      else:
        raise ValueError("unknown scheduled sampling method: %s" % mode)

      if isinstance(scheduled_sampling_func_var, tf.Tensor):
        tf.summary.scalar("scheduled_sampling_var", scheduled_sampling_func_var)
      partial_func = functools.partial(
          scheduled_sampling_func,
          batch_size=batch_size,
          scheduled_sample_var=scheduled_sampling_func_var)
      return partial_func

  def get_scheduled_sample_inputs(self,
                                  done_warm_start,
                                  groundtruth_items,
                                  generated_items,
                                  scheduled_sampling_func):
    """Scheduled sampling.

    Args:
      done_warm_start: whether we are done with warm start or not.
      groundtruth_items: list of ground truth items.
      generated_items: list of generated items.
      scheduled_sampling_func: scheduled sampling function to choose between
        groundtruth items and generated items.

    Returns:
      A mix list of ground truth and generated items.
    """
    def sample():
      """Calculate the scheduled sampling params based on iteration number."""
      with tf.variable_scope("scheduled_sampling", reuse=tf.AUTO_REUSE):
        return [
            scheduled_sampling_func(item_gt, item_gen)
            for item_gt, item_gen in zip(groundtruth_items, generated_items)]

    cases = [
        (tf.logical_not(done_warm_start), lambda: groundtruth_items),
        (tf.logical_not(self.is_training), lambda: generated_items),
    ]
    output_items = tf.case(cases, default=sample, strict=True)

    return output_items

  def get_extra_internal_loss(self, extra_raw_gts, extra_gts, extra_pds):
    """Hacky code the get the loss on predicted frames from input frames.

       Recurrent models consume the frames one-by-one. Therefore
       if there is more than one input frame they also get predicted.
       T2T only calculates loss on the predicted target frames which
       means the loss is not being applied on the predicted input frames.
       This code is to fix this issue. Since the model is not aware of the
       modality it has to match the pre-porocessing happening in bottom
       function and therefore this becomes a very hacky code. This code
       should match the bottom and top and loss of modalities otherwise
       it will calculate the wrong loss.

    Args:
      extra_raw_gts: extra raw ground truth frames.
      extra_gts: extra normalized ground truth frames.
      extra_pds: extra predicted frames.

    Returns:
      Additional reconstruction loss.

    Raises:
      ValueError: in case of unknown loss transformation.
    """
    # TODO(trandustin): This logic should be moved elsewhere.
    if self.hparams.loss.get("targets") == modalities.video_l2_raw_loss:
      recon_loss = tf.losses.mean_squared_error(extra_gts, extra_pds)
    elif "targets" not in self.hparams.loss:
      shape = common_layers.shape_list(extra_pds)
      updated_shape = shape[:-1] + [3, 256]
      extra_pds = tf.reshape(extra_pds, updated_shape)
      # Merge time and batch
      logits = tf.reshape(extra_pds, [-1] + updated_shape[2:])
      targets = extra_raw_gts
      targets_shape = common_layers.shape_list(targets)
      targets = tf.reshape(targets, [-1] + targets_shape[2:])
      targets_weights_fn = self.hparams.weights_fn.get(
          "targets",
          modalities.get_weights_fn(self._target_modality))
      numerator, denominator = common_layers.padded_cross_entropy(
          logits,
          targets,
          self.hparams.label_smoothing,
          cutoff=getattr(self.hparams, "video_modality_loss_cutoff", 0.01),
          weights_fn=targets_weights_fn)
      recon_loss = numerator / denominator
    else:
      raise ValueError("internal loss only supports specific hparams.loss.")
    tf.summary.scalar("recon_extra", recon_loss)
    return recon_loss

  def get_sampled_frame(self, pred_frame):
    """Samples the frame based on modality.

      if the modality is L2/L1 then the next predicted frame is the
      next frame and there is no sampling but in case of Softmax loss
      the next actual frame should be sampled from predicted frame.

      This enables multi-frame target prediction with Softmax loss.

    Args:
      pred_frame: predicted frame.

    Returns:
      sampled frame.

    """
    # TODO(lukaszkaiser): the logic below heavily depend on the current
    # (a bit strange) video modalities - we should change that.

    sampled_frame = pred_frame
    if self.is_per_pixel_softmax:
      frame_shape = common_layers.shape_list(pred_frame)
      target_shape = frame_shape[:-1] + [self.hparams.problem.num_channels]
      sampled_frame = tf.reshape(pred_frame, target_shape + [256])
      sampled_frame = pixels_from_softmax(
          sampled_frame, temperature=self.hparams.pixel_sampling_temperature)
      # TODO(lukaszkaiser): this should be consistent with modality.bottom()
      # sampled_frame = common_layers.standardize_images(sampled_frame)
    return tf.to_float(sampled_frame)

  def __get_next_inputs(self, index, all_frames, all_actions, all_rewards):
    """Get inputs for next prediction iteration.

      If the model is recurrent then the inputs of the models are
      the current inputs. For non-recurrent models the input is the
      last N stacked frames/actions/rewards.

    Args:
      index: current prediction index. from 0 to number of target frames.
      all_frames: list of all frames including input and target frames.
      all_actions: list of all actions including input and target actions.
      all_rewards: list of all rewards including input and target rewards.

    Returns:
      frames: input frames for next_frame prediction.
      actions: input actions for next_frame prediction.
      rewards: input rewards for next_frame prediction.
      target_index: index of target frame in all_frames list.
    """
    if self.is_recurrent_model:
      target_index = index + 1
      nones = [None]
    else:
      target_index = index + self.hparams.video_num_input_frames
      nones = [None] * self.hparams.video_num_input_frames

    frames = all_frames[index:target_index]
    actions = all_actions[index:target_index] if self.has_actions else nones
    rewards = all_rewards[index:target_index] if self.has_rewards else nones

    return frames, actions, rewards, target_index

  def infer(self, features, *args, **kwargs):  # pylint: disable=arguments-differ
    """Produce predictions from the model by running it."""
    del args, kwargs
    # Inputs and features preparation needed to handle edge cases.
    if not features:
      features = {}
    hparams = self.hparams
    inputs_old = None
    if "inputs" in features and len(features["inputs"].shape) < 4:
      inputs_old = features["inputs"]
      features["inputs"] = tf.expand_dims(features["inputs"], 2)

    def logits_to_samples(logits, key):
      """Get samples from logits."""
      # If the last dimension is 1 then we're using L1/L2 loss.
      if common_layers.shape_list(logits)[-1] == 1:
        return tf.to_int32(tf.squeeze(logits, axis=-1))
      if key == "targets":
        return pixels_from_softmax(
            logits, gumbel_noise_factor=0.0,
            temperature=hparams.pixel_sampling_temperature)
      # Argmax in TF doesn't handle more than 5 dimensions yet.
      logits_shape = common_layers.shape_list(logits)
      argmax = tf.argmax(tf.reshape(logits, [-1, logits_shape[-1]]), axis=-1)
      return tf.reshape(argmax, logits_shape[:-1])

    # Get predictions.
    try:
      num_channels = hparams.problem.num_channels
    except AttributeError:
      num_channels = 1
    if "inputs" in features:
      inputs_shape = common_layers.shape_list(features["inputs"])
      targets_shape = [inputs_shape[0], hparams.video_num_target_frames,
                       inputs_shape[2], inputs_shape[3], num_channels]
    else:
      tf.logging.warn("Guessing targets shape as no inputs are given.")
      targets_shape = [hparams.batch_size,
                       hparams.video_num_target_frames, 1, 1, num_channels]

    features["targets"] = tf.zeros(targets_shape, dtype=tf.int32)
    reward_in_mod = "target_reward" in self.problem_hparams.modality
    action_in_mod = "target_action" in self.problem_hparams.modality
    if reward_in_mod:
      # TODO(lukaszkaiser): this is a hack. get the actual reward history.
      if "input_reward" not in features:
        features["input_reward"] = tf.zeros(
            [inputs_shape[0], inputs_shape[1], 1], dtype=tf.int32)
      features["target_reward"] = tf.zeros(
          [targets_shape[0], targets_shape[1], 1], dtype=tf.int32)
    if action_in_mod and "target_action" not in features:
      features["target_action"] = tf.zeros(
          [targets_shape[0], targets_shape[1], 1], dtype=tf.int32)
    logits, _ = self(features)  # pylint: disable=not-callable
    if isinstance(logits, dict):
      results = {}
      for k, v in six.iteritems(logits):
        results[k] = logits_to_samples(v, k)
        results["%s_logits" % k] = v
      # HACK: bypassing decoding issues.
      results["outputs"] = results["targets"]
      results["scores"] = results["targets"]
    else:
      results = logits_to_samples(logits, "targets")

    # Restore inputs to not confuse Estimator in edge cases.
    if inputs_old is not None:
      features["inputs"] = inputs_old

    # Return results.
    return results

  def __process(self, all_frames, all_actions, all_rewards, all_raw_frames):
    """Main video processing function."""
    hparams = self.hparams
    all_frames_copy = [tf.identity(frame) for frame in all_frames]
    orig_frame_shape = common_layers.shape_list(all_frames[0])
    batch_size = orig_frame_shape[0]
    ss_func = self.get_scheduled_sample_func(batch_size)
    target_frames = []
    extra_loss = 0.0

    # Any extra info required by the model goes into here.
    video_features = self.video_features(
        all_frames, all_actions, all_rewards, all_raw_frames)

    num_frames = len(all_frames)
    if self.is_recurrent_model:
      input_index_range = range(num_frames - 1)
    else:
      input_index_range = range(hparams.video_num_target_frames)

    # Setup the internal states as well as an auxiliary tf op
    # to enforce syncronization between prediction steps.
    if self.internal_states is None:
      internal_states = None
      sync_op = tf.no_op()
    else:
      internal_states = self.load_internal_states_ops()
      with tf.control_dependencies(flat_lists(internal_states)):
        sync_op = tf.no_op()

    res_frames, sampled_frames, res_rewards, res_policies, res_values = \
        [], [], [], [], []
    for i in input_index_range:
      with tf.control_dependencies([sync_op]):
        frames, actions, rewards, target_index = self.__get_next_inputs(
            i, all_frames, all_actions, all_rewards)
        target_frame = all_frames[target_index]
        target_frames.append(tf.identity(target_frame))

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
          float_frames = [tf.to_float(frame) for frame in frames]
          func_out = self.next_frame(
              float_frames, actions, rewards, tf.to_float(target_frame),
              internal_states, video_features)
          res_frame, res_reward, res_policy, res_value, res_extra_loss, \
              internal_states = func_out
          res_frames.append(res_frame)
          res_rewards.append(res_reward)
          res_policies.append(res_policy)
          res_values.append(res_value)
          extra_loss += res_extra_loss / float(len(input_index_range))

          # Syncronizing the internals states
          # Some Tensflow Magic to make sure everything happens as it should.
          with tf.control_dependencies([res_frame]):
            sync_op = tf.no_op()
            if self.is_predicting and self.is_recurrent_model and i == 0:
              # The internal state save happens at the end of the 1st iteration
              # which essentially allows recurrent models to continue
              # running after one prediction.
              # Necessary for planning/rl applications.
              save_ops = self.save_internal_states_ops(internal_states)
              with tf.control_dependencies(flat_lists(save_ops)):
                sync_op = tf.no_op()

        # Only for Softmax loss: sample frame so we can keep iterating.
        sampled_frame = self.get_sampled_frame(res_frame)
        sampled_frames.append(sampled_frame)

        # Check whether we are done with context frames or not
        if self.is_recurrent_model:
          done_warm_start = (i >= hparams.video_num_input_frames - 1)
        else:
          done_warm_start = True  # Always true for non-reccurent networks.

        if self.is_predicting and done_warm_start:
          all_frames[target_index] = sampled_frame

        # Scheduled sampling during training.
        if self.is_training:
          groundtruth_items = [tf.to_float(target_frame)]
          generated_items = [sampled_frame]
          ss_frame, = self.get_scheduled_sample_inputs(
              done_warm_start, groundtruth_items, generated_items, ss_func)
          all_frames[target_index] = ss_frame

    video_extra_loss = self.video_extra_loss(
        sampled_frames, target_frames, internal_states, video_features)
    tf.summary.scalar("video_extra_loss", video_extra_loss)
    extra_loss += video_extra_loss

    if self.is_recurrent_model:
      has_input_predictions = hparams.video_num_input_frames > 1
      if self.is_training and hparams.internal_loss and has_input_predictions:
        # add the loss for input frames as well.
        extra_gts = all_frames_copy[1:hparams.video_num_input_frames]
        extra_raw_gts = all_raw_frames[1:hparams.video_num_input_frames]
        extra_pds = res_frames[:hparams.video_num_input_frames-1]
        recon_loss = self.get_extra_internal_loss(
            extra_raw_gts, extra_gts, extra_pds)
        extra_loss += recon_loss
      # Cut the predicted input frames.
      res_frames = res_frames[hparams.video_num_input_frames-1:]
      res_rewards = res_rewards[hparams.video_num_input_frames-1:]
      res_policies = res_policies[hparams.video_num_input_frames-1:]
      res_values = res_values[hparams.video_num_input_frames-1:]
      sampled_frames = sampled_frames[hparams.video_num_input_frames-1:]
      target_frames = target_frames[hparams.video_num_input_frames-1:]

    self.visualize_predictions(
        sampled_frames, [tf.to_float(f) for f in target_frames])

    output_frames = tf.stack(res_frames, axis=1)
    targets = output_frames

    if any((self.has_rewards, self.has_policies, self.has_values)):
      targets = {"targets": output_frames}
      if self.has_rewards:
        targets["target_reward"] = tf.stack(res_rewards, axis=1)
      if self.has_policies:
        targets["target_policy"] = tf.stack(res_policies, axis=1)
      if self.has_values:
        targets["target_value"] = tf.stack(res_values, axis=1)

    return targets, extra_loss

  def loss(self, *args, **kwargs):
    if "policy_network" in self.hparams.values():
      return 0.0
    else:
      return super(NextFrameBase, self).loss(*args, **kwargs)

  def body(self, features):
    self.has_actions = "input_action" in features
    self.has_rewards = "target_reward" in features
    self.has_policies = "target_policy" in features
    self.has_values = "target_value" in features
    hparams = self.hparams

    def merge(inputs, targets):
      """Split inputs and targets into lists."""
      inputs = tf.unstack(inputs, axis=1)
      targets = tf.unstack(targets, axis=1)
      assert len(inputs) == hparams.video_num_input_frames
      assert len(targets) == hparams.video_num_target_frames
      return inputs + targets

    frames = merge(features["inputs"], features["targets"])
    frames_raw = merge(features["inputs_raw"], features["targets_raw"])
    actions, rewards = None, None
    if self.has_actions:
      actions = merge(features["input_action"], features["target_action"])
    if self.has_rewards:
      rewards = merge(features["input_reward"], features["target_reward"])

    # Reset the internal states if the reset_internal_states has been
    # passed as a feature and has greater value than 0.
    if self.is_recurrent_model and self.internal_states is not None:
      def reset_func():
        reset_ops = flat_lists(self.reset_internal_states_ops())
        with tf.control_dependencies(reset_ops):
          return tf.no_op()
      if self.is_predicting and "reset_internal_states" in features:
        reset = features["reset_internal_states"]
        reset = tf.greater(tf.reduce_sum(reset), 0.5)
        reset_ops = tf.cond(reset, reset_func, tf.no_op)
      else:
        reset_ops = tf.no_op()
      with tf.control_dependencies([reset_ops]):
        frames[0] = tf.identity(frames[0])

    with tf.control_dependencies([frames[0]]):
      return self.__process(frames, actions, rewards, frames_raw)


def next_frame_base():
  """Common HParams for next_frame models."""
  hparams = common_hparams.basic_params1()
  # Loss cutoff.
  hparams.add_hparam("video_modality_loss_cutoff", 0.01)
  # Additional resizing the frames before feeding them to model.
  hparams.add_hparam("preprocess_resize_frames", None)
  # How many data points to suffle. Ideally should be part of problem not model!
  hparams.add_hparam("shuffle_buffer_size", 128)
  # Tiny mode. For faster tests.
  hparams.add_hparam("tiny_mode", False)
  # In case a model supports smaller/faster version.
  hparams.add_hparam("small_mode", False)
  # In case a model has stochastic version.
  hparams.add_hparam("stochastic_model", False)
  # Internal loss for recurrent models.
  hparams.add_hparam("internal_loss", True)
  # choose from: concat, multiplicative, multi_additive
  hparams.add_hparam("action_injection", "multi_additive")
  # Scheduled sampling method. Choose between
  # ground_truth_only, prediction_only, prob, count, prob_inverse_exp.
  hparams.add_hparam("scheduled_sampling_mode", "prediction_only")
  hparams.add_hparam("scheduled_sampling_decay_steps", 10000)
  hparams.add_hparam("scheduled_sampling_max_prob", 1.0)
  hparams.add_hparam("scheduled_sampling_k", 900.0)
  return hparams
