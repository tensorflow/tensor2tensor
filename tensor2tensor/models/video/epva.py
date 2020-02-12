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

"""Model architecture for video prediction model.

based on following paper:
"Hierarchical Long-term Video Prediction without Supervision"
http://web.eecs.umich.edu/~honglak/icml2018-unsupHierarchicalVideoPred.pdf
by Nevan Wichers, Ruben Villegas, Dumitru Erhan and Honglak Lee.

This code is based on the original code:
https://github.com/brain-research/long-term-video-prediction-without-supervision
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import reduce

from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_video
from tensor2tensor.models.video import epva_params  # pylint: disable=unused-import
from tensor2tensor.models.video import sv2p
from tensor2tensor.utils import contrib
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf

from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.slim.python.slim.nets import vgg

tfl = tf.layers
tfcl = contrib.layers()

IMG_WIDTH = 64
IMG_HEIGHT = 64
VGG_IMAGE_SIZE = 224
COLOR_NORMALIZATION_VECTOR = [123.68, 116.78, 103.94]


def van_image_enc_2d(x, first_depth, reuse=False, hparams=None):
  """The image encoder for the VAN.

  Similar architecture as Ruben's paper
  (http://proceedings.mlr.press/v70/villegas17a/villegas17a.pdf).

  Args:
    x: The image to encode.
    first_depth: The depth of the first layer. Depth is increased in subsequent
      layers.
    reuse: To reuse in variable scope or not.
    hparams: The python hparams.

  Returns:
    The encoded image.
  """
  with tf.variable_scope('van_image_enc', reuse=reuse):
    enc_history = [x]

    enc = tf.layers.conv2d(
        x, first_depth, 3, padding='same', activation=tf.nn.relu, strides=1)
    enc = contrib.layers().layer_norm(enc)
    enc = tf.layers.conv2d(
        enc, first_depth, 3, padding='same', activation=tf.nn.relu, strides=1)
    enc = tf.nn.max_pool(enc, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    enc = tf.nn.dropout(enc, hparams.van_keep_prob)
    enc = contrib.layers().layer_norm(enc)
    enc_history.append(enc)

    enc = tf.layers.conv2d(
        enc,
        first_depth * 2,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    enc = tf.layers.conv2d(
        enc,
        first_depth * 2,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    enc = tf.nn.max_pool(enc, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    enc = tf.nn.dropout(enc, hparams.van_keep_prob)
    enc = contrib.layers().layer_norm(enc)
    enc_history.append(enc)

    enc = tf.layers.conv2d(
        enc,
        first_depth * 4,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    enc = tf.layers.conv2d(
        enc,
        first_depth * 4,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    enc = tf.layers.conv2d(
        enc,
        first_depth * 4,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    enc = tf.nn.max_pool(enc, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    return enc, enc_history


def van_enc_2d(x, first_depth, reuse=False):
  """The higher level structure encoder for the VAN.

  The high level structure is a vector instead of an image.

  Args:
    x: The higher level structure to encode.
    first_depth: The depth of the first layer. Depth is increased in subsequent
      layers.
    reuse: To reuse in variable scope or not.

  Returns:
    The encoded image.
  """
  with tf.variable_scope('van_enc', reuse=reuse):
    a = 4  # depends on the inputs size
    b = 4
    # a, b = 4,4
    enc = tf.nn.relu(x)
    enc = tf.layers.dense(enc, first_depth * a * b, tf.nn.relu)
    enc = contrib.layers().layer_norm(enc)

    enc = tf.reshape(enc, [-1, a, b, first_depth])

    enc = tf.layers.conv2d_transpose(
        enc, first_depth, 3, padding='same', activation=tf.nn.relu, strides=1)
    enc = contrib.layers().layer_norm(enc)
    enc = tf.layers.conv2d_transpose(
        enc,
        first_depth * 2,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=2)
    van_higher_level_2 = tf.reshape(enc, [-1, a * 2 * b * 2 * first_depth * 2])

    enc = tf.layers.conv2d_transpose(
        enc,
        first_depth * 2,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    enc = contrib.layers().layer_norm(enc)
    enc = tf.layers.conv2d_transpose(
        enc,
        first_depth * 4,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    van_higher_level_4 = tf.reshape(enc, [-1, a * 2 * b * 2 * first_depth * 4])

    van_higher_level = tf.concat([x, van_higher_level_2, van_higher_level_4], 1)

    return enc, van_higher_level


def van_dec_2d(x, skip_connections, output_shape, first_depth, hparams=None):
  """The VAN decoder.

  Args:
    x: The analogy information to decode.
    skip_connections: The encoder layers which can be used as skip connections.
    output_shape: The shape of the desired output image.
    first_depth: The depth of the first layer of the van image encoder.
    hparams: The python hparams.

  Returns:
    The decoded image prediction.
  """
  with tf.variable_scope('van_dec'):
    dec = tf.layers.conv2d_transpose(
        x, first_depth * 4, 3, padding='same', activation=tf.nn.relu, strides=2)
    dec = tf.nn.dropout(dec, hparams.van_keep_prob)
    dec = contrib.layers().layer_norm(dec)
    dec = tf.layers.conv2d_transpose(
        dec,
        first_depth * 4,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    dec = tf.nn.dropout(dec, hparams.van_keep_prob)
    dec = tf.layers.conv2d_transpose(
        dec,
        first_depth * 2,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    dec = tf.nn.dropout(dec, hparams.van_keep_prob)
    dec = contrib.layers().layer_norm(dec)

    dec = tf.layers.conv2d_transpose(
        dec,
        first_depth * 2,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=2)
    dec = tf.nn.dropout(dec, hparams.van_keep_prob)
    dec = tf.layers.conv2d_transpose(
        dec, first_depth, 3, padding='same', activation=tf.nn.relu, strides=1)
    dec = tf.nn.dropout(dec, hparams.van_keep_prob)
    dec = contrib.layers().layer_norm(dec)

    dec = tf.layers.conv2d_transpose(
        dec,
        output_shape[3] + 1,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=2)
    dec = tf.nn.dropout(dec, hparams.van_keep_prob)

    out_mask = tf.layers.conv2d_transpose(
        dec, output_shape[3] + 1, 3, strides=1, padding='same', activation=None)

    mask = tf.nn.sigmoid(out_mask[:, :, :, 3:4])
    out = out_mask[:, :, :, :3]

    return out * mask + skip_connections[0] * (1 - mask)


def analogy_computation_2d(f_first_enc,
                           f_first_frame,
                           f_current_enc,
                           first_depth):
  """Implements the deep analogy computation."""
  with tf.variable_scope('analogy_computation'):

    frame_enc_diff = f_first_frame - f_first_enc

    frame_enc_diff_enc = tf.layers.conv2d(
        frame_enc_diff,
        first_depth * 4,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    f_current_enc_enc = tf.layers.conv2d(
        f_current_enc,
        first_depth * 4,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)

    analogy = tf.concat([frame_enc_diff_enc, f_current_enc_enc], 3)
    analogy = tf.layers.conv2d(
        analogy,
        first_depth * 4,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    analogy = contrib.layers().layer_norm(analogy)
    analogy = tf.layers.conv2d(
        analogy,
        first_depth * 4,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)
    return tf.layers.conv2d(
        analogy,
        first_depth * 4,
        3,
        padding='same',
        activation=tf.nn.relu,
        strides=1)


def van(first_enc,
        first_frame,
        current_enc,
        gt_image,
        reuse=False,
        scope_prefix='',
        hparams=None):
  """Implements a VAN.

  Args:
    first_enc: The first encoding.
    first_frame: The first ground truth frame.
    current_enc: The encoding of the frame to generate.
    gt_image: The ground truth image, only used for regularization.
    reuse: To reuse in variable scope or not.
    scope_prefix: The prefix before the scope name.
    hparams: The python hparams.

  Returns:
    The generated image.
  """
  with tf.variable_scope(scope_prefix + 'van', reuse=reuse):
    output_shape = first_frame.get_shape().as_list()
    output_shape[0] = -1

    first_depth = 64

    f_first_enc, _ = van_enc_2d(first_enc, first_depth)
    f_first_frame, image_enc_history = van_image_enc_2d(
        first_frame, first_depth, hparams=hparams)
    f_current_enc, van_higher_level = van_enc_2d(
        current_enc, first_depth, reuse=True)
    f_gt_image, _ = van_image_enc_2d(gt_image, first_depth, True,
                                     hparams=hparams)

    analogy_t = analogy_computation_2d(
        f_first_enc, f_first_frame, f_current_enc, first_depth)
    enc_img = f_current_enc + analogy_t

    img = van_dec_2d(
        enc_img, image_enc_history, output_shape, first_depth, hparams=hparams)

    batch_size = tf.to_float(tf.shape(first_enc)[0])
    r_loss = tf.nn.l2_loss(f_gt_image - f_current_enc - analogy_t) / batch_size

    return img, r_loss, van_higher_level


def encoder_vgg(x, enc_final_size, reuse=False, scope_prefix='', hparams=None,
                is_training=True):
  """VGG network to use as encoder without the top few layers.

  Can be pretrained.

  Args:
    x: The image to encode. In the range 0 to 1.
    enc_final_size: The desired size of the encoding.
    reuse: To reuse in variable scope or not.
    scope_prefix: The prefix before the scope name.
    hparams: The python hparams.
    is_training: boolean value indicating if training is happening.

  Returns:
    The generated image.
  """
  with tf.variable_scope(scope_prefix + 'encoder', reuse=reuse):

    # Preprocess input
    x *= 256
    x = x - COLOR_NORMALIZATION_VECTOR

    with arg_scope(vgg.vgg_arg_scope()):
      # Padding because vgg_16 accepts images of size at least VGG_IMAGE_SIZE.
      x = tf.pad(x, [[0, 0], [0, VGG_IMAGE_SIZE - IMG_WIDTH],
                     [0, VGG_IMAGE_SIZE - IMG_HEIGHT], [0, 0]])
      _, end_points = vgg.vgg_16(
          x,
          num_classes=enc_final_size,
          is_training=is_training)
      pool5_key = [key for key in end_points.keys() if 'pool5' in key]
      assert len(pool5_key) == 1
      enc = end_points[pool5_key[0]]
      # Undoing padding.
      enc = tf.slice(enc, [0, 0, 0, 0], [-1, 2, 2, -1])

    enc_shape = enc.get_shape().as_list()
    enc_shape[0] = -1
    enc_size = enc_shape[1] * enc_shape[2] * enc_shape[3]

    enc_flat = tf.reshape(enc, (-1, enc_size))
    enc_flat = tf.nn.dropout(enc_flat, hparams.enc_keep_prob)

    enc_flat = tf.layers.dense(
        enc_flat,
        enc_final_size,
        kernel_initializer=tf.truncated_normal_initializer(stddev=1e-4,))

    if hparams.enc_pred_use_l2norm:
      enc_flat = tf.nn.l2_normalize(enc_flat, 1)

  return enc_flat


def predictor(enc_flat,
              action,
              lstm_states,
              pred_depth,
              reuse=False,
              scope_prefix='',
              hparams=None):
  """LSTM predictor network."""
  with tf.variable_scope(scope_prefix + 'predict', reuse=reuse):

    enc_final_size = enc_flat.get_shape().as_list()[1]
    action_size = action.get_shape().as_list()[1]
    initial_size = (enc_final_size + action_size)

    batch_size = tf.shape(enc_flat)[0]

    init_stddev = 1e-2

    pre_pred = tf.concat([enc_flat, action], 1)
    pre_pred = tf.layers.dense(
        pre_pred,
        initial_size,
        kernel_initializer=tf.truncated_normal_initializer(stddev=init_stddev))

    # This is only needed or the GAN version.
    if hparams.pred_noise_std > 0:
      # Add the noise like this so a pretrained model can be used.
      pred_noise = tf.random_normal(
          shape=[batch_size, 100], stddev=hparams.pred_noise_std)
      pre_pred += tf.layers.dense(
          pred_noise,
          initial_size,
          kernel_initializer=tf.truncated_normal_initializer(
              stddev=init_stddev),
          name='noise_dense')

    pre_pred = tf.nn.relu(pre_pred)

    if lstm_states[pred_depth - 2] is None:
      back_connect = tf.tile(
          tf.get_variable(
              'back_connect_init',
              shape=[1, initial_size * 2],
              initializer=tf.truncated_normal_initializer(stddev=init_stddev))
          , (batch_size, 1))
    else:
      back_connect = lstm_states[pred_depth - 2]

    lstm_init_stddev = 1e-4

    part_pred, lstm_states[0] = common_video.lstm_cell(
        tf.concat([pre_pred, back_connect], 1),
        lstm_states[0],
        initial_size,
        use_peepholes=True,
        initializer=tf.truncated_normal_initializer(stddev=lstm_init_stddev),
        num_proj=initial_size)
    part_pred = contrib.layers().layer_norm(part_pred)
    pred = part_pred

    for pred_layer_num in range(1, pred_depth, 2):
      part_pred, lstm_states[pred_layer_num] = common_video.lstm_cell(
          pred,
          lstm_states[pred_layer_num],
          initial_size,
          use_peepholes=True,
          initializer=tf.truncated_normal_initializer(stddev=lstm_init_stddev),
          num_proj=initial_size)
      pred += part_pred

      part_pred, lstm_states[pred_layer_num + 1] = common_video.lstm_cell(
          tf.concat([pred, pre_pred], 1),
          lstm_states[pred_layer_num + 1],
          initial_size,
          use_peepholes=True,
          initializer=tf.truncated_normal_initializer(stddev=lstm_init_stddev),
          num_proj=initial_size)
      part_pred = contrib.layers().layer_norm(part_pred)
      pred += part_pred

    pred = tf.layers.dense(
        pred,
        enc_final_size,
        kernel_initializer=tf.truncated_normal_initializer(stddev=init_stddev))

    if hparams.enc_pred_use_l2norm:
      pred = tf.nn.l2_normalize(pred, 1)

    return pred


def construct_model(images,
                    actions=None,
                    context_frames=2,
                    hparams=None,
                    is_training=True):
  """Constructs the tensorflow graph of the hierarchical model."""

  pred_depth = 20

  enc_out_all, pred_out_all, van_out_all, van_on_enc_all = [], [], [], []

  lstm_states = [None] * (pred_depth + 2)

  enc_out = encoder_vgg(
      images[0], hparams.enc_size, False, scope_prefix='timestep/',
      hparams=hparams, is_training=is_training)
  enc_out = tf.identity(enc_out, 'enc_out')
  enc_out_all.append(enc_out)

  num_timesteps = len(actions) - 1
  sum_freq = int(num_timesteps / 4 + 1)

  reuse = False
  for timestep, action in zip(range(len(actions) - 1), actions[:-1]):
    done_warm_start = timestep > context_frames - 1

    with tf.variable_scope('timestep', reuse=reuse):
      if done_warm_start:
        pred_input = pred_out_all[-1]
      else:
        pred_input = enc_out_all[-1]
      pred_out = predictor(
          pred_input, action, lstm_states, pred_depth, False, hparams=hparams)
      pred_out = tf.identity(pred_out, 'pred_out')
      if timestep % sum_freq == 0:  # and not hparams.use_tpu:
        tf.summary.histogram('pred_out', pred_out)
      pred_out_all.append(pred_out)

      if timestep % sum_freq == 0:  # and not hparams.use_tpu:
        tf.summary.histogram('lstm_state', lstm_states[0])
      van_out, _, _ = van(
          enc_out_all[0],
          images[0],
          pred_out,
          images[timestep + 1],
          tf.AUTO_REUSE,
          hparams=hparams)
      van_out = tf.identity(van_out, 'van_out')
      van_out_all.append(van_out)

      enc_out = encoder_vgg(
          images[timestep + 1], hparams.enc_size, True, hparams=hparams,
          is_training=is_training)
      enc_out = tf.identity(enc_out, 'enc_out')
      if timestep % sum_freq == 0:  # and not hparams.use_tpu:
        tf.summary.histogram('enc_out', enc_out)
      enc_out_all.append(enc_out)

      van_input = images[0]
      enc_noise = tf.zeros_like(enc_out)
      if timestep % sum_freq == 0:  # and not hparams.use_tpu:
        tf.summary.histogram('enc_noise', enc_noise)
      van_on_enc, _, _ = van(
          enc_out_all[0],
          van_input,
          enc_out + enc_noise,
          images[timestep + 1],
          tf.AUTO_REUSE,
          hparams=hparams)
      van_on_enc = tf.identity(van_on_enc, 'van_on_enc')
      van_on_enc_all.append(van_on_enc)

      reuse = True

  return enc_out_all, pred_out_all, van_out_all, van_on_enc_all


def peak_signal_to_noise_ratio(true, pred):
  """Image quality metric based on maximal signal power vs. power of the noise.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    peak signal to noise ratio (PSNR)
  """
  return 10.0 * tf.log(1.0 / mean_squared_error(true, pred)) / tf.log(10.0)


def mean_squared_error(true, pred):
  """L2 distance between tensors true and pred.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
  result = tf.reduce_sum(
      tf.squared_difference(true, pred)) / tf.to_float(tf.size(pred))
  return result


def l1_error(true, pred):
  """L1 distance between tensors true and pred."""
  return tf.reduce_sum(tf.abs(true - pred)) / tf.to_float(tf.size(pred))


def calc_loss_psnr(gen_images, images, name, hparams=None, use_l1_loss=False):
  """Calculates loss and psnr for predictions over multiple timesteps."""
  del hparams
  with tf.name_scope(name):
    loss, error, psnr_all = 0.0, 0.0, 0.0
    for _, x, gx in zip(range(len(gen_images)), images, gen_images):
      recon_cost = mean_squared_error(x, gx)
      if use_l1_loss:
        recon_cost = l1_error(x, gx)

      error_i = l1_error(x, gx)
      psnr_i = peak_signal_to_noise_ratio(x, gx)
      psnr_all += psnr_i
      error += error_i
      loss += recon_cost

    psnr_all /= tf.to_float(len(gen_images))
    loss /= tf.to_float(len(gen_images))
    error /= tf.to_float(len(gen_images))

    # if not hparams.use_tpu:
    tf.summary.scalar('psnr_all', psnr_all)
    tf.summary.scalar('loss', loss)

    return loss, psnr_all


@registry.register_model
class NextFrameEpva(sv2p.NextFrameSv2pLegacy):
  """Hierarchical Long-term Video Prediction without Supervision"""

  def body(self, features):
    hparams = self.hparams
    input_shape = common_layers.shape_list(features['inputs'])
    batch_size, _, frame_width, frame_height, frame_channels = input_shape  # pylint: disable=unused-variable

    # Swap time and batch axes.
    input_frames = common_video.swap_time_and_batch_axes(
        tf.to_float(features['inputs']))
    target_frames = common_video.swap_time_and_batch_axes(features['targets'])

    # Get actions if exist otherwise use zeros
    input_actions = self.get_input_if_exists(
        features, 'input_action', batch_size, hparams.video_num_input_frames)
    target_actions = self.get_input_if_exists(
        features, 'target_action', batch_size, hparams.video_num_target_frames)

    # Get rewards if exist otherwise use zeros
    # TODO(blazej) enable rewards.
    # input_rewards = self.get_input_if_exists(
    #     features, 'input_reward', batch_size, hparams.video_num_input_frames)
    # target_rewards = self.get_input_if_exists(
    #     features, 'target_reward', batch_size,hparams.video_num_target_frames)
    # all_rewards = tf.concat([input_rewards, target_rewards], axis=0)

    all_actions = tf.concat([input_actions, target_actions], axis=0)
    # flatten actions tensor to have the shape: framesXbatch_sizeXaction_dims.
    actions_shape = common_layers.shape_list(all_actions)
    all_actions = tf.reshape(
        all_actions,
        [actions_shape[0], -1,
         reduce(lambda x, y: x * y, actions_shape[2:])])
    all_frames = tf.concat([input_frames, target_frames], axis=0)

    all_frames = tf.unstack(all_frames, axis=0)
    all_actions = tf.unstack(all_actions, axis=0)

    # TODO(blazej) - most likely this downsize is too strong.
    all_frames = [
        tf.image.resize_images(
            image, (IMG_HEIGHT, IMG_WIDTH),
            method=tf.image.ResizeMethod.BICUBIC)
        for image in all_frames
    ]

    enc_out_all, pred_out_all, _, van_on_enc_all = construct_model(
        all_frames,
        all_actions,
        context_frames=hparams.context_frames,
        hparams=hparams,
        is_training=self.is_training)

    enc_pred_loss, _ = calc_loss_psnr(
        enc_out_all[1:],
        pred_out_all,
        'enc_pred_loss',
        hparams=hparams,
        use_l1_loss=hparams.enc_pred_use_l1_loss)

    van_on_enc_loss, _ = calc_loss_psnr(
        van_on_enc_all,
        all_frames[1:],
        'van_on_enc_loss',
        hparams=hparams)

    enc_pred_loss_scale_delay = max(hparams.enc_pred_loss_scale_delay, 1)
    enc_pred_loss_scale = tf.nn.sigmoid(
        (tf.to_float(tf.train.get_or_create_global_step()
                    ) - enc_pred_loss_scale_delay) /
        (enc_pred_loss_scale_delay * .1)) * hparams.enc_pred_loss_scale
    tf.summary.scalar('enc_pred_loss_scale', enc_pred_loss_scale)
    epva_loss = enc_pred_loss * enc_pred_loss_scale + van_on_enc_loss
    tf.summary.scalar('epva_loss', epva_loss)

    predictions = tf.stack(van_on_enc_all)

    if hparams.clip_pixel_values:
      predictions = tf.clip_by_value(predictions, 0.0, 1.0)

    # TODO(mbz): clean this up!
    def fix_video_dims_and_concat_on_x_axis(x):
      x = tf.transpose(x, [1, 3, 4, 0, 2])
      x = tf.reshape(x, [batch_size, frame_height, frame_channels, -1])
      x = tf.transpose(x, [0, 3, 1, 2])
      return x

    frames_gd = fix_video_dims_and_concat_on_x_axis(target_frames)
    frames_pd = fix_video_dims_and_concat_on_x_axis(predictions)
    side_by_side_video = tf.concat([frames_gd, frames_pd], axis=1)
    tf.summary.image('full_video', side_by_side_video)

    predictions = tf.unstack(predictions)
    predictions = [
        tf.image.resize_images(
            image, (frame_width, frame_height),
            method=tf.image.ResizeMethod.BICUBIC)
        for image in predictions
    ]
    predictions = tf.stack(predictions)

    predictions = common_video.swap_time_and_batch_axes(predictions)
    predictions = tf.slice(predictions,
                           [0, hparams.video_num_input_frames-1, 0, 0, 0],
                           [-1]*5)

    return predictions, {'extra': epva_loss}
