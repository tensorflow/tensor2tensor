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
"""Layers common to multiple models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers
import tensorflow as tf

tfl = tf.layers
tfcl = tf.contrib.layers


def swap_time_and_batch_axes(inputs):
  """Swaps time and batch axis (the first two axis)."""
  transposed_axes = tf.concat([[1, 0], tf.range(2, tf.rank(inputs))], axis=0)
  return tf.transpose(inputs, transposed_axes)


def encode_to_shape(inputs, shape, scope):
  """Encode the given tensor to given image shape."""
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    w, h = shape[1], shape[2]
    x = inputs
    x = tf.contrib.layers.flatten(x)
    x = tfl.dense(x, w * h, activation=None, name="enc_dense")
    x = tf.reshape(x, (-1, w, h, 1))
    return x


def decode_to_shape(inputs, shape, scope):
  """Encode the given tensor to given image shape."""
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    x = inputs
    x = tf.contrib.layers.flatten(x)
    x = tfl.dense(x, shape[2], activation=None, name="dec_dense")
    x = tf.expand_dims(x, axis=1)
    return x


def basic_lstm(inputs, state, num_units, name=None):
  """Basic LSTM."""
  input_shape = common_layers.shape_list(inputs)
  cell = tf.contrib.rnn.BasicLSTMCell(num_units, name=name)
  if state is None:
    state = cell.zero_state(input_shape[0], tf.float32)
  outputs, new_state = cell(inputs, state)
  return outputs, new_state


def conv_lstm_2d(inputs, state, output_channels,
                 kernel_size=5, name=None, spatial_dims=None):
  """2D Convolutional LSTM."""
  input_shape = common_layers.shape_list(inputs)
  batch_size, input_channels = input_shape[0], input_shape[-1]
  if spatial_dims is None:
    input_shape = input_shape[1:]
  else:
    input_shape = spatial_dims + [input_channels]

  cell = tf.contrib.rnn.ConvLSTMCell(
      2, input_shape, output_channels,
      [kernel_size, kernel_size], name=name)
  if state is None:
    state = cell.zero_state(batch_size, tf.float32)
  outputs, new_state = cell(inputs, state)
  return outputs, new_state


def scheduled_sample_count(ground_truth_x,
                           generated_x,
                           batch_size,
                           scheduled_sample_var):
  """Sample batch with specified mix of groundtruth and generated data points.

  Args:
    ground_truth_x: tensor of ground-truth data points.
    generated_x: tensor of generated data points.
    batch_size: batch size
    scheduled_sample_var: number of ground-truth examples to include in batch.
  Returns:
    New batch with num_ground_truth sampled from ground_truth_x and the rest
    from generated_x.
  """
  num_ground_truth = scheduled_sample_var
  idx = tf.random_shuffle(tf.range(batch_size))
  ground_truth_idx = tf.gather(idx, tf.range(num_ground_truth))
  generated_idx = tf.gather(idx, tf.range(num_ground_truth, batch_size))

  ground_truth_examps = tf.gather(ground_truth_x, ground_truth_idx)
  generated_examps = tf.gather(generated_x, generated_idx)
  return tf.dynamic_stitch([ground_truth_idx, generated_idx],
                           [ground_truth_examps, generated_examps])


def scheduled_sample_prob(ground_truth_x,
                          generated_x,
                          batch_size,
                          scheduled_sample_var):
  """Probability based scheduled sampling.

  Args:
    ground_truth_x: tensor of ground-truth data points.
    generated_x: tensor of generated data points.
    batch_size: batch size
    scheduled_sample_var: probability of choosing from ground_truth.
  Returns:
    New batch with randomly selected data points.
  """
  probability_threshold = scheduled_sample_var
  probability_of_generated = tf.random_uniform([batch_size])
  array_ind = tf.to_int32(probability_of_generated > probability_threshold)
  indices = tf.range(batch_size) + array_ind * batch_size
  xy = tf.concat([ground_truth_x, generated_x], axis=0)
  output = tf.gather(xy, indices)
  return output


def dna_transformation(prev_image, dna_input, dna_kernel_size, relu_shift):
  """Apply dynamic neural advection to previous image.

  Args:
    prev_image: previous image to be transformed.
    dna_input: hidden lyaer to be used for computing DNA transformation.
    dna_kernel_size: dna kernel size.
    relu_shift: shift for ReLU function.
  Returns:
    List of images transformed by the predicted CDNA kernels.
  """
  # Construct translated images.
  prev_image_pad = tf.pad(prev_image, [[0, 0], [2, 2], [2, 2], [0, 0]])
  image_height = int(prev_image.get_shape()[1])
  image_width = int(prev_image.get_shape()[2])

  inputs = []
  for xkern in range(dna_kernel_size):
    for ykern in range(dna_kernel_size):
      inputs.append(
          tf.expand_dims(
              tf.slice(prev_image_pad, [0, xkern, ykern, 0],
                       [-1, image_height, image_width, -1]), [3]))
  inputs = tf.concat(axis=3, values=inputs)

  # Normalize channels to 1.
  kernel = tf.nn.relu(dna_input - relu_shift) + relu_shift
  kernel = tf.expand_dims(
      kernel / tf.reduce_sum(kernel, [3], keep_dims=True), [4])
  return tf.reduce_sum(kernel * inputs, [3], keep_dims=False)


def cdna_transformation(prev_image, cdna_input, num_masks, color_channels,
                        dna_kernel_size, relu_shift):
  """Apply convolutional dynamic neural advection to previous image.

  Args:
    prev_image: previous image to be transformed.
    cdna_input: hidden lyaer to be used for computing CDNA kernels.
    num_masks: number of masks and hence the number of CDNA transformations.
    color_channels: the number of color channels in the images.
    dna_kernel_size: dna kernel size.
    relu_shift: shift for ReLU function.
  Returns:
    List of images transformed by the predicted CDNA kernels.
  """
  batch_size = tf.shape(cdna_input)[0]
  height = int(prev_image.get_shape()[1])
  width = int(prev_image.get_shape()[2])

  # Predict kernels using linear function of last hidden layer.
  cdna_kerns = tfl.dense(
      cdna_input, dna_kernel_size * dna_kernel_size * num_masks,
      name="cdna_params",
      activation=None)

  # Reshape and normalize.
  cdna_kerns = tf.reshape(
      cdna_kerns, [batch_size, dna_kernel_size, dna_kernel_size, 1, num_masks])
  cdna_kerns = (tf.nn.relu(cdna_kerns - relu_shift) + relu_shift)
  norm_factor = tf.reduce_sum(cdna_kerns, [1, 2, 3], keep_dims=True)
  cdna_kerns /= norm_factor

  # Treat the color channel dimension as the batch dimension since the same
  # transformation is applied to each color channel.
  # Treat the batch dimension as the channel dimension so that
  # depthwise_conv2d can apply a different transformation to each sample.
  cdna_kerns = tf.transpose(cdna_kerns, [1, 2, 0, 4, 3])
  cdna_kerns = tf.reshape(
      cdna_kerns, [dna_kernel_size, dna_kernel_size, batch_size, num_masks])
  # Swap the batch and channel dimensions.
  prev_image = tf.transpose(prev_image, [3, 1, 2, 0])

  # Transform image.
  transformed = tf.nn.depthwise_conv2d(
      prev_image, cdna_kerns, [1, 1, 1, 1], "SAME")

  # Transpose the dimensions to where they belong.
  transformed = tf.reshape(
      transformed, [color_channels, height, width, batch_size, num_masks])
  transformed = tf.transpose(transformed, [3, 1, 2, 0, 4])
  transformed = tf.unstack(transformed, axis=-1)
  return transformed


def vgg_layer(inputs,
              nout,
              kernel_size=3,
              activation=tf.nn.leaky_relu,
              padding="SAME",
              is_training=False,
              scope=None):
  """A layer of VGG network with batch norm.

  Args:
    inputs: image tensor
    nout: number of output channels
    kernel_size: size of the kernel
    activation: activation function
    padding: padding of the image
    is_training: whether it is training mode or not
    scope: variable scope of the op
  Returns:
    net: output of layer
  """
  with tf.variable_scope(scope):
    net = tfl.conv2d(inputs, nout, kernel_size=kernel_size, padding=padding,
                     activation=None, name="conv")
    net = tfl.batch_normalization(net, training=is_training, name="bn")
    net = activation(net)
  return net


def tile_and_concat(image, latent, concat_latent=True):
  """Tile latent and concatenate to image across depth.

  Args:
    image: 4-D Tensor, (batch_size X height X width X channels)
    latent: 2-D Tensor, (batch_size X latent_dims)
    concat_latent: If set to False, the image is returned as is.

  Returns:
    concat_latent: 4-D Tensor, (batch_size X height X width X channels+1)
      latent tiled and concatenated to the image across the channels.
  """
  if not concat_latent:
    return image
  image_shape = common_layers.shape_list(image)
  latent_shape = common_layers.shape_list(latent)
  height, width = image_shape[1], image_shape[2]
  latent_dims = latent_shape[1]

  height_multiples = height // latent_dims
  pad = height - (height_multiples * latent_dims)
  latent = tf.reshape(latent, (-1, latent_dims, 1, 1))
  latent = tf.tile(latent, (1, height_multiples, width, 1))
  latent = tf.pad(latent, [[0, 0], [pad // 2, pad // 2], [0, 0], [0, 0]])
  return tf.concat([image, latent], axis=-1)




def tinyify(array, tiny_mode):
  if tiny_mode:
    return [1 for _ in array]
  return array


def get_gaussian_tensor(mean, log_var):
  z = tf.random_normal(tf.shape(mean), 0, 1, dtype=tf.float32)
  z = mean + tf.exp(log_var / 2.0) * z
  return z


def conv_latent_tower(images, time_axis, latent_channels=1, min_logvar=-5,
                      is_training=False, random_latent=False, tiny_mode=False):
  """Builds convolutional latent tower for stochastic model.

  At training time this tower generates a latent distribution (mean and std)
  conditioned on the entire video. This latent variable will be fed to the
  main tower as an extra variable to be used for future frames prediction.
  At inference time, the tower is disabled and only returns latents sampled
  from N(0,1).
  If the multi_latent flag is on, a different latent for every timestep would
  be generated.

  Args:
    images: tensor of ground truth image sequences
    time_axis: the time axis  in images tensor
    latent_channels: number of latent channels
    min_logvar: minimum value for log_var
    is_training: whether or not it is training mode
    random_latent: whether or not generate random latents
    tiny_mode: whether or not it is tiny_mode
  Returns:
    latent_mean: predicted latent mean
    latent_logvar: predicted latent log variance
  """
  conv_size = tinyify([32, 64, 64], tiny_mode)
  with tf.variable_scope("latent", reuse=tf.AUTO_REUSE):
    images = tf.to_float(images)
    images = tf.unstack(images, axis=time_axis)
    images = tf.concat(images, axis=3)

    x = images
    x = common_layers.make_even_size(x)
    x = tfl.conv2d(x, conv_size[0], [3, 3], strides=(2, 2),
                   padding="SAME", activation=tf.nn.relu, name="latent_conv1")
    x = tfcl.batch_norm(x, updates_collections=None,
                        is_training=is_training, scope="latent_bn1")
    x = common_layers.make_even_size(x)
    x = tfl.conv2d(x, conv_size[1], [3, 3], strides=(2, 2),
                   padding="SAME", activation=tf.nn.relu, name="latent_conv2")
    x = tfcl.batch_norm(x, updates_collections=None,
                        is_training=is_training, scope="latent_bn2")
    x = tfl.conv2d(x, conv_size[2], [3, 3], strides=(1, 1),
                   padding="SAME", activation=tf.nn.relu, name="latent_conv3")
    x = tfcl.batch_norm(x, updates_collections=None,
                        is_training=is_training, scope="latent_bn3")

    nc = latent_channels
    mean = tfl.conv2d(x, nc, [3, 3], strides=(2, 2),
                      padding="SAME", activation=None, name="latent_mean")
    logv = tfl.conv2d(x, nc, [3, 3], strides=(2, 2),
                      padding="SAME", activation=tf.nn.relu, name="latent_std")
    logvar = logv + min_logvar

    # No latent tower at inference time, just standard gaussian.
    if not is_training:
      return tf.zeros_like(mean), tf.zeros_like(logvar)

    # No latent in the first phase
    ret_mean, ret_logvar = tf.cond(
        random_latent,
        lambda: (tf.zeros_like(mean), tf.zeros_like(logvar)),
        lambda: (mean, logvar))

    return ret_mean, ret_logvar


def beta_schedule(schedule, global_step, final_beta, decay_start, decay_end):
  """Get KL multiplier (beta) based on the schedule."""
  # TODO(mechcoder): Add log_annealing schedule.
  if schedule == "constant":
    beta = tf.cond(
        tf.less(global_step, decay_start), lambda: 0.0, lambda: final_beta)
  elif schedule == "linear_anneal":
    # Linearly anneal beta from 0.0 to self.hparams.latent_loss_multiplier.
    # between self.hparams.num_iterations_2nd_stage to anneal_end.
    # beta = latent_loss * (1 - (global_step - 2nd_stage) / (anneal_end - 2nd_stage))  # pylint:disable=line-too-long
    if decay_start > decay_end:
      raise ValueError("decay_end is smaller than decay_end.")

    def anneal_loss(step_num):
      step_num = tf.cast(step_num, dtype=tf.float32)
      fraction = (float(decay_end) - step_num) / (decay_end - decay_start)
      return final_beta * (1 - fraction)

    beta = tf.case(
        pred_fn_pairs={
            tf.less(global_step, decay_start): lambda: 0.0,
            tf.greater(global_step, decay_end): lambda: final_beta},
        default=lambda: anneal_loss(global_step))
  else:
    raise ValueError("Unknown beta schedule.")
  return beta
