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

"""Utilities for video."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensor2tensor.layers import common_layers
from tensor2tensor.utils import contrib
import tensorflow.compat.v1 as tf

from tensorflow.python.ops import summary_op_util  # pylint: disable=g-direct-tensorflow-import

# After tf-nightly 1.14.1.dev20190314 summary_op_util.skip_summary was extracted
# out to the distribute module.
try:
  from tensorflow.python.distribute import summary_op_util as distribute_summary_op_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
except ImportError:
  distribute_summary_op_util = summary_op_util

tfl = common_layers.layers()


def swap_time_and_batch_axes(inputs):
  """Swaps time and batch axis (the first two axis)."""
  transposed_axes = tf.concat([[1, 0], tf.range(2, tf.rank(inputs))], axis=0)
  return tf.transpose(inputs, transposed_axes)


def encode_to_shape(inputs, shape, scope):
  """Encode the given tensor to given image shape."""
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    w, h = shape[1], shape[2]
    x = inputs
    x = tfl.flatten(x)
    x = tfl.dense(x, w * h, activation=None, name="enc_dense")
    x = tf.reshape(x, (-1, w, h, 1))
    return x


def decode_to_shape(inputs, shape, scope):
  """Encode the given tensor to given image shape."""
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    x = inputs
    x = tfl.flatten(x)
    x = tfl.dense(x, shape[2], activation=None, name="dec_dense")
    x = tf.expand_dims(x, axis=1)
    return x


def basic_lstm(inputs, state, num_units, name=None):
  """Basic LSTM."""
  input_shape = common_layers.shape_list(inputs)
  # reuse parameters across time-steps.
  cell = tf.nn.rnn_cell.BasicLSTMCell(
      num_units, name=name, reuse=tf.AUTO_REUSE)
  if state is None:
    state = cell.zero_state(input_shape[0], tf.float32)
  outputs, new_state = cell(inputs, state)
  return outputs, new_state


def lstm_cell(inputs,
              state,
              num_units,
              use_peepholes=False,
              cell_clip=0.0,
              initializer=None,
              num_proj=None,
              num_unit_shards=None,
              num_proj_shards=None,
              reuse=None,
              name=None):
  """Full LSTM cell."""
  input_shape = common_layers.shape_list(inputs)
  cell = tf.nn.rnn_cell.LSTMCell(num_units,
                                 use_peepholes=use_peepholes,
                                 cell_clip=cell_clip,
                                 initializer=initializer,
                                 num_proj=num_proj,
                                 num_unit_shards=num_unit_shards,
                                 num_proj_shards=num_proj_shards,
                                 reuse=reuse,
                                 name=name,
                                 state_is_tuple=False)
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

  cell = contrib.rnn().ConvLSTMCell(
      2, input_shape, output_channels, [kernel_size, kernel_size], name=name)
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

  output = tf.dynamic_stitch([ground_truth_idx, generated_idx],
                             [ground_truth_examps, generated_examps])
  # if batch size is known set it.
  if isinstance(batch_size, int):
    output.set_shape([batch_size] + common_layers.shape_list(output)[1:])
  return output


def inject_additional_input(layer, inputs, name, mode="concat"):
  """Injects the additional input into the layer.

  Args:
    layer: layer that the input should be injected to.
    inputs: inputs to be injected.
    name: TF scope name.
    mode: how the infor should be added to the layer:
      "concat" concats as additional channels.
      "multiplicative" broadcasts inputs and multiply them to the channels.
      "multi_additive" broadcasts inputs and multiply and add to the channels.

  Returns:
    updated layer.

  Raises:
    ValueError: in case of unknown mode.
  """
  layer_shape = common_layers.shape_list(layer)
  input_shape = common_layers.shape_list(inputs)
  zeros_mask = tf.zeros(layer_shape, dtype=tf.float32)
  if mode == "concat":
    emb = encode_to_shape(inputs, layer_shape, name)
    layer = tf.concat(values=[layer, emb], axis=-1)
  elif mode == "multiplicative":
    filters = layer_shape[-1]
    input_reshaped = tf.reshape(inputs, [-1, 1, 1, input_shape[-1]])
    input_mask = tf.layers.dense(input_reshaped, filters, name=name)
    input_broad = input_mask + zeros_mask
    layer *= input_broad
  elif mode == "multi_additive":
    filters = layer_shape[-1]
    input_reshaped = tf.reshape(inputs, [-1, 1, 1, input_shape[-1]])
    input_mul = tf.layers.dense(input_reshaped, filters, name=name + "_mul")
    layer *= tf.nn.sigmoid(input_mul)
    input_add = tf.layers.dense(input_reshaped, filters, name=name + "_add")
    layer += input_add
  else:
    raise ValueError("Unknown injection mode: %s" % mode)

  return layer


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
  return tf.where(probability_of_generated > probability_threshold,
                  generated_x, ground_truth_x)


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
              is_training=True,
              has_batchnorm=False,
              scope=None):
  """A layer of VGG network with batch norm.

  Args:
    inputs: image tensor
    nout: number of output channels
    kernel_size: size of the kernel
    activation: activation function
    padding: padding of the image
    is_training: whether it is training mode or not
    has_batchnorm: whether batchnorm is applied or not
    scope: variable scope of the op
  Returns:
    net: output of layer
  """
  with tf.variable_scope(scope):
    net = tfl.conv2d(inputs, nout, kernel_size=kernel_size, padding=padding,
                     activation=None, name="conv")
    if has_batchnorm:
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


def _encode_gif(images, fps):
  """Encodes numpy images into gif string.

  Args:
    images: A 4-D `uint8` `np.array` (or a list of 3-D images) of shape
      `[time, height, width, channels]` where `channels` is 1 or 3.
    fps: frames per second of the animation

  Returns:
    The encoded gif string.

  Raises:
    IOError: If the ffmpeg command returns an error.
  """
  writer = WholeVideoWriter(fps)
  writer.write_multi(images)
  return writer.finish()


def ffmpeg_works():
  """Tries to encode images with ffmpeg to check if it works."""
  images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
  try:
    _encode_gif(images, 2)
    return True
  except (IOError, OSError):
    return False


def py_gif_summary(tag, images, max_outputs, fps, return_summary_value=False):
  """Outputs a `Summary` protocol buffer with gif animations.

  Args:
    tag: Name of the summary.
    images: A 5-D `uint8` `np.array` of shape `[batch_size, time, height, width,
      channels]` where `channels` is 1 or 3.
    max_outputs: Max number of batch elements to generate gifs for.
    fps: frames per second of the animation.
    return_summary_value: If set to True, return a list of tf.Summary.Value
                          objects in addition to the protocol buffer.

  Returns:
    The serialized `Summary` protocol buffer.

  Raises:
    ValueError: If `images` is not a 5-D `uint8` array with 1 or 3 channels.
  """
  images = np.asarray(images)
  if images.dtype != np.uint8:
    raise ValueError("Tensor must have dtype uint8 for gif summary.")
  if images.ndim != 5:
    raise ValueError("Tensor must be 5-D for gif summary.")
  batch_size, _, height, width, channels = images.shape
  if channels not in (1, 3):
    raise ValueError("Tensors must have 1 or 3 channels for gif summary.")

  summ = tf.Summary()
  all_summ_values = []
  num_outputs = min(batch_size, max_outputs)
  for i in range(num_outputs):
    image_summ = tf.Summary.Image()
    image_summ.height = height
    image_summ.width = width
    image_summ.colorspace = channels  # 1: grayscale, 3: RGB
    try:
      image_summ.encoded_image_string = _encode_gif(images[i], fps)
    except (IOError, OSError) as e:
      tf.logging.warning(
          "Unable to encode images to a gif string because either ffmpeg is "
          "not installed or ffmpeg returned an error: %s. Falling back to an "
          "image summary of the first frame in the sequence.", e)
      try:
        from PIL import Image  # pylint: disable=g-import-not-at-top
        import io  # pylint: disable=g-import-not-at-top
        with io.BytesIO() as output:
          Image.fromarray(images[i][0]).save(output, "PNG")
          image_summ.encoded_image_string = output.getvalue()
      except ImportError as e:
        tf.logging.warning(
            "Gif summaries requires ffmpeg or PIL to be installed: %s", e)
        image_summ.encoded_image_string = ""
    if num_outputs == 1:
      summ_tag = "{}/gif".format(tag)
    else:
      summ_tag = "{}/gif/{}".format(tag, i)
    curr_summ_value = tf.Summary.Value(tag=summ_tag, image=image_summ)
    all_summ_values.append(curr_summ_value)
    summ.value.add(tag=summ_tag, image=image_summ)
  summ_str = summ.SerializeToString()
  if return_summary_value:
    return all_summ_values, summ_str
  return summ_str


def gif_summary(name, tensor, max_outputs=3, fps=10, collections=None,
                family=None):
  """Outputs a `Summary` protocol buffer with gif animations.

  Args:
    name: Name of the summary.
    tensor: A 5-D `uint8` `Tensor` of shape `[batch_size, time, height, width,
      channels]` where `channels` is 1 or 3.
    max_outputs: Max number of batch elements to generate gifs for.
    fps: frames per second of the animation
    collections: Optional list of tf.GraphKeys.  The collections to add the
      summary to.  Defaults to [tf.GraphKeys.SUMMARIES]
    family: Optional; if provided, used as the prefix of the summary tag name,
      which controls the tab name used for display on Tensorboard.

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.

  Raises:
    ValueError: if the given tensor has the wrong shape.
  """
  tensor = tf.convert_to_tensor(tensor)
  if len(tensor.get_shape()) != 5:
    raise ValueError("Assuming videos given as tensors in the format "
                     "[batch, time, height, width, channels] but got one "
                     "of shape: %s" % str(tensor.get_shape()))
  tensor = tf.cast(tensor, tf.uint8)
  if distribute_summary_op_util.skip_summary():
    return tf.constant("")
  with summary_op_util.summary_scope(
      name, family, values=[tensor]) as (tag, scope):
    val = tf.py_func(
        py_gif_summary,
        [tag, tensor, max_outputs, fps],
        tf.string,
        stateful=False,
        name=scope)
    summary_op_util.collect(val, collections, [tf.GraphKeys.SUMMARIES])
  return val




def tinyify(array, tiny_mode, small_mode):
  if tiny_mode:
    return [1 for _ in array]
  if small_mode:
    return [max(x // 4, 1) for x in array]
  return array


def get_gaussian_tensor(mean, log_var):
  z = tf.random_normal(tf.shape(mean), 0, 1, dtype=tf.float32)
  z = mean + tf.exp(log_var / 2.0) * z
  return z


def conv_latent_tower(images, time_axis, latent_channels=1, min_logvar=-5,
                      is_training=False, random_latent=False,
                      tiny_mode=False, small_mode=False):
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
    tiny_mode: whether or not it is tiny_mode. tiny_mode sets the number
        of conv channels to 1 at each layer. useful for testing the
        integration tests.
    small_mode: whether or not it is small_mode. small mode is the same model
        with less conv and lstm layers and also lower number of channels.
        suitable for videos with less complexity and testing.
  Returns:
    latent_mean: predicted latent mean
    latent_logvar: predicted latent log variance
  """
  conv_size = tinyify([32, 64, 64], tiny_mode, small_mode)
  with tf.variable_scope("latent", reuse=tf.AUTO_REUSE):
    images = tf.to_float(images)
    images = tf.unstack(images, axis=time_axis)
    images = tf.concat(images, axis=3)

    x = images
    x = common_layers.make_even_size(x)
    x = tfl.conv2d(x, conv_size[0], [3, 3], strides=(2, 2),
                   padding="SAME", activation=tf.nn.relu, name="latent_conv1")
    x = contrib.layers().layer_norm(x)
    if not small_mode:
      x = tfl.conv2d(x, conv_size[1], [3, 3], strides=(2, 2),
                     padding="SAME", activation=tf.nn.relu, name="latent_conv2")
      x = contrib.layers().layer_norm(x)
    x = tfl.conv2d(x, conv_size[2], [3, 3], strides=(1, 1),
                   padding="SAME", activation=tf.nn.relu, name="latent_conv3")
    x = contrib.layers().layer_norm(x)

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
  if decay_start > decay_end:
    raise ValueError("decay_end is smaller than decay_end.")

  # Since some of the TF schedules do not support incrementing a value,
  # in all of the schedules, we anneal the beta from final_beta to zero
  # and then reverse it at the bottom.
  if schedule == "constant":
    decayed_value = 0.0
  elif schedule == "linear":
    decayed_value = tf.train.polynomial_decay(
        learning_rate=final_beta,
        global_step=global_step - decay_start,
        decay_steps=decay_end - decay_start,
        end_learning_rate=0.0)
  elif schedule == "noisy_linear_cosine_decay":
    decayed_value = tf.train.noisy_linear_cosine_decay(
        learning_rate=final_beta,
        global_step=global_step - decay_start,
        decay_steps=decay_end - decay_start)
  # TODO(mechcoder): Add log_annealing schedule.
  else:
    raise ValueError("Unknown beta schedule.")

  increased_value = final_beta - decayed_value
  increased_value = tf.maximum(0.0, increased_value)

  beta = tf.case(
      pred_fn_pairs={
          tf.less(global_step, decay_start): lambda: 0.0,
          tf.greater(global_step, decay_end): lambda: final_beta},
      default=lambda: increased_value)
  return beta


def extract_random_video_patch(videos, num_frames=-1):
  """For every video, extract a random consecutive patch of num_frames.

  Args:
    videos: 5-D Tensor, (NTHWC)
    num_frames: Integer, if -1 then the entire video is returned.
  Returns:
    video_patch: 5-D Tensor, (NTHWC) with T = num_frames.
  Raises:
    ValueError: If num_frames is greater than the number of total frames in
                the video.
  """
  if num_frames == -1:
    return videos
  batch_size, num_total_frames, h, w, c = common_layers.shape_list(videos)
  if num_total_frames < num_frames:
    raise ValueError("Expected num_frames <= %d, got %d" %
                     (num_total_frames, num_frames))

  # Randomly choose start_inds for each video.
  frame_start = tf.random_uniform(
      shape=(batch_size,), minval=0, maxval=num_total_frames - num_frames + 1,
      dtype=tf.int32)

  # [start[0], start[0] + 1, ... start[0] + num_frames - 1] + ...
  # [start[batch_size-1], ... start[batch_size-1] + num_frames - 1]
  range_inds = tf.expand_dims(tf.range(num_frames), axis=0)
  frame_inds = range_inds + tf.expand_dims(frame_start, axis=1)
  frame_inds = tf.reshape(frame_inds, [-1])

  # [0]*num_frames + [1]*num_frames + ... [batch_size-1]*num_frames
  batch_inds = tf.expand_dims(tf.range(batch_size), axis=1)
  batch_inds = tf.tile(batch_inds, [1, num_frames])
  batch_inds = tf.reshape(batch_inds, [-1])

  gather_inds = tf.stack((batch_inds, frame_inds), axis=1)
  video_patches = tf.gather_nd(videos, gather_inds)
  return tf.reshape(video_patches, (batch_size, num_frames, h, w, c))


class VideoWriter(object):
  """Base helper class for writing videos."""

  def write(self, frame, encoded_frame=None):
    """Writes a single video frame."""
    raise NotImplementedError

  def write_multi(self, frames, encoded_frames=None):
    """Writes multiple video frames."""
    if encoded_frames is None:
      # Infinite iterator.
      encoded_frames = iter(lambda: None, 1)
    for (frame, encoded_frame) in zip(frames, encoded_frames):
      self.write(frame, encoded_frame)

  def finish(self):
    """Finishes writing frames and returns output, if any.

    Frees any resources acquired by the writer.
    """
    pass

  def save_to_disk(self, output):
    """Saves output to disk.

    Args:
      output: result of finish().
    """
    raise NotImplementedError

  def finish_to_disk(self):
    """Finishes writing frames and saves output to disk, if any."""
    output = self.finish()  # pylint: disable=assignment-from-no-return
    if output is not None:
      self.save_to_disk(output)

  def __del__(self):
    """Frees any resources acquired by the writer."""
    self.finish()


class WholeVideoWriter(VideoWriter):
  """Helper class for writing whole videos."""

  def __init__(self, fps, output_path=None, file_format="gif"):
    self.fps = fps
    self.output_path = output_path
    self.file_format = file_format
    self.proc = None
    self._out_chunks = []
    self._err_chunks = []
    self._out_thread = None
    self._err_thread = None

  def __init_ffmpeg(self, image_shape):
    """Initializes ffmpeg to write frames."""
    import itertools  # pylint: disable=g-import-not-at-top
    from subprocess import Popen, PIPE  # pylint: disable=g-import-not-at-top,g-multiple-import,g-importing-member
    ffmpeg = "ffmpeg"
    height, width, channels = image_shape
    self.cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-r", "%.02f" % self.fps,
        "-s", "%dx%d" % (width, height),
        "-pix_fmt", {1: "gray", 3: "rgb24"}[channels],
        "-i", "-",
        "-filter_complex", "[0:v]split[x][z];[x]fifo[w];[z]palettegen,fifo[y];"
                           "[w][y]paletteuse,fifo",
        "-r", "%.02f" % self.fps,
        "-f", self.file_format,
        "-qscale", "0",
        "-"
    ]
    self.proc = Popen(
        self.cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, bufsize=-1
    )
    (self._out_thread, self._err_thread) = itertools.starmap(
        self._start_reader_thread, [
            (self.proc.stdout, self._out_chunks),
            (self.proc.stderr, self._err_chunks)
        ]
    )

  def _start_reader_thread(self, stream, chunks):
    """Starts a thread for reading output from FFMPEG.

    The thread reads consecutive chunks from the stream and saves them in
    the given list.

    Args:
      stream: output stream of the FFMPEG process.
      chunks: list to save output chunks to.

    Returns:
      Thread
    """
    import io  # pylint: disable=g-import-not-at-top
    import threading  # pylint: disable=g-import-not-at-top
    def target():
      while True:
        chunk = stream.read(io.DEFAULT_BUFFER_SIZE)
        if not chunk:
          break
        chunks.append(chunk)
    thread = threading.Thread(target=target)
    thread.start()
    return thread

  def write(self, frame, encoded_frame=None):
    if self.proc is None:
      self.__init_ffmpeg(frame.shape)
    self.proc.stdin.write(frame.tostring())

  def finish(self):
    """Finishes transconding and returns the video.

    Returns:
      bytes

    Raises:
      IOError: in case of transcoding error.
    """
    if self.proc is None:
      return None
    self.proc.stdin.close()
    for thread in (self._out_thread, self._err_thread):
      thread.join()
    (out, err) = [
        b"".join(chunks) for chunks in (self._out_chunks, self._err_chunks)
    ]
    self.proc.stdout.close()
    self.proc.stderr.close()
    if self.proc.returncode:
      err = "\n".join([" ".join(self.cmd), err.decode("utf8")])
      raise IOError(err)
    del self.proc
    self.proc = None
    return out

  def save_to_disk(self, output):
    if self.output_path is None:
      raise ValueError(
          "This writer doesn't support saving to disk (output_path not "
          "specified)."
      )
    with tf.gfile.Open(self.output_path, "w") as f:
      f.write(output)


class BatchWholeVideoWriter(VideoWriter):
  """Helper class for writing videos in batch."""

  def __init__(self, fps, path_template, file_format="gif"):
    self.fps = fps
    self.path_template = path_template
    self.file_format = file_format
    self.writers = None

  def write(self, batch_frame, batch_encoded_frame=None):
    del batch_encoded_frame
    if self.writers is None:
      self.writers = [
          WholeVideoWriter(  # pylint: disable=g-complex-comprehension
              self.fps, self.path_template.format(i), self.file_format
          )
          for i in range(len(batch_frame))
      ]
    for i, frame in enumerate(batch_frame):
      self.writers[i].write(frame)

  def finish(self):
    outs = [w.finish() for w in self.writers]
    return outs

  def save_to_disk(self, outputs):
    for (writer, output) in zip(self.writers, outputs):
      writer.save_to_disk(output)


class IndividualFrameWriter(VideoWriter):
  """Helper class for writing individual video frames."""

  def __init__(self, output_dir):
    self.output_dir = output_dir
    self._counter = 0

  def write(self, frame=None, encoded_frame=None):
    import os  # pylint: disable=g-import-not-at-top
    if encoded_frame is None:
      raise ValueError("This writer only supports encoded frames.")
    path = os.path.join(self.output_dir, "frame_%05d.png" % self._counter)
    with tf.gfile.Open(path, "wb") as f:
      f.write(encoded_frame)
      self._counter += 1
