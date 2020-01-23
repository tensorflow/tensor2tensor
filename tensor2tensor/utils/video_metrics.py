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

"""Computes the metrics for video prediction and generation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import six


import tensorflow.compat.v1 as tf


def load_image_map_function(filename, frame_shape):
  image = tf.read_file(filename)
  image = tf.image.decode_png(image)
  image = tf.image.resize_images(image, frame_shape[0:2])
  image.set_shape(frame_shape)
  return image


def load_videos(template, video_length, frame_shape):
  """Loads videos from files.

  Args:
    template: template string for listing the image files.
    video_length: length of the video.
    frame_shape: shape of each frame.

  Returns:
    dataset: the tf dataset frame by frame.
    dataset_len: number of the items which is the number of image files.

  Raises:
    ValueError: if no files found.
  """
  filenames = tf.gfile.Glob(template)
  if not filenames:
    raise ValueError("no files found.")
  filenames = sorted(filenames)
  dataset_len = len(filenames)
  filenames = tf.constant(filenames)
  dataset = tf.data.Dataset.from_tensor_slices(filenames)
  dataset = dataset.apply(tf.data.experimental.map_and_batch(
      lambda filename: load_image_map_function(filename, frame_shape),
      video_length, drop_remainder=True))
  return dataset, dataset_len


def file_pattern(output_dir, problem_name, prefix):
  return os.path.join(output_dir, "{}_{}*.png".format(problem_name, prefix))


def get_target_and_output_filepatterns(output_dir, problem_name):
  return (file_pattern(output_dir, problem_name, "outputs"),
          file_pattern(output_dir, problem_name, "targets"))


def get_zipped_dataset_from_png_files(
    output_files, target_files, video_length, frame_shape):
  outputs, len_ = load_videos(output_files, video_length, frame_shape)
  targets, len_ = load_videos(target_files, video_length, frame_shape)
  zipped_dataset = tf.data.Dataset.zip((outputs, targets))
  num_videos = len_ // video_length
  iterator = zipped_dataset.make_one_shot_iterator()
  return iterator, None, num_videos


def save_results(results, output_dir, problem_name):
  for name, array in six.iteritems(results):
    output_filename = "{}_{}.npy".format(problem_name, name)
    output_filename = os.path.join(output_dir, output_filename)
    with tf.gfile.Open(output_filename, "wb") as fname:
      np.save(fname, array)


def psnr_and_ssim(output, target):
  """Compute the PSNR and SSIM.

  Args:
    output: 4-D Tensor, shape=(num_frames, height, width, num_channels)
    target: 4-D Tensor, shape=(num_frames, height, width, num_channels)
  Returns:
    psnr: 1-D Tensor, shape=(num_frames,)
    ssim: 1-D Tensor, shape=(num_frames,)
  """
  output = tf.cast(output, dtype=tf.int32)
  target = tf.cast(target, dtype=tf.int32)
  psnr = tf.image.psnr(output, target, max_val=255)
  ssim = tf.image.ssim(output, target, max_val=255)
  return psnr, ssim


def stack_data_given_key(predictions, key):
  x = [p[key] for p in predictions]
  x = np.stack(x, axis=0)
  return x


def get_zipped_dataset_from_predictions(predictions):
  """Creates dataset from in-memory predictions."""
  targets = stack_data_given_key(predictions, "targets")
  outputs = stack_data_given_key(predictions, "outputs")
  num_videos, num_steps = targets.shape[:2]

  # Truncate output time-steps to match target time-steps
  outputs = outputs[:, :num_steps]

  targets_placeholder = tf.placeholder(targets.dtype, targets.shape)
  outputs_placeholder = tf.placeholder(outputs.dtype, outputs.shape)
  dataset = tf.data.Dataset.from_tensor_slices(
      (targets_placeholder, outputs_placeholder))
  iterator = dataset.make_initializable_iterator()
  feed_dict = {targets_placeholder: targets,
               outputs_placeholder: outputs}
  return iterator, feed_dict, num_videos


def compute_one_decoding_video_metrics(iterator, feed_dict, num_videos):
  """Computes the average of all the metric for one decoding.

  Args:
    iterator: dataset iterator.
    feed_dict: feed dict to initialize iterator.
    num_videos: number of videos.

  Returns:
    all_psnr: 2-D Numpy array, shape=(num_samples, num_frames)
    all_ssim: 2-D Numpy array, shape=(num_samples, num_frames)
  """
  output, target = iterator.get_next()
  metrics = psnr_and_ssim(output, target)

  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    initalizer = iterator._initializer  # pylint: disable=protected-access
    if initalizer is not None:
      sess.run(initalizer, feed_dict=feed_dict)

    all_psnr, all_ssim = [], []
    for i in range(num_videos):
      print("Computing video: %d" % i)
      psnr_np, ssim_np = sess.run(metrics)
      all_psnr.append(psnr_np)
      all_ssim.append(ssim_np)
    all_psnr = np.array(all_psnr)
    all_ssim = np.array(all_ssim)
    return all_psnr, all_ssim


def reduce_to_best_decode(metrics, reduce_func):
  """Extracts the best-decode from the metrics according to reduce_func.

  Args:
    metrics: 3-D numpy array, shape=(num_decodes, num_samples, num_frames)
    reduce_func: callable, np.argmax or np.argmin.
  Returns:
    best_metrics: 2-D numpy array, shape=(num_samples, num_frames).
    best_decode_ind: 1-D numpy array, shape=(num_samples,)
  """
  num_videos = metrics.shape[1]
  # Take mean of the metric across the frames to approximate the video
  # closest to the ground truth.
  mean_across_frames = np.mean(metrics, axis=-1)

  # For every sample, use the decode that has a maximum mean-metric.
  best_decode_ind = reduce_func(mean_across_frames, axis=0)
  best_metrics = metrics[best_decode_ind, np.arange(num_videos), :]
  return best_metrics, best_decode_ind


def compute_all_metrics_statistics(all_results):
  """Computes statistics of metrics across multiple decodings.

  Args:
    all_results: dict of 3-D numpy arrays.
                 Each array has shape=(num_decodes, num_samples, num_frames).
  Returns:
    statistics: dict of 1-D numpy arrays, shape=(num_frames).
                First the statistic (max/mean/std) is computed across the
                decodes, then the mean is taken across num_samples.
    decode_inds: dict of 1-D numpy arrays, shape=(num_samples,)
                 Each element represents the index of the decode corresponding
                 to the best statistic.
  """
  statistics = {}
  decode_inds = {}
  all_metrics = all_results.keys()

  for key in all_metrics:
    values = all_results[key]
    statistics[key + "_MEAN"] = np.mean(values, axis=0)
    statistics[key + "_STD"] = np.std(values, axis=0)
    min_stats, min_decode_ind = reduce_to_best_decode(values, np.argmin)
    statistics[key + "_MIN"] = min_stats
    decode_inds[key + "_MIN_DECODE"] = min_decode_ind
    max_stats, max_decode_ind = reduce_to_best_decode(values, np.argmax)
    statistics[key + "_MAX"] = max_stats
    decode_inds[key + "_MAX_DECODE"] = max_decode_ind

  # Computes mean of each statistic across the dataset.
  for key in statistics:
    statistics[key] = np.mean(statistics[key], axis=0)
  return statistics, decode_inds


def compute_video_metrics_from_predictions(predictions, decode_hparams):
  """Computes metrics from predictions.

  Args:
    predictions: list of list of dicts.
                 outer length: num_decodes, inner_length: num_samples
    decode_hparams: Decode hparams. instance of HParams.
  Returns:
    statistics: dict of Tensors, key being the metric with each Tensor
                having the shape (num_samples, num_frames).
  """
  all_results = {}


  ssim_all_decodes, psnr_all_decodes = [], []
  for single_decode in predictions:
    args = get_zipped_dataset_from_predictions(single_decode)
    psnr_single, ssim_single = compute_one_decoding_video_metrics(*args)
    psnr_all_decodes.append(psnr_single)
    ssim_all_decodes.append(ssim_single)
  psnr_all_decodes = np.array(psnr_all_decodes)
  ssim_all_decodes = np.array(ssim_all_decodes)
  all_results.update({"PSNR": psnr_all_decodes, "SSIM": ssim_all_decodes})
  return compute_all_metrics_statistics(all_results)


def compute_video_metrics_from_png_files(
    output_dirs, problem_name, video_length, frame_shape):
  """Computes the average of all the metric for one decoding.

  This function assumes that all the predicted and target frames
  have been saved on the disk and sorting them by name will result
  to consecutive frames saved in order.

  Args:
    output_dirs: directory with all the saved frames.
    problem_name: prefix of the saved frames usually name of the problem.
    video_length: length of the videos.
    frame_shape: shape of each frame in HxWxC format.

  Returns:
    Dictionary which contains the average of each metric per frame.
  """
  ssim_all_decodes, psnr_all_decodes = [], []
  for output_dir in output_dirs:
    output_files, target_files = get_target_and_output_filepatterns(
        output_dir, problem_name)
    args = get_zipped_dataset_from_png_files(
        output_files, target_files, video_length, frame_shape)
    psnr_single, ssim_single = compute_one_decoding_video_metrics(*args)
    psnr_all_decodes.append(psnr_single)
    ssim_all_decodes.append(ssim_single)

  psnr_all_decodes = np.array(psnr_all_decodes)
  ssim_all_decodes = np.array(ssim_all_decodes)
  all_results = {"PSNR": psnr_all_decodes, "SSIM": ssim_all_decodes}
  return compute_all_metrics_statistics(all_results)


def compute_and_save_video_metrics(
    output_dirs, problem_name, video_length, frame_shape):
  """Compute and saves the video metrics."""
  statistics, all_results = compute_video_metrics_from_png_files(
      output_dirs, problem_name, video_length, frame_shape)
  for results, output_dir in zip(all_results, output_dirs):
    save_results(results, output_dir, problem_name)

  parent_dir = os.path.join(output_dirs[0], os.pardir)
  final_dir = os.path.join(parent_dir, "decode")
  tf.gfile.MakeDirs(parent_dir)

  save_results(statistics, final_dir, problem_name)
