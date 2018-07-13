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
"""Computes the metrics for video prediction and generation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import six
import tensorflow as tf


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
  dataset = dataset.apply(tf.contrib.data.map_and_batch(
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


def compute_metrics(output_video, target_video):
  max_pixel_value = 255.0
  output_video = tf.to_float(output_video)
  target_video = tf.to_float(target_video)
  psnr = tf.image.psnr(output_video, target_video, max_pixel_value)
  ssim = tf.image.ssim(output_video, target_video, max_pixel_value)
  return {"PSNR": psnr, "SSIM": ssim}


def stack_data_given_key(predictions, key):
  x = [p[key] for p in predictions]
  x = np.stack(x, axis=0)
  return x


def get_zipped_dataset_from_predictions(predictions):
  """Creates dataset from in-memory predictions."""
  targets = stack_data_given_key(predictions, "targets")
  outputs = stack_data_given_key(predictions, "outputs")
  num_videos = len(targets)

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
    Dictionary which contains the average of each metric per frame.
  """
  output, target = iterator.get_next()

  metrics_dict = compute_metrics(output, target)
  metrics_names, metrics = zip(*six.iteritems(metrics_dict))
  means, update_ops = tf.metrics.mean_tensor(metrics)

  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    initalizer = iterator._initializer  # pylint: disable=protected-access
    if initalizer is not None:
      sess.run(initalizer, feed_dict=feed_dict)

    # Compute mean over dataset
    for i in range(num_videos):
      print("Computing video: %d" % i)
      sess.run(update_ops)
    averaged_metrics = sess.run(means)

    results = dict(zip(metrics_names, averaged_metrics))
    return results


def compute_all_metrics_statistics(all_results):
  """Computes statistics of metrics across multiple decodings."""
  statistics = {}
  for key in all_results[0].keys():
    values = [result[key] for result in all_results]
    values = np.vstack(values)
    statistics[key + "_MEAN"] = np.mean(values, axis=0)
    statistics[key + "_STD"] = np.std(values, axis=0)
    statistics[key + "_MIN"] = np.min(values, axis=0)
    statistics[key + "_MAX"] = np.max(values, axis=0)
  return statistics


def compute_video_metrics_from_predictions(predictions):
  all_results = []
  for prediction in predictions:
    args = get_zipped_dataset_from_predictions(prediction)
    all_results.append(compute_one_decoding_video_metrics(*args))
  statistics = compute_all_metrics_statistics(all_results)
  return statistics


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
  all_results = []
  for output_dir in output_dirs:
    output_files, target_files = get_target_and_output_filepatterns(
        output_dir, problem_name)
    args = get_zipped_dataset_from_png_files(
        output_files, target_files, video_length, frame_shape)
    all_results.append(compute_one_decoding_video_metrics(*args))
  statistics = compute_all_metrics_statistics(all_results)
  return statistics, all_results


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
