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


def get_zipped_dataset(output_files, target_files, video_length, frame_shape):
  outputs, len_ = load_videos(output_files, video_length, frame_shape)
  targets, len_ = load_videos(target_files, video_length, frame_shape)
  zipped_dataset = tf.data.Dataset.zip((outputs, targets))
  num_videos = len_ // video_length
  return zipped_dataset, num_videos


def save_results(results, output_dir, problem_name):
  for name, array in six.iteritems(results):
    output_filename = "{}_{}.npy".format(problem_name, name)
    output_filename = os.path.join(output_dir, output_filename)
    with tf.gfile.Open(output_filename, "wb") as fname:
      np.save(fname, array)


def compute_metrics(output_video, target_video):
  max_pixel_value = 255.0
  psnr = tf.image.psnr(output_video, target_video, max_pixel_value)
  ssim = tf.image.ssim(output_video, target_video, max_pixel_value)
  return {"PSNR": psnr, "SSIM": ssim}


def compute_video_metrics(output_dir, problem_name, video_length, frame_shape):
  """Computes the average of all the metric over the whole dataset.

  This function assumes that all the predicted and target frames
  have been saved on the disk and sorting them by name will result
  to consecutive frames saved in order.

  Args:
    output_dir: directory with all the saved frames.
    problem_name: prefix of the saved frames usually name of the problem.
    video_length: length of the videos.
    frame_shape: shape of each frame in HxWxC format.

  Returns:
    Dictionary which contains the average of each metric per frame.
  """
  output_files, target_files = get_target_and_output_filepatterns(
      output_dir, problem_name)
  dataset, num_videos = get_zipped_dataset(
      output_files, target_files, video_length, frame_shape)
  output, target = dataset.make_one_shot_iterator().get_next()
  metrics_dict = compute_metrics(output, target)
  metrics_names, metrics = zip(*six.iteritems(metrics_dict))
  means, update_ops = tf.metrics.mean_tensor(metrics)

  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())

    # Compute mean over dataset
    for i in range(num_videos):
      print("Computing video: %d" % i)
      sess.run(update_ops)
    averaged_metrics = sess.run(means)

    results = dict(zip(metrics_names, averaged_metrics))
    return results


def compute_and_save_video_metrics(
    output_dir, problem_name, video_length, frame_shape):
  results = compute_video_metrics(
      output_dir, problem_name, video_length, frame_shape)
  save_results(results, output_dir, problem_name)

