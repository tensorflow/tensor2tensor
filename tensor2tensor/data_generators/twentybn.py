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

"""Data generator for twenty bn video data-set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports

from tensor2tensor.data_generators import image_utils
from tensor2tensor.utils import registry

import tensorflow as tf


_FILE_VIDEO_PATTERN = '20bn-something-something-v1'
_FILE_LABEL_PATTERN = 'something-something-v1-'

_TWENTYBN_IMAGE_SIZE = 32


def resize_video_frames(images, size):
  resized_images = []
  for image in images:
    resized_images.append(
        tf.to_int64(tf.image.resize_images(
            image, [size, size], tf.image.ResizeMethod.BILINEAR)))
  return resized_images


def twentybn_generator(tmp_dir, training):
  """Video generator for twenty-bn dataset.

  Args:
    tmp_dir: path to temporary storage directory.
    training: a Boolean; if true, we use the train set, otherwise the dev set.


  Yields:
    A dictionary representing the images with the following fields:
    * image/encoded: the string encoding the images of a video as JPG,
    * image/format: the string "jpg" representing image format,
    * image/class/label: an integer representing the label,
  """
  data_suffix = 'train' if training else 'validation'

  def process_labels():
    all_labels = {}
    with tf.gfile.Open(tmp_dir + _FILE_LABEL_PATTERN + 'labels.csv') as f:
      for (i, label) in enumerate(f):
        all_labels[label] = i+1
    return all_labels

  def read_id_to_labels():
    id_to_label = {}
    with tf.gfile.Open(tmp_dir + _FILE_LABEL_PATTERN +
                       data_suffix + '.csv') as f:
      for line in f:
        values = line.split(';')
        id_to_label[int(values[0])] = values[1]
    return id_to_label

  # Get the label string to class id dictionary.
  all_labels = process_labels()
  # Get the video ids to label string dictionary.
  id_to_labels = read_id_to_labels()

  # Read video frames as images.
  for vname, label_id in id_to_labels.items():
    path = os.path.join(os.path.join(tmp_dir, _FILE_VIDEO_PATTERN), str(vname))
    label = all_labels[label_id]
    images = []
    image_files = tf.gfile.Glob(os.path.join(path, '*.jpg'))

    for filename in image_files:
      with tf.gfile.Open(filename, 'rb') as f:
        encoded_image_data = f.read()
        images.append(encoded_image_data)
    yield {
        'image/encoded': images,
        'image/format': ['jpg'],
        'image/class/label': [int(label)]
    }


@registry.register_problem
class VideoTwentybn(image_utils.Image2ClassProblem):
  """Videonet."""

  @property
  def is_small(self):
    return True

  @property
  def num_classes(self):
    return 174

  @property
  def train_shards(self):
    return 100

  @property
  def dev_shards(self):
    return 10

  def preprocess_example(self, example, unused_mode, unused_hparams):
    example['inputs'] = resize_video_frames(example['inputs'],
                                            _TWENTYBN_IMAGE_SIZE)
    return example

  def generator(self, data_dir, tmp_dir, is_training):
    if is_training:
      return twentybn_generator(tmp_dir, True)
    else:
      return twentybn_generator(tmp_dir, False)
