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
"""OCR."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import struct
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_problem
class OcrTest(image_utils.Image2TextProblem):
  """OCR test problem."""

  @property
  def is_small(self):
    return True

  @property
  def is_character_level(self):
    return True

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def train_shards(self):
    return 1

  @property
  def dev_shards(self):
    return 1

  def preprocess_example(self, example, mode, _):
    # Resize from usual size ~1350x60 to 90x4 in this test.
    img = example["inputs"]
    img = tf.to_int64(
        tf.image.resize_images(img, [90, 4], tf.image.ResizeMethod.AREA))
    img = tf.image.per_image_standardization(img)
    example["inputs"] = img
    return example

  def generator(self, data_dir, tmp_dir, is_training):
    # In this test problem, we assume that the data is in tmp_dir/ocr/ in
    # files names 0.png, 0.txt, 1.png, 1.txt and so on until num_examples.
    num_examples = 2
    ocr_dir = os.path.join(tmp_dir, "ocr/")
    tf.logging.info("Looking for OCR data in %s." % ocr_dir)
    for i in range(num_examples):
      image_filepath = os.path.join(ocr_dir, "%d.png" % i)
      text_filepath = os.path.join(ocr_dir, "%d.txt" % i)
      with tf.gfile.Open(text_filepath, "rb") as f:
        label = f.read()
      with tf.gfile.Open(image_filepath, "rb") as f:
        encoded_image_data = f.read()
      # In PNG files width and height are stored in these bytes.
      width, height = struct.unpack(">ii", encoded_image_data[16:24])
      yield {
          "image/encoded": [encoded_image_data],
          "image/format": ["png"],
          "image/class/label": label.strip(),
          "image/height": [height],
          "image/width": [width]
      }
