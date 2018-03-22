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

"""Base classes and utilities for video datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf


def resize_video_frames(images, size):
  resized_images = []
  for image in images:
    resized_images.append(
        tf.to_int64(tf.image.resize_images(
            image, [size, size], tf.image.ResizeMethod.BILINEAR)))
  return resized_images


class VideoProblem(problem.Problem):
  """Base class for problems with videos."""

  @property
  def num_channels(self):
    """Number of color channels."""
    return 3

  def example_reading_spec(self, label_repr=None):
    data_fields = {
        "image/encoded": tf.FixedLenFeature((), tf.string),
        "image/format": tf.FixedLenFeature((), tf.string),
    }

    data_items_to_decoders = {
        "inputs":
            tf.contrib.slim.tfexample_decoder.Image(
                image_key="image/encoded",
                format_key="image/format",
                channels=self.num_channels),
    }

    return data_fields, data_items_to_decoders

  def eval_metrics(self):
    eval_metrics = [
        metrics.Metrics.ACC, metrics.Metrics.ACC_TOP5,
        metrics.Metrics.NEG_LOG_PERPLEXITY
    ]
    return eval_metrics


class Video2ClassProblem(VideoProblem):
  """Base class for image classification problems."""

  @property
  def is_small(self):
    raise NotImplementedError()

  @property
  def num_classes(self):
    raise NotImplementedError()

  @property
  def train_shards(self):
    raise NotImplementedError()

  @property
  def dev_shards(self):
    return 1

  @property
  def class_labels(self):
    return ["ID_%d" % i for i in range(self.num_classes)]

  @property
  def image_size(self):
    raise NotImplementedError()

  def feature_encoders(self, data_dir):
    del data_dir
    return {
        "inputs": text_encoder.ImageEncoder(),
        "targets": text_encoder.ClassLabelEncoder(self.class_labels)
    }

  def generator(self, data_dir, tmp_dir, is_training):
    raise NotImplementedError()

  def example_reading_spec(self):
    label_key = "image/class/label"
    data_fields, data_items_to_decoders = (
        super(Video2ClassProblem, self).example_reading_spec())
    data_fields[label_key] = tf.FixedLenFeature((1,), tf.int64)

    data_items_to_decoders[
        "targets"] = tf.contrib.slim.tfexample_decoder.Tensor(label_key)
    return data_fields, data_items_to_decoders

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.input_modality = {"inputs": (registry.Modalities.IMAGE, 256)}
    p.target_modality = (registry.Modalities.CLASS_LABEL, self.num_classes)
    p.input_space_id = problem.SpaceID.IMAGE
    p.target_space_id = problem.SpaceID.IMAGE_LABEL

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    generator_utils.generate_dataset_and_shuffle(
        self.generator(data_dir, tmp_dir, True),
        self.training_filepaths(data_dir, self.train_shards, shuffled=False),
        self.generator(data_dir, tmp_dir, False),
        self.dev_filepaths(data_dir, self.dev_shards, shuffled=False))
