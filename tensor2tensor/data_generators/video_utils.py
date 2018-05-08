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

import os

# Dependency imports

import six

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
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

  def __init__(self, *args, **kwargs):
    super(VideoProblem, self).__init__(*args, **kwargs)
    # Path to a directory to dump generated frames as png for debugging.
    # If empty, no debug frames will be generated.
    self.debug_dump_frames_path = ""

  @property
  def num_channels(self):
    """Number of color channels in each frame."""
    return 3

  @property
  def frame_height(self):
    """Height of each frame."""
    raise NotImplementedError

  @property
  def frame_width(self):
    """Width of each frame."""
    raise NotImplementedError

  @property
  def total_number_of_frames(self):
    """The total number of frames, needed for sharding."""
    raise NotImplementedError

  @property
  def num_input_frames(self):
    """Number of frames to batch on one input."""
    return 1

  @property
  def num_target_frames(self):
    """Number of frames to batch on one target."""
    return 1

  @property
  def extra_reading_spec(self):
    """Additional data fields to store on disk and their decoders."""
    return {}, {}

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 10,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  def preprocess_example(self, example, mode, hparams):
    """Runtime preprocessing, e.g., resize example["frame"]."""
    return example

  @property
  def is_generate_per_split(self):
    """A single call to `generate_samples` generates for all `dataset_splits`.

    Set to True if you already have distinct subsets of data for each dataset
    split specified in `self.dataset_splits`. `self.generate_samples` will be
    called once for each split.

    Set to False if you have a unified dataset that you'd like to have split out
    into training and evaluation data automatically. `self.generate_samples`
    will be called only once and the data will be sharded across the dataset
    splits specified in `self.dataset_splits`.

    Returns:
      bool
    """
    raise NotImplementedError()

  def example_reading_spec(self, label_repr=None):
    extra_data_fields, extra_data_items_to_decoders = self.extra_reading_spec

    data_fields = {
        "image/encoded": tf.FixedLenFeature((), tf.string),
        "image/format": tf.FixedLenFeature((), tf.string),
    }
    data_fields.update(extra_data_fields)

    data_items_to_decoders = {
        "frame":
            tf.contrib.slim.tfexample_decoder.Image(
                image_key="image/encoded",
                format_key="image/format",
                shape=[self.frame_height, self.frame_width, self.num_channels],
                channels=self.num_channels),
    }
    data_items_to_decoders.update(extra_data_items_to_decoders)

    return data_fields, data_items_to_decoders

  def preprocess(self, dataset, mode, hparams):
    def split_on_batch(x):
      """Split x on batch dimension into x[:size, ...] and x[size:, ...]."""
      length = len(x.get_shape())
      size = self.num_input_frames
      if length < 1:
        raise ValueError("Batched tensor of length < 1.")
      if length == 1:
        return x[:size], x[size:]
      if length == 2:
        return x[:size, :], x[size:, :]
      if length == 3:
        return x[:size, :, :], x[size:, :, :]
      if length == 4:
        return x[:size, :, :, :], x[size:, :, :, :]
      # TODO(lukaszkaiser): use tf.split for the general case.
      raise ValueError("Batch splitting on general dimensions not done yet.")

    def features_from_batch(batched_prefeatures):
      """Construct final features from the batched inputs.

      This function gets prefeatures.

      Args:
        batched_prefeatures: single-frame features (from disk) as batch tensors.

      Returns:
        Features dictionary with joint features per-frame.
      """
      features = {}
      for k, v in six.iteritems(batched_prefeatures):
        if k == "frame":  # We rename past frames to inputs and targets.
          s1, s2 = split_on_batch(v)
          # Reshape just to make sure shapes are right and set.
          s1 = tf.reshape(s1, [self.num_input_frames, self.frame_height,
                               self.frame_width, self.num_channels])
          s2 = tf.reshape(s2, [self.num_target_frames, self.frame_height,
                               self.frame_width, self.num_channels])
          features["inputs"] = s1
          features["targets"] = s2
        else:
          s1, s2 = split_on_batch(v)
          features["input_%s" % k] = s1
          features["target_%s" % k] = s2
      return features

    # Batch and construct features.
    def _preprocess(example):
      return self.preprocess_example(example, mode, hparams)
    preprocessed_dataset = dataset.map(_preprocess)

    num_frames = self.num_input_frames + self.num_target_frames
    # We jump by a random position at the beginning to add variety.
    random_skip = tf.random_uniform([], maxval=num_frames, dtype=tf.int64)
    preprocessed_dataset = preprocessed_dataset.skip(random_skip)
    batch_dataset = preprocessed_dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(num_frames))
    dataset = batch_dataset.map(features_from_batch).shuffle(8)
    return dataset

  def eval_metrics(self):
    eval_metrics = [
        metrics.Metrics.ACC, metrics.Metrics.ACC_PER_SEQ,
        metrics.Metrics.NEG_LOG_PERPLEXITY, metrics.Metrics.IMAGE_SUMMARY]
    return eval_metrics

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Generate samples of the frames with possible extra data.

    Args:
      data_dir: final data directory. Typically only used in this method to copy
        over user-supplied vocab files if there are extra fields needing them.
      tmp_dir: temporary directory that you can use for downloading and scratch.
      dataset_split: problem.DatasetSplit, which data split to generate samples
        for (for example, training and evaluation). You can assume it's TRAIN
        if self.

    Yields:
      Sample: dict<str feature_name, feature value>; we assume that there is
        a "frame" feature with unencoded frame which is a numpy arrays of shape
        [frame_height, frame_width, num_channels] and which will be transcoded
        into an image format by generate_encodeded_samples.
    """
    raise NotImplementedError()

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    """Generate samples of the encoded frames with possible extra data.

    By default this function just encodes the numpy array returned as "frame"
    from `self.generate_samples` into a PNG image. Override this function to
    get other encodings on disk.

    Args:
      data_dir: final data directory. Typically only used in this method to copy
        over user-supplied vocab files if there are extra fields needing them.
      tmp_dir: temporary directory that you can use for downloading and scratch.
      dataset_split: problem.DatasetSplit, which data split to generate samples
        for (for example, training and evaluation).

    Yields:
      Sample: dict<str feature_name, feature value> which is in disk encoding.

    Raises:
      ValueError: if the frame has a different number of channels than required.
    """
    for features in self.generate_samples(data_dir, tmp_dir, dataset_split):
      unencoded_frame = features.pop("frame")
      height, width, channels = unencoded_frame.shape
      if channels != self.num_channels:
        raise ValueError("Generated frame has %d channels while the class "
                         "assumes %d channels." % (channels, self.num_channels))
      if height != self.frame_height:
        raise ValueError("Generated frame has height %d while the class "
                         "assumes height %d." % (height, self.frame_height))
      if width != self.frame_width:
        raise ValueError("Generated frame has width %d while the class "
                         "assumes width %d." % (width, self.frame_width))
      encoded_frame = six.next(
          image_utils.encode_images_as_png([unencoded_frame]))
      features["image/encoded"] = [encoded_frame]
      features["image/format"] = ["png"]
      features["image/height"] = [height]
      features["image/width"] = [width]
      yield features

  def generate_encoded_samples_debug(self, data_dir, tmp_dir, dataset_split):
    """Generate samples of the encoded frames and dump for debug if needed."""
    counter = 0
    for sample in self.generate_encoded_samples(
        data_dir, tmp_dir, dataset_split):
      if self.debug_dump_frames_path:
        if not tf.gfile.Exists(self.debug_dump_frames_path):
          tf.gfile.MkDir(self.debug_dump_frames_path)
        path = os.path.join(self.debug_dump_frames_path,
                            "frame_%05d.png" % counter)
        with tf.gfile.Open(path, "wb") as f:
          if "image/encoded_debug" in sample:
            img_to_save = sample["image/encoded_debug"][0]
          else:
            img_to_save = sample["image/encoded"][0]
          f.write(img_to_save)
        counter += 1
      yield sample

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    """The function generating the data."""
    filepath_fns = {
        problem.DatasetSplit.TRAIN: self.training_filepaths,
        problem.DatasetSplit.EVAL: self.dev_filepaths,
        problem.DatasetSplit.TEST: self.test_filepaths,
    }

    # We set shuffled=True as we don't want to shuffle on disk later.
    split_paths = [(split["split"], filepath_fns[split["split"]](
        data_dir, split["shards"], shuffled=True))
                   for split in self.dataset_splits]
    all_paths = []
    for _, paths in split_paths:
      all_paths.extend(paths)

    if self.is_generate_per_split:
      for split, paths in split_paths:
        generator_utils.generate_files(
            self.generate_encoded_samples_debug(
                data_dir, tmp_dir, split), paths)
    else:
      generator_utils.generate_files(
          self.generate_encoded_samples_debug(
              data_dir, tmp_dir, problem.DatasetSplit.TRAIN),
          all_paths,
          cycle_every_n=self.total_number_of_frames // len(all_paths))


# TODO(lukaszkaiser): remove this version after everything is ported.
class VideoProblemOld(problem.Problem):
  """Base class for problems with videos: previous version."""

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


class Video2ClassProblem(VideoProblemOld):
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
