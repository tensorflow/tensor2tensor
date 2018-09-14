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
import numpy as np
import six

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import common_video
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry
from tensor2tensor.utils import video_metrics

import tensorflow as tf


def resize_video_frames(images, size):
  resized_images = []
  for image in images:
    resized_images.append(
        tf.to_int64(
            tf.image.resize_images(image, [size, size],
                                   tf.image.ResizeMethod.BILINEAR)))
  return resized_images


def display_video_hooks(hook_args):
  """Hooks to display videos at decode time."""
  predictions = hook_args.predictions
  fps = hook_args.decode_hparams.frames_per_second

  all_summaries = []
  for decode_ind, decode in enumerate(predictions):

    target_videos = video_metrics.stack_data_given_key(decode, "targets")
    output_videos = video_metrics.stack_data_given_key(decode, "outputs")
    input_videos = video_metrics.stack_data_given_key(decode, "inputs")
    target_videos = np.asarray(target_videos, dtype=np.uint8)
    output_videos = np.asarray(output_videos, dtype=np.uint8)
    input_videos = np.asarray(input_videos, dtype=np.uint8)

    input_videos = np.concatenate((input_videos, target_videos), axis=1)
    output_videos = np.concatenate((input_videos, output_videos), axis=1)
    input_summ_vals, _ = common_video.py_gif_summary(
        "decode_%d/input" % decode_ind,
        input_videos,
        max_outputs=10,
        fps=fps,
        return_summary_value=True)
    output_summ_vals, _ = common_video.py_gif_summary(
        "decode_%d/output" % decode_ind,
        output_videos,
        max_outputs=10,
        fps=fps,
        return_summary_value=True)
    all_summaries.extend(input_summ_vals)
    all_summaries.extend(output_summ_vals)
  return all_summaries


def summarize_video_metrics(hook_args):
  """Computes video metrics summaries using the decoder output."""
  problem_name = hook_args.problem.name
  current_problem = hook_args.problem
  hparams = hook_args.hparams
  output_dirs = hook_args.output_dirs
  predictions = hook_args.predictions
  frame_shape = [
      current_problem.frame_height, current_problem.frame_width,
      current_problem.num_channels
  ]
  metrics_graph = tf.Graph()
  with metrics_graph.as_default():
    if predictions:
      metrics_results = video_metrics.compute_video_metrics_from_predictions(
          predictions)
    else:
      metrics_results, _ = video_metrics.compute_video_metrics_from_png_files(
          output_dirs, problem_name, hparams.video_num_target_frames,
          frame_shape)

  summary_values = []
  for name, array in six.iteritems(metrics_results):
    for ind, val in enumerate(array):
      tag = "metric_{}/{}".format(name, ind)
      summary_values.append(tf.Summary.Value(tag=tag, simple_value=val))
  return summary_values


class VideoProblem(problem.Problem):
  """Base class for problems with videos."""

  def __init__(self, *args, **kwargs):
    super(VideoProblem, self).__init__(*args, **kwargs)
    # Path to a directory to dump generated frames as png for debugging.
    # If empty, no debug frames will be generated.
    self.debug_dump_frames_path = ""
    # Whether to skip random inputs at the beginning or not.
    self.settable_random_skip = True
    self.settable_use_not_breaking_batching = True
    self.shuffle = True

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
  def frame_shape(self):
    """Shape of a frame: a list [height , width , channels]."""
    return [self.frame_height, self.frame_width, self.num_channels]

  @property
  def total_number_of_frames(self):
    """The total number of frames, needed for sharding."""
    # It can also be a lower number -- we will switch shards every
    # total_number_of_frames // num_shards time, so for example if
    # you know that every video is 30 frames long and you have 100 shards
    # then it's sufficient to set this to 30 * 100 so no shard-switching
    # occurs during the generation of a video. For videos of variable length,
    # just make this large so switching shards mid-video is very rare.
    raise NotImplementedError

  @property
  def random_skip(self):
    """Whether to skip random inputs at the beginning or not."""
    return True

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

  @property
  def only_keep_videos_from_0th_frame(self):
    return True

  @property
  def use_not_breaking_batching(self):
    return True

  def preprocess_example(self, example, mode, hparams):
    """Runtime preprocessing, e.g., resize example["frame"]."""
    if getattr(hparams, "preprocess_resize_frames", None) is not None:
      example["frame"] = tf.image.resize_images(
          example["frame"], hparams.preprocess_resize_frames,
          tf.image.ResizeMethod.BILINEAR)
    return example

  @property
  def decode_hooks(self):
    return [summarize_video_metrics, display_video_hooks]

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

  def example_reading_spec(self):
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

  def serving_input_fn(self, hparams):
    """For serving/predict, assume that only video frames are provided."""
    video_input_frames = tf.placeholder(
        dtype=tf.float32,
        shape=[
            None, hparams.video_num_input_frames, self.frame_width,
            self.frame_height, self.num_channels
        ])

    # TODO(michalski): add support for passing input_action and input_reward.
    return tf.estimator.export.ServingInputReceiver(
        features={"inputs": video_input_frames},
        receiver_tensors=video_input_frames)

  def preprocess(self, dataset, mode, hparams, interleave=True):

    def split_on_batch(x):
      """Split x on batch dimension into x[:size, ...] and x[size:, ...]."""
      length = len(x.get_shape())
      size = hparams.video_num_input_frames
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

    def avoid_break_batching(dataset):
      """Smart preprocessing to avoid break between videos!

      Simple batching of images into videos may result into broken videos
      with two parts from two different videos. This preprocessing avoids
      this using the frame number.

      Args:
        dataset: raw not-batched dataset.

      Returns:
        batched not-broken videos.

      """

      def check_integrity_and_batch(*datasets):
        """Checks whether a sequence of frames are from the same video.

        Args:
          *datasets: datasets each skipping 1 frame from the previous one.

        Returns:
          batched data and the integrity flag.
        """
        not_broken = tf.constant(True)
        if "frame_number" in datasets[0]:
          frame_numbers = [dataset["frame_number"][0] for dataset in datasets]

          not_broken = tf.equal(frame_numbers[-1] - frame_numbers[0],
                                num_frames - 1)
          if self.only_keep_videos_from_0th_frame:
            not_broken = tf.logical_and(not_broken, tf.equal(
                frame_numbers[0], 0))
        else:
          tf.logging.warning("use_not_breaking_batching is True but "
                             "no frame_number is in the dataset.")

        features = {}
        for key in datasets[0].keys():
          values = [dataset[key] for dataset in datasets]
          batch = tf.stack(values)
          features[key] = batch
        return features, not_broken

      ds = [dataset.skip(i) for i in range(num_frames)]
      dataset = tf.data.Dataset.zip(tuple(ds))
      dataset = dataset.map(check_integrity_and_batch)
      dataset = dataset.filter(lambda _, not_broken: not_broken)
      dataset = dataset.map(lambda features, _: features)

      return dataset

    preprocessed_dataset = dataset.map(_preprocess)
    num_frames = (
        hparams.video_num_input_frames + hparams.video_num_target_frames)
    # We jump by a random position at the beginning to add variety.
    if (self.random_skip and self.settable_random_skip and interleave and
        mode == tf.estimator.ModeKeys.TRAIN):
      random_skip = tf.random_uniform([], maxval=num_frames, dtype=tf.int64)
      preprocessed_dataset = preprocessed_dataset.skip(random_skip)
    if (self.use_not_breaking_batching and
        self.settable_use_not_breaking_batching):
      batch_dataset = avoid_break_batching(preprocessed_dataset)
    else:
      batch_dataset = preprocessed_dataset.apply(
          tf.contrib.data.batch_and_drop_remainder(num_frames))
    dataset = batch_dataset.map(features_from_batch)
    if self.shuffle and interleave and mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.shuffle(hparams.get("shuffle_buffer_size", 128))
    return dataset

  def eval_metrics(self):
    eval_metrics = [
        metrics.Metrics.ACC, metrics.Metrics.ACC_PER_SEQ,
        metrics.Metrics.NEG_LOG_PERPLEXITY, metrics.Metrics.IMAGE_SUMMARY
    ]
    return eval_metrics

  def validate_frame(self, frame):
    height, width, channels = frame.shape
    if channels != self.num_channels:
      raise ValueError("Generated frame has %d channels while the class "
                       "assumes %d channels." % (channels, self.num_channels))
    if height != self.frame_height:
      raise ValueError("Generated frame has height %d while the class "
                       "assumes height %d." % (height, self.frame_height))
    if width != self.frame_width:
      raise ValueError("Generated frame has width %d while the class "
                       "assumes width %d." % (width, self.frame_width))

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Generate samples of the frames with possible extra data.

    Args:
      data_dir: final data directory. Typically only used in this method to copy
        over user-supplied vocab files if there are extra fields needing them.
      tmp_dir: temporary directory that you can use for downloading and scratch.
      dataset_split: problem.DatasetSplit, which data split to generate samples
        for (for example, training and evaluation). You can assume it's TRAIN if
        self.

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
    if self.debug_dump_frames_path:
      writer = common_video.VideoWriter(fps=10, file_format="avi")

    with tf.Graph().as_default():
      image_t = tf.placeholder(dtype=tf.uint8, shape=(None, None, None))
      encoded_image_t = tf.image.encode_png(image_t)
      with tf.Session() as sess:
        for features in self.generate_samples(data_dir, tmp_dir, dataset_split):
          unencoded_frame = features.pop("frame")
          self.validate_frame(unencoded_frame)
          height, width, _ = unencoded_frame.shape
          encoded_frame = sess.run(
              encoded_image_t, feed_dict={image_t: unencoded_frame})
          features["image/encoded"] = [encoded_frame]
          features["image/format"] = ["png"]
          features["image/height"] = [height]
          features["image/width"] = [width]

          has_debug_image = "image/debug" in features
          if has_debug_image:
            unencoded_debug = features.pop("image/debug")
            encoded_debug = sess.run(
                encoded_image_t, feed_dict={image_t: unencoded_debug})
            features["image/encoded_debug"] = [encoded_debug]

          if self.debug_dump_frames_path:
            img = unencoded_debug if has_debug_image else unencoded_frame
            writer.write(img)

          yield features

    if self.debug_dump_frames_path:
      if not tf.gfile.Exists(self.debug_dump_frames_path):
        tf.gfile.MkDir(self.debug_dump_frames_path)
      path = os.path.join(self.debug_dump_frames_path, "video.avi")
      writer.finish_to_file(path)

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
            self.generate_encoded_samples(data_dir, tmp_dir, split),
            paths,
            cycle_every_n=self.total_number_of_frames // len(paths))
    else:
      generator_utils.generate_files(
          self.generate_encoded_samples(data_dir, tmp_dir,
                                        problem.DatasetSplit.TRAIN),
          all_paths,
          cycle_every_n=self.total_number_of_frames // len(all_paths))


# TODO(lukaszkaiser): remove this version after everything is ported.
class VideoProblemOld(problem.Problem):
  """Base class for problems with videos: previous version."""

  @property
  def num_channels(self):
    """Number of color channels."""
    return 3

  def example_reading_spec(self):
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
