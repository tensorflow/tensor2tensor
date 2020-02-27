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

"""Utilities for linear interpolation over the next_frame_glow latent space."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import flags
import numpy as np
from six.moves import zip
from tensor2tensor.bin import t2t_trainer  # pylint: disable=unused-import
from tensor2tensor.data_generators import image_utils
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_video
from tensor2tensor.models.research import glow_ops
from tensor2tensor.utils import contrib
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib
import tensorflow.compat.v1 as tf

# Flags placeholders.
flags.DEFINE_string("checkpoint_path", None,
                    "Path to the model checkpoint. Overrides output_dir.")
flags.DEFINE_bool("keep_timestamp", False,
                  "Set the mtime of the decoded file to the "
                  "checkpoint_path+'.index' mtime.")
flags.DEFINE_bool("decode_interactive", False,
                  "Interactive local inference mode.")
flags.DEFINE_integer("decode_shards", 1, "Number of decoding replicas.")
flags.DEFINE_string("score_file", "", "File to score. Each line in the file "
                    "must be in the format input \t target.")
flags.DEFINE_bool("decode_in_memory", False, "Decode in memory.")

flags = tf.flags
FLAGS = flags.FLAGS


arg_scope = contrib.framework().arg_scope


def decode_hparams(overrides=""):
  """Hparams for decoding."""
  hparams = decoding.decode_hparams()
  # Number of interpolations between [0.0, 1.0].
  hparams.add_hparam("num_interp", 11)
  # Which level(s) to interpolate.
  hparams.add_hparam("level_interp", [0, 1, 2])
  # "all" or "ranked", interpolate all channels or a "ranked".
  hparams.add_hparam("channel_interp", "all")
  # interpolate channels ranked according to squared L2 norm.
  hparams.add_hparam("rank_interp", 1)
  # Whether on not to save frames as summaries
  hparams.add_hparam("save_frames", True)
  hparams.parse(overrides)
  return hparams


def preprocess_frame(frame):
  """Preprocess frame.

  1. Converts [0, 255] to [-0.5, 0.5]
  2. Adds uniform noise.

  Args:
    frame: 3-D Tensor representing pixels.
  Returns:
    frame: 3-D Tensor with values in between [-0.5, 0.5]
  """
  # Normalize from [0.0, 1.0] -> [-0.5, 0.5]
  frame = common_layers.convert_rgb_to_real(frame)
  frame = frame - 0.5
  frame, _ = glow_ops.uniform_binning_correction(frame)
  return frame


def frame_to_latents(frame, hparams):
  """Encode frames to latents."""
  # Preprocess
  frame = preprocess_frame(frame)

  # Encode [X_t] to [z^1_t, z^2_t .. z^l_t]
  glow_vals = glow_ops.encoder_decoder(
      "codec", frame, hparams, eps=None, reverse=False)
  z_top, _, level_eps, _, _ = glow_vals
  return z_top, level_eps


def latents_to_frames(z_top_interp, level_eps_interp, hparams):
  """Decodes latents to frames."""
  # Decode [z^1_t, z^2_t .. z^l_t] to [X_t]
  images, _, _, _ = glow_ops.encoder_decoder(
      "codec", z_top_interp, hparams, eps=level_eps_interp, reverse=True)
  images = glow_ops.postprocess(images)
  return images


def interpolate(features, hparams, decode_hp):
  """Interpolate between the first input frame and last target frame.

  Args:
    features: dict of tensors
    hparams: HParams, training hparams.
    decode_hp: HParams, decode hparams.
  Returns:
    images: interpolated images, 4-D Tensor, shape=(num_interp, H, W, C)
    first_frame: image, 3-D Tensor, shape=(1, H, W, C)
    last_frame: image, 3-D Tensor, shape=(1, H, W, C)
  """
  inputs, targets = features["inputs"], features["targets"]
  inputs = tf.unstack(inputs, axis=1)
  targets = tf.unstack(targets, axis=1)
  coeffs = np.linspace(0.0, 1.0, decode_hp.num_interp)

  # (X_1, X_t) -> (z_1, z_t)
  first_frame, last_frame = inputs[0], targets[-1]
  first_top_z, first_level_eps = frame_to_latents(first_frame, hparams)
  last_top_z, last_level_eps = frame_to_latents(last_frame, hparams)

  # Interpolate latents at all levels.
  first_lats = first_level_eps + [first_top_z]
  last_lats = last_level_eps + [last_top_z]
  interp_lats = []
  lat_iterator = enumerate(zip(first_lats, last_lats))
  for level_ind, (first_lat, last_lat) in lat_iterator:
    if level_ind in decode_hp.level_interp:
      if decode_hp.channel_interp == "all":
        interp_lat = glow_ops.linear_interpolate(first_lat, last_lat, coeffs)
      else:
        interp_lat = glow_ops.linear_interpolate_rank(
            first_lat, last_lat, coeffs, decode_hp.rank_interp)
    else:
      interp_lat = tf.tile(first_lat, [decode_hp.num_interp, 1, 1, 1])
    interp_lats.append(interp_lat)

  level_eps_interp = interp_lats[:hparams.n_levels-1]
  z_top_interp = interp_lats[-1]
  images = latents_to_frames(z_top_interp, level_eps_interp, hparams)
  return images, first_frame, last_frame


def get_summaries_log_dir(decode_hp, output_dir, dataset_split):
  """Get nested summaries_log_dir based on decode_hp."""
  child_dir = decode_hp.summaries_log_dir
  level_dir = "".join([str(level) for level in decode_hp.level_interp])
  if decode_hp.channel_interp == "all":
    rank_dir = "all"
  else:
    rank_dir = "rank_%d" % decode_hp.rank_interp
  child_dir = "%s/%s_%s" % (child_dir, level_dir, rank_dir)
  if dataset_split is not None:
    child_dir += "_{}".format(dataset_split)
  return os.path.join(output_dir, child_dir)


def interpolations_to_summary(sample_ind, interpolations, first_frame,
                              last_frame, hparams, decode_hp):
  """Converts interpolated frames into tf summaries.

  The summaries consists of:
    1. Image summary corresponding to the first frame.
    2. Image summary corresponding to the last frame.
    3. The interpolated frames as a gif summary.

  Args:
    sample_ind: int
    interpolations: Numpy array, shape=(num_interp, H, W, 3)
    first_frame: Numpy array, shape=(HWC)
    last_frame: Numpy array, shape=(HWC)
    hparams: HParams, train hparams
    decode_hp: HParams, decode hparams
  Returns:
    summaries: list of tf Summary Values.
  """
  parent_tag = "sample_%d" % sample_ind
  frame_shape = hparams.problem.frame_shape
  interp_shape = [hparams.batch_size, decode_hp.num_interp] + frame_shape
  interpolations = np.reshape(interpolations, interp_shape)
  interp_tag = "%s/interp/%s" % (parent_tag, decode_hp.channel_interp)
  if decode_hp.channel_interp == "ranked":
    interp_tag = "%s/rank_%d" % (interp_tag, decode_hp.rank_interp)
  summaries, _ = common_video.py_gif_summary(
      interp_tag, interpolations, return_summary_value=True,
      max_outputs=decode_hp.max_display_outputs,
      fps=decode_hp.frames_per_second)

  if decode_hp.save_frames:
    first_frame_summ = image_utils.image_to_tf_summary_value(
        first_frame, "%s/first" % parent_tag)
    last_frame_summ = image_utils.image_to_tf_summary_value(
        last_frame, "%s/last" % parent_tag)
    summaries.append(first_frame_summ)
    summaries.append(last_frame_summ)
  return summaries


def main(_):
  decode_hp = decode_hparams(FLAGS.decode_hparams)
  trainer_lib.set_random_seed(FLAGS.random_seed)
  if FLAGS.output_dir is None:
    raise ValueError("Expected output_dir to be set to a valid path.")

  hparams = trainer_lib.create_hparams(
      FLAGS.hparams_set, FLAGS.hparams, data_dir=FLAGS.data_dir,
      problem_name=FLAGS.problem)
  if hparams.batch_size != 1:
    raise ValueError("Set batch-size to be equal to 1")

  # prepare dataset using Predict mode.
  dataset_split = "test" if FLAGS.eval_use_test_set else None
  dataset = hparams.problem.dataset(
      tf.estimator.ModeKeys.PREDICT, shuffle_files=False, hparams=hparams,
      data_dir=FLAGS.data_dir, dataset_split=dataset_split)
  dataset = dataset.batch(hparams.batch_size)
  dataset = dataset.make_one_shot_iterator().get_next()

  # Obtain frame interpolations.
  ops = [glow_ops.get_variable_ddi, glow_ops.actnorm, glow_ops.get_dropout]
  var_scope = tf.variable_scope("next_frame_glow/body", reuse=tf.AUTO_REUSE)
  with arg_scope(ops, init=False), var_scope:
    interpolations, first_frame, last_frame = interpolate(
        dataset, hparams, decode_hp)

  var_list = tf.global_variables()
  saver = tf.train.Saver(var_list)

  # Get latest checkpoints from model_dir.
  ckpt_path = tf.train.latest_checkpoint(FLAGS.output_dir)
  final_dir = get_summaries_log_dir(decode_hp, FLAGS.output_dir, dataset_split)
  summary_writer = tf.summary.FileWriter(final_dir)
  global_step = decoding.latest_checkpoint_step(FLAGS.output_dir)

  sample_ind = 0
  num_samples = decode_hp.num_samples
  all_summaries = []

  with tf.train.MonitoredTrainingSession() as sess:
    saver.restore(sess, ckpt_path)

    while not sess.should_stop() and sample_ind < num_samples:
      interp_np, first_frame_np, last_frame_np = sess.run(
          [interpolations, first_frame, last_frame])

      interp_summ = interpolations_to_summary(sample_ind, interp_np,
                                              first_frame_np[0],
                                              last_frame_np[0],
                                              hparams, decode_hp)
      all_summaries.extend(interp_summ)
      sample_ind += 1
    all_summaries = tf.Summary(value=list(all_summaries))
    summary_writer.add_summary(all_summaries, global_step)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
