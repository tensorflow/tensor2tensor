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

r"""View the problem.

This binary saves the videos in the problem(dataset) into gifs.

The imagemagick package should be installed for conversion to gifs.

Example usage to view dataset:

  video2gif \
      --data_dir ~/data \
      --problem=gym_water_world_random5k \
      --hparams_set=next_frame_stochastic \
      --output_dir /usr/local/google/home/mbz/t2t_train/ww/ \
      --data_dir /usr/local/google/home/mbz/temp/ \
      --num_samples 10
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
from tensor2tensor.bin import t2t_trainer          # pylint: disable=unused-import
from tensor2tensor.data_generators import problem  # pylint: disable=unused-import
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib

import tensorflow.compat.v1 as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("num_samples", -1, "Number of saved samples.")


def create_gif(name):
  cmd = "convert -delay 15 {0}* {0}.gif".format(name)
  os.system(cmd)


def main(_):
  problem_name = FLAGS.problem
  if "video" not in problem_name and "gym" not in problem_name:
    print("This tool only works for video problems.")
    return

  mode = tf.estimator.ModeKeys.TRAIN
  hparams = trainer_lib.create_hparams(
      FLAGS.hparams_set,
      FLAGS.hparams,
      data_dir=os.path.expanduser(FLAGS.data_dir),
      problem_name=problem_name)

  dataset = hparams.problem.input_fn(mode, hparams)
  features = dataset.make_one_shot_iterator().get_next()

  tf.gfile.MakeDirs(FLAGS.output_dir)
  base_template = os.path.join(FLAGS.output_dir, FLAGS.problem)
  count = 0
  with tf.train.MonitoredTrainingSession() as sess:
    while not sess.should_stop():
      # TODO(mbz): figure out what the second output is.
      data, _ = sess.run(features)
      video_batch = np.concatenate((data["inputs"], data["targets"]), axis=1)

      for video in video_batch:
        print("Saving {}/{}".format(count, FLAGS.num_samples))
        name = "%s_%05d" % (base_template, count)
        decoding.save_video(video, name + "_{:05d}.png")
        create_gif(name)
        count += 1

        if count == FLAGS.num_samples:
          sys.exit(0)

if __name__ == "__main__":
  tf.app.run()
