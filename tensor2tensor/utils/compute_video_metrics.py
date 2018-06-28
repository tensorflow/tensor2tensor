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
"""Computes and saves the metrics for video prediction and generation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.bin import t2t_decoder
from tensor2tensor.utils import video_metrics
import tensorflow as tf


FLAGS = tf.flags.FLAGS


def main(_):
  hparams = t2t_decoder.create_hparams()
  problem = hparams.problem
  frame_shape = [problem.frame_height,
                 problem.frame_width,
                 problem.num_channels]
  video_metrics.compute_and_save_video_metrics(
      FLAGS.output_dir,
      FLAGS.problem,
      hparams.video_num_target_frames,
      frame_shape)


if __name__ == "__main__":
  tf.app.run(main)
