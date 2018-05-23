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
"""Data generators for video problems with artificially generated frames."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np

from tensor2tensor.data_generators import video_utils
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_problem
class VideoStochasticShapes10k(video_utils.VideoProblem):
  """Shapes moving in a stochastic way."""

  @property
  def num_input_frames(self):
    """Number of frames to batch on one input."""
    return 4

  @property
  def num_target_frames(self):
    """Number of frames to predict in one step."""
    return 1

  @property
  def is_generate_per_split(self):
    """Whether we have a train/test split or just hold out data."""
    return False  # Just hold out some generated data for evals.

  @property
  def frame_height(self):
    return 64

  @property
  def frame_width(self):
    return 64

  @property
  def total_number_of_frames(self):
    return 10000

  @property
  def extra_reading_spec(self):
    """Additional data fields to store on disk and their decoders."""
    data_fields = {
        "frame_number": tf.FixedLenFeature([1], tf.int64),
    }
    decoders = {
        "frame_number": tf.contrib.slim.tfexample_decoder.Tensor(
            tensor_key="frame_number"),
    }
    return data_fields, decoders

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.input_modality = {
        "inputs": ("video", 256),
        "input_frame_number": ("symbol:identity", 1)
    }
    p.target_modality = ("video", 256)

  def generate_samples(self, data_dir, tmp_dir, unused_dataset_split):
    frame_number = 0
    for _ in range(self.total_number_of_frames):
      frame = np.zeros([self.frame_height, self.frame_width, self.num_channels],
                       dtype=np.uint8)
      yield {"frame": frame, "frame_number": [frame_number]}
      frame_number += 1
