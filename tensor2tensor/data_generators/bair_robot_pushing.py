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
"""Berkeley (BAIR) robot pushing dataset.

Self-Supervised Visual Planning with Temporal Skip Connections
Frederik Ebert, Chelsea Finn, Alex X. Lee, and Sergey Levine.
https://arxiv.org/abs/1710.05268

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import numpy as np

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import video_utils
from tensor2tensor.utils import registry

import tensorflow as tf

DATA_URL = (
    "http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar")


# Lazy load PIL.Image
def PIL_Image():  # pylint: disable=invalid-name
  from PIL import Image  # pylint: disable=g-import-not-at-top
  return Image


@registry.register_problem
class VideoBairRobotPushing(video_utils.VideoProblem):
  """Berkeley (BAIR) robot pushing dataset."""

  @property
  def num_channels(self):
    return 3

  @property
  def frame_height(self):
    return 64

  @property
  def frame_width(self):
    return 64

  @property
  def is_generate_per_split(self):
    return True

  @property
  def total_number_of_frames(self):
    # TODO(mbz): correct this number to be the real total number of frames.
    return 30 * 10 * 1000

  def parse_frames(self, filenames):
    image_key = "{}/image_aux1/encoded"
    action_key = "{}/action"
    state_key = "{}/endeffector_pos"

    for f in filenames:
      print("Parsing ", f)
      for serialized_example in tf.python_io.tf_record_iterator(f):
        x = tf.train.Example()
        x.ParseFromString(serialized_example)
        # there are 4 features per frame
        # main image, aux image, actions and states
        nf = len(x.features.feature.keys()) // 4

        for i in range(nf):
          image_name = image_key.format(i)
          action_name = action_key.format(i)
          state_name = state_key.format(i)

          byte_str = x.features.feature[image_name].bytes_list.value[0]
          img = PIL_Image().frombytes(
              "RGB", (self.frame_width, self.frame_height), byte_str)
          arr = np.array(img.getdata())
          frame = arr.reshape(
              self.frame_width, self.frame_height, self.num_channels)

          state = x.features.feature[state_name].float_list.value
          action = x.features.feature[action_name].float_list.value

          yield i, frame, state, action

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    path = generator_utils.maybe_download(
        tmp_dir, os.path.basename(DATA_URL), DATA_URL)

    tar = tarfile.open(path)
    tar.extractall(tmp_dir)
    tar.close()

    if dataset_split == problem.DatasetSplit.TRAIN:
      base_dir = os.path.join(tmp_dir, "softmotion30_44k/train/*")
    else:
      base_dir = os.path.join(tmp_dir, "softmotion30_44k/test/*")

    filenames = tf.gfile.Glob(base_dir)
    for frame_number, frame, state, action in self.parse_frames(filenames):
      yield {
          "frame_number": [frame_number],
          "frame": frame,
          "state": state,
          "action": action,
      }

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.input_modality = {
        # Pixels are in 0..255 range.
        "inputs": ("video", 256),
    }
    p.target_modality = {
        "targets": ("video", 256),
    }
