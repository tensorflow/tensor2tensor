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
"""Google robot pushing dataset.

Unsupervised Learning for Physical Interaction through Video Prediction
Chelsea Finn, Ian Goodfellow, Sergey Levine
https://arxiv.org/abs/1605.07157

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import numpy as np

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import video_utils
from tensor2tensor.utils import registry

import tensorflow as tf

BASE_URL = "https://storage.googleapis.com/brain-robotics-data/push/"
DATA_TRAIN = (264, "push_train/push_train.tfrecord-{:05d}-of-00264")
DATA_TEST_SEEN = (5, "/push_testseen/push_testseen.tfrecord-{:05d}-of-00005")
DATA_TEST_NOVEL = (5, "/push_testnovel/push_testnovel.tfrecord-{:05d}-of-00005")


# Lazy load PIL.Image
def PIL_Image():  # pylint: disable=invalid-name
  from PIL import Image  # pylint: disable=g-import-not-at-top
  return Image


@registry.register_problem
class VideoGoogleRobotPushing(video_utils.VideoProblem):
  """Google robot pushing dataset."""

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
  def total_number_of_frames(self):
    # TODO(mbz): correct this number to be the real total number of frames.
    return 50 * 10 * 1000

  @property
  def max_number_of_frames_per_video(self):
    return 60

  @property
  def is_generate_per_split(self):
    return True

  def parse_frames(self, filename):
    image_key = "move/{}/image/encoded"
    action_key = "move/{}/commanded_pose/vec_pitch_yaw"
    state_key = "move/{}/endeffector/vec_pitch_yaw"

    for serialized_example in tf.python_io.tf_record_iterator(filename):
      x = tf.train.Example()
      x.ParseFromString(serialized_example)
      # there are 6 features per frame
      nf = len(x.features.feature.keys()) // 6
      # it seems features after 60 don't have any image
      nf = min(nf, self.max_number_of_frames_per_video)

      for i in range(nf):
        image_name = image_key.format(i)
        action_name = action_key.format(i)
        state_name = state_key.format(i)

        byte_str = x.features.feature[image_name].bytes_list.value[0]
        img = PIL_Image().open(io.BytesIO(byte_str))
        # The original images are much bigger than 64x64
        img = img.resize((self.frame_width, self.frame_height),
                         resample=PIL_Image().BILINEAR)
        arr = np.array(img.getdata())
        frame = arr.reshape(
            self.frame_width, self.frame_height, self.num_channels)

        state = x.features.feature[state_name].float_list.value
        action = x.features.feature[action_name].float_list.value

        yield i, frame, state, action

  def get_urls(self, count, url_part):
    template = os.path.join(BASE_URL, url_part)
    return [template.format(i) for i in range(count)]

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    if dataset_split == problem.DatasetSplit.TRAIN:
      urls = self.get_urls(DATA_TRAIN[0], DATA_TRAIN[1])
    else:
      urls = self.get_urls(DATA_TEST_SEEN[0], DATA_TEST_SEEN[1])
      urls += self.get_urls(DATA_TEST_NOVEL[0], DATA_TEST_NOVEL[1])

    for url in urls:
      path = generator_utils.maybe_download(tmp_dir, os.path.basename(url), url)
      for frame_number, frame, state, action in self.parse_frames(path):
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
