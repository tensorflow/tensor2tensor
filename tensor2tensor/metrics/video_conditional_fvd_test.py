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

"""Tests for video_conditional_fvd."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.metrics import video_conditional_fvd
import tensorflow.compat.v1 as tf


class VideoConditionalFvdTest(tf.test.TestCase):

  def test_sample(self):
    dataset = video_conditional_fvd.VideoEvaluationDataset(
        n_input_frames=4,
        n_output_frames=10,
        get_video_batch_fn=None)
    model = video_conditional_fvd.Model(
        apply_fn=None,
        load_fn=None)
    video_conditional_fvd.evaluate_model(dataset, model, 10, 16)


if __name__ == '__main__':
  tf.test.main()
