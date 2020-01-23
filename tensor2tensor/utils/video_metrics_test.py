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

"""video metrics test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensor2tensor.utils import video_metrics
import tensorflow.compat.v1 as tf


class VideoMetricsTest(tf.test.TestCase):

  def test_reduce_to_best_decode(self):
    # num_decodes=2, num_samples=3, num_frames=4
    decode1 = [
        [30.0, 32.0, 33.0, 34.0],
        [22.0, 19.0, 12.0, 13.0],
        [30.0, 10.0, 30.0, 10.0]]
    decode2 = [
        [22.0, 19.0, 12.0, 13.0],
        [30.0, 32.0, 33.0, 34.0],
        [25.0, 25.0, 25.0, 25.0]]
    all_decodes = [decode1, decode2]
    all_decodes = np.array(all_decodes)
    best_decode, best_decode_ind = video_metrics.reduce_to_best_decode(
        all_decodes, np.argmax)
    worst_decode, worst_decode_ind = video_metrics.reduce_to_best_decode(
        all_decodes, np.argmin)
    exp_best_decode = [
        [30.0, 32.0, 33.0, 34.0],
        [30.0, 32.0, 33.0, 34.0],
        [25.0, 25.0, 25.0, 25.0]]
    exp_worst_decode = [
        [22.0, 19.0, 12.0, 13.0],
        [22.0, 19.0, 12.0, 13.0],
        [30.0, 10.0, 30.0, 10.0]]
    self.assertTrue(np.allclose(best_decode, exp_best_decode))
    self.assertTrue(np.allclose(worst_decode, exp_worst_decode))
    self.assertTrue(np.allclose(best_decode_ind, [0, 1, 1]))
    self.assertTrue(np.allclose(worst_decode_ind, [1, 0, 0]))


if __name__ == '__main__':
  tf.test.main()
