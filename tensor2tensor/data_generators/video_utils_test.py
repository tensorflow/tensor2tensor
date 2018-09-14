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
"""video_utils test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tensor2tensor.data_generators import video_utils
from tensor2tensor.utils import decoding

import tensorflow as tf


class VideoUtilsTest(tf.test.TestCase):

  def testConvertPredictionsToVideoSummaries(self):
    # Initialize predictions.
    rng = np.random.RandomState(0)
    inputs = rng.randint(0, 255, (2, 32, 32, 3))
    outputs = rng.randint(0, 255, (5, 32, 32, 3))
    targets = rng.randint(0, 255, (5, 32, 32, 3))

    # batch it up.
    prediction = [{"outputs": outputs, "inputs": inputs, "targets": targets}]*50
    predictions = [prediction]
    decode_hparams = decoding.decode_hparams()

    decode_hooks = decoding.DecodeHookArgs(
        estimator=None, problem=None, output_dirs=None,
        hparams=decode_hparams, decode_hparams=decode_hparams,
        predictions=predictions)
    summaries = video_utils.display_video_hooks(decode_hooks)
    # 10 input vids + 10 output vids + 10 frame-by-frame.
    self.assertEqual(len(summaries), 30)
    for summary in summaries:
      self.assertTrue(isinstance(summary, tf.Summary.Value))


if __name__ == "__main__":
  tf.test.main()
