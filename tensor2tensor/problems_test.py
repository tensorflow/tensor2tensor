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

"""tensor2tensor.problems test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor import problems

import tensorflow as tf

MODES = tf.estimator.ModeKeys


class ProblemsTest(tf.test.TestCase):

  def testBuildDataset(self):
    # See all the available problems
    self.assertTrue(len(problems.available()) > 10)

    # Retrieve a problem by name
    problem = problems.problem("translate_ende_wmt8k")

    # Access train and dev datasets through Problem
    train_dataset = problem.dataset(MODES.TRAIN)
    dev_dataset = problem.dataset(MODES.EVAL)

    # Access vocab size and other info (e.g. the data encoders used to
    # encode/decode data for the feature, used below) through feature_info.
    feature_info = problem.feature_info
    self.assertTrue(feature_info["inputs"].vocab_size > 0)
    self.assertTrue(feature_info["targets"].vocab_size > 0)

    train_example = train_dataset.make_one_shot_iterator().get_next()
    dev_example = dev_dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
      train_ex_val, _ = sess.run([train_example, dev_example])
      _ = feature_info["inputs"].encoder.decode(train_ex_val["inputs"])
      _ = feature_info["targets"].encoder.decode(train_ex_val["targets"])


if __name__ == "__main__":
  tf.test.main()
