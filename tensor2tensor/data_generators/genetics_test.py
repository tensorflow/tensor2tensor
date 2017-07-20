# Copyright 2017 The Tensor2Tensor Authors.
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

"""Tests for Genetics problems."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np

from tensor2tensor.data_generators import genetics

import tensorflow as tf


class GeneticsTest(tf.test.TestCase):

  def _oneHotBases(self, bases):
    one_hots = []
    for base_id in bases:
      one_hot = [False] * 4
      if base_id < 4:
        one_hot[base_id] = True
      one_hots.append(one_hot)
    return np.array(one_hots)

  def testRecordToExample(self):
    inputs = self._oneHotBases([0, 1, 3, 4, 1, 0])
    mask = np.array([True, False, True])
    outputs = np.array([[1.0, 2.0, 3.0], [5.0, 1.0, 0.2], [5.1, 2.3, 2.3]])
    ex_dict = genetics.to_example_dict(inputs, mask, outputs)

    self.assertAllEqual([2, 3, 5, 6, 3, 2, 1], ex_dict["inputs"])
    self.assertAllEqual([1.0, 0.0, 1.0], ex_dict["targets_mask"])
    self.assertAllEqual([1.0, 2.0, 3.0, 5.0, 1.0, 0.2, 5.1, 2.3, 2.3],
                        ex_dict["targets"])
    self.assertAllEqual([3, 3], ex_dict["targets_shape"])

  def testGenerateShardArgs(self):
    num_examples = 37
    num_shards = 4
    outfiles = [str(i) for i in range(num_shards)]
    shard_args = genetics.generate_shard_args(outfiles, num_examples)

    starts, ends, fnames = zip(*shard_args)
    self.assertAllEqual([0, 9, 18, 27], starts)
    self.assertAllEqual([9, 18, 27, 37], ends)
    self.assertAllEqual(fnames, outfiles)


if __name__ == "__main__":
  tf.test.main()
