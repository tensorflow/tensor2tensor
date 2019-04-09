# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

"""Tests for tensor2tensor.trax.rlax.ppo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensor2tensor.trax.rlax import ppo
from tensorflow import test


class PpoTest(test.TestCase):

  def test_rewards_to_go(self):
    time_steps = 4
    # [1., 1., 1., 1.]
    rewards = np.ones((time_steps,))
    # No discounting.
    self.assertAllEqual(ppo.rewards_to_go(rewards, gamma=1.0),
                        np.array([4., 3., 2., 1.]))
    # Discounting.
    self.assertAllEqual(ppo.rewards_to_go(rewards, gamma=0.5),
                        np.array([1.875, 1.75, 1.5, 1.]))


if __name__ == "__main__":
  test.main()
