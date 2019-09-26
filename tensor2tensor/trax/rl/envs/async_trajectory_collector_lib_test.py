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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.trax.rl import ppo
from tensor2tensor.trax.rl.envs import async_trajectory_collector_lib as async_lib
from tensorflow import test
from tensorflow.io import gfile


class AsyncTrajectoryCollectorLibTest(test.TestCase):

  def test_get_newer_policy_model_file(self):
    output_dir = self.get_temp_dir()

    def write_policy_model_file(epoch):
      fname = ppo.get_policy_model_file_from_epoch(output_dir, epoch)
      with gfile.GFile(fname, "w") as f:
        f.write("some data")
      return fname

    # No file exists currently.
    self.assertIsNone(async_lib.get_newer_policy_model_file(output_dir))

    # Write a policy model file.
    epoch = 0
    policy_model_filename = write_policy_model_file(epoch)

    # See that we get it.
    actual_policy_file, actual_epoch = (
        async_lib.get_newer_policy_model_file(output_dir, min_epoch=-1))

    self.assertEqual(actual_policy_file, policy_model_filename)
    self.assertEqual(actual_epoch, epoch)

    # If we now ask for a larger epoch, we don't get it.
    self.assertIsNone(
        async_lib.get_newer_policy_model_file(output_dir, min_epoch=0))

    # Write a newer epoch and expect to get that with appropriate min_epoch.
    epoch = 1
    policy_model_filename = write_policy_model_file(epoch)
    actual_policy_file, actual_epoch = (
        async_lib.get_newer_policy_model_file(output_dir, min_epoch=0))
    self.assertEqual(actual_policy_file, policy_model_filename)
    self.assertEqual(actual_epoch, epoch)


if __name__ == "__main__":
  test.main()
