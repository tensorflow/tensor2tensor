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

"""Tests for rl_utils."""

import os

from tensor2tensor.rl.restarter import Restarter

import tensorflow.compat.v1 as tf


TEST_MODE_1 = "mode1"
TEST_MODE_2 = "mode2"
TEST_NUM_STEPS = 2


class RestarterTest(tf.test.TestCase):

  def setUp(self):
    self.out_dir = tf.test.get_temp_dir()
    tf.gfile.DeleteRecursively(self.out_dir)
    tf.gfile.MkDir(self.out_dir)

  def create_checkpoint(self, global_step):
    checkpoint_name = "model.ckpt-{}".format(global_step)
    for suffix in ("index", "meta", "data-00000-of-00001"):
      filename = "{}.{}".format(checkpoint_name, suffix)
      # Just create the file.
      with tf.gfile.Open(os.path.join(self.out_dir, filename), "w") as f:
        f.write("")
    tf.train.update_checkpoint_state(self.out_dir, checkpoint_name)

  def run_single_mode(self, mode, target_local_step, target_global_step):
    restarter = Restarter(mode, self.out_dir, target_local_step)
    with restarter.training_loop():
      self.create_checkpoint(target_global_step)

  def assert_first_run(self, restarter, steps_to_go, target_global_step):
    self.assertFalse(restarter.should_skip)
    self.assertFalse(restarter.restarting)
    self.assertEqual(restarter.steps_to_go, steps_to_go)
    self.assertEqual(restarter.target_global_step, target_global_step)

  def test_runs_in_single_mode(self):
    restarter = Restarter(
        TEST_MODE_1, self.out_dir, target_local_step=TEST_NUM_STEPS
    )
    self.assert_first_run(
        restarter, steps_to_go=TEST_NUM_STEPS, target_global_step=TEST_NUM_STEPS
    )

  def test_runs_in_two_modes(self):
    global_step = TEST_NUM_STEPS
    local_steps = {
        TEST_MODE_1: TEST_NUM_STEPS,
        TEST_MODE_2: 0
    }
    self.run_single_mode(TEST_MODE_1, local_steps[TEST_MODE_1], global_step)

    for mode in [TEST_MODE_2, TEST_MODE_1]:
      global_step += TEST_NUM_STEPS
      local_steps[mode] += TEST_NUM_STEPS
      restarter = Restarter(
          mode, self.out_dir, target_local_step=local_steps[mode]
      )
      self.assert_first_run(
          restarter, steps_to_go=TEST_NUM_STEPS, target_global_step=global_step
      )
      with restarter.training_loop():
        self.create_checkpoint(global_step)

  def test_skips_already_done(self):
    self.run_single_mode(
        TEST_MODE_1, target_local_step=TEST_NUM_STEPS,
        target_global_step=TEST_NUM_STEPS
    )

    restarter = Restarter(
        TEST_MODE_1, self.out_dir, target_local_step=TEST_NUM_STEPS
    )
    # We should skip the training as those steps are already completed.
    self.assertTrue(restarter.should_skip)

  def test_restarts_after_interruption(self):
    # Run some initial training first.
    self.run_single_mode(
        TEST_MODE_1, target_local_step=TEST_NUM_STEPS,
        target_global_step=TEST_NUM_STEPS
    )
    global_step = TEST_NUM_STEPS

    restarter = Restarter(
        TEST_MODE_2, self.out_dir, target_local_step=2
    )
    with self.assertRaises(RuntimeError):
      global_step += 1
      with restarter.training_loop():
        self.create_checkpoint(global_step)
        # Simulate training interruption after the first step.
        raise RuntimeError
    restarter = Restarter(
        TEST_MODE_2, self.out_dir, target_local_step=2
    )

    self.assertFalse(restarter.should_skip)
    self.assertTrue(restarter.restarting)
    # Training should resume after the first step.
    self.assertEqual(restarter.steps_to_go, 1)

if __name__ == "__main__":
  tf.test.main()
