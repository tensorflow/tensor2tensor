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

"""Tests for tensor2tensor.envs.rendered_env_problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.envs import env_problem
from tensor2tensor.envs import env_problem_utils
from tensor2tensor.envs import rendered_env_problem
from tensor2tensor.envs.mujoco_problems import ReacherEnvProblem
import tensorflow.compat.v1 as tf


class RenderedEnvProblemTest(tf.test.TestCase):

  def test_generate_timesteps(self):
    env = ReacherEnvProblem()
    env.initialize(batch_size=2)
    env_problem_utils.play_env_problem_randomly(env, num_steps=5)
    env.trajectories.complete_all_trajectories()

    frame_number = 0
    for time_step in env._generate_time_steps(
        env.trajectories.completed_trajectories):
      # original observation should not be in time_step
      self.assertNotIn(env_problem.OBSERVATION_FIELD, time_step)
      # validate frame
      self.assertIn(rendered_env_problem._IMAGE_ENCODED_FIELD, time_step)
      self.assertIn(rendered_env_problem._IMAGE_HEIGHT_FIELD, time_step)
      self.assertIn(rendered_env_problem._IMAGE_WIDTH_FIELD, time_step)
      self.assertIn(rendered_env_problem._IMAGE_FORMAT_FIELD, time_step)
      self.assertIn(rendered_env_problem._FRAME_NUMBER_FIELD, time_step)

      decoded_frame = tf.image.decode_png(
          time_step[rendered_env_problem._IMAGE_ENCODED_FIELD][0])

      decoded_frame = self.evaluate(decoded_frame)

      self.assertListEqual(
          [env.frame_height, env.frame_width, env.num_channels],
          list(decoded_frame.shape))
      self.assertListEqual([rendered_env_problem._FORMAT],
                           time_step[rendered_env_problem._IMAGE_FORMAT_FIELD])
      self.assertListEqual([frame_number],
                           time_step[rendered_env_problem._FRAME_NUMBER_FIELD])
      self.assertListEqual([env.frame_width],
                           time_step[rendered_env_problem._IMAGE_WIDTH_FIELD])
      self.assertListEqual([env.frame_height],
                           time_step[rendered_env_problem._IMAGE_HEIGHT_FIELD])
      frame_number += 1
      frame_number %= 6


if __name__ == "__main__":
  tf.test.main()
