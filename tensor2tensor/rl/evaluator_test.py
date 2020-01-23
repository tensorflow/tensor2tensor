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

"""Tests the evaluator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.rl import evaluator
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


class EvalTest(tf.test.TestCase):

  def test_evaluate_pong_random_agent(self):
    loop_hparams = registry.hparams("rlmb_tiny")
    planner_hparams = registry.hparams("planner_tiny")
    temp_dir = tf.test.get_temp_dir()
    evaluator.evaluate(
        loop_hparams, planner_hparams, temp_dir, temp_dir, temp_dir,
        agent_type="random", eval_mode="agent_real", eval_with_learner=False,
        log_every_steps=None, debug_video_path=""
    )


if __name__ == "__main__":
  tf.test.main()
