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
"""Generate trajectories to disk with random or ckpt agent.

TODO: Usage
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import gym_problems_specs
from tensor2tensor.utils import registry

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "", "Data directory.")
flags.DEFINE_string("tmp_dir", "/tmp/t2t_datagen",
                    "Temporary storage directory.")
flags.DEFINE_string("game", None, "Atari game to generate data for.")
flags.DEFINE_integer("num_env_steps", 5000, "Number of steps to roll out.")
flags.DEFINE_boolean("eval", False, "Whether to run in eval mode.")


def main(_):

  tf.gfile.MakeDirs(FLAGS.data_dir)
  tf.gfile.MakeDirs(FLAGS.tmp_dir)

  # Create problem if not already defined
  problem_name = "gym_discrete_problem_with_agent_on_%s" % FLAGS.game
  if problem_name not in registry.list_problems():
    gym_problems_specs.create_problems_for_game(FLAGS.game)

  # Generate
  tf.logging.info("Running %s environment for %d steps for trajectories.",
                  FLAGS.game, FLAGS.num_env_steps)
  problem = registry.problem(problem_name)
  problem.settable_num_steps = FLAGS.num_env_steps
  problem.settable_eval_phase = FLAGS.eval
  problem.generate_data(FLAGS.data_dir, FLAGS.tmp_dir)

  # Log stats
  if problem.statistics.number_of_dones:
    mean_reward = (problem.statistics.sum_of_rewards /
                   problem.statistics.number_of_dones)
    tf.logging.info("Mean reward: %.2f, Num dones: %d",
                    mean_reward,
                    problem.statistics.number_of_dones)


if __name__ == "__main__":
  tf.app.run(main)
