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

"""Server that acts as a remote env."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
from absl import logging
from tensor2tensor.envs import env_problem_utils
from tensor2tensor.envs import server_utils

FLAGS = flags.FLAGS

flags.DEFINE_bool("xm", False, "Copy atari roms?")
flags.DEFINE_integer("env_service_port", 7777, "Port on which to run.")
flags.DEFINE_string("env_problem_name", None, "Name of the EnvProblem to make.")
flags.DEFINE_string("max_timestep",
                    None,
                    "If set to an integer, maximum number of time-steps in a "
                    "trajectory. The bare env is TimeLimit wrapped.")
flags.DEFINE_boolean("resize", False, "If true, resize the game frame")
flags.DEFINE_integer("resized_height", 105, "Resized height of the game frame.")
flags.DEFINE_integer("resized_width", 80, "Resized width of the game frame.")
flags.DEFINE_string("output_dir", "", "Output dir.")
flags.DEFINE_bool("use_tpu", False, "Whether we're running on TPU.")
flags.DEFINE_integer("replica", 0, "Basically to append to output_dir")
flags.DEFINE_bool("clip_rewards",
                  True,
                  "Whether to clip and discretize the rewards.")

# Since we're only dealing with 1 GPU machines here.
_MAX_CONCURRENCY = 1
_ADDRESS_FORMAT = "[::]:{}"


def main(argv):
  del argv
  output_dir = FLAGS.output_dir

  output_dir = os.path.join(output_dir, str(FLAGS.replica))

  env = env_problem_utils.make_env(
      batch_size=1,
      env_problem_name=FLAGS.env_problem_name,
      resize=FLAGS.resize,
      resized_height=FLAGS.resized_height,
      resized_width=FLAGS.resized_width,
      max_timestep=FLAGS.max_timestep,
      clip_rewards=FLAGS.clip_rewards)

  logging.info("Replica[%s] is ready to serve requests.", FLAGS.replica)
  server_utils.serve(output_dir, env, FLAGS.env_service_port)


if __name__ == "__main__":
  app.run(main)
