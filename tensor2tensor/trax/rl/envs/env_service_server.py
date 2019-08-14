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

"""Server that acts as a remote env.

NOTE: This is a fork from T2T's `env_service_server.py` since we need to
link in some TRAX specific envs and gin configuration. This also enables
eager execution.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging
import gin
from tensor2tensor.envs import env_problem_utils
from tensor2tensor.envs import server_utils
from tensor2tensor.rl.google import atari_utils
from tensor2tensor.trax.rl import envs  # pylint: disable=unused-import
import tensorflow as tf



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

# Gin related flags.
flags.DEFINE_multi_string("gin_config_file",
                          None,
                          "Configuration file with parameters (.gin).")
flags.DEFINE_multi_string("gin_config_string",
                          [],
                          "Configuration parameters (gin string).")


# TODO(afrozm): Check this.
flags.DEFINE_bool("enable_eager_execution", False, "")

# Since we're only dealing with 1 GPU machines here.
_MAX_CONCURRENCY = 1
_ADDRESS_FORMAT = "[::]:{}"


def initialize_gin():
  gin_bindings = FLAGS.gin_config_string
  if not (FLAGS.gin_config_file or gin_bindings):
    return
  gin.parse_config_files_and_bindings(FLAGS.gin_config_file, gin_bindings)


def main(argv):
  del argv

  if FLAGS.enable_eager_execution:
    tf.enable_eager_execution()

  output_dir = FLAGS.output_dir

  # Initialize Gin.
  initialize_gin()

  output_dir = os.path.join(output_dir, str(FLAGS.replica))

  env_kwargs = {"output_dir": output_dir}

  env = env_problem_utils.make_env(
      batch_size=1,
      env_problem_name=FLAGS.env_problem_name,
      resize=FLAGS.resize,
      resized_height=FLAGS.resized_height,
      resized_width=FLAGS.resized_width,
      max_timestep=FLAGS.max_timestep,
      clip_rewards=FLAGS.clip_rewards,
      **env_kwargs)

  logging.info("Replica[%s] is ready to serve requests.", FLAGS.replica)
  server_utils.serve(output_dir, env, FLAGS.env_service_port)


if __name__ == "__main__":
  app.run(main)
