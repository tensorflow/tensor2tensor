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

"""PPO binary over a gym env."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
from tensor2tensor.trax.rlax import ppo

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "CartPole-v0", "Name of the environment to make.")
flags.DEFINE_integer("epochs", 100, "Number of epochs to run for.")
flags.DEFINE_integer("log_level", logging.INFO, "Log level.")


def main(unused_argv):
  logging.set_verbosity(FLAGS.log_level)
  ppo.training_loop(env_name=FLAGS.env, epochs=FLAGS.epochs)


if __name__ == "__main__":
  app.run(main)
