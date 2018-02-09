# coding=utf-8
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

"""Training of RL agent with PPO algorithm."""

import tensorflow as tf

from tensor2tensor.rl import rl_trainer_lib


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("event_dir", None,
                    "Where to store the event file.")
flags.DEFINE_string("environment", "pendulum",
                    "Which environment should be used for training.")


def main(_):
  name_to_env = {
      "pendulum": rl_trainer_lib.pendulum_params,
      "cartpole": rl_trainer_lib.cartpole_params,
  }
  if FLAGS.environment not in name_to_env:
    raise ValueError(
        "Environment with name %s not configured." % FLAGS.environment)
  rl_trainer_lib.train(name_to_env[FLAGS.environment](), FLAGS.event_dir)


if __name__ == "__main__":
  tf.app.run()
