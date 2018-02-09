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

# Dependency imports

from tensor2tensor.rl import rl_trainer_lib

import tensorflow as tf


def main(_):
  rl_trainer_lib.train(rl_trainer_lib.example_params())


if __name__ == "__main__":
  tf.app.run()
