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

r"""Training of model-based RL agent assuming a fully trained world model.

Example invocation:

python -m tensor2tensor.rl.trainer_model_based_agent_only \
    --loop_hparams_set=rl_modelrl_base \
    --world_model_dir=$HOME/world_model/ \
    --data_dir=$HOME/data/ \
    --output_dir=$HOME/ppo_agent_only/ \
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.bin import t2t_trainer  # pylint: disable=unused-import
from tensor2tensor.data_generators import gym_env
from tensor2tensor.rl import trainer_model_based
from tensor2tensor.rl import trainer_model_based_params


import tensorflow.compat.v1 as tf


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("world_model_dir", "",
                    "Directory containing checkpoints of the world model.")


def get_simulated_problem_name(game):
  game_with_mode = game
  if game in gym_env.ATARI_GAMES:
    game_with_mode += "_deterministic-v4"
  return "gym_simulated_discrete_problem_with_agent_on_%s" % game_with_mode


def main(_):
  hparams = trainer_model_based_params.create_loop_hparams()
  problem_name = get_simulated_problem_name(hparams.game)
  world_model_dir = FLAGS.world_model_dir
  agent_model_dir = FLAGS.output_dir
  event_dir = FLAGS.output_dir
  epoch_data_dir = FLAGS.data_dir  # only required for initial frames

  trainer_model_based.train_agent(
      problem_name,
      agent_model_dir,
      event_dir,
      world_model_dir,
      epoch_data_dir,
      hparams,
      0,
      epoch=0,
      is_final_epoch=True)

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
