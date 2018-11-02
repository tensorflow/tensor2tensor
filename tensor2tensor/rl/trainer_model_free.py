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

r"""Training of RL agent with PPO algorithm.

Example invocation:

python -m tensor2tensor.rl.trainer_model_free \
    --output_dir=$HOME/t2t/rl_v1 \
    --hparams_set=pong_model_free \
    --loop_hparams='num_agents=15'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.data_generators import gym_env
from tensor2tensor.models.research import rl
from tensor2tensor.rl import rl_trainer_lib
from tensor2tensor.utils import flags as t2t_flags  # pylint: disable=unused-import
from tensor2tensor.utils import trainer_lib

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

# To maintain compatibility with some internal libs, we guard against these flag
# definitions possibly erring. Apologies for the ugliness.
try:
  flags.DEFINE_string("output_dir", "", "Base output directory for run.")
except:  # pylint: disable=bare-except
  pass


def initialize_env_specs(hparams):
  """Initializes env_specs using T2TGymEnvs."""
  if getattr(hparams, "game", None):
    game_name = gym_env.camel_case_name(hparams.game)
    env = gym_env.T2TGymEnv("{}Deterministic-v4".format(game_name),
                            batch_size=hparams.num_agents)
    env.start_new_epoch(0)
    hparams.add_hparam("env_fn", rl.make_real_env_fn(env))
    eval_env = gym_env.T2TGymEnv("{}Deterministic-v4".format(game_name),
                                 batch_size=hparams.num_eval_agents)
    eval_env.start_new_epoch(0)
    hparams.add_hparam("eval_env_fn", rl.make_real_env_fn(eval_env))
  return hparams


def train(hparams, output_dir, report_fn=None):
  hparams = initialize_env_specs(hparams)
  rl_trainer_lib.train(hparams, output_dir, output_dir, report_fn=report_fn)


def main(_):
  hparams = trainer_lib.create_hparams(FLAGS.hparams_set, FLAGS.hparams)
  train(hparams, FLAGS.output_dir)


if __name__ == "__main__":
  tf.app.run()
