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
    --loop_hparams='batch_size=15'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensor2tensor.data_generators import gym_env
from tensor2tensor.models.research import rl
from tensor2tensor.rl.ppo_learner import PPOLearner
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


LEARNERS = {
    "ppo": PPOLearner
}


def update_hparams_from_hparams(target_hparams, source_hparams, prefix):
  """Copy a subset of hparams to target_hparams."""
  for (param_name, param_value) in six.iteritems(source_hparams.values()):
    if param_name.startswith(prefix):
      target_hparams.set_hparam(param_name[len(prefix):], param_value)


def initialize_env_specs(hparams):
  """Initializes env_specs using T2TGymEnvs."""
  if getattr(hparams, "game", None):
    game_name = gym_env.camel_case_name(hparams.game)
    env = gym_env.T2TGymEnv("{}Deterministic-v4".format(game_name),
                            batch_size=hparams.batch_size)
    env.start_new_epoch(0)
    hparams.add_hparam("env_fn", rl.make_real_env_fn(env))
    eval_env = gym_env.T2TGymEnv("{}Deterministic-v4".format(game_name),
                                 batch_size=hparams.eval_batch_size)
    eval_env.start_new_epoch(0)
    hparams.add_hparam("eval_env_fn", rl.make_real_env_fn(eval_env))
  return hparams


def train(hparams, output_dir, report_fn=None):
  hparams = initialize_env_specs(hparams)
  learner = LEARNERS[hparams.base_algo](
      hparams.frame_stack_size, FLAGS.output_dir, output_dir
  )
  policy_hparams = trainer_lib.create_hparams(hparams.base_algo_params)
  update_hparams_from_hparams(
      policy_hparams, hparams, hparams.base_algo + "_"
  )
  learner.train(
      hparams.env_fn, policy_hparams, simulated=False, save_continuously=True,
      epoch=0, eval_env_fn=hparams.eval_env_fn, report_fn=report_fn
  )


def main(_):
  hparams = trainer_lib.create_hparams(FLAGS.hparams_set, FLAGS.hparams)
  train(hparams, FLAGS.output_dir)


if __name__ == "__main__":
  tf.app.run()
