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
from tensor2tensor.data_generators import gym_problems
from tensor2tensor.models.research import rl as rl_models
from tensor2tensor.rl import rl_trainer_lib
from tensor2tensor.utils import flags as t2t_flags  # pylint: disable=unused-import
from tensor2tensor.utils import registry
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


@registry.register_hparams
def pong_model_free():
  """TODO(piotrmilos): Document this."""
  hparams = tf.contrib.training.HParams(
      epochs_num=4,
      eval_every_epochs=2,
      num_agents=10,
      optimization_epochs=3,
      epoch_length=30,
      entropy_loss_coef=0.003,
      learning_rate=8e-05,
      optimizer="Adam",
      policy_network=rl_models.feed_forward_cnn_small_categorical_fun,
      gae_lambda=0.985,
      num_eval_agents=1,
      max_gradients_norm=0.5,
      gae_gamma=0.985,
      optimization_batch_size=4,
      clipping_coef=0.2,
      value_loss_coef=1,
      save_models_every_epochs=False)
  hparams.add_hparam("environment_spec",
                     gym_problems.standard_atari_env_spec("PongNoFrameskip-v4"))
  hparams.add_hparam(
      "environment_eval_spec",
      gym_problems.standard_atari_env_eval_spec("PongNoFrameskip-v4"))

  return hparams


def main(_):
  hparams = trainer_lib.create_hparams(FLAGS.hparams_set, FLAGS.hparams)
  rl_trainer_lib.train(hparams, FLAGS.output_dir, FLAGS.output_dir)


if __name__ == "__main__":
  tf.app.run()
