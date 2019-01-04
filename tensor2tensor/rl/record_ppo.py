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

"""

Run this script with the same parameters as trainer_model_based.py /
trainer_model_free.py. Note that values of most of them have no effect,
so running just

python -m tensor2tensor/rl/record_ppo.py \
    --output_dir=path/to/your/experiment \
    --loop_hparams_set=rlmb_base

might work for you.

More advanced example:

python -m tensor2tensor/rl/record_ppo.py \
    --output_dir=path/to/your/experiment \
    --loop_hparams_set=rlmb_base \
    --loop_hparams=game=<right game in case of problems> \
    --video_dir=my/video/dir \
    --env=real \
    --simulated_episode_len=50 \
    --num_episodes=5

Check flags definitions under imports for more details.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from gym.wrappers import TimeLimit

from tensor2tensor.rl.envs.simulated_batch_gym_env import FlatBatchEnv
from player_utils import wrap_with_monitor, PPOPolicyInferencer, \
  load_data_and_make_simulated_env
# Import flags from t2t_trainer and trainer_model_based
from tensor2tensor.bin import t2t_trainer  # pylint: disable=unused-import
import tensor2tensor.rl.trainer_model_based_params # pylint: disable=unused-import

from tensor2tensor.data_generators.gym_env import T2TGymEnv
from tensor2tensor.utils import registry
import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string("video_dir", "/tmp/record_ppo_out",
                    "Where to save recorded trajectories.")
flags.DEFINE_string("epoch", "last",
                    "Data from which epoch to use.")
flags.DEFINE_boolean("simulated_env", True,
                     "Either to use 'simulated' or 'real' env.")
flags.DEFINE_integer("simulated_episode_len", 100,
                     "Timesteps limit for simulated env")
flags.DEFINE_integer("num_episodes", 20,
                     "How many episodes record.")


def main(_):
  # TODO(konradczechowski): add initial frame stack for policy?
  hparams = registry.hparams(FLAGS.loop_hparams_set)
  hparams.parse(FLAGS.loop_hparams)
  # Not important for experiments past 2018
  if "wm_policy_param_sharing" not in hparams.values().keys():
    hparams.add_hparam("wm_policy_param_sharing", False)
  output_dir = FLAGS.output_dir
  video_dir = FLAGS.video_dir
  epoch = FLAGS.epoch if FLAGS.epoch == "last" else int(FLAGS.epoch)

  if FLAGS.simulated_env:
    env = load_data_and_make_simulated_env(output_dir, hparams,
                                           which_epoch_data=epoch)
    env = TimeLimit(env, max_episode_steps=FLAGS.simulated_episode_len)
  else:
    env = T2TGymEnv.setup_and_load_epoch(
        hparams, data_dir=os.path.join(output_dir, "data"),
        which_epoch_data=None)
    env = FlatBatchEnv(env)

  env = wrap_with_monitor(env, video_dir=video_dir)
  ppo = PPOPolicyInferencer(hparams,
                            action_space=env.action_space,
                            observation_space=env.observation_space,
                            policy_dir=os.path.join(output_dir, "policy"))

  ppo.reset_frame_stack()
  ob = env.reset()
  for _ in range(FLAGS.num_episodes):
    done = False
    while not done:
      logits, _ = ppo.infer(ob)
      probs = np.exp(logits) / np.sum(np.exp(logits))
      action = np.random.choice(probs.size, p=probs[0])
      ob, _, done, _ = env.step(action)
    ob = env.reset()
    ppo.reset_frame_stack()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
