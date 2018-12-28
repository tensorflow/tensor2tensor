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
    --video_dir="my/video/dir" \
    --env=real \
    --simulated_episode_len="50" \
    --num_episodes="5" \

Check flags definitions under imports for more details.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from gym.wrappers import TimeLimit

from envs.simulated_batch_gym_env import FlatBatchEnv
from player_utils import SimulatedEnv, wrap_with_monitor, PPOPolicyInferencer, \
  load_t2t_env, join_and_check
from tensor2tensor.rl.trainer_model_based import FLAGS
from tensor2tensor.utils import registry
import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string("video_dir", "/tmp/record_ppo_out",
                    "Where to save recorded trajectories.")
flags.DEFINE_string("epoch", "last",
                    "Data from which epoch to use.")
flags.DEFINE_string("env", "simulated",
                    "Either to use 'simulated' or 'real' env.")
flags.DEFINE_string("simulated_episode_len", "100",
                    "Timesteps limit for simulated env")
flags.DEFINE_string("num_episodes", "20",
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
  simulated_episode_len = int(FLAGS.simulated_episode_len)
  num_episodes = int(FLAGS.num_episodes)

  if FLAGS.env == "simulated":
    env = SimulatedEnv(output_dir, hparams, which_epoch_data=epoch)
    env = TimeLimit(env, max_episode_steps=simulated_episode_len)
  elif FLAGS.env == "real":
    env = load_t2t_env(hparams,
                       data_dir=join_and_check(output_dir, "data"),
                       which_epoch_data=None)
    env = FlatBatchEnv(env)
  else:
    raise ValueError("Invalid 'env' flag {}".format(FLAGS.env))

  env = wrap_with_monitor(env, video_dir=video_dir)
  ppo = PPOPolicyInferencer(hparams,
                           action_space=env.action_space,
                           observation_space=env.observation_space,
                           policy_dir=join_and_check(output_dir, "policy"))

  ppo.reset_frame_stack()
  ob = env.reset()
  for _ in range(num_episodes):
    done = False
    while not done:
      logits, vf = ppo.infer(ob)
      probs = np.exp(logits) / np.sum(np.exp(logits))
      action = np.random.choice(probs.size, p=probs[0])
      ob, rew, done, _ = env.step(action)
    ob = env.reset()
    ppo.reset_frame_stack()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
