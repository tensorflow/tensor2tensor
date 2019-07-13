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

r"""PPO binary over a gym env.

Sample invocation:

ENV_PROBLEM_NAME=Acrobot-v1
BATCH_SIZE=32
EPOCHS=100
RANDOM_SEED=0
BOUNDARY=100

python trax/rlax/ppo_main.py \
  --env_problem_name=${ENV_PROBLEM_NAME} \
  --batch_size=${BATCH_SIZE} \
  --config=ppo.training_loop.epochs=${EPOCHS} \
  --config=ppo.training_loop.random_seed=${RANDOM_SEED} \
  --config=ppo.training_loop.boundary=${BOUNDARY} \
  --output_dir=${HOME}/ppo_acrobot \
  --vmodule=*/tensor2tensor/*=1 \
  --alsologtostderr \
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import multiprocessing
import os

from absl import app
from absl import flags
from absl import logging
import gin
import jax
from jax.config import config
import numpy as onp
from tensor2tensor.envs import gym_env_problem
from tensor2tensor.envs import rendered_env_problem
from tensor2tensor.rl import gym_utils
from tensor2tensor.trax import layers
from tensor2tensor.trax import models
from tensor2tensor.trax.rlax import envs  # pylint: disable=unused-import
from tensor2tensor.trax.rlax import ppo


FLAGS = flags.FLAGS

flags.DEFINE_string("env_problem_name", None, "Name of the EnvProblem to make.")

# -1: returns env as is.
# None: unwraps and returns without TimeLimit wrapper.
# Any other number: imposes this restriction.
flags.DEFINE_string(
    "max_timestep", None,
    "If set to an integer, maximum number of time-steps in a "
    "trajectory. The bare env is wrapped with TimeLimit wrapper.")

flags.DEFINE_boolean(
    "jax_debug_nans", False,
    "Setting to true will help to debug nans and disable jit.")
flags.DEFINE_boolean("disable_jit", False, "Setting to true will disable jit.")

# If resize is True, then we create RenderedEnvProblem, so this has to be set to
# False for something like CartPole.
flags.DEFINE_boolean("resize", False, "If true, resize the game frame")
flags.DEFINE_integer("resized_height", 105, "Resized height of the game frame.")
flags.DEFINE_integer("resized_width", 80, "Resized width of the game frame.")

flags.DEFINE_bool(
    "two_towers", True,
    "In the combined network case should we make one tower or"
    "two.")

# Learning rate of the combined net, policy net and value net.
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")

flags.DEFINE_string("output_dir", "", "Output dir.")
flags.DEFINE_multi_string("config_file", None,
                          "Configuration file with parameters (.gin).")
flags.DEFINE_multi_string("config", None,
                          "Configuration parameters (gin string).")
flags.DEFINE_bool("use_tpu", False, "Whether we're running on TPU.")
flags.DEFINE_bool("xm", False, "Copy atari roms?")
flags.DEFINE_integer("batch_size", 32,
                     "Number of parallel environments during training.")
flags.DEFINE_integer("eval_batch_size", 4, "Batch size for evaluation.")
flags.DEFINE_bool("clip_rewards", True,
                  "Whether to clip and discretize the rewards.")
flags.DEFINE_boolean("parallelize_envs", False,
                     "If true, sets parallelism to number of cpu cores.")


def common_layers():
  # TODO(afrozm): Refactor.
  if "NoFrameskip" in FLAGS.env_problem_name:
    return atari_layers()

  return [layers.Dense(64), layers.Tanh(), layers.Dense(64), layers.Tanh()]


def atari_layers():
  return [models.AtariCnn()]


def make_env(batch_size=8, **env_kwargs):
  """Creates the env."""

  if FLAGS.clip_rewards:
    env_kwargs.update({"reward_range": (-1, 1), "discrete_rewards": True})
  else:
    env_kwargs.update({"discrete_rewards": False})

  # TODO(afrozm): Should we leave out some cores?
  parallelism = multiprocessing.cpu_count() if FLAGS.parallelize_envs else 1

  # No resizing needed, so let's be on the normal EnvProblem.
  if not FLAGS.resize:  # None or False
    return gym_env_problem.GymEnvProblem(
        base_env_name=FLAGS.env_problem_name,
        batch_size=batch_size,
        parallelism=parallelism,
        **env_kwargs)

  max_timestep = None
  try:
    max_timestep = int(FLAGS.max_timestep)
  except Exception:  # pylint: disable=broad-except
    pass

  wrapper_fn = functools.partial(
      gym_utils.gym_env_wrapper, **{
          "rl_env_max_episode_steps": max_timestep,
          "maxskip_env": True,
          "rendered_env": True,
          "rendered_env_resize_to": (FLAGS.resized_height, FLAGS.resized_width),
          "sticky_actions": False,
          "output_dtype": onp.int32 if FLAGS.use_tpu else None,
      })

  return rendered_env_problem.RenderedEnvProblem(
      base_env_name=FLAGS.env_problem_name,
      batch_size=batch_size,
      parallelism=parallelism,
      env_wrapper_fn=wrapper_fn,
      **env_kwargs)


def get_optimizer_fn(learning_rate):
  return functools.partial(ppo.optimizer_fn, step_size=learning_rate)


def main(argv):
  del argv
  logging.info("Starting PPO Main.")

  if FLAGS.jax_debug_nans:
    config.update("jax_debug_nans", True)

  if FLAGS.use_tpu:
    config.update("jax_platform_name", "tpu")
  else:
    config.update("jax_platform_name", "gpu")


  gin_configs = FLAGS.config or []
  gin.parse_config_files_and_bindings(FLAGS.config_file, gin_configs)

  # TODO(pkozakowski): Find a better way to determine this.
  if "OnlineTuneEnv" in FLAGS.env_problem_name:
    # TODO(pkozakowski): Separate env output dirs by train/eval and epoch.
    env_kwargs = {"output_dir": os.path.join(FLAGS.output_dir, "envs")}
  else:
    env_kwargs = {}

  # Make an env here.
  env = make_env(batch_size=FLAGS.batch_size, **env_kwargs)
  assert env

  eval_env = make_env(batch_size=FLAGS.eval_batch_size, **env_kwargs)
  assert eval_env

  def run_training_loop():
    """Runs the training loop."""
    logging.info("Starting the training loop.")

    policy_and_value_net_fn = functools.partial(
        ppo.policy_and_value_net,
        bottom_layers_fn=common_layers,
        two_towers=FLAGS.two_towers)
    policy_and_value_optimizer_fn = get_optimizer_fn(FLAGS.learning_rate)

    ppo.training_loop(
        output_dir=FLAGS.output_dir,
        env=env,
        eval_env=eval_env,
        env_name=str(FLAGS.env_problem_name),
        policy_and_value_net_fn=policy_and_value_net_fn,
        policy_and_value_optimizer_fn=policy_and_value_optimizer_fn,
    )

  if FLAGS.jax_debug_nans or FLAGS.disable_jit:
    with jax.disable_jit():
      run_training_loop()
  else:
    run_training_loop()


if __name__ == "__main__":
  app.run(main)
