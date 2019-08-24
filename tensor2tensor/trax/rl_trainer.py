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

r"""Trainer for RL environments.

For now we only support PPO as RL algorithm.

Sample invocation:

TRAIN_BATCH_SIZE=32
python trax/rl_trainer.py \
  --config_file=trax/rl/configs/acrobot.gin \
  --train_batch_size=${TRAIN_BATCH_SIZE} \
  --output_dir=${HOME}/ppo_acrobot \
  --vmodule=*/tensor2tensor/*=1 \
  --alsologtostderr
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
from tensor2tensor import envs  # pylint: disable=unused-import
from tensor2tensor.envs import gym_env_problem
from tensor2tensor.envs import rendered_env_problem
from tensor2tensor.rl import gym_utils
from tensor2tensor.rl.google import atari_utils  # GOOGLE-INTERNAL:
from tensor2tensor.trax.rl import envs as rl_envs  # pylint: disable=unused-import
from tensor2tensor.trax.rl import trainers as rl_trainers


FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    "jax_debug_nans", False,
    "Setting to true will help to debug nans and disable jit.")
flags.DEFINE_boolean("disable_jit", False, "Setting to true will disable jit.")

flags.DEFINE_string("output_dir", "", "Output dir.")
flags.DEFINE_multi_string("config_file", None,
                          "Configuration file with parameters (.gin).")
flags.DEFINE_multi_string("config", None,
                          "Configuration parameters (gin string).")
flags.DEFINE_bool("use_tpu", False, "Whether we're running on TPU.")
flags.DEFINE_bool("xm", False, "Copy atari roms?")
flags.DEFINE_integer("train_batch_size", 32,
                     "Number of parallel environments during training.")
flags.DEFINE_integer("eval_batch_size", 4, "Batch size for evaluation.")
flags.DEFINE_boolean("parallelize_envs", False,
                     "If true, sets parallelism to number of cpu cores.")


# TODO(afrozm): Find a better way to do these configurations.
flags.DEFINE_string("train_server_bns", "", "Train Server's BNS.")
flags.DEFINE_string("eval_server_bns", "", "Eval Server's BNS.")


def make_env(name, batch_size, max_timestep, clip_rewards, rendered_env,
             resize_dims, **env_kwargs):
  """Creates the env."""

  if clip_rewards:
    env_kwargs.update({"reward_range": (-1, 1), "discrete_rewards": True})
  else:
    env_kwargs.update({"discrete_rewards": False})

  # TODO(afrozm): Should we leave out some cores?
  parallelism = multiprocessing.cpu_count() if FLAGS.parallelize_envs else 1

  # No resizing needed, so let's be on the normal EnvProblem.
  if not rendered_env:
    return gym_env_problem.GymEnvProblem(
        base_env_name=name,
        batch_size=batch_size,
        parallelism=parallelism,
        **env_kwargs)

  wrapper_fn = functools.partial(
      gym_utils.gym_env_wrapper, **{
          "rl_env_max_episode_steps": max_timestep,
          "maxskip_env": True,
          "rendered_env": True,
          "rendered_env_resize_to": resize_dims,
          "sticky_actions": False,
          "output_dtype": onp.int32 if FLAGS.use_tpu else None,
      })

  return rendered_env_problem.RenderedEnvProblem(
      base_env_name=name,
      batch_size=batch_size,
      parallelism=parallelism,
      env_wrapper_fn=wrapper_fn,
      **env_kwargs)


# Not just "train" to avoid a conflict with trax.train in GIN files.
@gin.configurable(blacklist=[
    "output_dir", "train_batch_size", "eval_batch_size"])
def train_rl(
    output_dir,
    train_batch_size,
    eval_batch_size,
    env_name="Acrobot-v1",
    max_timestep=None,
    clip_rewards=False,
    rendered_env=False,
    resize_dims=(105, 80),
    trainer_class=rl_trainers.PPO,
    n_epochs=10000,
):
  """Train the RL agent.

  Args:
    output_dir: Output directory.
    train_batch_size: Number of parallel environments to use for training.
    eval_batch_size: Number of parallel environments to use for evaluation.
    env_name: Name of the environment.
    max_timestep: Int or None, the maximum number of timesteps in a trajectory.
      The environment is wrapped in a TimeLimit wrapper.
    clip_rewards: Whether to clip and discretize the rewards.
    rendered_env: Whether the environment has visual input. If so,
      a RenderedEnvProblem will be used.
    resize_dims: Pair (height, width), dimensions to resize the visual
      observations to.
    trainer_class: RLTrainer class to use.
    n_epochs: Number epochs to run the training for.
  """

  if FLAGS.jax_debug_nans:
    config.update("jax_debug_nans", True)

  if FLAGS.use_tpu:
    config.update("jax_platform_name", "tpu")
  else:
    config.update("jax_platform_name", "gpu")


  # TODO(pkozakowski): Find a better way to determine this.
  train_env_kwargs = {}
  eval_env_kwargs = {}
  if "OnlineTuneEnv" in env_name:
    # TODO(pkozakowski): Separate env output dirs by train/eval and epoch.
    train_env_kwargs = {
        "output_dir": os.path.join(output_dir, "envs/train")
    }
    eval_env_kwargs = {
        "output_dir": os.path.join(output_dir, "envs/eval")
    }

  if "ClientEnv" in env_name:
    train_env_kwargs["per_env_kwargs"] = [{
        "remote_env_address": os.path.join(FLAGS.train_server_bns, str(replica))
    } for replica in range(train_batch_size)]

    eval_env_kwargs["per_env_kwargs"] = [{
        "remote_env_address": os.path.join(FLAGS.eval_server_bns, str(replica))
    } for replica in range(eval_batch_size)]

  common_env_kwargs = {
      "name": env_name,
      "max_timestep": max_timestep,
      "clip_rewards": clip_rewards,
      "rendered_env": rendered_env,
      "resize_dims": resize_dims,
  }
  train_env_kwargs.update(common_env_kwargs)
  eval_env_kwargs.update(common_env_kwargs)

  # Make an env here.
  train_env = make_env(batch_size=train_batch_size, **train_env_kwargs)
  assert train_env

  eval_env = make_env(batch_size=eval_batch_size, **eval_env_kwargs)
  assert eval_env

  def run_training_loop():
    """Runs the training loop."""
    logging.info("Starting the training loop.")

    trainer = trainer_class(
        output_dir=output_dir,
        train_env=train_env,
        eval_env=eval_env,
    )
    trainer.training_loop(n_epochs=n_epochs)

  if FLAGS.jax_debug_nans or FLAGS.disable_jit:
    with jax.disable_jit():
      run_training_loop()
  else:
    run_training_loop()


def main(argv):
  del argv
  logging.info("Starting RL training.")

  gin_configs = FLAGS.config or []
  gin.parse_config_files_and_bindings(FLAGS.config_file, gin_configs)

  train_rl(
      output_dir=FLAGS.output_dir,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
  )


if __name__ == "__main__":
  app.run(main)
