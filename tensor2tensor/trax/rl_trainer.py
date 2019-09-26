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

import multiprocessing
import os

from absl import app
from absl import flags
from absl import logging
import gin
import jax
from jax.config import config
from tensor2tensor import envs  # pylint: disable=unused-import
from tensor2tensor.envs import env_problem_utils
from tensor2tensor.rl.google import atari_utils  # GOOGLE-INTERNAL:
from tensor2tensor.trax import rl  # pylint: disable=unused-import
from tensor2tensor.trax.rl import envs as rl_envs  # pylint: disable=unused-import
from tensor2tensor.trax.rl import trainers as rl_trainers


FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    "jax_debug_nans", False,
    "Setting to true will help to debug nans and disable jit.")
flags.DEFINE_boolean("disable_jit", False, "Setting to true will disable jit.")

flags.DEFINE_string("output_dir", "", "Output dir.")
flags.DEFINE_string("envs_output_dir", "", "Output dir for the envs.")
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
flags.DEFINE_string("trajectory_dump_dir", "",
                    "Directory to dump trajectories to.")

# TODO(afrozm): Find a better way to do these configurations.
flags.DEFINE_string("train_server_bns", "", "Train Server's BNS.")
flags.DEFINE_string("eval_server_bns", "", "Eval Server's BNS.")

flags.DEFINE_bool("async_mode", False, "Async mode.")


# Not just "train" to avoid a conflict with trax.train in GIN files.
@gin.configurable(blacklist=[
    "output_dir", "train_batch_size", "eval_batch_size", "trajectory_dump_dir"
])
def train_rl(
    output_dir,
    train_batch_size,
    eval_batch_size,
    env_name="ClientEnv-v0",
    max_timestep=None,
    clip_rewards=False,
    rendered_env=False,
    resize_dims=(105, 80),
    trainer_class=rl_trainers.PPO,
    n_epochs=10000,
    trajectory_dump_dir=None,
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
    rendered_env: Whether the environment has visual input. If so, a
      RenderedEnvProblem will be used.
    resize_dims: Pair (height, width), dimensions to resize the visual
      observations to.
    trainer_class: RLTrainer class to use.
    n_epochs: Number epochs to run the training for.
    trajectory_dump_dir: Directory to dump trajectories to.
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
    envs_output_dir = FLAGS.envs_output_dir or os.path.join(output_dir, "envs")
    train_env_output_dir = os.path.join(envs_output_dir, "train")
    eval_env_output_dir = os.path.join(envs_output_dir, "eval")
    train_env_kwargs = {"output_dir": train_env_output_dir}
    eval_env_kwargs = {"output_dir": eval_env_output_dir}

  if "ClientEnv" in env_name:
    train_env_kwargs["per_env_kwargs"] = [{
        "remote_env_address": os.path.join(FLAGS.train_server_bns, str(replica))
    } for replica in range(train_batch_size)]

    eval_env_kwargs["per_env_kwargs"] = [{
        "remote_env_address": os.path.join(FLAGS.eval_server_bns, str(replica))
    } for replica in range(eval_batch_size)]

  # TODO(afrozm): Should we leave out some cores?
  parallelism = multiprocessing.cpu_count() if FLAGS.parallelize_envs else 1

  train_env = env_problem_utils.make_env(
      batch_size=train_batch_size,
      env_problem_name=env_name,
      resize=rendered_env,
      resize_dims=resize_dims,
      max_timestep=max_timestep,
      clip_rewards=clip_rewards,
      parallelism=parallelism,
      use_tpu=FLAGS.use_tpu,
      **train_env_kwargs)
  assert train_env

  eval_env = env_problem_utils.make_env(
      batch_size=eval_batch_size,
      env_problem_name=env_name,
      resize=rendered_env,
      resize_dims=resize_dims,
      max_timestep=max_timestep,
      clip_rewards=clip_rewards,
      parallelism=parallelism,
      use_tpu=FLAGS.use_tpu,
      **eval_env_kwargs)
  assert eval_env

  def run_training_loop():
    """Runs the training loop."""
    logging.info("Starting the training loop.")

    trainer = trainer_class(
        output_dir=output_dir,
        train_env=train_env,
        eval_env=eval_env,
        trajectory_dump_dir=trajectory_dump_dir,
        async_mode=FLAGS.async_mode,
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
      trajectory_dump_dir=(FLAGS.trajectory_dump_dir or None),
  )


if __name__ == "__main__":
  app.run(main)
