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

"""A trajectory collector that polls on policy files and keeps collecting trajectories."""

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
from tensor2tensor.envs import env_problem_utils
from tensor2tensor.rl.google import atari_utils  # GOOGLE-INTERNAL:
from tensor2tensor.trax import rl  # pylint: disable=unused-import
from tensor2tensor.trax.rl import envs as rl_envs  # pylint: disable=unused-import
from tensor2tensor.trax.rl.envs import async_trajectory_collector_lib as async_lib
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_multi_string("config_file", None,
                          "Configuration file with parameters (.gin).")
flags.DEFINE_multi_string("config", None,
                          "Configuration parameters (gin string).")
flags.DEFINE_bool("use_tpu", False, "Whether we're running on TPU.")
flags.DEFINE_bool("xm", False, "Copy atari roms?")

flags.DEFINE_bool(
    "try_abort", True,
    "Should we try to abort a trajectory collection if a newer "
    "policy is available.")

flags.DEFINE_string("output_dir", "", "Output dir.")
flags.DEFINE_string("envs_output_dir", "", "Output dir for the envs.")

flags.DEFINE_boolean(
    "jax_debug_nans", False,
    "Setting to true will help to debug nans and disable jit.")
flags.DEFINE_boolean("disable_jit", False, "Setting to true will disable jit.")

flags.DEFINE_boolean("parallelize_envs", False,
                     "If true, sets parallelism to number of cpu cores.")

flags.DEFINE_integer("replica", 0, "Basically to append to trajectory name.")
flags.DEFINE_bool("enable_eager_execution", False, "")

flags.DEFINE_integer(
    "max_trajectories_to_collect", -1,
    "-1 for infinite, otherwise whatever number was specified.")


# TODO(afrozm): This code snippet is strewn across many places, unify it.
def initialize_gin():
  gin_configs = FLAGS.config or []
  gin.parse_config_files_and_bindings(FLAGS.config_file, gin_configs)


def get_output_dir():
  """Return output_dir."""
  output_dir = FLAGS.output_dir
  return output_dir


def update_jax_config():
  """Update JAX config based on flags."""

  if FLAGS.jax_debug_nans:
    config.update("jax_debug_nans", True)

  if FLAGS.use_tpu:
    config.update("jax_platform_name", "tpu")
  else:
    config.update("jax_platform_name", "gpu")


@gin.configurable(blacklist=[
    "output_dir",
])
def create_envs_and_collect_trajectories(
    output_dir,
    env_name="OnlineTuneEnv-v0",
    max_timestep=None,
    clip_rewards=False,
    rendered_env=False,
    resize_dims=(105, 80),
):
  """Creates the envs and continuously collects trajectories."""


  train_batch_size = 1
  eval_batch_size = 1

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

  parallelism = multiprocessing.cpu_count() if FLAGS.parallelize_envs else 1
  train_parallelism = min(train_batch_size, parallelism)
  eval_parallelism = min(eval_batch_size, parallelism)

  train_env = env_problem_utils.make_env(
      batch_size=train_batch_size,
      env_problem_name=env_name,
      resize=rendered_env,
      resize_dims=resize_dims,
      max_timestep=max_timestep,
      clip_rewards=clip_rewards,
      parallelism=train_parallelism,
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
      parallelism=eval_parallelism,
      use_tpu=FLAGS.use_tpu,
      **eval_env_kwargs)
  assert eval_env

  def run_collect_loop():
    async_lib.continuously_collect_trajectories(
        output_dir,
        train_env,
        eval_env,
        trajectory_dump_dir=None,
        env_id=FLAGS.replica,
        try_abort=FLAGS.try_abort,
        max_trajectories_to_collect=(None
                                     if FLAGS.max_trajectories_to_collect < 0
                                     else FLAGS.max_trajectories_to_collect))

  if FLAGS.jax_debug_nans or FLAGS.disable_jit:
    with jax.disable_jit():
      run_collect_loop()
  else:
    run_collect_loop()


def main(argv):
  del argv

  if FLAGS.enable_eager_execution:
    tf.enable_eager_execution()

  logging.info("Initializing Gin.")
  initialize_gin()

  logging.info("Update JAX config.")
  update_jax_config()

  logging.info("Getting output_dir")
  output_dir = get_output_dir()
  logging.info("Got output_dir = %s", output_dir)

  logging.info("Starting Trajectory collection.")
  create_envs_and_collect_trajectories(output_dir)


if __name__ == "__main__":
  app.run(main)
