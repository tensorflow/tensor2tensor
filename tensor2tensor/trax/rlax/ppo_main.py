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
COMBINED_NETWORK=false
EPOCHS=100
BATCH_SIZE=32
RANDOM_SEED=0
BOUNDARY=100

python trax/rlax/ppo_main.py \
  --env_problem_name=${ENV_PROBLEM_NAME} \
  --combined_policy_and_value_function=${COMBINED_NETWORK} \
  --epochs=${EPOCHS} \
  --batch_size=${BATCH_SIZE} \
  --random_seed=${RANDOM_SEED} \
  --boundary=${BOUNDARY} \
  --vmodule=*/tensor2tensor/*=1 \
  --alsologtostderr \
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import app
from absl import flags
from absl import logging
import jax
from jax.config import config
import numpy as onp
from tensor2tensor.envs import env_problem
from tensor2tensor.envs import rendered_env_problem
from tensor2tensor.rl import gym_utils
from tensor2tensor.trax import layers
from tensor2tensor.trax.models import atari_cnn
from tensor2tensor.trax.rlax import ppo


FLAGS = flags.FLAGS

flags.DEFINE_string("env_problem_name", None, "Name of the EnvProblem to make.")

flags.DEFINE_integer("epochs", 100, "Number of epochs to run for.")
flags.DEFINE_string("random_seed", None, "Random seed.")
flags.DEFINE_integer("batch_size", 32, "Batch of trajectories needed.")

flags.DEFINE_integer(
    "boundary", 20, "We pad trajectories at integer multiples of this number.")
# -1: returns env as is.
# None: unwraps and returns without TimeLimit wrapper.
# Any other number: imposes this restriction.
flags.DEFINE_string(
    "max_timestep", None,
    "If set to an integer, maximum number of time-steps in a "
    "trajectory. The bare env is wrapped with TimeLimit wrapper.")

# This is different from max_timestep is that in the above, the env is wrapped
# in a TimeLimit wrapper, vs here we use this in the collect function.
flags.DEFINE_integer(
    "truncation_timestep", None,
    "If set to an integer, maximum number of time-steps in a "
    "trajectory. Used in the collect procedure.")
flags.DEFINE_integer(
    "truncation_timestep_eval", 20000,
    "If set to an integer, maximum number of time-steps in an evaluation "
    "trajectory. Used in the collect procedure.")

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

# Number of optimizer steps of the combined net, policy net and value net.
flags.DEFINE_integer("n_optimizer_steps", 100, "Number of optimizer steps.")
flags.DEFINE_integer(
    "print_every_optimizer_steps", 1,
    "How often to log during the policy optimization process.")

# Learning rate of the combined net, policy net and value net.
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")

# Target KL is used for doing early stopping in the
flags.DEFINE_float("target_kl", 0.01, "Policy iteration early stopping")
flags.DEFINE_float("value_coef", 1.0,
                   "Coefficient of Value Loss term in combined loss.")
flags.DEFINE_float("entropy_coef", 0.01,
                   "Coefficient of the Entropy Bonus term in combined loss.")
flags.DEFINE_float("gamma", 0.99, "Policy iteration early stopping")
flags.DEFINE_float("lambda_", 0.95, "Policy iteration early stopping")
flags.DEFINE_float("epsilon", 0.1, "Policy iteration early stopping")

flags.DEFINE_string("output_dir", "", "Output dir.")
flags.DEFINE_bool("use_tpu", False, "Whether we're running on TPU.")
flags.DEFINE_bool("enable_early_stopping", True,
                  "Whether to enable early stopping.")
flags.DEFINE_bool("xm", False, "Copy atari roms?")
flags.DEFINE_integer("eval_every_n", 100, "How frequently to eval the policy.")
flags.DEFINE_integer("eval_batch_size", 4, "Batch size for evaluation.")
flags.DEFINE_integer("n_evals", 1, "Number of times to evaluate.")
flags.DEFINE_float(
    "done_frac_for_policy_save", 0.5,
    "Fraction of the trajectories that should be done to "
    "checkpoint the policy.")
flags.DEFINE_integer("len_history_for_policy", 4,
                     "How much of history to give to the policy.")


def common_layers():
  # TODO(afrozm): Refactor.
  if "NoFrameskip" in FLAGS.env_problem_name:
    return atari_layers()

  return [layers.Dense(64), layers.Tanh(), layers.Dense(64), layers.Tanh()]


def atari_layers():
  return [atari_cnn.AtariCnn()]


def make_env(batch_size=8):
  """Creates the env."""

  # No resizing needed, so let's be on the normal EnvProblem.
  if not FLAGS.resize:  # None or False
    return env_problem.EnvProblem(
        base_env_name=FLAGS.env_problem_name,
        batch_size=batch_size,
        reward_range=(-1, 1))

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
      env_wrapper_fn=wrapper_fn,
      reward_range=(-1, 1))


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


  # Make an env here.
  env = make_env(batch_size=FLAGS.batch_size)
  assert env

  eval_env = make_env(batch_size=FLAGS.eval_batch_size)
  assert eval_env

  def run_training_loop():
    """Runs the training loop."""
    logging.info("Starting the training loop.")

    policy_and_value_net_fn = functools.partial(
        ppo.policy_and_value_net,
        bottom_layers_fn=common_layers,
        two_towers=FLAGS.two_towers)
    policy_and_value_optimizer_fn = get_optimizer_fn(FLAGS.learning_rate)

    random_seed = None
    try:
      random_seed = int(FLAGS.random_seed)
    except Exception:  # pylint: disable=broad-except
      pass

    ppo.training_loop(
        env=env,
        epochs=FLAGS.epochs,
        policy_and_value_net_fn=policy_and_value_net_fn,
        policy_and_value_optimizer_fn=policy_and_value_optimizer_fn,
        n_optimizer_steps=FLAGS.n_optimizer_steps,
        print_every_optimizer_steps=FLAGS.print_every_optimizer_steps,
        batch_size=FLAGS.batch_size,
        target_kl=FLAGS.target_kl,
        boundary=FLAGS.boundary,
        max_timestep=FLAGS.truncation_timestep,
        max_timestep_eval=FLAGS.truncation_timestep_eval,
        random_seed=random_seed,
        c1=FLAGS.value_coef,
        c2=FLAGS.entropy_coef,
        gamma=FLAGS.gamma,
        lambda_=FLAGS.lambda_,
        epsilon=FLAGS.epsilon,
        enable_early_stopping=FLAGS.enable_early_stopping,
        output_dir=FLAGS.output_dir,
        eval_every_n=FLAGS.eval_every_n,
        done_frac_for_policy_save=FLAGS.done_frac_for_policy_save,
        eval_env=eval_env,
        n_evals=FLAGS.n_evals,
        env_name=str(FLAGS.env_problem_name),
        len_history_for_policy=int(FLAGS.len_history_for_policy),
    )

  if FLAGS.jax_debug_nans or FLAGS.disable_jit:
    with jax.disable_jit():
      run_training_loop()
  else:
    run_training_loop()


if __name__ == "__main__":
  app.run(main)
