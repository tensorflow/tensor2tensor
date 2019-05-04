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
import gym
import jax
from jax.config import config
from tensor2tensor.envs import env_problem
from tensor2tensor.envs import rendered_env_problem
from tensor2tensor.rl import gym_utils
from tensor2tensor.trax import layers
from tensor2tensor.trax.rlax import ppo

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", None, "Name of the environment to make.")
flags.DEFINE_string("env_problem_name", None, "Name of the EnvProblem to make.")

flags.DEFINE_integer("epochs", 100, "Number of epochs to run for.")
flags.DEFINE_integer("random_seed", 0, "Random seed.")
flags.DEFINE_integer("batch_size", 32, "Batch of trajectories needed.")

flags.DEFINE_integer(
    "boundary", 20, "We pad trajectories at integer multiples of this number.")
# -1: returns env as is.
# None: unwraps and returns without TimeLimit wrapper.
# Any other number: imposes this restriction.
flags.DEFINE_integer(
    "max_timestep", None,
    "If set to an integer, maximum number of time-steps in a "
    "trajectory.")

flags.DEFINE_boolean(
    "jax_debug_nans", False,
    "Setting to true will help to debug nans and disable jit.")
flags.DEFINE_boolean("disable_jit", False, "Setting to true will disable jit.")

# If resize is True, then we create RenderedEnvProblem, so this has to be set to
# False for something like CartPole.
flags.DEFINE_boolean("resize", False, "If true, resize the game frame")
flags.DEFINE_integer("resized_height", 105, "Resized height of the game frame.")
flags.DEFINE_integer("resized_width", 80, "Resized width of the game frame.")

flags.DEFINE_boolean(
    "combined_policy_and_value_function", False,
    "If True there is a single network that determines policy"
    "and values.")

flags.DEFINE_boolean("flatten_non_batch_time_dims", False,
                     "If true, we flatten except the first two dimensions.")

# Number of optimizer steps of the combined net, policy net and value net.
flags.DEFINE_integer("num_optimizer_steps", 100, "Number of optimizer steps.")
flags.DEFINE_integer("policy_only_num_optimizer_steps", 80,
                     "Number of optimizer steps policy only.")
flags.DEFINE_integer("value_only_num_optimizer_steps", 80,
                     "Number of optimizer steps value only.")

# Learning rate of the combined net, policy net and value net.
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate.")
flags.DEFINE_float("policy_only_learning_rate", 1e-3,
                   "Learning rate for policy network only.")
flags.DEFINE_float("value_only_learning_rate", 1e-3,
                   "Learning rate for value network only.")

# Target KL is used for doing early stopping in the
flags.DEFINE_float("target_kl", 0.01, "Policy iteration early stopping")


def common_layers():
  cur_layers = []
  if FLAGS.flatten_non_batch_time_dims:
    cur_layers = [layers.Div(divisor=255.0), layers.Flatten(num_axis_to_keep=2)]
  return cur_layers + [
      layers.Dense(64),
      layers.Tanh(),
      layers.Dense(64),
      layers.Tanh()
  ]


def make_env():
  """Creates the env."""
  if FLAGS.env_name:
    return gym.make(FLAGS.env_name)

  assert FLAGS.env_problem_name

  # No resizing needed, so let's be on the normal EnvProblem.
  if not FLAGS.resize:  # None or False
    return env_problem.EnvProblem(
        base_env_name=FLAGS.env_problem_name,
        batch_size=FLAGS.batch_size,
        reward_range=(-1, 1))

  wrapper_fn = functools.partial(
      gym_utils.gym_env_wrapper, **{
          "rl_env_max_episode_steps": FLAGS.max_timestep,
          "maxskip_env": True,
          "rendered_env": True,
          "rendered_env_resize_to": (FLAGS.resized_height, FLAGS.resized_width),
          "sticky_actions": False
      })

  return rendered_env_problem.RenderedEnvProblem(
      base_env_name=FLAGS.env_problem_name,
      batch_size=FLAGS.batch_size,
      env_wrapper_fn=wrapper_fn,
      reward_range=(-1, 1))


def get_optimizer_fun(learning_rate):
  return functools.partial(ppo.optimizer_fun, step_size=learning_rate)


def main(argv):
  del argv

  if FLAGS.jax_debug_nans:
    config.update("jax_debug_nans", True)

  # Make an env here.
  env = make_env()
  assert env

  def run_training_loop():
    """Runs the training loop."""
    policy_net_fun = None
    value_net_fun = None
    policy_and_value_net_fun = None
    policy_optimizer_fun = None
    value_optimizer_fun = None
    policy_and_value_optimizer_fun = None

    if FLAGS.combined_policy_and_value_function:
      policy_and_value_net_fun = functools.partial(
          ppo.policy_and_value_net, bottom_layers=common_layers())
      policy_and_value_optimizer_fun = get_optimizer_fun(FLAGS.learning_rate)
    else:
      policy_net_fun = functools.partial(
          ppo.policy_net, bottom_layers=common_layers())
      value_net_fun = functools.partial(
          ppo.value_net, bottom_layers=common_layers())
      policy_optimizer_fun = get_optimizer_fun(FLAGS.policy_only_learning_rate)
      value_optimizer_fun = get_optimizer_fun(FLAGS.value_only_learning_rate)

    ppo.training_loop(
        env=env,
        epochs=FLAGS.epochs,
        policy_net_fun=policy_net_fun,
        value_net_fun=value_net_fun,
        policy_and_value_net_fun=policy_and_value_net_fun,
        policy_optimizer_fun=policy_optimizer_fun,
        value_optimizer_fun=value_optimizer_fun,
        policy_and_value_optimizer_fun=policy_and_value_optimizer_fun,
        batch_size=FLAGS.batch_size,
        num_optimizer_steps=FLAGS.num_optimizer_steps,
        policy_only_num_optimizer_steps=FLAGS.policy_only_num_optimizer_steps,
        value_only_num_optimizer_steps=FLAGS.value_only_num_optimizer_steps,
        target_kl=FLAGS.target_kl,
        boundary=FLAGS.boundary,
        max_timestep=FLAGS.max_timestep,
        random_seed=FLAGS.random_seed)

  if FLAGS.jax_debug_nans or FLAGS.disable_jit:
    with jax.disable_jit():
      run_training_loop()
  else:
    run_training_loop()


if __name__ == "__main__":
  app.run(main)
