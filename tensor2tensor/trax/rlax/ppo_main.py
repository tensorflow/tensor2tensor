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

"""PPO binary over a gym env."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import app
from absl import flags
import jax
from jax.config import config
from tensor2tensor.trax import layers
from tensor2tensor.trax.rlax import ppo

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", None, "Name of the environment to make.")
flags.DEFINE_string("t2t_gym_env", None, "Name of the T2TGymEnv to make.")
flags.DEFINE_integer("epochs", 100, "Number of epochs to run for.")
flags.DEFINE_integer("random_seed", 0, "Random seed.")
flags.DEFINE_integer("batch_size", 32, "Batch of trajectories needed.")
flags.DEFINE_integer("num_optimizer_steps", 100, "Number of optimizer steps.")
flags.DEFINE_integer("boundary", 20,
                     "We pad trajectories at integer multiples of this number.")
flags.DEFINE_integer("max_timestep", None,
                     "If set to an integer, maximum number of time-steps in a "
                     "trajectory.")
flags.DEFINE_float("policy_and_value_net_learning_rate", 1e-3, "Learning rate.")
flags.DEFINE_float("policy_net_learning_rate", 3e-4,
                   "Learning rate for the policy net only.")
flags.DEFINE_float("value_net_learning_rate", 1e-3,
                   "Learning rate for the value net only.")
flags.DEFINE_boolean("jax_debug_nans", False,
                     "Setting to true will help to debug nans and disable jit.")
flags.DEFINE_boolean("disable_jit", False, "Setting to true will disable jit.")
flags.DEFINE_boolean("combined_policy_and_value_function", False,
                     "If True there is a single network that determines policy"
                     "and values.")
flags.DEFINE_integer("flatten_non_batch_time_dims", False,
                     "If true, we flatten except the first two dimensions.")


def common_layers():
  cur_layers = []
  if FLAGS.flatten_non_batch_time_dims:
    cur_layers = [layers.Div(divisor=255.0), layers.Flatten(num_axis_to_keep=2)]
  return cur_layers + [layers.Dense(16), layers.Relu(),
                       layers.Dense(4), layers.Relu()]


def run_training_loop():
  """Run the PPO training loop."""

  policy_net_fun = None
  value_net_fun = None
  policy_and_value_net_fun = None
  policy_optimizer_fun = None
  value_optimizer_fun = None
  policy_and_value_optimizer_fun = None

  if FLAGS.combined_policy_and_value_function:
    policy_and_value_net_fun = functools.partial(
        ppo.policy_and_value_net, bottom_layers=common_layers())
    policy_and_value_optimizer_fun = get_optimizer_fun(
        FLAGS.policy_and_value_net_learning_rate)
  else:
    policy_net_fun = functools.partial(ppo.policy_net,
                                       bottom_layers=common_layers())
    value_net_fun = functools.partial(ppo.value_net,
                                      bottom_layers=common_layers())
    policy_optimizer_fun = get_optimizer_fun(FLAGS.policy_net_learning_rate)
    value_optimizer_fun = get_optimizer_fun(FLAGS.value_net_learning_rate)

  ppo.training_loop(
      env_name=FLAGS.env_name,
      epochs=FLAGS.epochs,
      policy_net_fun=policy_net_fun,
      value_net_fun=value_net_fun,
      policy_and_value_net_fun=policy_and_value_net_fun,
      policy_optimizer_fun=policy_optimizer_fun,
      value_optimizer_fun=value_optimizer_fun,
      policy_and_value_optimizer_fun=policy_and_value_optimizer_fun,
      batch_size=FLAGS.batch_size,
      num_optimizer_steps=FLAGS.num_optimizer_steps,
      boundary=FLAGS.boundary,
      max_timestep=FLAGS.max_timestep,
      random_seed=FLAGS.random_seed)


def get_optimizer_fun(learning_rate):
  return functools.partial(ppo.optimizer_fun, step_size=learning_rate)


def main(argv):
  del argv

  if FLAGS.jax_debug_nans:
    config.update("jax_debug_nans", True)

  if FLAGS.jax_debug_nans or FLAGS.disable_jit:
    with jax.disable_jit():
      run_training_loop()
  else:
    run_training_loop()

if __name__ == "__main__":
  app.run(main)
