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

"""SimulatedBatchEnv in a Gym-like interface."""

import gin
import numpy as np

from dopamine.agents.dqn import dqn_agent
from dopamine.atari import run_experiment
import tensorflow as tf
from tensor2tensor.rl.envs.simulated_batch_gym_env import FlatBatchEnv, SimulatedBatchGymEnv


def create_agent(sess, environment, summary_writer=None):
  """Creates a DQN agent.

  Simplified version of `dopamine.atari.train.create_agent`
  """
  return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n,
                            summary_writer=summary_writer,
                            tf_device='/cpu:*')  # TODO: put gpu here!!!!


def get_create_env_simulated_fun(hparams):
  def create_env_fun(game_name, sticky_actions=True):
    # Possibly use wrappers as used by atari training in dopamine
    return FlatBatchEnv(SimulatedBatchGymEnv(hparams, 1))
  return create_env_fun


def get_create_env_real_fun(hparams):
  env = hparams.environment_spec.env

  def create_env_fun(_1, _2):
    return FlatBatchEnv(env)

  return create_env_fun


#TODO(pm): rename
def dopamine_trainer(hparams, model_dir):
  """ Simplified version of `dopamine.atari.train.create_runner` """

  # TODO: pass and clean up hparams
  steps_to_make = 1000
  if hparams.environment_spec.simulated_env:
    get_create_env_fun = get_create_env_simulated_fun
    #TODO(pm); this is unused
    num_iterations = np.ceil(steps_to_make / 1000)
    training_steps = 200
  else:
    get_create_env_fun = get_create_env_real_fun
    num_iterations = 1
    training_steps = 100
  # TODO: this must be 0 for real_env (to not generate addtionall data for
  # world-model, but maybe we can use it in simulated env?
  evaluation_steps = 0

  with gin.unlock_config():# This is slight wierdness due to multiple runs DQN
    # TODO(pm): Think how to integrate with dopamine.
    gin_files = ['/home/piotr.milos/code2/tensor-2-tensor-with-mrunner/deps/dopamine/dopamine/agents/dqn/configs/dqn.gin']
    run_experiment.load_gin_configs(gin_files, [])

  with tf.Graph().as_default():
    runner = run_experiment.Runner(model_dir, create_agent,
                                   create_environment_fn=get_create_env_fun(hparams),
                                   num_iterations=1,
                                   training_steps=training_steps,
                                   evaluation_steps=evaluation_steps,
                                   max_steps_per_episode=20  # TODO: remove this
                                   )

    runner.run_experiment()
