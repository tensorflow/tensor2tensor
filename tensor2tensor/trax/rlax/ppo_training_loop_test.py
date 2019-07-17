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

"""Tests for tensor2tensor.trax.rlax.ppo's training_loop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import functools
import itertools
import os
import tempfile

import gin
import gym
import numpy as np

from tensor2tensor.envs import gym_env_problem
from tensor2tensor.rl import gym_utils
from tensor2tensor.trax import inputs as trax_inputs
from tensor2tensor.trax import layers
from tensor2tensor.trax import learning_rate as lr
from tensor2tensor.trax import models
from tensor2tensor.trax import optimizers as trax_opt
from tensor2tensor.trax import trax
from tensor2tensor.trax.rlax import envs  # pylint: disable=unused-import
from tensor2tensor.trax.rlax import ppo
from tensor2tensor.trax.rlax import simulated_env_problem
from tensorflow import test
from tensorflow.io import gfile


class PpoTrainingLoopTest(test.TestCase):

  def get_wrapped_env(self, name="CartPole-v0", max_episode_steps=2):
    wrapper_fn = functools.partial(
        gym_utils.gym_env_wrapper,
        **{
            "rl_env_max_episode_steps": max_episode_steps,
            "maxskip_env": False,
            "rendered_env": False,
            "rendered_env_resize_to": None,  # Do not resize frames
            "sticky_actions": False,
            "output_dtype": None,
        })

    return gym_env_problem.GymEnvProblem(base_env_name=name,
                                         batch_size=1,
                                         env_wrapper_fn=wrapper_fn,
                                         discrete_rewards=False)

  @contextlib.contextmanager
  def tmp_dir(self):
    tmp = tempfile.mkdtemp(dir=self.get_temp_dir())
    yield tmp
    gfile.rmtree(tmp)

  def _run_training_loop(self, env, eval_env, output_dir):
    n_epochs = 2
    # Run the training loop.
    ppo.training_loop(
        env=env,
        eval_env=eval_env,
        epochs=n_epochs,
        policy_and_value_net_fn=functools.partial(
            ppo.policy_and_value_net,
            bottom_layers_fn=lambda: [layers.Dense(1)]),
        policy_and_value_optimizer_fn=ppo.optimizer_fn,
        n_optimizer_steps=1,
        output_dir=output_dir,
        env_name="SomeEnv",
        random_seed=0)

  def test_training_loop_cartpole(self):
    with self.tmp_dir() as output_dir:
      self._run_training_loop(
          env=self.get_wrapped_env("CartPole-v0", 2),
          eval_env=self.get_wrapped_env("CartPole-v0", 2),
          output_dir=output_dir,
      )

  def test_training_loop_onlinetune(self):
    with self.tmp_dir() as output_dir:
      gin.bind_parameter("OnlineTuneEnv.model", functools.partial(
          models.MLP,
          n_hidden_layers=0,
          n_output_classes=1,
      ))
      gin.bind_parameter("OnlineTuneEnv.inputs", functools.partial(
          trax_inputs.random_inputs,
          input_shape=(1, 1),
          input_dtype=np.float32,
          output_shape=(1, 1),
          output_dtype=np.float32,
      ))
      gin.bind_parameter("OnlineTuneEnv.train_steps", 2)
      gin.bind_parameter("OnlineTuneEnv.eval_steps", 2)
      gin.bind_parameter(
          "OnlineTuneEnv.output_dir", os.path.join(output_dir, "envs"))
      self._run_training_loop(
          env=self.get_wrapped_env("OnlineTuneEnv-v0", 2),
          eval_env=self.get_wrapped_env("OnlineTuneEnv-v0", 2),
          output_dir=output_dir,
      )

  def test_training_loop_simulated(self):
    n_actions = 5
    history_shape = (3, 2, 3)
    action_shape = (3,)
    obs_shape = (3, 3)
    reward_shape = (3, 1)

    def model(mode):
      del mode
      return layers.Serial(
          layers.Parallel(
              layers.Flatten(),  # Observation stack.
              layers.Embedding(d_feature=1, vocab_size=n_actions),  # Action.
          ),
          layers.Concatenate(),
          layers.Dense(n_units=1),
          layers.Dup(),
          layers.Parallel(
              layers.Dense(n_units=obs_shape[1]),  # New observation.
              None,  # Reward.
          )
      )

    def inputs(n_devices):
      del n_devices
      stream = itertools.repeat((
          (np.zeros(history_shape), np.zeros(action_shape, dtype=np.int32)),
          (np.zeros(obs_shape), np.zeros(reward_shape)),
      ))
      return trax_inputs.Inputs(
          train_stream=lambda: stream,
          train_eval_stream=lambda: stream,
          eval_stream=lambda: stream,
          input_shape=(history_shape[1:], action_shape[1:]),
          input_dtype=(np.float32, np.int32),
      )

    def loss(*args, **kwargs):
      del args
      del kwargs
      return 0.0

    with self.tmp_dir() as output_dir:
      # Run fake training just to save the parameters.
      trainer = trax.Trainer(
          model=model,
          loss_fn=loss,
          inputs=inputs,
          optimizer=trax_opt.SM3,
          lr_schedule=lr.MultifactorSchedule,
          output_dir=output_dir,
      )
      trainer.train_epoch(epoch_steps=1, eval_steps=1)

      # Repeat the initial observations over and over again.
      stream = itertools.repeat(np.zeros(history_shape))
      env_fn = functools.partial(
          simulated_env_problem.SimulatedEnvProblem,
          model=model,
          history_length=history_shape[1],
          trajectory_length=3,
          batch_size=history_shape[0],
          observation_space=gym.spaces.Box(
              low=-np.inf, high=np.inf, shape=(obs_shape[1],)),
          action_space=gym.spaces.Discrete(n=n_actions),
          reward_range=(-1, 1),
          discrete_rewards=False,
          initial_observation_stream=stream,
          output_dir=output_dir,
      )

      self._run_training_loop(
          env=env_fn(),
          eval_env=env_fn(),
          output_dir=output_dir,
      )


if __name__ == "__main__":
  test.main()
