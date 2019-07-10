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
import os
import tempfile

import gin
import numpy as np

from tensor2tensor.envs import env_problem
from tensor2tensor.rl import gym_utils
from tensor2tensor.trax import inputs as trax_inputs
from tensor2tensor.trax import layers
from tensor2tensor.trax import models
from tensor2tensor.trax.rlax import envs  # pylint: disable=unused-import
from tensor2tensor.trax.rlax import ppo
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

    return env_problem.EnvProblem(base_env_name=name,
                                  batch_size=1,
                                  env_wrapper_fn=wrapper_fn,
                                  reward_range=(-1, 1),
                                  discrete_rewards=False)

  @contextlib.contextmanager
  def tmp_dir(self):
    tmp = tempfile.mkdtemp(dir=self.get_temp_dir())
    yield tmp
    gfile.rmtree(tmp)

  def _run_training_loop(self, env_name, output_dir):
    env = self.get_wrapped_env(env_name, 2)
    eval_env = self.get_wrapped_env(env_name, 2)
    n_epochs = 2
    batch_size = 2
    # Run the training loop.
    ppo.training_loop(
        env=env,
        eval_env=eval_env,
        epochs=n_epochs,
        policy_and_value_net_fn=functools.partial(
            ppo.policy_and_value_net,
            bottom_layers_fn=lambda: [layers.Dense(1)]),
        policy_and_value_optimizer_fn=ppo.optimizer_fn,
        batch_size=batch_size,
        n_optimizer_steps=1,
        output_dir=output_dir,
        env_name=env_name,
        random_seed=0)

  def test_training_loop_cartpole(self):
    with self.tmp_dir() as output_dir:
      self._run_training_loop("CartPole-v0", output_dir)

  def test_training_loop_onlinetune(self):
    with self.tmp_dir() as output_dir:
      gin.bind_parameter("OnlineTuneEnv.model", functools.partial(
          models.MLP, n_hidden_layers=0, n_output_classes=1))
      gin.bind_parameter("OnlineTuneEnv.inputs", functools.partial(
          trax_inputs.random_inputs,
          input_shape=(1, 1),
          input_dtype=np.float32,
          output_shape=(1, 1),
          output_dtype=np.float32))
      gin.bind_parameter("OnlineTuneEnv.train_steps", 2)
      gin.bind_parameter("OnlineTuneEnv.eval_steps", 2)
      gin.bind_parameter(
          "OnlineTuneEnv.output_dir", os.path.join(output_dir, "envs"))
      self._run_training_loop("OnlineTuneEnv-v0", output_dir)


if __name__ == "__main__":
  test.main()
