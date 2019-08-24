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

"""Tests for tensor2tensor.trax.rl.simple_trainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import gin

from tensor2tensor.envs import gym_env_problem
from tensor2tensor.rl import gym_utils
from tensor2tensor.trax import models
from tensor2tensor.trax.rl import envs  # pylint: disable=unused-import
from tensor2tensor.trax.rl import simulated_env_problem
from tensor2tensor.trax.rl import trainers
from tensorflow import test


class SimpleTrainerTest(test.TestCase):

  def _make_wrapped_env(self, name, max_episode_steps=2):
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
                                         batch_size=2,
                                         env_wrapper_fn=wrapper_fn,
                                         discrete_rewards=False)

  def test_training_loop_acrobot(self):
    gin.bind_parameter("BoxSpaceSerializer.precision", 2)
    gin.bind_parameter("trax.train.train_steps", 1)
    gin.bind_parameter("trax.train.eval_steps", 1)
    trainer = trainers.SimPLe(
        train_env=self._make_wrapped_env("Acrobot-v1"),
        eval_env=self._make_wrapped_env("Acrobot-v1"),
        output_dir=self.get_temp_dir(),
        policy_trainer_class=functools.partial(
            trainers.PPO,
            policy_and_value_model=functools.partial(
                models.FrameStackMLP,
                n_frames=1,
                hidden_sizes=(),
                output_size=1,
            ),
            n_optimizer_steps=1,
        ),
        n_real_epochs=1,
        data_eval_frac=0.5,
        model_train_batch_size=2,
        simulated_env_problem_class=functools.partial(
            simulated_env_problem.SerializedSequenceSimulatedEnvProblem,
            model=functools.partial(
                models.TransformerLM,
                d_model=2,
                n_layers=0,
                max_len=64,
            ),
            reward_fn=simulated_env_problem.acrobot_reward_fn,
            done_fn=simulated_env_problem.acrobot_done_fn,
            vocab_size=4,
            max_trajectory_length=4,
        ),
        simulated_batch_size=2,
        n_simulated_epochs=1,
    )
    trainer.training_loop(n_epochs=1)


if __name__ == "__main__":
  test.main()
