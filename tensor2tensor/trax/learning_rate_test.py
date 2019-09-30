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

"""Tests for tensor2tensor.trax.learning_rate."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp

from tensor2tensor.trax import history as trax_history
from tensor2tensor.trax import learning_rate
from tensor2tensor.trax.backend import numpy as np
from tensor2tensor.trax.backend import random as jax_random
from tensor2tensor.trax.models import atari_cnn
from tensor2tensor.trax.rl import online_tune
from tensor2tensor.trax.rl import ppo
from tensorflow import test


class PolicyScheduleTest(test.TestCase):

  def _make_schedule(
      self,
      history,
      control_configs,
      observation_metrics=(("eval", "metrics/accuracy"),),
      action_multipliers=(1.0,),
  ):
    policy_and_value_model = atari_cnn.FrameStackMLP
    net = ppo.policy_and_value_net(
        n_actions=len(action_multipliers),
        n_controls=len(control_configs),
        vocab_size=None,
        bottom_layers_fn=policy_and_value_model,
        two_towers=False,
    )
    rng = jax_random.get_prng(seed=0)
    obs_dim = len(observation_metrics)
    (params, state) = net.initialize_once((1, 1, obs_dim), np.float32, rng)
    policy_dir = self.get_temp_dir()
    # Optimizer slots should not be used for anything.
    slots = None
    opt_state = (params, slots)
    ppo.save_opt_state(policy_dir, opt_state, state, epoch=0, total_opt_step=0)
    return learning_rate.PolicySchedule(
        history,
        observation_metrics=observation_metrics,
        include_controls_in_observation=False,
        action_multipliers=action_multipliers,
        control_configs=control_configs,
        policy_and_value_model=policy_and_value_model,
        policy_and_value_two_towers=False,
        policy_dir=policy_dir,
    )

  def test_returns_start_lr_when_there_are_no_metrics(self):
    history = trax_history.History()
    start_lr = 1e-3
    schedule = self._make_schedule(
        history,
        control_configs=(("learning_rate", start_lr, (1e-9, 1.0), False),),
    )
    self.assertEqual(schedule(0)["learning_rate"], start_lr)

  def test_changes_lr_when_there_are_some_metrics(self):
    history = trax_history.History()
    history.append("eval", "metrics/accuracy", step=0, value=0.8)
    history.append(
        *online_tune.control_metric("learning_rate"), step=0, value=1e-4
    )
    schedule = self._make_schedule(
        history,
        control_configs=(("learning_rate", 1e-3, (1e-9, 1.0), False),),
        observation_metrics=(("eval", "metrics/accuracy"),),
        action_multipliers=(0.5, 2.0),
    )
    new_lr = schedule(123)["learning_rate"]
    self.assertTrue(
        onp.allclose(new_lr, 5e-5) or onp.allclose(new_lr, 2e-4)
    )

  def test_works_with_multiple_controls(self):
    history = trax_history.History()
    history.append("eval", "metrics/accuracy", step=0, value=0.8)
    history.append(
        *online_tune.control_metric("learning_rate"), step=0, value=1e-4
    )
    history.append(
        *online_tune.control_metric("weight_decay_rate"), step=0, value=1e-5
    )
    schedule = self._make_schedule(
        history,
        observation_metrics=(("eval", "metrics/accuracy"),),
        control_configs=(
            ("learning_rate", 1e-3, (1e-9, 1.0), False),
            ("weight_decay_rate", 1e-5, (1e-9, 1.0), False),
        ),
        action_multipliers=(1.0,),
    )
    new_controls = schedule(123)
    self.assertIn("learning_rate", new_controls)
    self.assertIn("weight_decay_rate", new_controls)


if __name__ == "__main__":
  test.main()
