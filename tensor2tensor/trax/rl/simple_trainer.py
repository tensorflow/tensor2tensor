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

"""SimPLe trainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import random

from absl import logging
from jax import numpy as np
from tensor2tensor.trax import inputs as trax_inputs
from tensor2tensor.trax import trax
from tensor2tensor.trax.rl import base_trainer
from tensor2tensor.trax.rl import simulated_env_problem


class SimPLe(base_trainer.BaseTrainer):
  """SimPLe trainer."""

  def __init__(
      self,
      train_env,
      eval_env,
      output_dir,
      policy_trainer_class,
      n_real_epochs=10,
      data_eval_frac=0.05,
      model_train_batch_size=64,
      simulated_env_problem_class=(
          simulated_env_problem.SerializedSequenceSimulatedEnvProblem),
      simulated_batch_size=16,
      n_simulated_epochs=1000,
  ):
    super(SimPLe, self).__init__(train_env, eval_env, output_dir)
    self._policy_dir = os.path.join(output_dir, "policy")
    self._policy_trainer = policy_trainer_class(
        train_env=train_env,
        eval_env=eval_env,
        output_dir=self._policy_dir,
    )
    self._n_real_epochs = n_real_epochs
    self._model_train_batch_size = model_train_batch_size
    self._data_eval_frac = data_eval_frac
    self._train_trajectories = []
    self._eval_trajectories = []
    self._model_dir = os.path.join(output_dir, "model")
    self._sim_env = simulated_env_problem_class(
        batch_size=None,
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        reward_range=train_env.reward_range,
        discrete_rewards=train_env.discrete_rewards,
        history_stream=None,  # TODO(pkozakowski): Support this.
        output_dir=self._model_dir,
    )
    self._simulated_batch_size = simulated_batch_size
    self._n_simulated_epochs = n_simulated_epochs
    self._epoch = 0

  @property
  def epoch(self):
    return self._epoch

  def train_epoch(self):
    self.collect_trajectories()
    self.train_model()
    self.train_policy()
    self._epoch += 1

  def evaluate(self):
    self._policy_trainer.evaluate()

  def save(self):
    # Nothing to do, as we save stuff continuously.
    pass

  def flush_summaries(self):
    # TODO(pkozakowski): Report some metrics, timing?
    pass

  def collect_trajectories(self):
    logging.info("Epoch %d: collecting data", self._epoch)

    self._policy_trainer.train_env = self.train_env
    self._policy_trainer.training_loop(self._n_real_epochs)
    self.train_env.trajectories.complete_all_trajectories()
    trajectories = self.train_env.trajectories.completed_trajectories
    pivot = int(len(trajectories) * (1 - self._data_eval_frac))
    self._train_trajectories.extend(trajectories[:pivot])
    self._eval_trajectories.extend(trajectories[pivot:])
    # TODO(pkozakowski): Save trajectories to disk. Support restoring.

  def _data_stream(self, trajectories, batch_size):
    def make_batch(examples):
      """Stack a structure of np arrays nested in lists/tuples."""
      assert examples
      if isinstance(examples[0], (list, tuple)):
        return type(examples[0])(
            make_batch([example[i] for example in examples])
            for i in range(len(examples[0]))
        )
      else:
        batch = np.stack(examples, axis=0)
        pad_width = (
            [(0, batch_size - len(examples))] +
            [(0, 0)] * (len(batch.shape) - 1)
        )
        # Pad with zeros. This doesn't change anything, because we have weights
        # in the examples.
        return np.pad(batch, pad_width, mode="constant")

    examples = [
        example  # pylint: disable=g-complex-comprehension
        for trajectory_examples in map(
            self._sim_env.trajectory_to_training_examples, trajectories)
        for example in trajectory_examples
    ]
    while True:
      random.shuffle(examples)
      for from_index in range(0, len(examples), batch_size):
        example_list = examples[from_index:(from_index + batch_size)]
        yield make_batch(example_list)

  def train_model(self):
    logging.info("Epoch %d: training model", self._epoch)

    train_stream = lambda: self._data_stream(  # pylint: disable=g-long-lambda
        self._train_trajectories, self._model_train_batch_size)
    eval_stream = lambda: self._data_stream(  # pylint: disable=g-long-lambda
        self._eval_trajectories, self._model_train_batch_size)
    # Ignore n_devices for now.
    inputs = lambda _: trax_inputs.Inputs(  # pylint: disable=g-long-lambda
        train_stream=train_stream,
        train_eval_stream=train_stream,
        eval_stream=eval_stream,
        input_shape=self._sim_env.model_input_shape,
        input_dtype=self._sim_env.model_input_dtype,
    )
    trax.train(
        model=self._sim_env.model,
        inputs=inputs,
        output_dir=self._model_dir,
        has_weights=True,
    )

  def train_policy(self):
    logging.info("Epoch %d: training policy", self._epoch)

    self._sim_env.initialize(
        batch_size=self._simulated_batch_size,
        history_stream=itertools.repeat(None),
    )
    self._policy_trainer.train_env = self._sim_env
    self._policy_trainer.training_loop(self._n_simulated_epochs)
