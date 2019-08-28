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
import cloudpickle as pickle
import numpy as np
from tensor2tensor.trax import inputs as trax_inputs
from tensor2tensor.trax import trax
from tensor2tensor.trax.rl import base_trainer
from tensor2tensor.trax.rl import simulated_env_problem
from tensorflow.io import gfile


class SimPLe(base_trainer.BaseTrainer):
  """SimPLe trainer."""

  def __init__(
      self,
      train_env,
      eval_env,
      output_dir,
      policy_trainer_class,
      n_real_epochs=10,
      data_eval_frac=0.125,
      model_train_batch_size=64,
      n_model_train_steps=1000,
      simulated_env_problem_class=(
          simulated_env_problem.SerializedSequenceSimulatedEnvProblem),
      simulated_batch_size=16,
      n_simulated_epochs=1000,
      trajectory_dump_dir=None,
      **kwargs
  ):
    super(SimPLe, self).__init__(
        train_env, eval_env, output_dir, **kwargs)
    self._policy_dir = os.path.join(output_dir, "policy")
    self._policy_trainer = policy_trainer_class(
        train_env=train_env,
        eval_env=eval_env,
        output_dir=self._policy_dir,
    )
    self._n_real_epochs = n_real_epochs
    self._model_train_batch_size = model_train_batch_size
    self._n_model_train_steps = n_model_train_steps
    self._data_eval_frac = data_eval_frac
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

    # If trajectory_dump_dir is not provided explicitly, save the trajectories
    # in output_dir.
    if trajectory_dump_dir is None:
      trajectory_dump_dir = os.path.join(output_dir, "trajectories")
    self._trajectory_dump_root_dir = trajectory_dump_dir

    self._simple_epoch = 0
    self._policy_epoch = 0
    self._model_train_step = 0

  @property
  def epoch(self):
    return self._simple_epoch

  def train_epoch(self):
    self.collect_trajectories()
    self.train_model()
    self.train_policy()
    self._simple_epoch += 1

  def evaluate(self):
    self._policy_trainer.evaluate()

  def save(self):
    # Nothing to do, as we save stuff continuously.
    pass

  def flush_summaries(self):
    # TODO(pkozakowski): Report some metrics, timing?
    pass

  def collect_trajectories(self):
    logging.info("Epoch %d: collecting data", self._simple_epoch)

    self._policy_trainer.train_env = self.train_env
    self._policy_trainer.trajectory_dump_dir = os.path.join(
        self._trajectory_dump_root_dir, str(self.epoch))
    self._policy_epoch += self._n_real_epochs
    self._policy_trainer.training_loop(self._policy_epoch)

  def _load_trajectories(self, trajectory_dir):
    train_trajectories = []
    eval_trajectories = []
    # Search the entire directory subtree for trajectories.
    for (subdir, _, filenames) in gfile.walk(trajectory_dir):
      for filename in filenames:
        shard_path = os.path.join(subdir, filename)
        with gfile.GFile(shard_path, "rb") as f:
          trajectories = pickle.load(f)
          pivot = int(len(trajectories) * (1 - self._data_eval_frac))
          train_trajectories.extend(trajectories[:pivot])
          eval_trajectories.extend(trajectories[pivot:])
    assert train_trajectories, "Haven't found any training data."
    assert eval_trajectories, "Haven't found any evaluation data."
    return (train_trajectories, eval_trajectories)

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
    logging.info("Epoch %d: training model", self._simple_epoch)

    # Load data from all epochs.
    # TODO(pkozakowski): Handle the case when the data won't fit in the memory.
    (train_trajectories, eval_trajectories) = self._load_trajectories(
        self._trajectory_dump_root_dir)
    train_stream = lambda: self._data_stream(  # pylint: disable=g-long-lambda
        train_trajectories, self._model_train_batch_size)
    eval_stream = lambda: self._data_stream(  # pylint: disable=g-long-lambda
        eval_trajectories, self._model_train_batch_size)
    # Ignore n_devices for now.
    inputs = lambda _: trax_inputs.Inputs(  # pylint: disable=g-long-lambda
        train_stream=train_stream,
        train_eval_stream=train_stream,
        eval_stream=eval_stream,
        input_shape=self._sim_env.model_input_shape,
        input_dtype=self._sim_env.model_input_dtype,
    )

    self._model_train_step += self._n_model_train_steps
    trax.train(
        model=self._sim_env.model,
        inputs=inputs,
        train_steps=self._model_train_step,
        output_dir=self._model_dir,
        has_weights=True,
    )

  def train_policy(self):
    logging.info("Epoch %d: training policy", self._simple_epoch)

    self._sim_env.initialize(
        batch_size=self._simulated_batch_size,
        history_stream=itertools.repeat(None),
    )
    self._policy_trainer.train_env = self._sim_env
    self._policy_epoch += self._n_simulated_epochs
    self._policy_trainer.training_loop(self._policy_epoch)
