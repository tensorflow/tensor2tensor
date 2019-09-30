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

import functools
import itertools
import os
import random
import time

from absl import logging
import gin
from matplotlib import pyplot as plt
from tensor2tensor.trax import inputs as trax_inputs
from tensor2tensor.trax import jaxboard
from tensor2tensor.trax import trax
from tensor2tensor.trax.rl import base_trainer
from tensor2tensor.trax.rl import simple
from tensor2tensor.trax.rl import simulated_env_problem
from tensorflow.io import gfile


class SimPLe(base_trainer.BaseTrainer):
  """SimPLe trainer."""

  def __init__(self,
               train_env,
               eval_env,
               output_dir,
               policy_trainer_class,
               n_real_epochs=10,
               data_eval_frac=0.125,
               model_train_batch_size=64,
               n_model_initial_train_steps=1000,
               n_model_train_steps_per_epoch=1000,
               simulated_env_problem_class=(
                   simulated_env_problem.SerializedSequenceSimulatedEnvProblem),
               simulated_batch_size=16,
               n_simulated_epochs=1000,
               trajectory_dump_dir=None,
               initial_trajectory_dir=None,
               initial_trajectory_mix_prob=0.5,
               initial_model=None,
               init_policy_from_world_model=False,
               **kwargs):
    super(SimPLe, self).__init__(train_env, eval_env, output_dir, **kwargs)
    self._policy_dir = os.path.join(output_dir, "policy")
    self._model_dir = os.path.join(output_dir, "model")
    # Initialize the policy trainer lazily, so in case of initializing the
    # policy from world model checkpoint, the trainer will try to load the
    # checkpoint _after_ it's been created in train_model().
    self._policy_trainer_fn = functools.partial(
        policy_trainer_class,
        train_env=train_env,
        eval_env=eval_env,
        output_dir=self._policy_dir,
        async_mode=self._async_mode,
        init_policy_from_world_model_output_dir=(
            self._model_dir if init_policy_from_world_model else None
        ),
    )
    self._policy_trainer = None
    self._n_real_epochs = n_real_epochs
    self._model_train_batch_size = model_train_batch_size
    self._n_model_initial_train_steps = n_model_initial_train_steps
    self._n_model_train_steps_per_epoch = n_model_train_steps_per_epoch
    self._data_eval_frac = data_eval_frac

    gfile.makedirs(self._model_dir)
    if initial_model is not None:
      gfile.copy(
          initial_model,
          os.path.join(self._model_dir, "model.pkl"),
          overwrite=True,
      )
    self._initial_model = initial_model
    self._initial_trajectories = None

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

    self._initial_trajectory_dir = initial_trajectory_dir
    self._initial_trajectory_mix_prob = initial_trajectory_mix_prob

    self._summary_writer = jaxboard.SummaryWriter(self._output_dir)

    self._simple_epoch = 0
    self._policy_epoch = 0
    self._model_train_step = 0

  @property
  def policy_trainer(self):
    if self._policy_trainer is None:
      self._policy_trainer = self._policy_trainer_fn()
    return self._policy_trainer

  @property
  def epoch(self):
    return self._simple_epoch

  def train_epoch(self, evaluate=True):
    if self._simple_epoch > 0 or not self._has_initial_data:
      logging.info(
          "Collect trajectories by running the policy in the real environment.")
      self.collect_trajectories(evaluate=evaluate)
    if self._simple_epoch > 0 or not self._initial_model:
      logging.info(
          "Train the model of the environment on the collected trajectories.")
      skipped = self.train_model()
      if evaluate and not skipped:
        logging.info("Evaluate the trained model.")
        self.evaluate_model()
    logging.info("Train the policy inside the simulated environment generated "
                 "by the model.")
    self.train_policy()

    self._simple_epoch += 1

  def evaluate(self):
    self.policy_trainer.evaluate()

  def save(self):
    # Nothing to do, as we save stuff continuously.
    pass

  def flush_summaries(self):
    self._summary_writer.flush()

  def collect_trajectories(self, evaluate):
    logging.info("SimPLe epoch [% 6d]: collecting data.", self._simple_epoch)
    start_time = time.time()

    self.policy_trainer.train_env = self.train_env
    self.policy_trainer.trajectory_dump_dir = os.path.join(
        self._trajectory_dump_root_dir, str(self.epoch))
    self._policy_epoch += self._n_real_epochs
    self.policy_trainer.training_loop(self._policy_epoch, evaluate=evaluate)

    logging.vlog(
        1, "Collecting trajectories took %0.2f sec.", time.time() - start_time)

  def train_model(self):
    """Train the model.

    Returns:
      whether the training was skipped due to a restart.
    """
    logging.info("SimPLe epoch [% 6d]: training model.", self._simple_epoch)
    start_time = time.time()

    (train_stream, eval_stream) = self._make_input_streams()
    # Ignore n_devices for now.
    inputs = lambda _: trax_inputs.Inputs(  # pylint: disable=g-long-lambda
        train_stream=(lambda: train_stream),
        train_eval_stream=(lambda: train_stream),
        eval_stream=(lambda: eval_stream),
        input_shape=self._sim_env.model_input_shape,
        input_dtype=self._sim_env.model_input_dtype,
        # TODO(lukaszkaiser): correct those, they may differ from inputs.
        target_shape=self._sim_env.model_input_shape,
        target_dtype=self._sim_env.model_input_dtype)

    if self._simple_epoch == 0:
      train_steps = self._n_model_initial_train_steps
    else:
      train_steps = self._n_model_train_steps_per_epoch
    self._model_train_step += train_steps
    with gin.config_scope("world_model"):
      state = trax.train(
          model=self._sim_env.model,
          inputs=inputs,
          train_steps=self._model_train_step,
          output_dir=self._model_dir,
          has_weights=True,
      )

    logging.vlog(
        1, "Training model took %0.2f sec.", time.time() - start_time)
    return state.step > self._model_train_step

  def train_policy(self):
    logging.info("SimPLe epoch [% 6d]: training policy.", self._simple_epoch)
    start_time = time.time()

    self._sim_env.initialize(
        batch_size=self._simulated_batch_size,
        history_stream=itertools.repeat(None),
    )
    # We never want async mode in the simulated env.
    original_async_mode = self.policy_trainer.async_mode
    self.policy_trainer.async_mode = False
    self.policy_trainer.train_env = self._sim_env
    # Don't dump trajectories from the simulated environment.
    self.policy_trainer.trajectory_dump_dir = None
    self._policy_epoch += self._n_simulated_epochs
    self.policy_trainer.training_loop(self._policy_epoch, evaluate=False)
    # Revert back to the original async mode in the policy trainer.
    self.policy_trainer.async_mode = original_async_mode

    logging.vlog(
        1, "Training policy took %0.2f sec.", time.time() - start_time)

  @property
  def _has_own_data(self):
    return self._simple_epoch > 0 or self._initial_trajectory_dir is None

  @property
  def _has_initial_data(self):
    return self._initial_trajectory_dir is not None

  def _load_trajectories(self, initial):
    # Cache the initial trajectories in memory, as loading them can take a lot
    # of time and they don't change.
    if initial:
      if self._initial_trajectories is not None:
        return self._initial_trajectories
      trajectory_dir = self._initial_trajectory_dir
    else:
      trajectory_dir = self._trajectory_dump_root_dir

    trajectories = simple.load_trajectories(
        trajectory_dir, self._data_eval_frac
    )

    if initial:
      self._initial_trajectories = trajectories
    return trajectories

  def _make_input_streams(self):
    def make_example_streams(initial):
      (train_trajs, eval_trajs) = self._load_trajectories(initial)
      generate_examples = functools.partial(
          simple.generate_examples,
          trajectory_to_training_examples_fn=(
              self._sim_env.trajectory_to_training_examples),
      )
      return tuple(map(generate_examples, (train_trajs, eval_trajs)))

    # We mix two data sources: trajectories collected in this SimPLe training
    # loop ("own" data) and trajectories collected before, outside of this
    # training loop ("initial" data).
    mix_prob = self._initial_trajectory_mix_prob

    if self._has_initial_data:
      start_time = time.time()
      # Load the initial, precollected data.
      (init_train_stream, init_eval_stream) = make_example_streams(initial=True)
      logging.vlog(
          1, "Loading initial trajectories took %0.2f sec.",
          time.time() - start_time
      )
    else:
      (init_train_stream, init_eval_stream) = (None, None)
      mix_prob = 0.0  # Take just our own collected data.

    if self._has_own_data:
      start_time = time.time()
      # Load trajectories collected in all epochs so far.
      (own_train_stream, own_eval_stream) = make_example_streams(initial=False)
      logging.vlog(
          1, "Loading own trajectories took %0.2f sec.",
          time.time() - start_time
      )
    else:
      # We start the loop with training the model, so we don't have our own
      # collected data yet.
      (own_train_stream, own_eval_stream) = (None, None)
      mix_prob = 1.0  # Take just the initial data.

    def mix_and_batch(streams):
      (init_stream, own_stream) = streams
      mixed_stream = simple.mix_streams(init_stream, own_stream, mix_prob)
      return simple.batch_stream(mixed_stream, self._model_train_batch_size)

    return tuple(
        map(mix_and_batch, (
            (init_train_stream, own_train_stream),
            (init_eval_stream, own_eval_stream),
        )))

  def evaluate_model(self):
    logging.info("SimPLe epoch [% 6d]: evaluating model.", self._simple_epoch)
    start_time = time.time()

    self._sim_env.initialize(
        batch_size=self._simulated_batch_size,
        history_stream=itertools.repeat(None),
    )

    (_, eval_trajectories) = self._load_trajectories(
        # If we have any trajectories collected in this run, evaluate on them.
        # Otherwise, use the initial dataset.
        initial=(not self._has_own_data)
    )
    chosen_trajectories = [
        random.choice(eval_trajectories)
        for _ in range(self._sim_env.batch_size)
    ]
    summaries = simple.evaluate_model(self._sim_env, chosen_trajectories, plt)
    if summaries is not None:
      for (name, value) in summaries.items():
        self._summary_writer.scalar(
            "simple/{}".format(name), value, step=self._simple_epoch)
      self._summary_writer.plot(
          "simple/model_eval_plot", plt, step=self._simple_epoch)
      self.flush_summaries()

    logging.vlog(
        1, "Evaluating model took %0.2f sec.", time.time() - start_time)
