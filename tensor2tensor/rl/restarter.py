# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""Training restarter."""

import contextlib
import os

import tensorflow.compat.v1 as tf


class Restarter(object):
  """Handles training restarts.

  Particularly useful when sharing parameters (and checkpoints) between models.

  Args:
    model_mode (str): Model "mode". Different modes have different local step
        counters, but the same global step counter. Also used in log messages.
    checkpoint_dir (str): Model checkpoint directory. Global step is inferred
        from the name of the last checkpoint.
    target_local_step (int): Local step to train the model up to.

  Attributes:
    model_mode (str): See args.
    checkpoint_dir (str): See args.
    target_local_step (int): See args.
    target_global_step (int): Calculated global step to train the model up to.
    should_skip (bool): Whether training should be skipped because the number of
        local steps already done is higher than the target. This happens during
        restarts.
    steps_to_go: how many steps to go.
    restarting (bool): Whether the current epoch of training has been
        interrupted and is being restarted.
  """

  def __init__(self, model_mode, checkpoint_dir, target_local_step):
    self.model_mode = model_mode
    self.checkpoint_dir = checkpoint_dir
    self.target_local_step = target_local_step
    self.target_global_step = None
    self.should_skip = False
    self.restarting = False

    self._counter_path = os.path.join(
        checkpoint_dir, "{}_step_counter".format(model_mode)
    )

    self._global_step = self._get_global_step()
    tf.logging.info(
        "Will load %s checkpoint %d", self.model_mode, self._global_step
    )

    (self._local_step_at_start, global_step_at_start) = self._read_counters()

    self.steps_to_go = target_local_step - self._local_step_at_start
    if self.steps_to_go <= 0:
      tf.logging.info(
          "Skipping training %s, requested %d steps, already done %d",
          self.model_mode, target_local_step, self._local_step_at_start
      )
      self.should_skip = True
      return

    if global_step_at_start != -1:
      # Restart.
      steps_done_this_epoch = self._global_step - global_step_at_start
      self.steps_to_go -= steps_done_this_epoch
      tf.logging.info(
          "Restarting training %s, %d steps already done this epoch",
          self.model_mode, steps_done_this_epoch
      )
      self.restarting = True

    self.target_global_step = self._global_step + self.steps_to_go

  @contextlib.contextmanager
  def training_loop(self):
    """Context manager wrapping the training loop, updates step counters."""
    if not self.restarting:
      self._write_counters(self._local_step_at_start, self._global_step)

    tf.logging.info(
        "Training %s up to %d, %d to go", self.model_mode,
        self.target_local_step, self.steps_to_go
    )

    yield

    self._write_counters(self.target_local_step, -1)

  def _get_global_step(self):
    checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
    if checkpoint:
      return int(checkpoint.split("-")[-1])
    else:
      return 0

  def _read_counters(self):
    try:
      with tf.gfile.Open(self._counter_path, "r") as f:
        return tuple(
            int(counter) for counter in f.read().split(" ")
        )
    except tf.errors.NotFoundError:
      return (0, -1)

  def _write_counters(self, local_step, global_step):
    with tf.gfile.Open(self._counter_path, "w") as f:
      f.write("{} {}".format(local_step, global_step))
