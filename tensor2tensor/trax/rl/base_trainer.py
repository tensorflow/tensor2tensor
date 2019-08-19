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

"""Base class for RL trainers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from tensorflow.io import gfile


class BaseTrainer(object):
  """Base class for RL trainers."""

  def __init__(self, train_env, eval_env, output_dir):
    self._train_env = train_env
    self._eval_env = eval_env
    self._output_dir = output_dir
    gfile.makedirs(self._output_dir)

  @property
  def epoch(self):
    raise NotImplementedError

  def train_epoch(self):
    raise NotImplementedError

  def evaluate(self):
    raise NotImplementedError

  def save(self):
    raise NotImplementedError

  def flush_summaries(self):
    raise NotImplementedError

  def training_loop(self, n_epochs):
    logging.info("Starting the RL training loop.")
    for _ in range(self.epoch, n_epochs):
      self.train_epoch()
    self.save()
    self.evaluate()
    self.flush_summaries()
