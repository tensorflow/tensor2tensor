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

"""trax test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import functools
import tempfile

import numpy as np

from tensor2tensor.trax import inputs
from tensor2tensor.trax import models
from tensor2tensor.trax import trax

from tensorflow import test
from tensorflow.io import gfile


def test_inputs(num_classes):
  """Make trax.inputs.Inputs."""
  batch_size = 2
  input_shape = (6, 6, 3)

  def input_stream():
    while True:
      yield (np.random.rand(*([batch_size] + list(input_shape))),
             np.random.randint(num_classes, size=batch_size))

  return inputs.Inputs(
      train_stream=input_stream,
      eval_stream=input_stream,
      input_shape=input_shape)


class TraxTest(test.TestCase):

  @contextlib.contextmanager
  def tmp_dir(self):
    tmp = tempfile.mkdtemp(dir=self.get_temp_dir())
    yield tmp
    gfile.rmtree(tmp)

  @property
  def train_args(self):
    num_classes = 4
    return dict(
        model=functools.partial(models.MLP,
                                hidden_size=16,
                                num_output_classes=num_classes),
        inputs=lambda: test_inputs(num_classes),
        train_steps=3,
        eval_steps=2)

  def _test_train(self, train_args):
    with self.tmp_dir() as output_dir:
      state = trax.train(output_dir, **train_args)

      # Assert total train steps
      self.assertEqual(train_args["train_steps"], state.step)

      # Assert 2 epochs ran
      train_acc = state.history.get("train", "metrics/accuracy")
      eval_acc = state.history.get("eval", "metrics/accuracy")
      self.assertEqual(len(train_acc), len(eval_acc))
      self.assertEqual(2, len(eval_acc))

  def test_train(self):
    self._test_train(self.train_args)


if __name__ == "__main__":
  test.main()
