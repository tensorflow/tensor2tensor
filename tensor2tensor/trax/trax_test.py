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

from tensor2tensor.trax import inputs as inputs_lib
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

  return inputs_lib.Inputs(
      train_stream=input_stream,
      eval_stream=input_stream,
      input_shape=input_shape)


class TraxTest(test.TestCase):

  @contextlib.contextmanager
  def tmp_dir(self):
    tmp = tempfile.mkdtemp(dir=self.get_temp_dir())
    yield tmp
    gfile.rmtree(tmp)

  def test_train_eval_predict(self):
    with self.tmp_dir() as output_dir:
      # Prepare model and inputs
      num_classes = 4
      train_steps = 2
      eval_steps = 2
      model = functools.partial(models.MLP,
                                hidden_size=16,
                                num_output_classes=num_classes)
      inputs = lambda: test_inputs(num_classes)

      # Train and evaluate
      state = trax.train(output_dir,
                         model=model,
                         inputs=inputs,
                         train_steps=train_steps,
                         eval_steps=eval_steps)

      # Assert total train steps
      self.assertEqual(train_steps, state.step)

      # Assert 2 evaluations ran
      train_acc = state.history.get("train", "metrics/accuracy")
      eval_acc = state.history.get("eval", "metrics/accuracy")
      self.assertEqual(len(train_acc), len(eval_acc))
      self.assertEqual(2, len(eval_acc))

      # Predict with final params
      _, predict_fun = model()
      inputs = inputs().train_stream()
      predict_fun(state.params, next(inputs)[0])


if __name__ == "__main__":
  test.main()
