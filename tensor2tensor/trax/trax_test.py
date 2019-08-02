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

import gin
from jax import test_util  # pylint: disable=unused-import
from jax.config import config
import numpy as np

from tensor2tensor.trax import backend
from tensor2tensor.trax import inputs as inputs_lib
from tensor2tensor.trax import layers
from tensor2tensor.trax import models
from tensor2tensor.trax import optimizers as trax_opt
from tensor2tensor.trax import trax

from tensorflow import test
from tensorflow.io import gfile


def test_inputs(n_classes, with_weights=False):
  """Make trax.inputs.Inputs."""
  batch_size = 2
  input_shape = (6, 6, 3)

  def input_stream():
    while True:
      inputs = np.random.rand(*([batch_size] + list(input_shape)))
      targets = np.random.randint(n_classes, size=batch_size)
      weights = np.random.rand(batch_size)
      if with_weights:
        yield inputs, targets, weights
      else:
        yield inputs, targets

  return inputs_lib.Inputs(
      train_stream=input_stream,
      train_eval_stream=input_stream,
      eval_stream=input_stream,
      input_shape=input_shape,
      input_dtype=np.float32)


class TraxTest(test.TestCase):

  @contextlib.contextmanager
  def tmp_dir(self):
    tmp = tempfile.mkdtemp(dir=self.get_temp_dir())
    yield tmp
    gfile.rmtree(tmp)

  def test_train_eval_predict(self):
    with self.tmp_dir() as output_dir:
      # Prepare model and inputs
      n_classes = 4
      train_steps = 2
      eval_steps = 2
      model_fn = functools.partial(models.MLP,
                                   d_hidden=16,
                                   n_output_classes=n_classes)
      inputs = lambda _: test_inputs(n_classes)

      # Train and evaluate
      state = trax.train(output_dir,
                         model=model_fn,
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
      inputs = inputs(1).train_stream()
      model = layers.Serial(model_fn())
      model(next(inputs)[0], state.opt_state.params)

  def test_train_eval_predict_sm3(self):
    with self.tmp_dir() as output_dir:
      # Prepare model and inputs
      n_classes = 4
      train_steps = 2
      eval_steps = 2
      model_fn = functools.partial(models.MLP,
                                   d_hidden=16,
                                   n_output_classes=n_classes)
      inputs = lambda _: test_inputs(n_classes)

      # Train and evaluate
      state = trax.train(output_dir,
                         model=model_fn,
                         inputs=inputs,
                         train_steps=train_steps,
                         eval_steps=eval_steps,
                         optimizer=trax_opt.SM3)

      # Assert total train steps
      self.assertEqual(train_steps, state.step)

      # Assert 2 evaluations ran
      train_acc = state.history.get("train", "metrics/accuracy")
      eval_acc = state.history.get("eval", "metrics/accuracy")
      self.assertEqual(len(train_acc), len(eval_acc))
      self.assertEqual(2, len(eval_acc))

      # Predict with final params
      inputs = inputs(1).train_stream()
      model = layers.Serial(model_fn())
      model(next(inputs)[0], state.opt_state.params)

  def test_train_restart(self):
    with self.tmp_dir() as output_dir:
      # Prepare model and inputs
      n_classes = 4
      train_steps = 2
      eval_steps = 2
      model_fn = functools.partial(models.MLP,
                                   d_hidden=16,
                                   n_output_classes=n_classes)
      inputs = lambda _: test_inputs(n_classes)

      # Train and evaluate
      trax.train(output_dir,
                 model=model_fn,
                 inputs=inputs,
                 train_steps=train_steps,
                 eval_steps=eval_steps)

      # Restart training
      state = trax.train(output_dir,
                         model=model_fn,
                         inputs=inputs,
                         train_steps=train_steps,
                         eval_steps=eval_steps)

      # Assert total train steps
      self.assertEqual(state.step, 2 * train_steps)

  def test_train_with_weights(self):
    with self.tmp_dir() as output_dir:
      gin.bind_parameter("unpack_batch.has_weights", True)

      # Prepare model and inputs
      n_classes = 4
      train_steps = 2
      eval_steps = 2
      model_fn = functools.partial(models.MLP,
                                   d_hidden=16,
                                   n_output_classes=n_classes)
      inputs = lambda _: test_inputs(n_classes, with_weights=True)

      # Train and evaluate
      state = trax.train(output_dir,
                         model=model_fn,
                         inputs=inputs,
                         train_steps=train_steps,
                         eval_steps=eval_steps)

      # Assert total train steps
      self.assertEqual(state.step, train_steps)


class MaskedMeanTest(test.TestCase):

  def test_computes_basic_mean(self):
    inputs = [np.array([1, 2, 3])]
    targets = [np.zeros(3)]
    weights = [1]
    with backend.use_backend("numpy"):
      mean = trax.masked_mean(inputs, targets, weights)
      np.testing.assert_allclose(mean, 2)

  def test_computes_mean_with_weights(self):
    inputs = [np.array([1, 2, 3])]
    targets = [np.zeros(3)]
    weights = [np.array([3, 1, 0])]
    with backend.use_backend("numpy"):
      mean = trax.masked_mean(inputs, targets, weights)
      np.testing.assert_allclose(mean, 1.25)

  def test_computes_mean_with_mask(self):
    inputs = [np.array([1, 2, 3])]
    targets = [np.array([1, 0, 0])]
    weights = [1]
    with backend.use_backend("numpy"):
      mean = trax.masked_mean(inputs, targets, weights, mask_id=1)
      np.testing.assert_allclose(mean, 2.5)

  def test_computes_mean_with_weights_and_mask(self):
    inputs = [np.array([1, 2, 4])]
    targets = [np.array([1, 0, 0])]
    weights = [np.array([10, 4, 1])]
    with backend.use_backend("numpy"):
      mean = trax.masked_mean(inputs, targets, weights, mask_id=1)
      np.testing.assert_allclose(mean, 2.4)


if __name__ == "__main__":
  config.config_with_absl()
  test.main()
