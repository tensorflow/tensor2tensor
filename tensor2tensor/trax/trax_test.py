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
from absl.testing import parameterized

from jax import test_util  # pylint: disable=unused-import
from jax.config import config
from jax.lib import xla_bridge
import numpy as onp

from tensor2tensor.trax import backend
from tensor2tensor.trax import inputs as inputs_lib
from tensor2tensor.trax import layers
from tensor2tensor.trax import learning_rate as lr
from tensor2tensor.trax import models
from tensor2tensor.trax import optimizers as trax_opt
from tensor2tensor.trax import trax
from tensor2tensor.trax.backend import numpy as np

import tensorflow as tf
from tensorflow import test
from tensorflow.io import gfile


def test_inputs(n_classes, with_weights=False):
  """Make trax.inputs.Inputs."""
  batch_size = 2 * xla_bridge.device_count()
  input_shape = (6, 6, 3)

  def input_stream():
    key = backend.random.get_prng(0)
    while True:
      keys = backend.random.split(key, 4)
      key = keys[0]
      inputs = backend.random.uniform(keys[1], [batch_size] + list(input_shape))
      targets = backend.random.randint(keys[2], [batch_size], dtype=np.int32,
                                       minval=0, maxval=n_classes)
      weights = backend.random.uniform(keys[3], [batch_size])
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


BACKENDS = ["jax"]


class TraxTest(test.TestCase, parameterized.TestCase):

  @contextlib.contextmanager
  def tmp_dir(self):
    tmp = tempfile.mkdtemp(dir=self.get_temp_dir())
    yield tmp
    gfile.rmtree(tmp)

  # TODO(wangpeng): Remove `skipTest`'s when tf-numpy's `pmap` is in place

  @parameterized.parameters(BACKENDS)
  def test_train_eval_predict(self, backend_name):
    if xla_bridge.device_count() > 1 and backend_name == "tf":
      self.skipTest("tf-numpy backend doesn't support multi-devices yet.")
    with backend.use_backend(backend_name), self.tmp_dir() as output_dir:
      # Prepare model and inputs
      n_classes = 4
      train_steps = 2
      eval_steps = 2
      # Adds Dropout and BatchNorm to test state handling.
      mlp = functools.partial(models.MLP,
                              d_hidden=16,
                              n_output_classes=n_classes)
      def model_fn(mode="train"):
        return layers.Model(layers.Dropout(mode=mode, rate=0.1),
                            layers.BatchNorm(mode=mode),
                            mlp(mode=mode))

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
      self.assertLen(eval_acc, 2)

      # Predict with final params
      inputs = inputs(1).train_stream()
      model = layers.Serial(model_fn())
      model(next(inputs)[0], state.opt_state.params)

  @parameterized.parameters(BACKENDS)
  def test_train_eval_predict_sm3(self, backend_name):
    if xla_bridge.device_count() > 1 and backend_name == "tf":
      self.skipTest("tf-numpy backend doesn't support multi-devices yet.")
    with backend.use_backend(backend_name), self.tmp_dir() as output_dir:
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
      self.assertLen(eval_acc, 2)

      # Predict with final params
      inputs = inputs(1).train_stream()
      model = layers.Serial(model_fn())
      model(next(inputs)[0], state.opt_state.params)

  @parameterized.parameters(BACKENDS)
  def test_train_restart(self, backend_name):
    if xla_bridge.device_count() > 1 and backend_name == "tf":
      self.skipTest("tf-numpy backend doesn't support multi-devices yet.")
    with backend.use_backend(backend_name), self.tmp_dir() as output_dir:
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

  @parameterized.parameters(BACKENDS)
  def test_train_with_weights(self, backend_name):
    if xla_bridge.device_count() > 1 and backend_name == "tf":
      self.skipTest("tf-numpy backend doesn't support multi-devices yet.")
    with backend.use_backend(backend_name), self.tmp_dir() as output_dir:
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
                         eval_steps=eval_steps,
                         has_weights=True)

      # Assert total train steps
      self.assertEqual(state.step, train_steps)

  @parameterized.parameters(BACKENDS)
  def test_reset_twice(self, backend_name):
    if xla_bridge.device_count() > 1 and backend_name == "tf":
      self.skipTest("tf-numpy backend doesn't support multi-devices yet.")
    with backend.use_backend(backend_name), self.tmp_dir() as output_dir1, \
          self.tmp_dir() as output_dir2:
      n_classes = 4
      model_fn = functools.partial(models.MLP,
                                   d_hidden=16,
                                   n_output_classes=n_classes)
      inputs = lambda _: test_inputs(n_classes)

      trainer = trax.Trainer(
          model=model_fn,
          loss_fn=trax.loss,
          optimizer=trax_opt.SM3,
          lr_schedule=lr.MultifactorSchedule,
          inputs=inputs,
      )

      trainer.reset(output_dir1)
      trainer.evaluate(1)
      trainer.reset(output_dir2)
      trainer.evaluate(1)


MASKED_MEAN_TEST_BACKENDS = ["numpy"]


class MaskedMeanTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters(MASKED_MEAN_TEST_BACKENDS)
  def test_computes_basic_mean(self, backend_name):
    with backend.use_backend(backend_name):
      inputs = [np.array([1, 2, 3])]
      targets = [np.zeros(3)]
      weights = [1]
      mean = trax.masked_mean(inputs, targets, weights)
      onp.testing.assert_allclose(mean, 2)

  @parameterized.parameters(MASKED_MEAN_TEST_BACKENDS)
  def test_computes_mean_with_weights(self, backend_name):
    with backend.use_backend(backend_name):
      inputs = [np.array([1, 2, 3])]
      targets = [np.zeros(3)]
      weights = [np.array([3, 1, 0])]
      mean = trax.masked_mean(inputs, targets, weights)
      onp.testing.assert_allclose(mean, 1.25)

  @parameterized.parameters(MASKED_MEAN_TEST_BACKENDS)
  def test_computes_mean_with_mask(self, backend_name):
    with backend.use_backend(backend_name):
      inputs = [np.array([1, 2, 3])]
      targets = [np.array([1, 0, 0])]
      weights = [1]
      mean = trax.masked_mean(inputs, targets, weights, mask_id=1)
      onp.testing.assert_allclose(mean, 2.5)

  @parameterized.parameters(MASKED_MEAN_TEST_BACKENDS)
  def test_computes_mean_with_weights_and_mask(self, backend_name):
    with backend.use_backend(backend_name):
      inputs = [np.array([1, 2, 4])]
      targets = [np.array([1, 0, 0])]
      weights = [np.array([10, 4, 1])]
      mean = trax.masked_mean(inputs, targets, weights, mask_id=1)
      onp.testing.assert_allclose(mean, 2.4)


if __name__ == "__main__":
  config.config_with_absl()
  test.main()
