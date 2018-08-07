# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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
"""Toy model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_hparams
from tensor2tensor.mesh_tensorflow import mesh_tensorflow as mtf
from tensor2tensor.mesh_tensorflow import mtf_layers
from tensor2tensor.mesh_tensorflow import mtf_model
from tensor2tensor.utils import registry
import tensorflow as tf

from tensorflow.contrib.tpu.python.ops import tpu_ops


@registry.register_model
class MtfToy(mtf_model.MtfModel):
  """Toy model to test mesh_tensorflow."""

  def mtf_model_fn(self, features, mesh):
    hparams = self._hparams
    # tf_x = tf.random_uniform([hparams.batch_size, hparams.io_size])
    tf_x = tf.matmul(
        tf.reshape(
            tf.lin_space(0., 1.0, hparams.batch_size), [hparams.batch_size, 1]),
        tf.reshape(
            tf.lin_space(0., 1.0, hparams.io_size), [1, hparams.io_size]))
    batch_dim = mtf.Dimension("batch", hparams.batch_size)

    hidden_dim = mtf.Dimension("hidden", hparams.hidden_size)
    io_dim = mtf.Dimension("io", hparams.io_size)
    x = mtf.infeed_fully_replicated(
        mesh, tf_x, mtf.TensorShape([batch_dim, io_dim]))
    h = mtf_layers.dense(x, hidden_dim, name="layer1", use_bias=False)
    y = mtf_layers.dense(h, io_dim, name="layer2", use_bias=False)

    loss = mtf.reduce_sum(mtf.square(y - x))
    return None, loss


@registry.register_model
class MtfSimple(mtf_model.MtfModel):
  """Toy model to test mesh_tensorflow."""

  def mtf_model_fn(self, features, mesh):
    hparams = self._hparams
    # tf_x = tf.random_uniform([hparams.batch_size, hparams.io_size])
    tf_x = tf.matmul(
        tf.reshape(
            tf.lin_space(0., 1.0, hparams.batch_size), [hparams.batch_size, 1]),
        tf.reshape(
            tf.lin_space(0., 1.0, hparams.io_size), [1, hparams.io_size]))
    batch_dim = mtf.Dimension("batch", hparams.batch_size)
    hidden_dim = mtf.Dimension("hidden", hparams.hidden_size)
    io_dim = mtf.Dimension("io", hparams.io_size)

    x = mtf.infeed_fully_replicated(
        mesh, tf_x, mtf.TensorShape([batch_dim, io_dim]))
    h = mtf_layers.dense(x, hidden_dim, name="layer1", use_bias=False)
    y = mtf_layers.dense(h, io_dim, name="layer2", use_bias=False)
    loss = mtf.reduce_sum(mtf.square(y - x))
    return None, loss


@registry.register_model
class MtfToyNormal(mtf_model.MtfModel):
  """Toy model to test mesh_tensorflow."""

  def mtf_model_fn(self, features, mesh):
    hparams = self._hparams
    hparams.batch_size = 10
    hparams.io_size = 4
    hparams.hidden_size = 2
    tf_x = tf.matmul(
        tf.reshape(
            tf.lin_space(0., 1.0, hparams.batch_size), [hparams.batch_size, 1]),
        tf.reshape(
            tf.lin_space(0., 1.0, hparams.io_size), [1, hparams.io_size]))
    # tf_x = tf.random_uniform([hparams.batch_size, hparams.io_size])

    hidden_1_variable = tf.get_variable(
        "a",
        shape=[hparams.io_size, hparams.hidden_size],
        initializer=tf.random_normal_initializer())
    hidden_2_variable = tf.get_variable(
        "b",
        shape=[hparams.hidden_size, hparams.io_size],
        initializer=tf.random_normal_initializer())

    hidden_layer_1 = tf.matmul(tf_x, hidden_1_variable)
    hidden_layer_2 = tf.matmul(hidden_layer_1, hidden_2_variable)
    hidden_layer_2 = tpu_ops.cross_replica_sum(hidden_layer_2)
    loss = tf.reduce_mean(tf.square(hidden_layer_2 - tf_x))
    return None, loss


def set_sgd_optimizer(hparams):
  hparams.optimizer = "SGD"
  hparams.learning_rate_schedule = "constant"
  hparams.learning_rate_constant = 0.01


def set_adafactor_optimizer(hparams):
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_schedule = "rsqrt_decay"
  hparams.optimizer_adafactor_factored = True
  hparams.learning_rate_warmup_steps = 1000


@registry.register_hparams
def mtf_toy_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.no_data_parallelism = True
  hparams.use_fixed_batch_size = True
  hparams.add_hparam("mtf_mode", True)
  hparams.batch_size = 64
  set_adafactor_optimizer(hparams)
  hparams.add_hparam("io_size", 32)
  hparams.hidden_size = 32
  hparams.add_hparam("mesh_shape", "4.2")
  hparams.add_hparam("layout", "batch:0;hidden:1")
  return hparams


@registry.register_hparams
def mtf_toy_data_parallel():
  """Set of hyperparameters."""
  hparams = mtf_toy_base()
  hparams.add_hparam("layout", "batch:0")
  return hparams


@registry.register_hparams
def mtf_toy_model_parallel():
  """Set of hyperparameters."""
  hparams = mtf_toy_base()
  hparams.add_hparam("layout", "hidden:0")
  return hparams


@registry.register_hparams
def mtf_toy_data_parallel_m2():
  """Set of hyperparameters."""
  hparams = mtf_toy_data_parallel()
  hparams.mesh_shape = "2"
  return hparams


@registry.register_hparams
def mtf_toy_model_parallel_m2():
  """Set of hyperparameters."""
  hparams = mtf_toy_model_parallel()
  hparams.mesh_shape = "2"
  return hparams


@registry.register_hparams
def mtf_toy_m32():
  """Set of hyperparameters."""
  hparams = mtf_toy_base()
  hparams.mesh_shape = "8;4"
  return hparams
