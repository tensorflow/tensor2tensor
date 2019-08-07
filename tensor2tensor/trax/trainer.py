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

"""trax trainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

from absl import app
from absl import flags
from absl import logging

import gin
import jax
from tensor2tensor.trax import backend
from tensor2tensor.trax import trax

import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string("dataset", None, "Which dataset to use.")
flags.DEFINE_string("model", None, "Which model to train.")
flags.DEFINE_string("data_dir", None, "Path to the directory with data.")
flags.DEFINE_string("output_dir", None,
                    "Path to the directory to save logs and checkpoints.")
flags.DEFINE_multi_string("config_file", None,
                          "Configuration file with parameters (.gin).")
flags.DEFINE_multi_string("config", None,
                          "Configuration parameters (gin string).")
flags.DEFINE_integer("log_level", logging.INFO, "Log level.")
flags.DEFINE_bool("use_tpu", False, "Whether we're running on TPU.")
flags.DEFINE_bool("tf_eager", True, "Whether we're running TF in eager mode.")
flags.DEFINE_bool("tf_xla", True, "Whether to turn on XLA for TF.")
flags.DEFINE_bool("tf_opt_pin_to_host", False, "Whether to turn on TF "
                  "pin-to-host optimization.")
flags.DEFINE_bool("tf_opt_layout", False, "Whether to turn on TF layout "
                  "optimization.")


def _default_output_dir():
  """Default output directory."""
  try:
    dataset_name = gin.query_parameter("inputs.dataset_name")
  except ValueError:
    dataset_name = "random"
  dir_name = "{model_name}_{dataset_name}_{timestamp}".format(
      model_name=gin.query_parameter("train.model").configurable.name,
      dataset_name=dataset_name,
      timestamp=datetime.datetime.now().strftime("%Y%m%d_%H%M"),
  )
  dir_path = os.path.join("~", "trax", dir_name)
  print()
  trax.log("No --output_dir specified")
  return dir_path


def _setup_gin():
  """Setup gin configuration."""
  # Imports for configurables
  # pylint: disable=g-import-not-at-top,unused-import,g-bad-import-order,reimported,unused-variable
  from tensor2tensor.trax import models as _trax_models
  from tensor2tensor.trax import optimizers as _trax_opt
  # pylint: disable=g-import-not-at-top,unused-import,g-bad-import-order,reimported,unused-variable

  configs = FLAGS.config or []
  # Override with --dataset and --model
  if FLAGS.dataset:
    configs.append("inputs.dataset_name='%s'" % FLAGS.dataset)
    if FLAGS.data_dir:
      configs.append("inputs.data_dir='%s'" % FLAGS.data_dir)
  if FLAGS.model:
    configs.append("train.model=@trax.models.%s" % FLAGS.model)
  gin.parse_config_files_and_bindings(FLAGS.config_file, configs)


def main(_):
  logging.set_verbosity(FLAGS.log_level)

  if FLAGS.tf_eager:
    tf.enable_eager_execution()

  if FLAGS.tf_xla:
    tf.config.optimizer.set_jit(True)

  tf.config.optimizer.set_experimental_options(
      {"pin_to_host_optimization": FLAGS.tf_opt_pin_to_host}
  )

  tf.config.optimizer.set_experimental_options(
      {"layout_optimizer": FLAGS.tf_opt_layout}
  )

  _setup_gin()

  if FLAGS.tf_eager and backend.get_name() in ("numpy", "jax"):
    # Numpy backend doesn't benefit from having the input pipeline run on GPU,
    # and jax backend has GPU memory contention if TF uses the GPU. Gin must be
    # set up first before determining the backend.
    tf.config.experimental.set_visible_devices([], "GPU")

  # Setup output directory
  output_dir = FLAGS.output_dir or _default_output_dir()
  trax.log("Using --output_dir %s" % output_dir)
  output_dir = os.path.expanduser(output_dir)

  # If on TPU, let JAX know.
  if FLAGS.use_tpu:
    jax.config.update("jax_platform_name", "tpu")

  trax.train(output_dir=output_dir)


if __name__ == "__main__":
  app.run(main)
