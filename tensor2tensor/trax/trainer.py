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

from tensor2tensor.trax import trax

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


def _default_output_dir():
  """Default output directory."""
  dir_name = "{model_name}_{dataset_name}_{timestamp}".format(
      model_name=gin.query_parameter("train.model").configurable.name,
      dataset_name=gin.query_parameter("inputs.dataset_name"),
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
  from tensor2tensor.trax import inputs as _trax_inputs
  from tensor2tensor.trax import models as _trax_models
  from tensor2tensor.trax import optimizers as _trax_opt
  # pylint: disable=g-import-not-at-top,unused-import,g-bad-import-order,reimported,unused-variable

  configs = FLAGS.config or []
  # Override with --dataset and --model
  if FLAGS.dataset:
    configs.append("inputs.dataset_name='%s'" % FLAGS.dataset)
    configs.append("inputs.data_dir='%s'" % FLAGS.data_dir)
    configs.append("train.inputs=@trax.inputs.inputs")
  if FLAGS.model:
    configs.append("train.model=@trax.models.%s" % FLAGS.model)
  gin.parse_config_files_and_bindings(FLAGS.config_file, configs)


def main(_):
  _setup_gin()

  # Setup output directory
  output_dir = FLAGS.output_dir or _default_output_dir()
  trax.log("Using --output_dir %s" % output_dir)
  output_dir = os.path.expanduser(output_dir)

  trax.train(output_dir=output_dir)


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)
