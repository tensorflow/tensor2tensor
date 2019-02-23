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

r"""J2J trainer.

Examples:

- train a basic model on mnist:
    jax/j2j_trainer.py --dataset=mnist --model=mlp
      --config="train_fn.train_steps=4000" --output_dir ~/j2j/test1
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging

import gin
from tensor2tensor.jax import j2j

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


def _setup_gin():
  configs = FLAGS.config or []
  # Override with --dataset and --model
  if FLAGS.dataset:
    configs.append("train_fn.dataset='%s'" % FLAGS.dataset)
  if FLAGS.model:
    configs.append("train_fn.model=@" + FLAGS.model)
  gin.parse_config_files_and_bindings(FLAGS.config_file, configs)


def main(_):
  _setup_gin()

  # Setup directories
  data_dir, output_dir = FLAGS.data_dir, FLAGS.output_dir
  data_dir = data_dir and os.path.expanduser(data_dir)
  output_dir = output_dir and os.path.expanduser(output_dir)

  j2j.train_fn(data_dir, output_dir=output_dir)


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)
