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

# For iterators over datasets so we can do "for example in dataset".
tf.enable_v2_behavior()


def j2j_train(model_name, dataset_name,
              data_dir=None, output_dir=None, config_file=None, config=None):
  """Main function to train the given model on the given dataset.

  Args:
    model_name: The name of the model to train.
    dataset_name: The name of the dataset to train on.
    data_dir: Directory where the data is located.
    output_dir: Directory where to put the logs and checkpoints.
    config_file: the gin configuration file to use.
    config: string (in gin format) to override gin parameters.
  """
  gin.bind_parameter("train_fn.dataset", dataset_name)
  if FLAGS.model:
    config = []  if config is None else config
    config += ["train_fn.model=@models." + model_name]
  gin.parse_config_files_and_bindings(config_file, config)
  if output_dir:
    if not tf.gfile.Exists(output_dir):
      tf.gfile.MkDir(output_dir)
    config_path = os.path.join(output_dir, "gin.config")
    # TODO(lukaszkaiser): why is the file empty if there's no provided config?
    with tf.gfile.Open(config_path, "w") as f:
      f.write(gin.operative_config_str())
  j2j.train_fn(data_dir, output_dir=output_dir)


def main(argv):
  del argv
  logging.set_verbosity(logging.INFO)
  data_dir, output_dir = FLAGS.data_dir, FLAGS.output_dir
  data_dir = data_dir and os.path.expanduser(data_dir)
  output_dir = output_dir and os.path.expanduser(output_dir)
  j2j_train(FLAGS.model, FLAGS.dataset,
            data_dir=data_dir, output_dir=output_dir,
            config_file=FLAGS.config_file, config=FLAGS.config)


if __name__ == "__main__":
  app.run(main)
