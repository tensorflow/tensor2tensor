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

r"""T2T trainer for TF 2.0.

This trainer only supports a subset of models and features for now.

Examples:

- train a basic model on mnist:
    v2/t2t_trainer.py --dataset=mnist --model=basic_fc_relu
      --config="train_fn.train_steps=4000"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
from tensor2tensor.v2 import t2t
import tensorflow as tf

tf.enable_v2_behavior()

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


def main(argv):
  del argv
  data_dir, output_dir = FLAGS.data_dir, FLAGS.output_dir
  if data_dir is not None:
    data_dir = os.path.expanduser(data_dir)
  if output_dir is not None:
    output_dir = os.path.expanduser(output_dir)
  t2t.t2t_train(FLAGS.model, FLAGS.dataset,
                data_dir=data_dir, output_dir=output_dir,
                config_file=FLAGS.config_file, config=FLAGS.config)


if __name__ == "__main__":
  app.run(main)
