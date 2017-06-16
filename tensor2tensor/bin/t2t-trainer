#!/usr/bin/env python
# Copyright 2017 Google Inc.
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

r"""Trainer for T2T models.

This binary perform training, evaluation, and inference using
the Estimator API with tf.learn Experiment objects.

To train your model, for example:
  t2t-trainer \
      --data_dir ~/data \
      --problems=algorithmic_identity_binary40 \
      --model=transformer
      --hparams_set=transformer_base
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.utils import trainer_utils as utils

import tensorflow as tf

FLAGS = tf.flags.FLAGS


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  utils.log_registry()
  utils.validate_flags()
  utils.run(
      data_dir=FLAGS.data_dir,
      model=FLAGS.model,
      output_dir=FLAGS.output_dir,
      train_steps=FLAGS.train_steps,
      eval_steps=FLAGS.eval_steps,
      schedule=FLAGS.schedule)


if __name__ == "__main__":
  tf.app.run()
