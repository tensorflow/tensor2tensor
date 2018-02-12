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

"""Export a trained model for serving."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir

import tensorflow as tf

FLAGS = tf.flags.FLAGS


def create_estimator(run_config, hparams):
  return trainer_lib.create_estimator(
      FLAGS.model,
      hparams,
      run_config,
      decode_hparams=decoding.decode_hparams(FLAGS.decode_hparams))


def create_hparams():
  return trainer_lib.create_hparams(
      FLAGS.hparams_set,
      FLAGS.hparams,
      data_dir=os.path.expanduser(FLAGS.data_dir),
      problem_name=FLAGS.problems)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  trainer_lib.set_random_seed(FLAGS.random_seed)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)

  ckpt_dir = os.path.expanduser(FLAGS.output_dir)

  hparams = create_hparams()
  hparams.no_data_parallelism = True  # To clear the devices
  run_config = t2t_trainer.create_run_config(hparams)

  estimator = create_estimator(run_config, hparams)

  problem = hparams.problem_instances[0]
  strategy = trainer_lib.create_export_strategy(problem, hparams)

  export_dir = os.path.join(ckpt_dir, "export", strategy.name)
  strategy.export(
      estimator,
      export_dir,
      checkpoint_path=tf.train.latest_checkpoint(ckpt_dir))


if __name__ == "__main__":
  tf.app.run()
