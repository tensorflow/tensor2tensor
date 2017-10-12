# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

r"""Train on TPU.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor import models  # pylint: disable=unused-import
from tensor2tensor.data_generators import all_problems  # pylint: disable=unused-import
from tensor2tensor.tpu import tpu_trainer_lib as lib
from tensor2tensor.utils import trainer_utils

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("tpu_num_shards", 8, "Number of tpu shards.")
flags.DEFINE_string("output_dir", "", "Base output directory for run.")
flags.DEFINE_string("master", "", "Address of TensorFlow master.")
flags.DEFINE_integer("eval_steps", 10, "Number of steps in evaluation.")
flags.DEFINE_integer("iterations_per_loop", 1000,
                     "Number of iterations in a TPU training loop.")


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.set_random_seed(123)

  assert len(FLAGS.problems.split("-")) == 1

  hparams = trainer_utils.create_hparams(
      FLAGS.hparams_set, FLAGS.data_dir, passed_hparams=FLAGS.hparams)
  trainer_utils.add_problem_hparams(hparams, FLAGS.problems)

  problem = hparams.problem_instances[0]

  model_fn = lib.get_model_fn(FLAGS.model, hparams)
  input_fn = lib.get_input_fn(FLAGS.data_dir, problem, hparams)

  estimator = lib.make_estimator(
      model_fn=model_fn,
      output_dir=FLAGS.output_dir,
      master=FLAGS.master,
      num_shards=FLAGS.tpu_num_shards,
      batch_size=hparams.tpu_batch_size_per_shard * FLAGS.tpu_num_shards,
      log_device_placement=FLAGS.log_device_placement,
      iterations_per_loop=FLAGS.iterations_per_loop)
  if FLAGS.train_steps:
    estimator.train(
        lambda params: input_fn(tf.estimator.ModeKeys.TRAIN, params),
        steps=FLAGS.train_steps)
  if FLAGS.eval_steps:
    estimator.evaluate(
        lambda params: input_fn(tf.estimator.ModeKeys.EVAL, params),
        steps=FLAGS.eval_steps)


if __name__ == "__main__":
  tf.app.run()
