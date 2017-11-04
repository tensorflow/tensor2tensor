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
from tensor2tensor import problems as problems_lib  # pylint: disable=unused-import
from tensor2tensor.tpu import tpu_trainer_lib as lib
from tensor2tensor.utils import registry

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("tpu_num_shards", 8, "Number of tpu shards.")
flags.DEFINE_integer("iterations_per_loop", 1000,
                     "Number of iterations in a TPU training loop.")

# To maintain compatibility with some internal libs, we guard against these flag
# definitions possibly erroring. Apologies for the ugliness.
try:
  flags.DEFINE_string("master", "", "Address of TensorFlow master.")
  flags.DEFINE_string("output_dir", "", "Base output directory for run.")
  flags.DEFINE_integer("eval_steps", 200, "Number of steps in evaluation.")
except:  # pylint: disable=bare-except
  pass


def get_problem_name():
  problems = FLAGS.problems.split("-")
  assert len(problems) == 1
  return problems[0]


def create_hparams():
  hparams = registry.hparams(FLAGS.hparams_set)()
  if FLAGS.hparams:
    hparams = hparams.parse(FLAGS.hparams)
  return hparams


def create_experiment_fn():
  return lib.make_experiment_fn(FLAGS.model, get_problem_name(), FLAGS.data_dir,
                                FLAGS.train_steps, FLAGS.eval_steps,
                                FLAGS.local_eval_frequency)


def create_run_config():
  return lib.create_run_config(
      model_dir=FLAGS.output_dir,
      master=FLAGS.master,
      iterations_per_loop=FLAGS.iterations_per_loop,
      num_shards=FLAGS.tpu_num_shards,
      log_device_placement=FLAGS.log_device_placement)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.set_random_seed(123)

  exp_fn = create_experiment_fn()
  exp = exp_fn(create_run_config(), create_hparams())
  exp.continuous_train_and_eval()


if __name__ == "__main__":
  tf.app.run()
