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

from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_utils
from tensor2tensor.utils import usr_dir

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("t2t_usr_dir", "",
                    "Path to a Python module that will be imported. The "
                    "__init__.py file should include the necessary imports. "
                    "The imported files should contain registrations, "
                    "e.g. @registry.register_model calls, that will then be "
                    "available to the t2t-trainer.")
flags.DEFINE_string("tmp_dir", "/tmp/t2t_datagen",
                    "Temporary storage directory.")
flags.DEFINE_bool("generate_data", False, "Generate data before training?")


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
  trainer_utils.log_registry()
  trainer_utils.validate_flags()
  tf.gfile.MakeDirs(FLAGS.output_dir)

  # Generate data if requested.
  if FLAGS.generate_data:
    tf.gfile.MakeDirs(FLAGS.data_dir)
    tf.gfile.MakeDirs(FLAGS.tmp_dir)
    for problem_name in FLAGS.problems.split("-"):
      tf.logging.info("Generating data for %s" % problem_name)
      problem = registry.problem(problem_name)
      problem.generate_data(FLAGS.data_dir, FLAGS.tmp_dir)

  # Run the trainer.
  trainer_utils.run(
      data_dir=FLAGS.data_dir,
      model=FLAGS.model,
      output_dir=FLAGS.output_dir,
      train_steps=FLAGS.train_steps,
      eval_steps=FLAGS.eval_steps,
      schedule=FLAGS.schedule)


if __name__ == "__main__":
  tf.app.run()
