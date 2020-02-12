# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""Script to continuously average last N checkpoints in a given directory."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
import os
import shutil
import numpy as np
import six
from six.moves import zip  # pylint: disable=redefined-builtin
from tensor2tensor.utils import bleu_hook
import tensorflow.compat.v1 as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "",
                    "Directory to load model checkpoints from.")
flags.DEFINE_string("output_dir", "avg/",
                    "Directory to output the averaged checkpoints to.")
flags.DEFINE_integer("n", 8, "How many checkpoints should be averaged?")
flags.DEFINE_integer("min_steps", 0, "Ignore checkpoints with less steps.")
flags.DEFINE_integer("wait_minutes", 0,
                     "Wait upto N minutes for a new checkpoint.")


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  model_dir = os.path.expanduser(FLAGS.model_dir)
  output_dir = os.path.expanduser(FLAGS.output_dir)
  out_base_file = os.path.join(output_dir, "model.ckpt")

  # Copy flags.txt with the original time, so t2t-bleu can report correct
  # relative time.
  tf.gfile.MakeDirs(FLAGS.output_dir)
  if (not os.path.exists(os.path.join(output_dir, "flags.txt")) and
      os.path.exists(os.path.join(model_dir, "flags.txt"))):
    shutil.copy2(os.path.join(model_dir, "flags.txt"),
                 os.path.join(output_dir, "flags.txt"))

  models_processed = 0
  queue = deque()
  for model in bleu_hook.stepfiles_iterator(model_dir, FLAGS.wait_minutes,
                                            FLAGS.min_steps):
    if models_processed == 0:
      var_list = tf.train.list_variables(model.filename)
      avg_values = {}
      for (name, shape) in var_list:
        if not (name.startswith("global_step") or
                name.startswith("train_stats/")):
          avg_values[name] = np.zeros(shape)
    models_processed += 1

    tf.logging.info("Loading [%d]: %s" % (models_processed, model.filename))
    reader = tf.train.load_checkpoint(model.filename)
    for name in avg_values:
      avg_values[name] += reader.get_tensor(name) / FLAGS.n
    queue.append(model)
    if len(queue) < FLAGS.n:
      continue

    out_file = "%s-%d" % (out_base_file, model.steps)
    tf_vars = []
    tf.logging.info("Averaging %s" % (out_file))
    for (name, value) in six.iteritems(avg_values):
      # TODO(martinpopel): dtype=var_dtypes[name]
      tf_vars.append(tf.get_variable(name, shape=value.shape))
    placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]

    global_step = tf.get_variable(
        "global_step",
        initializer=tf.constant(model.steps, dtype=tf.int64),
        trainable=False)
    with tf.variable_scope("train_stats"):
      tf.get_variable("problem_0_steps", initializer=0, trainable=False)
    saver = tf.train.Saver(tf.global_variables())

    tf.logging.info("Running session for %s" % (out_file))
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for p, assign_op, (name, value) in zip(
          placeholders, assign_ops, six.iteritems(avg_values)):
        sess.run(assign_op, {p: value})
      tf.logging.info("Storing to %s" % out_file)
      saver.save(sess, out_base_file, global_step=global_step)
    os.utime(out_file + ".index", (model.mtime, model.mtime))

    tf.reset_default_graph()
    first_model = queue.popleft()

    reader = tf.train.load_checkpoint(first_model.filename)
    for name in avg_values:
      avg_values[name] -= reader.get_tensor(name) / FLAGS.n

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
