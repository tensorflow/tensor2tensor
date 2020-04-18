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

"""Script to average values of variables in a list of checkpoint files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import six
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow.compat.v1 as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoints", "",
                    "Comma-separated list of checkpoints to average.")
flags.DEFINE_integer("num_last_checkpoints", 0,
                     "Averages the last N saved checkpoints."
                     " If the checkpoints flag is set, this is ignored.")
flags.DEFINE_string("prefix", "",
                    "Prefix (e.g., directory) to append to each checkpoint.")
flags.DEFINE_string("output_path", "/tmp/averaged.ckpt",
                    "Path to output the averaged checkpoint to.")


def checkpoint_exists(path):
  return (tf.gfile.Exists(path) or tf.gfile.Exists(path + ".meta") or
          tf.gfile.Exists(path + ".index"))


def main(_):
  if FLAGS.checkpoints:
    # Get the checkpoints list from flags and run some basic checks.
    checkpoints = [c.strip() for c in FLAGS.checkpoints.split(",")]
    checkpoints = [c for c in checkpoints if c]
    if not checkpoints:
      raise ValueError("No checkpoints provided for averaging.")
    if FLAGS.prefix:
      checkpoints = [FLAGS.prefix + c for c in checkpoints]
  else:
    assert FLAGS.num_last_checkpoints >= 1, "Must average at least one model"
    assert FLAGS.prefix, ("Prefix must be provided when averaging last"
                          " N checkpoints")
    checkpoint_state = tf.train.get_checkpoint_state(
        os.path.dirname(FLAGS.prefix))
    # Checkpoints are ordered from oldest to newest.
    checkpoints = checkpoint_state.all_model_checkpoint_paths[
        -FLAGS.num_last_checkpoints:]

  checkpoints = [c for c in checkpoints if checkpoint_exists(c)]
  if not checkpoints:
    if FLAGS.checkpoints:
      raise ValueError(
          "None of the provided checkpoints exist. %s" % FLAGS.checkpoints)
    else:
      raise ValueError("Could not find checkpoints at %s" %
                       os.path.dirname(FLAGS.prefix))

  # Read variables from all checkpoints and average them.
  tf.logging.info("Reading variables and averaging checkpoints:")
  for c in checkpoints:
    tf.logging.info("%s ", c)
  var_list = tf.train.list_variables(checkpoints[0])
  var_values, var_dtypes = {}, {}
  for (name, shape) in var_list:
    if not name.startswith("global_step"):
      var_values[name] = np.zeros(shape)
  for checkpoint in checkpoints:
    reader = tf.train.load_checkpoint(checkpoint)
    for name in var_values:
      tensor = reader.get_tensor(name)
      var_dtypes[name] = tensor.dtype
      var_values[name] += tensor
    tf.logging.info("Read from checkpoint %s", checkpoint)
  for name in var_values:  # Average.
    var_values[name] /= len(checkpoints)

  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    tf_vars = [
        tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v])
        for v in var_values
    ]
  placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
  assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
  global_step = tf.Variable(
      0, name="global_step", trainable=False, dtype=tf.int64)
  saver = tf.train.Saver(tf.all_variables())

  # Build a model consisting only of variables, set them to the average values.
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                           six.iteritems(var_values)):
      sess.run(assign_op, {p: value})
    # Use the built saver to save the averaged checkpoint.
    saver.save(sess, FLAGS.output_path, global_step=global_step)

  tf.logging.info("Averaged checkpoints saved in %s", FLAGS.output_path)


if __name__ == "__main__":
  tf.app.run()
