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

"""Computing rouge scores using pyrouge."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import shutil
from tempfile import mkdtemp

# Dependency imports

from pyrouge import Rouge155
import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("decodes_filename", None,
                       "File containing model generated summaries tokenized")
tf.flags.DEFINE_string("targets_filename", None,
                       "File containing model target summaries tokenized")


def write_to_file(filename, data):
  data = ".\n".join(data.split(". "))
  with open(filename, "w") as fp:
    fp.write(data)


def prep_data(decode_dir, target_dir):
  with open(FLAGS.decodes_filename, "rb") as fdecodes:
    with open(FLAGS.targets_filename, "rb") as ftargets:
      for i, (d, t) in enumerate(zip(fdecodes, ftargets)):
        write_to_file(os.path.join(decode_dir, "rouge.%06d.txt" % (i+1)), d)
        write_to_file(os.path.join(target_dir, "rouge.A.%06d.txt" % (i+1)), t)
        if (i+1 % 1000) == 0:
          tf.logging.into("Written %d examples to file" % i)


def main(_):
  rouge = Rouge155()
  rouge.log.setLevel(logging.ERROR)
  rouge.system_filename_pattern = "rouge.(\\d+).txt"
  rouge.model_filename_pattern = "rouge.[A-Z].#ID#.txt"

  tf.logging.set_verbosity(tf.logging.INFO)

  tmpdir = mkdtemp()
  tf.logging.info("tmpdir: %s" % tmpdir)
  # system = decodes/predictions
  system_dir = os.path.join(tmpdir, "system")
  # model = targets/gold
  model_dir = os.path.join(tmpdir, "model")
  os.mkdir(system_dir)
  os.mkdir(model_dir)

  rouge.system_dir = system_dir
  rouge.model_dir = model_dir

  prep_data(rouge.system_dir, rouge.model_dir)

  rouge_scores = rouge.convert_and_evaluate()
  rouge_scores = rouge.output_to_dict(rouge_scores)
  for prefix in ["rouge_1", "rouge_2", "rouge_l"]:
    for suffix in ["f_score", "precision", "recall"]:
      key = "_".join([prefix, suffix])
      tf.logging.info("%s: %.4f" % (key, rouge_scores[key]))

  # clean up after pyrouge
  shutil.rmtree(tmpdir)
  shutil.rmtree(rouge._config_dir)  # pylint: disable=protected-access
  shutil.rmtree(os.path.split(rouge._system_dir)[0])  # pylint: disable=protected-access


if __name__ == "__main__":
  tf.app.run()
