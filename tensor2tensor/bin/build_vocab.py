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

r"""Build vocab for a subclass of Text2TextProblem.

build_vocab \
    --problem=program_search_algolisp \
    --data_dir=~/t2t_data \
    --tmp_dir=~/t2t_data/tmp
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor import problems as problems_lib  # pylint: disable=unused-import
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
import tensorflow.compat.v1 as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "/tmp/t2t/data_dir",
                    "Directory to place the generated vocabulary file in.")

flags.DEFINE_string("tmp_dir", "/tmp/t2t/tmp_dir",
                    "Temporary storage directory.")

flags.DEFINE_string("problem", None,
                    "Problem to generate the vocabulary file for.")

flags.mark_flag_as_required("problem")


def main(_):
  problem = registry.problem(FLAGS.problem)

  # We make the assumption that the problem is a subclass of Text2TextProblem.
  assert isinstance(problem, text_problems.Text2TextProblem)

  data_dir = os.path.expanduser(FLAGS.data_dir)
  tmp_dir = os.path.expanduser(FLAGS.tmp_dir)

  tf.gfile.MakeDirs(data_dir)
  tf.gfile.MakeDirs(tmp_dir)

  tf.logging.info("Saving vocabulary to data_dir: %s" % data_dir)

  problem.get_or_create_vocab(data_dir, tmp_dir)

  tf.logging.info("Saved vocabulary file: " +
                  os.path.join(data_dir, problem.vocab_filename))


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
