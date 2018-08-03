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
"""Generate vocab from references and wikis."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators.wikisum import wikisum

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("out_dir", None, "Directory to write vocab to.")
flags.DEFINE_string("wikis_dir",
                    "gs://tensor2tensor-data/wikisum/wiki_content/",
                    "Directory with wiki_content.tfrecords shards.")
flags.DEFINE_string("refs_dir", None,
                    "Directory with process_X folders with reference shards.")
flags.DEFINE_bool("for_commoncrawl", False,
                  "Whether to use WikisumCommoncrawl or WikisumWeb.")


def main(_):
  if FLAGS.for_commoncrawl:
    problem = wikisum.WikisumCommoncrawl()
  else:
    problem = wikisum.WikisumWeb()
  problem.generate_vocab(FLAGS.out_dir, FLAGS.wikis_dir, FLAGS.refs_dir)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
