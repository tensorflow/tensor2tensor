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

"""Produce examples given a vocab, wikis, references, and dataset URLs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import range
from tensor2tensor.data_generators.wikisum import utils
from tensor2tensor.data_generators.wikisum import wikisum

import tensorflow.compat.v1 as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("num_tasks", 1000, "Number of parallel tasks.")
flags.DEFINE_integer("task_id", 0, "Task id in a parallel run.")
flags.DEFINE_string("out_dir", None, "Directory to write to.")
flags.DEFINE_string("wikis_dir",
                    "gs://tensor2tensor-data/wikisum/wiki_content/",
                    "Directory with wiki_content.tfrecords.")
flags.DEFINE_string("refs_dir", None, "Directory with process_X dirs")
flags.DEFINE_string("urls_dir", "gs://tensor2tensor-data/wikisum/wiki_urls/",
                    "Directory with wiki_urls.json")
flags.DEFINE_string("vocab_dir", None, "Directory with vocab file")
flags.DEFINE_bool("for_commoncrawl", False,
                  "Whether to use WikisumCommoncrawl or WikisumWeb.")


def main(_):
  if FLAGS.for_commoncrawl:
    problem = wikisum.WikisumCommoncrawl()
  else:
    problem = wikisum.WikisumWeb()

  out_filepaths = problem.out_filepaths(FLAGS.out_dir)
  out_filepaths = utils.shard(out_filepaths, FLAGS.num_tasks)[FLAGS.task_id]

  if not FLAGS.vocab_dir:
    FLAGS.vocab_dir = FLAGS.out_dir

  shard_ids = utils.shard(list(range(utils.NUM_SHARDS)),
                          FLAGS.num_tasks)[FLAGS.task_id]

  with utils.timing("produce_examples"):
    wikisum.produce_examples(
        shard_ids=shard_ids,
        wikis_dir=FLAGS.wikis_dir,
        refs_dir=FLAGS.refs_dir,
        urls_dir=FLAGS.urls_dir,
        vocab_path=os.path.join(FLAGS.vocab_dir, problem.vocab_filename),
        out_filepaths=out_filepaths)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
