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
"""Produce examples given a vocab, wikis, references, and dataset URLs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor.data_generators.wikisum_commoncrawl import utils as cc_utils
from tensor2tensor.data_generators.wikisum_commoncrawl import wikisum_commoncrawl

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("num_tasks", 1000, "Number of parallel tasks.")
flags.DEFINE_integer("task_id", 0, "Task id when running with borg multirun.")
flags.DEFINE_string("out_dir", None, "Directory to write to.")
flags.DEFINE_string("wikis_dir",
                    "gs://tensor2tensor-data/wikisum/wiki_content/",
                    "Directory with wiki_content.tfrecords.")
flags.DEFINE_string("refs_dir", None, "Directory with process_X dirs")
flags.DEFINE_string("urls_dir", "gs://tensor2tensor-data/wikisum/wiki_urls/",
                    "Directory with wiki_urls.json")
flags.DEFINE_string("vocab_dir", None, "Directory with vocab file")


def main(_):
  problem = wikisum_commoncrawl.WikisumCommoncrawl()

  out_filepaths = problem.out_filepaths(FLAGS.out_dir)
  out_filepaths = cc_utils.shard(out_filepaths, FLAGS.num_tasks)[FLAGS.task_id]

  shard_ids = cc_utils.shard(list(range(cc_utils.NUM_SHARDS)),
                             FLAGS.num_tasks)[FLAGS.task_id]

  wikisum_commoncrawl.produce_examples(
      shard_ids=shard_ids,
      wikis_dir=FLAGS.wikis_dir,
      refs_dir=FLAGS.refs_dir,
      urls_dir=FLAGS.urls_dir,
      vocab_path=os.path.join(FLAGS.vocab_dir, problem.vocab_filename),
      out_filepaths=out_filepaths)


if __name__ == "__main__":
  tf.app.run()
