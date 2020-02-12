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

"""Extract references from CommonCrawl files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from tensor2tensor.data_generators.wikisum import utils
from tensor2tensor.data_generators.wikisum import wikisum

import tensorflow.compat.v1 as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("num_tasks", 1000, "Number of parallel tasks.")
flags.DEFINE_integer("task_id", 0, "Task id in a parallel run.")
flags.DEFINE_string("metadata_dir",
                    "gs://tensor2tensor-data/wikisum/commoncrawl_metadata/",
                    "Path to metadata files specifying what references are in "
                    "which CommonCrawl files.")
flags.DEFINE_string("out_dir", None, "Directory to write references to.")
flags.DEFINE_string("commoncrawl_wet_dir", None,
                    "Path to CommonCrawl wet.gz files locally. If not "
                    "provided, will download.")


def main(_):
  assert FLAGS.out_dir
  assert FLAGS.metadata_dir
  out_dir = os.path.join(FLAGS.out_dir, "process_%d" % FLAGS.task_id)
  tf.gfile.MakeDirs(out_dir)

  with utils.timing("get_refs_commoncrawl"):
    # Get all WET files
    if FLAGS.commoncrawl_wet_dir:
      wet_files = tf.gfile.Glob(
          os.path.join(FLAGS.commoncrawl_wet_dir, "*.wet.gz"))
    else:
      tmp_dir = tempfile.gettempdir()
      wet_files = list(
          utils.wet_download_urls(utils.WET_PATHS_BY_DATE["0917"], tmp_dir))

    # Shard and select this task's work
    wet_files.sort()
    wet_files = utils.shard(wet_files, FLAGS.num_tasks)[FLAGS.task_id]
    tf.logging.info("Sharded out WET files. Processing %d files",
                    len(wet_files))

    wikisum.extract_references_from_wets(wet_files, FLAGS.metadata_dir, out_dir)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
