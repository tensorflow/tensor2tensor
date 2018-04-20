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
# pylint: disable=line-too-long
r"""Fetch all groups for a single shard id.

Because of an SSL memory leak in Python 3.5, fetching too many URLs in the same
Python process will OOM. This script wraps fetch_ref_urls and calls it through
subprocess for each group in the shard, where each group is ~5k URLs.

Launch with parallel_launch.py

Each job should finish in ~5 hours with the settings below.

GCS_BUCKET=gs://my-bucket
python parallel_launch.py \
    --num_instances=1000 \
    --cpu=4 \
    --mem=4 \
    --name=fetch-ref-urls \
    --code_dir=./ \
    --log_dir=$GCS_BUCKET/logs \
    --setup_command="pip3 install aiohttp cchardet aiodns bs4 -q --user" \
    --command_prefix="python3 wikisum_commoncrawl/fetch_ref_urls_all_groups.py --out_dir=$GCS_BUCKET/wiki_references --shard_id"
"""
# pylint: enable=line-too-long
import math
import os
import subprocess as sp

import fetch_ref_urls as fetch

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("command", "python3 wikisum_commoncrawl/fetch_ref_urls.py",
                    "Command to run fetch_ref_urls, without flags.")


def main(_):
  shard_urls = fetch.get_urls_for_shard(FLAGS.urls_dir, FLAGS.shard_id)
  num_groups = int(math.ceil(len(shard_urls) / fetch.URLS_PER_CLIENT))
  tf.logging.info("Launching fetch_ref_urls sequentially for %d groups in "
                  "shard %d. Total URLs: %d",
                  num_groups, FLAGS.shard_id, len(shard_urls))
  command_prefix = FLAGS.command.split() + [
      "--urls_dir=%s" % FLAGS.urls_dir,
      "--shard_id=%d" % FLAGS.shard_id,
      "--debug_num_urls=%d" % FLAGS.debug_num_urls,
  ]
  with fetch.timing("all_groups_fetch"):
    for i in range(num_groups):
      command = list(command_prefix)
      out_dir = os.path.join(FLAGS.out_dir, "process_%d" % i)
      command.append("--out_dir=%s" % out_dir)
      command.append("--group_id=%d" % i)
      try:
        # Even on 1 CPU, each group should finish within an hour.
        sp.check_call(command, timeout=60*60)
      except sp.TimeoutExpired:
        tf.logging.error("Group %d timed out", i)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
