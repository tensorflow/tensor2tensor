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

"""Aggregate stats from produce_examples."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import numpy as np

import six
from six.moves import zip
from tensor2tensor.data_generators.wikisum import wikisum

import tensorflow.compat.v1 as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("out_dir", None, "Directory with data and stats files.")
flags.DEFINE_bool("for_commoncrawl", False,
                  "Whether to use WikisumCommoncrawl or WikisumWeb.")
flags.DEFINE_bool("rm_per_shard_stats", True,
                  "Whether to remove the per-shard stats files after writing "
                  "out the aggregated stats.")


def aggregate_stats(stats_files):
  """Aggregate stats in per-shard stats files."""
  all_stats = {}
  for fname in stats_files:
    with tf.gfile.Open(fname) as f:
      stats = json.loads(f.read())
      for k, v in six.iteritems(stats):
        if k not in all_stats:
          if isinstance(v, list):
            all_stats[k] = []
          else:
            all_stats[k] = 0

        if isinstance(v, list):
          all_stats[k].extend(v)
        else:
          all_stats[k] += v

  stats = all_stats
  ref_coverage = float(stats["total_found_refs"]) / stats["total_original_refs"]
  len_bounds = [0, 2, 10, 100, 1000, 5000, 10000, 20000, 50000, 100000, 1000000]
  len_counts, len_bounds = np.histogram(stats["ref_lengths"], len_bounds)
  len_dist = len_counts.astype(np.float32) / len_counts.sum()
  wiki_coverage = (float(stats["num_wikis_written"]) /
                   stats["total_original_wikis"])
  wikis_skipped_no_ref = (float(stats["wikis_skipped_no_refs"]) /
                          stats["total_original_wikis"])
  wikis_skipped_no_lead = (float(stats["wikis_skipped_short_lead"]) /
                           stats["total_original_wikis"])
  wiki_ref_coverage = [
      float(found) / orig for found, orig
      in zip(stats["wiki_found_refs"], stats["wiki_original_refs"]) if found
  ]
  coverage_bounds = np.arange(21).astype(np.float32) / 20
  coverage_counts, coverage_bounds = np.histogram(wiki_ref_coverage,
                                                  coverage_bounds)
  coverage_dist = coverage_counts.astype(np.float32) / coverage_counts.sum()

  agg_stats = dict(
      total_original_wikis=stats["total_original_wikis"],
      total_original_refs=stats["total_original_refs"],
      wiki_coverage=wiki_coverage,
      wikis_skipped_no_ref=wikis_skipped_no_ref,
      wikis_skipped_no_lead=wikis_skipped_no_lead,
      overall_ref_coverage=ref_coverage,
      per_wiki_ref_coverage_dist=list((coverage_dist * 100).astype(int)),
      per_wiki_ref_coverage_bounds=list((coverage_bounds * 100).astype(int)),
      ref_len_dist=list((len_dist * 100).astype(int)),
      ref_len_bounds=list(len_bounds),
  )
  return agg_stats


def filename_to_task_id(fname):
  """Map filename to the task id that created it assuming 1k tasks."""
  # This matches the order and size in WikisumBase.out_filepaths
  fname = os.path.basename(fname)
  shard_id_increment = {
      "train": 0,
      "dev": 800,
      "test": 900,
  }
  parts = fname.split("-")
  split = parts[1]
  shard_id = parts[2]
  task_id = int(shard_id) + shard_id_increment[split]
  return task_id


def get_length(fname):
  return tf.gfile.Stat(fname).length


def validate_data_files(problem, data_files, min_size):
  """Validate presence and minimum size of files."""
  # Check that all files are present
  data_dir = os.path.split(data_files[0])[0]
  out_filepaths = problem.out_filepaths(data_dir)
  missing_filepaths = set(out_filepaths) - set(data_files)
  if missing_filepaths:
    tf.logging.error("Missing %d data files", len(missing_filepaths))

  # Check that each file is at least 100M
  too_small = []
  for data_file in data_files:
    length = get_length(data_file)
    if length < min_size:
      too_small.append(data_file)
  if too_small:
    tf.logging.error("%d files too small", len(too_small))

  bad_files = too_small + list(missing_filepaths)
  return bad_files


def main(_):
  if FLAGS.for_commoncrawl:
    problem = wikisum.WikisumCommoncrawl()
  else:
    problem = wikisum.WikisumWeb()
  prefix = problem.dataset_filename()
  data_files = tf.gfile.Glob(os.path.join(FLAGS.out_dir, "%s*" % prefix))
  missing_files = validate_data_files(
      problem, data_files,
      min_size=(60 if FLAGS.for_commoncrawl else 120) * 1e6)

  task_ids = [filename_to_task_id(fname) for fname in missing_files]
  ids_for_flag = ",".join([str(i) for i in task_ids])
  tf.logging.error("You should (re)generate %d of the data files. "
                   "Rerun produce_examples with --instance_ids='%s'.",
                   len(missing_files), ids_for_flag)

  # Compute and write out aggregated stats
  stats_files = tf.gfile.Glob(os.path.join(FLAGS.out_dir, "stats*"))
  agg_stats = aggregate_stats(stats_files)
  if not FLAGS.for_commoncrawl:
    coverage = agg_stats["overall_ref_coverage"] * 100
    if not coverage > 80:
      tf.logging.error("Overall reference coverage is expected to be > 80%. "
                       "It is %0.1f. You may want to rerun get_references_web.",
                       coverage)
  with tf.gfile.Open(
      os.path.join(FLAGS.out_dir, "stats.json"), "w") as f:
    f.write(json.dumps(agg_stats))
  if FLAGS.rm_per_shard_stats and not missing_files:
    for fname in stats_files:
      tf.gfile.Remove(fname)


if __name__ == "__main__":
  tf.app.run()
