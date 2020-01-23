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

r"""Data extraction/preprocessing for processing wiki history dumps for GEC.

We use a set of heuristics to distill prose from the wikipedia xml. We produce
source-target pairs of text reflecting wikipedia edits.

WikiRevision problem - fragment of older revision -> fragment of newer revision.

This implements data extraction from wikipedia as desribed in the paper,
Weakly Supervised Grammatical Error Correction using Iterative Decoding
(https://arxiv.org/pdf/1811.01710.pdf).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random

from absl import flags
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import wiki_revision_utils
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

flags.DEFINE_integer("wiki_revision_num_train_shards", 50,
                     "Set the number of training shards to be output.")
flags.DEFINE_integer("wiki_revision_num_dev_shards", 1,
                     "Set the number of dev shards to be output.")

flags.DEFINE_string(
    "wiki_revision_data_prefix", "",
    "Specify the prefix for input data. Expects 7z compressed Wikipedia XML "
    "files, available at https://dumps.wikimedia.org/enwiki/latest/.")
flags.DEFINE_string(
    "wiki_revision_vocab_file", "",
    "Specify a wordpieces vocabulary with which to encode the text. Will "
    "generate one from data if not specified.")

flags.DEFINE_integer(
    "wiki_revision_max_examples_per_shard", 0,
    "Use this to set a cap on examples per shard. "
    "0 is no cap.")

# Data filtration heuristics:
flags.DEFINE_integer("wiki_revision_max_page_size_exp", 26,
                     "Exponent for 2**X byte cap on page size.")
flags.DEFINE_float(
    "wiki_revision_max_equal_to_diff_ratio", 0,
    "Max ratio between count of equal, diff chars for generated "
    "examples. Ratio of 1 means examples with more diff chars "
    "than equal chars will be tossed out.")
flags.DEFINE_float(
    "wiki_revision_revision_skip_factor", 1.5,
    "If >1, process only logarithmically many revisions. "
    "This avoids blowup in runtime due to many-revision pages. "
    "See wiki_revision_utils.include_revision for details.")
flags.DEFINE_float("wiki_revision_percent_identical_examples", 0.04,
                   "Percent of generated examples for which source == target.")
flags.DEFINE_bool(
    "wiki_revision_introduce_errors", True, "Add errors to the data."
    "See wiki_revision_utils.introduce_errors for details.")


@registry.register_problem
class WikiRevision(text_problems.Text2TextProblem):
  """Old segment -> revised segment.

  Data filtration heuristics:
    wiki_revision_max_page_size_exp:
      pages above this # of bytes are thrown out

    wiki_revision_revision_skip_factor:
      rate of logarithmic downsampling of revision history list

    wiki_revision_percent_identical_examples:
      how many identitcal examples to admit, as percent of total examples

    wiki_revision_introduce_errors:
      whether or not to introduce spelling-type errors on the source side

    wiki_revision_max_equal_to_diff_ratio:
      whether or not to introduce spelling-type errors on the source side


  Vocab size=32k
  Maximum input/target length = 1024 wordpiece tokens
  """
  num_identity_examples = 0
  num_total_examples = 0
  num_identity_examples = 0
  num_pages = 0
  num_revisions_total = 0
  num_revisions_admitted = 0
  num_examples_thrown_out_identity = 0
  num_examples_thrown_out_too_long = 0
  num_examples_thrown_out_edit_distance = 0
  num_examples_with_introduced_error = 0
  num_introduced_errors = 0
  num_source_tokens = 0
  num_target_tokens = 0
  corpus_files = None

  @property
  def approx_vocab_size(self):
    return 2**15  # 32K

  @property
  def strip(self):
    """Whether to strip wikipedia-stuff to get plain text."""
    return True

  @property
  def wiki_revision_skip_factor(self):
    """If this value is >1.0, process only logarithmically many revisions."""
    return FLAGS.wiki_revision_revision_skip_factor

  @property
  def max_segment_length(self):
    """Maximum number of input/target wordpiece tokens."""
    return 256

  @property
  def max_examples_per_shard(self):
    """Maximum number of examples to generate per shard.  0=unlimited."""
    return FLAGS.wiki_revision_max_examples_per_shard

  def aggregate_job_stats(self):
    # Aggregate job stats for output.
    stat = []
    # Run stats.
    stat.append("Flags for job:\n"
                "Dev shards: {}\n"
                "Train shards: {}\n"
                "Revision skip factor: {}\n"
                "Max page size: 2**{}\n"
                "Introduce errors: {}\n"
                "Max edit ratio: {}\n"
                "Percent Identical Examples: {}\n"
                "".format(FLAGS.wiki_revision_num_dev_shards,
                          FLAGS.wiki_revision_num_train_shards,
                          FLAGS.wiki_revision_revision_skip_factor,
                          FLAGS.wiki_revision_max_page_size_exp,
                          FLAGS.wiki_revision_introduce_errors,
                          FLAGS.wiki_revision_max_equal_to_diff_ratio,
                          FLAGS.wiki_revision_percent_identical_examples))

    # File stats.
    stat.append("corpus files: {}\n"
                "\tnames: {}\n"
                "\tpages per input file: {:.1f}\n"
                "".format(
                    len(self.corpus_files), self.corpus_files,
                    (0 if not self.corpus_files else
                     self.num_pages / len(self.corpus_files))))
    # Page stats.
    stat.append(
        "pages processed: {}\n"
        "\trevisions per page: {:.2f}, total: {}\n"
        "\trevisions admitted per page: {:.2f}, percent of total: {:.2f}\n"
        "".format(
            self.num_pages, (0 if not self.num_pages else
                             self.num_revisions_total / self.num_pages),
            self.num_revisions_total,
            (0 if not self.num_pages else
             self.num_revisions_admitted / self.num_pages),
            (0 if not self.num_revisions_total else
             100 * self.num_revisions_admitted / self.num_revisions_total)))
    # Revision stats.
    stat.append(
        "revisions admitted: {}\n"
        "\texamples generated per revision: {:.2f}\n"
        "".format(self.num_revisions_admitted,
                  (0 if not self.num_revisions_admitted else
                   self.num_total_examples / self.num_revisions_admitted)))
    # Example stats.
    stat.append(
        "examples generated: {}\n"
        "\twith error introduced: {}, percent of total: {:.2f}\n"
        "\ttotal errors introduced: {}, errors per errorred example: {:.2f}\n"
        "\texamples thrown out: {}\n"
        "\t\ttoo long: {}\n"
        "\t\tidentity: {}\n"
        "\t\tedit distance: {}\n"
        "\tremaining identity examples: {}\n"
        "\tratio identity (actual, desired): {:.3f}, {}\n"
        "".format(
            self.num_total_examples, self.num_examples_with_introduced_error,
            (0 if not self.num_total_examples else 100 *
             self.num_examples_with_introduced_error / self.num_total_examples),
            self.num_introduced_errors,
            (0 if not self.num_examples_with_introduced_error else
             self.num_introduced_errors /
             self.num_examples_with_introduced_error),
            self.num_examples_thrown_out_too_long +
            self.num_examples_thrown_out_identity +
            self.num_examples_thrown_out_edit_distance,
            self.num_examples_thrown_out_too_long,
            self.num_examples_thrown_out_identity,
            self.num_examples_thrown_out_edit_distance,
            self.num_identity_examples,
            (0 if not self.num_total_examples else
             self.num_identity_examples / self.num_total_examples),
            FLAGS.wiki_revision_percent_identical_examples))
    # Token stats.
    stat.append("tokens generated: {}\n"
                "\tsource: {}\n"
                "\ttarget: {}\n"
                "\tper example: {:.2f}\n"
                "\t\tsource: {:.2f}\n"
                "\t\ttarget: {:.2f}\n"
                "".format(self.num_source_tokens + self.num_target_tokens,
                          self.num_source_tokens, self.num_target_tokens,
                          (0 if not self.num_total_examples else
                           (self.num_source_tokens + self.num_target_tokens) /
                           self.num_total_examples),
                          (0 if not self.num_total_examples else
                           self.num_source_tokens / self.num_total_examples),
                          (0 if not self.num_total_examples else
                           self.num_target_tokens / self.num_total_examples)))
    return "\n".join(stat)

  def generate_data(self, data_dir, tmp_dir, task_id=-1):

    if task_id == -1 or task_id is None:
      for i in range(FLAGS.wiki_revision_num_train_shards +
                     FLAGS.wiki_revision_num_dev_shards):
        self.generate_data(data_dir, tmp_dir, i)
        return

    tf.logging.info(
        "Flags for job (task_id {}): "
        "Dev shards: {}, Train shards: {}, "
        "Revision skip factor: {}, Max page size: 2**{}, Introduce errors: {},"
        "Percent Identical Examples: {}"
        "".format(task_id, FLAGS.wiki_revision_num_dev_shards,
                  FLAGS.wiki_revision_num_train_shards,
                  FLAGS.wiki_revision_revision_skip_factor,
                  FLAGS.wiki_revision_max_page_size_exp,
                  FLAGS.wiki_revision_introduce_errors,
                  FLAGS.wiki_revision_percent_identical_examples))

    if FLAGS.wiki_revision_vocab_file:
      encoder = wiki_revision_utils.get_encoder_from_vocab(
          FLAGS.wiki_revision_vocab_file)
    else:
      encoder = wiki_revision_utils.get_or_generate_vocabulary(
          data_dir, tmp_dir, FLAGS.wiki_revision_data_prefix,
          FLAGS.wiki_revision_max_page_size_exp, self.approx_vocab_size,
          self.strip)

    random.seed(123)
    if task_id < FLAGS.wiki_revision_num_train_shards:
      out_file = self.training_filepaths(
          data_dir, FLAGS.wiki_revision_num_train_shards,
          shuffled=False)[task_id]
    else:
      out_file = self.dev_filepaths(
          data_dir, FLAGS.wiki_revision_num_dev_shards,
          shuffled=False)[task_id - FLAGS.wiki_revision_num_train_shards]

    tf.logging.info("Generating files for path: %s", out_file)
    self.corpus_files = wiki_revision_utils.corpus_files_for_shard(
        task_id, FLAGS.wiki_revision_num_train_shards,
        FLAGS.wiki_revision_num_dev_shards, FLAGS.wiki_revision_data_prefix)
    example_generator = self.generator(encoder, self.corpus_files, tmp_dir)

    packed_example_generator = self._maybe_pack_examples(example_generator)
    generator_utils.generate_files(packed_example_generator, [out_file])
    generator_utils.shuffle_dataset([out_file])

    tf.logging.info(
        "Job stats: identity examples: {}, total examples {}, ratio: {}".format(
            self.num_identity_examples, self.num_total_examples,
            (1 + self.num_identity_examples) / (1 + self.num_total_examples)))

    job_stats_string = self.aggregate_job_stats()
    out_dir, filename = out_file.replace("-unshuffled", "").rsplit("/", 1)
    stats_prefix = "/stats_"
    stats_file_path = "".join([out_dir, stats_prefix, filename])
    if tf.gfile.Exists(
        stats_file_path) and tf.gfile.Open(stats_file_path).size() != 0:
      tf.logging.info("Skipping writing stats because output file exists.")
    else:
      with tf.gfile.Open(stats_file_path, "w") as out:
        tf.logging.info("Writing job stats to {}".format(stats_file_path))
        out.write(job_stats_string)

    tf.logging.info(job_stats_string)

  def generator(self, encoder, corpus_files, tmp_dir):
    for page in wiki_revision_utils.corpus_page_generator(
        corpus_files, tmp_dir, FLAGS.wiki_revision_max_page_size_exp):
      self.num_pages += 1
      examples = self.page_to_examples(page, encoder)
      for x in examples:
        yield x
      if self.num_total_examples % 100000 == 0:
        tf.logging.info(
            u"page count={} num_total_examples={} id={} title={}".format(
                self.num_pages, self.num_total_examples, page["id"],
                page["title"]))
      if (self.max_examples_per_shard and
          self.num_total_examples >= self.max_examples_per_shard):
        tf.logging.info(
            "Examples per shard {} >= max_examples_per_shard {}. Shutting down."
            .format(self.num_total_examples, self.max_examples_per_shard))
        break
    tf.logging.info(
        "Total pages: {}, total examples: {}, examples per page: {}".format(
            self.num_pages, self.num_total_examples, 0 if not self.num_pages
            else self.num_total_examples / self.num_pages))

  def page_to_examples(self, page, encoder):
    revisions = page["revisions"]
    self.num_revisions_total += len(revisions)
    if len(revisions) < 2:
      return []
    revisions = [
        wiki_revision_utils.get_text(r)
        for n, r in enumerate(revisions)
        if wiki_revision_utils.include_revision(
            n, self.wiki_revision_skip_factor) or n + 1 == len(revisions)
    ]
    self.num_revisions_admitted += len(revisions)

    ret = []
    for i in range(len(revisions) - 1):
      old_revision = revisions[i]
      new_revision = revisions[i + 1]

      if FLAGS.wiki_revision_introduce_errors:
        old_revision_text, num_added_err = wiki_revision_utils.introduce_errors(
            revisions[i])
        if num_added_err:
          self.num_introduced_errors += num_added_err
          self.num_examples_with_introduced_error += 1
      else:
        old_revision_text = revisions[i]
      new_revision_text = revisions[i + 1]
      if encoder:
        # Encode text into list of ids, if a text encoder is present.
        old_revision = encoder.encode(old_revision_text)
        new_revision = encoder.encode(new_revision_text)
      else:
        # Retain text (as list of characters), if a text encoder is not present.
        old_revision = old_revision_text
        new_revision = new_revision_text
      ret.extend(
          self.make_examples(
              encoder,
              old_revision,
              new_revision,
              max_length=self.max_segment_length,
              percent_identical_examples=FLAGS
              .wiki_revision_percent_identical_examples))
    return ret

  def make_examples(self,
                    encoder,
                    old_snapshot,
                    new_snapshot,
                    max_length=1024,
                    percent_identical_examples=0.01,
                    max_length_distance=0):
    """Produce training examples based on a pair of snapshots.

    Aligns the snapshots, then chops at a random subset of the alignment points
    to create (old snippet -> new snippet) examples.

    Most negative examples (those with no changes) are discarded, but we
    keep some of them, maintaining a proportion in the final data
    determined by percent_identical_examples.

    Args:
      encoder: the subword text encoder
      old_snapshot: a list of ids
      new_snapshot: a list of ids
      max_length: an integer.  Maximum length of "inputs" and "targets".
      percent_identical_examples: a float
      max_length_distance: an integer. Max token edit dist for admitted examples

    Returns:
      a list of feature dictionaries.  The dictionaries have
      "inputs" and "targets" populated. text_encoder.EOS is appended to both.
    """
    ret = []
    eos_sequence = [text_encoder.EOS_ID]
    # Pick a per-token cut probability with a log-uniform distribution between
    # 1/4 and 1/(max_length / 2)
    bound1 = -math.log(4.0)
    bound2 = -math.log(max_length / 2.0)
    cut_prob = math.exp(random.random() * (bound2 - bound1) + bound1)
    opcodes = wiki_revision_utils.fast_match_sequences(old_snapshot,
                                                       new_snapshot)
    cut_points = [(0, 0)]
    for tag, i1, i2, j1, j2 in opcodes:
      if tag == "equal":
        for i in range(i1, i2 + 1):
          if random.random() < cut_prob:
            cut_points.append((i, i + j1 - i1))
    cut_points.append((len(old_snapshot), len(new_snapshot)))
    src_tgt_pairs = []
    for cut_number in range(len(cut_points) - 1):
      i1, j1 = cut_points[cut_number]
      i2, j2 = cut_points[cut_number + 1]
      old_segment = old_snapshot[i1:i2]
      new_segment = new_snapshot[j1:j2]
      src_tgt_pairs.append((old_segment, new_segment))

    src_tgt_pairs, thrown_edit_count = wiki_revision_utils.edit_distance_filter(
        wiki_revision_utils.throw_empty_pairs(src_tgt_pairs),
        FLAGS.wiki_revision_max_equal_to_diff_ratio)

    self.num_examples_thrown_out_edit_distance += thrown_edit_count

    for source, target in src_tgt_pairs:
      # Add EOS segment.
      old_segment = source + eos_sequence
      new_segment = target + eos_sequence
      if len(old_segment) <= max_length and len(new_segment) <= max_length:
        if max_length_distance and (abs(len(old_segment) - len(new_segment)) >
                                    max_length_distance):
          self.num_examples_thrown_out_edit_distance += 1
          continue
        if old_segment == new_segment:
          # If current proportion of identity is below target
          # percent_identical_examples, then roll for a 50% chance to add an
          # identitical example. Random roll preserves nondeterminism.
          # percent_identical_examples, then add identitical example.
          # Random roll preserves nondeterminism in selecting identity examples.
          if (((self.num_identity_examples) / (1 + self.num_total_examples)) >
              percent_identical_examples) or random.random() > 0.5:
            self.num_examples_thrown_out_identity += 1
            continue
          else:
            self.num_identity_examples += 1
        self.num_total_examples += 1
        self.num_source_tokens += len(old_segment) - 1
        self.num_target_tokens += len(new_segment) - 1
        ret.append({"inputs": old_segment, "targets": new_segment})
      else:
        self.num_examples_thrown_out_too_long += 1

    return ret

  def eval_metrics(self):
    return [
        metrics.Metrics.ACC,
        metrics.Metrics.ACC_TOP5,
        metrics.Metrics.ACC_PER_SEQ,
        metrics.Metrics.NEG_LOG_PERPLEXITY,
    ]

  @property
  def invert_prob(self):
    """Ratio of e^2 positive forward to backward examples."""
    return 1.0 / (1.0 + math.exp(2.0))


@registry.register_problem
class WikiRevisionPacked1k(WikiRevision):
  """Packed version for TPU."""

  @property
  def packed_length(self):
    return 1024


@registry.register_problem
class WikiRevisionPacked256(WikiRevision):
  """Packed version for TPU."""

  @property
  def packed_length(self):
    return 256

  @property
  def max_segment_length(self):
    return 256
