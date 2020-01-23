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

"""Data generators for the CNN and Daily Mail datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import os
import random
import tarfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import wiki_lm
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf

# Links to data from http://cs.nyu.edu/~kcho/DMQA/
_CNN_STORIES_DRIVE_URL = ("https://drive.google.com/uc?"
                          "export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ")

_DAILYMAIL_STORIES_DRIVE_URL = ("https://drive.google.com/uc?export=download&id"
                                "=0BwmD_VLjROrfM1BxdkxVaTY2bWs")

# Note: using See et al. (2017) as reference for data generation
# For more info, use the links below

# Train/Dev/Test Splits for summarization data
_TRAIN_URLS = ("https://raw.githubusercontent.com/abisee/cnn-dailymail/"
               "master/url_lists/all_train.txt")
_DEV_URLS = ("https://raw.githubusercontent.com/abisee/cnn-dailymail/"
             "master/url_lists/all_val.txt")
_TEST_URLS = ("https://raw.githubusercontent.com/abisee/cnn-dailymail/"
              "master/url_lists/all_test.txt")

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

# Techniques for data prep from See et al. (2017)
dm_single_close_quote = u"\u2019"  # unicode
dm_double_close_quote = u"\u201d"
# Acceptable ways to end a sentence.
END_TOKENS = [
    u".", u"!", u"?", u"...", u"'", u"`", u"\"", dm_single_close_quote,
    dm_double_close_quote, u")"
]


def _maybe_download_corpora(tmp_dir, dataset_split):
  """Download corpora if necessary and unzip them.

  Args:
    tmp_dir: directory containing dataset.
    dataset_split: whether we're in train/dev/test mode.

  Returns:
    List of all files generated and path to file containing
      train/dev/test split info.
  """
  cnn_filename = "cnn_stories.tgz"
  cnn_finalpath = os.path.join(tmp_dir, "cnn/stories/")
  dailymail_filename = "dailymail_stories.tgz"
  dailymail_finalpath = os.path.join(tmp_dir, "dailymail/stories/")
  if not tf.gfile.Exists(cnn_finalpath):
    cnn_file = generator_utils.maybe_download_from_drive(
        tmp_dir, cnn_filename, _CNN_STORIES_DRIVE_URL)
    with tarfile.open(cnn_file, "r:gz") as cnn_tar:
      cnn_tar.extractall(tmp_dir)
  if not tf.gfile.Exists(dailymail_finalpath):
    dailymail_file = generator_utils.maybe_download_from_drive(
        tmp_dir, dailymail_filename, _DAILYMAIL_STORIES_DRIVE_URL)
    with tarfile.open(dailymail_file, "r:gz") as dailymail_tar:
      dailymail_tar.extractall(tmp_dir)

  cnn_files = tf.gfile.Glob(cnn_finalpath + "*")
  dailymail_files = tf.gfile.Glob(dailymail_finalpath + "*")
  all_files = cnn_files + dailymail_files

  if dataset_split == problem.DatasetSplit.TRAIN:
    urls_path = generator_utils.maybe_download(tmp_dir, "all_train.txt",
                                               _TRAIN_URLS)
  elif dataset_split == problem.DatasetSplit.EVAL:
    urls_path = generator_utils.maybe_download(tmp_dir, "all_val.txt",
                                               _DEV_URLS)
  else:
    urls_path = generator_utils.maybe_download(tmp_dir, "all_test.txt",
                                               _TEST_URLS)

  return all_files, urls_path


def example_splits(url_file, all_files):
  """Generate splits of the data."""

  def generate_hash(inp):
    """Generate a sha1 hash to match the raw url to the filename extracted."""
    h = hashlib.sha1()
    h.update(inp)
    return h.hexdigest()

  all_files_map = {f.split("/")[-1]: f for f in all_files}

  urls = [line.strip().encode("utf-8") for line in tf.gfile.Open(url_file)]

  filelist = []
  for url in urls:
    url_hash = generate_hash(url)
    filename = url_hash + ".story"
    if filename not in all_files_map:
      tf.logging.info("Missing file: %s" % url)
      continue
    filelist.append(all_files_map[filename])

  tf.logging.info("Found %d examples" % len(filelist))

  return filelist


def example_generator(all_files, urls_path, sum_token):
  """Generate examples."""

  def fix_run_on_sents(line):
    if u"@highlight" in line:
      return line
    if not line:
      return line
    if line[-1] in END_TOKENS:
      return line
    return line + u"."

  filelist = example_splits(urls_path, all_files)
  story_summary_split_token = u" <summary> " if sum_token else " "

  for story_file in filelist:
    story = []
    summary = []
    reading_highlights = False
    for line in tf.gfile.Open(story_file, "rb"):
      line = text_encoder.to_unicode_utf8(line.strip())
      line = fix_run_on_sents(line)
      if not line:
        continue
      elif line.startswith(u"@highlight"):
        if not story:
          break  # No article text.
        reading_highlights = True
      elif reading_highlights:
        summary.append(line)
      else:
        story.append(line)

    if (not story) or not summary:
      continue

    yield " ".join(story) + story_summary_split_token + " ".join(summary)


def _story_summary_split(story):
  split_str = u" <summary> "
  split_str_len = len(split_str)
  split_pos = story.find(split_str)
  return story[:split_pos], story[split_pos + split_str_len:]  # story, summary


def write_raw_text_to_files(all_files, urls_path, dataset_split, tmp_dir):
  """Write text to files."""

  def write_to_file(all_files, urls_path, tmp_dir, filename):
    """Write text to files."""
    with io.open(
        os.path.join(tmp_dir, filename + ".source"), "w",
        encoding="utf-8") as fstory:
      with io.open(
          os.path.join(tmp_dir, filename + ".target"), "w",
          encoding="utf-8") as fsummary:
        for example in example_generator(all_files, urls_path, sum_token=True):
          story, summary = _story_summary_split(example)
          fstory.write(story + "\n")
          fsummary.write(summary + "\n")

  if dataset_split == problem.DatasetSplit.TRAIN:
    filename = "cnndm.train"
  elif dataset_split == problem.DatasetSplit.EVAL:
    filename = "cnndm.dev"
  else:
    filename = "cnndm.test"

  tf.logging.info("Writing %s" % filename)
  write_to_file(all_files, urls_path, tmp_dir, filename)


@registry.register_problem
class SummarizeCnnDailymail32k(text_problems.Text2TextProblem):
  """Summarize CNN and Daily Mail articles to their summary highlights."""

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    del data_dir
    all_files, urls_path = _maybe_download_corpora(tmp_dir,
                                                   problem.DatasetSplit.TRAIN)
    return example_generator(all_files, urls_path, sum_token=False)

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 100,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 10,
    }, {
        "split": problem.DatasetSplit.TEST,
        "shards": 10,
    }]

  def is_generate_per_split(self):
    return True

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del data_dir
    all_files, urls_path = _maybe_download_corpora(tmp_dir, dataset_split)
    write_raw_text_to_files(all_files, urls_path, dataset_split, tmp_dir)
    for example in example_generator(all_files, urls_path, sum_token=True):
      story, summary = _story_summary_split(example)
      yield {"inputs": story, "targets": summary}


@registry.register_problem
class SummarizeCnnDailymailWikiLMSharedVocab(SummarizeCnnDailymail32k):
  """Summarize CNN and Daily Mail articles using the Wiki 32k vocab."""

  @property
  def use_vocab_from_other_problem(self):
    return wiki_lm.LanguagemodelEnWiki32k()


@registry.register_problem
class SummarizeCnnDailymailWikiLMSharedVocab64k(SummarizeCnnDailymail32k):
  """Summarize CNN and Daily Mail articles using the Wiki 64k vocab."""

  @property
  def use_vocab_from_other_problem(self):
    return wiki_lm.LanguagemodelEnWiki64k()


@registry.register_problem
class SummarizeCnnDailymailWikiLMMultiVocab64k(SummarizeCnnDailymail32k):
  """Summarize CNN and Daily Mail articles using multi-lingual 64k vocab."""

  @property
  def use_vocab_from_other_problem(self):
    return wiki_lm.LanguagemodelDeEnFrRoWiki64k()


@registry.register_problem
class SummarizeCnnDailymailMulti64kPacked1k(SummarizeCnnDailymail32k):
  """Summarize CNN and Daily Mail articles using multi-lingual 64k vocab."""

  @property
  def use_vocab_from_other_problem(self):
    return wiki_lm.LanguagemodelDeEnFrRoWiki64k()

  @property
  def packed_length(self):
    return 1024

  @property
  def num_training_examples(self):
    return 252600

  @property
  def inputs_prefix(self):
    return "CNN Daily Mail article to summary "

  @property
  def targets_prefix(self):
    return "CNN Daily Mail summary to article "


@registry.register_problem
class SummarizeFracCnnDailymailWikiLMSharedVocab64k(SummarizeCnnDailymail32k):
  """Summarize a fraction of CNN/DM articles using the Wiki 64k vocab."""

  @property
  def use_vocab_from_other_problem(self):
    return wiki_lm.LanguagemodelEnWiki64k()

  def fraction_of_data(self):
    return 1.

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del data_dir
    all_data = []
    all_files, urls_path = _maybe_download_corpora(tmp_dir, dataset_split)
    write_raw_text_to_files(all_files, urls_path, dataset_split, tmp_dir)
    for example in example_generator(all_files, urls_path, sum_token=True):
      story, summary = _story_summary_split(example)
      all_data.append((story, summary))

    if dataset_split == problem.DatasetSplit.TRAIN:
      random.shuffle(all_data)
      fractional_len = int(self.fraction_of_data() * len(all_data))
      all_data = all_data[:fractional_len]

    for story, summary in all_data:
      yield {"inputs": story, "targets": summary}


@registry.register_problem
class SummarizeFrac0p1CnnDailymailWikiLMSharedVocab64k(
    SummarizeFracCnnDailymailWikiLMSharedVocab64k):

  def fraction_of_data(self):
    return 0.001


@registry.register_problem
class SummarizeFrac1CnnDailymailWikiLMSharedVocab64k(
    SummarizeFracCnnDailymailWikiLMSharedVocab64k):

  def fraction_of_data(self):
    return 0.01


@registry.register_problem
class SummarizeFrac2CnnDailymailWikiLMSharedVocab64k(
    SummarizeFracCnnDailymailWikiLMSharedVocab64k):

  def fraction_of_data(self):
    return 0.02


@registry.register_problem
class SummarizeFrac5CnnDailymailWikiLMSharedVocab64k(
    SummarizeFracCnnDailymailWikiLMSharedVocab64k):

  def fraction_of_data(self):
    return 0.05


@registry.register_problem
class SummarizeFrac10CnnDailymailWikiLMSharedVocab64k(
    SummarizeFracCnnDailymailWikiLMSharedVocab64k):

  def fraction_of_data(self):
    return 0.1


@registry.register_problem
class SummarizeFrac20CnnDailymailWikiLMSharedVocab64k(
    SummarizeFracCnnDailymailWikiLMSharedVocab64k):

  def fraction_of_data(self):
    return 0.2


@registry.register_problem
class SummarizeFrac50CnnDailymailWikiLMSharedVocab64k(
    SummarizeFracCnnDailymailWikiLMSharedVocab64k):

  def fraction_of_data(self):
    return 0.5
