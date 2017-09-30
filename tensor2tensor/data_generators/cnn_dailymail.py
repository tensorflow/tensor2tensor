# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

import os
import tarfile

# Dependency imports

import six
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry

import tensorflow as tf


# Links to data from http://cs.nyu.edu/~kcho/DMQA/
_CNN_STORIES_DRIVE_URL = "https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ"

_DAILYMAIL_STORIES_DRIVE_URL = "https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs"


# End-of-sentence marker.
EOS = text_encoder.EOS_ID


def _maybe_download_corpora(tmp_dir):
  """Download corpora if necessary and unzip them.

  Args:
    tmp_dir: directory containing dataset.

  Returns:
    filepath of the downloaded corpus file.
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
  return [cnn_finalpath, dailymail_finalpath]


def story_generator(tmp_dir):
  paths = _maybe_download_corpora(tmp_dir)
  for path in paths:
    for story_file in tf.gfile.Glob(path + "*"):
      story = u""
      for line in tf.gfile.Open(story_file):
        line = unicode(line, "utf-8") if six.PY2 else line.decode("utf-8")
        story += line
      yield story


def _story_summary_split(story):
  end_pos = story.find("\n\n")  # Upto first empty line.
  assert end_pos != -1
  return story[:end_pos], story[end_pos:].strip()


@registry.register_problem
class SummarizeCnnDailymail32k(problem.Text2TextProblem):
  """Summarize CNN and Daily Mail articles to their first paragraph."""

  @property
  def is_character_level(self):
    return False

  @property
  def has_inputs(self):
    return True

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def num_shards(self):
    return 100

  @property
  def vocab_name(self):
    return "vocab.cnndailymail"

  @property
  def use_subword_tokenizer(self):
    return True

  @property
  def targeted_vocab_size(self):
    return 2**15  # 32768

  @property
  def use_train_shards_for_dev(self):
    return True

  def generator(self, data_dir, tmp_dir, _):
    encoder = generator_utils.get_or_generate_vocab_inner(
        data_dir, self.vocab_file, self.targeted_vocab_size,
        story_generator(tmp_dir))
    for story in story_generator(tmp_dir):
      summary, rest = _story_summary_split(story)
      encoded_summary = encoder.encode(summary) + [EOS]
      encoded_story = encoder.encode(rest) + [EOS]
      yield {"inputs": encoded_story, "targets": encoded_summary}
