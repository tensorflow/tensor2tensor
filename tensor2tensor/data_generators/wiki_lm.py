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

"""Data generators for untokenized wikipedia LM dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_problem
class LanguagemodelWiki32k(text_problems.Text2SelfProblem):
  """A language model on the untokenized wikipedia corpus."""

  # File names and Google drive ids for the training/dev/test data.
  train_name_id = ("wiki_train.txt.gz", "1-l02fI15ieMIZk8EnXhzhsvuEYRoznZ8")
  dev_name_id = ("wiki_dev.txt.gz", "1odhDxWKtAPKXwxRw1KCrmlrVewxdXYq7")
  test_name_id = ("wiki_test.txt.gz", "1i1Bg6XqvdRl1LuOiIWbg7ww8Y02Ip5VK")

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  @property
  def max_samples_for_vocab(self):
    return 63000

  def is_generate_per_split(self):
    return True

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    # Thresholds in the number of characters for LM examples
    lo_thresh = 10
    up_thresh = 256*8

    if dataset_split == problem.DatasetSplit.TRAIN:
      (fname, fid) = self.train_name_id
    else:
      (fname, fid) = self.dev_name_id

    wikifiles = []
    url = "https://drive.google.com/uc?export=download&id=" + fid
    download_path = generator_utils.maybe_download_from_drive(
        tmp_dir, fname, url)
    wiki_file = os.path.join(tmp_dir, fname[:-3])
    if not tf.gfile.Exists(wiki_file):
      generator_utils.gunzip_file(download_path, wiki_file)
    wikifiles.append(wiki_file)

    txt = ""
    for wiki_file in wikifiles:
      for line in tf.gfile.Open(wiki_file):
        line = line.strip()
        if len(txt) + len(line) > up_thresh:
          ret = txt
          txt = ""
          if len(ret) > lo_thresh and len(ret) < up_thresh:
            yield {"targets": ret}

        if not txt:
          txt = line
        else:
          txt = " ".join([txt, line])
