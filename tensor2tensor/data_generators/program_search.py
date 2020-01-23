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

"""Program Search Problems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import json
import os

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


@registry.register_problem
class ProgramSearchAlgolisp(text_problems.Text2TextProblem):
  """Problem class for Program Search Algolisp task.

  Synthesizing programs from description and examples.

  Please see: https://arxiv.org/pdf/1802.04335.pdf for the full description.
  """

  # The locations of the train, dev, and test set.
  DROPBOX = "https://www.dropbox.com"
  DATA_URLS = {
      problem.DatasetSplit.TRAIN: (
          DROPBOX + "/s/qhun6kml9yb2ui9/metaset3.train.jsonl.gz?dl=1"),
      problem.DatasetSplit.EVAL: (
          DROPBOX + "/s/aajkw83j2ps8bzx/metaset3.dev.jsonl.gz?dl=1"),
      problem.DatasetSplit.TEST: (
          DROPBOX + "/s/f1x9ybkjpf371cp/metaset3.test.jsonl.gz?dl=1"),
  }

  @staticmethod
  def _extract_filename_from_url(url):
    # Ex: TRAIN_URL --> metaset3.train.jsonl.gz

    # Get everything from the last / onwards.
    filename = os.path.basename(url)

    # Get rid of everything after the first ?
    return filename.split("?")[0]

  @staticmethod
  def _flatten_target_programs(iterable):
    # The target programs are read as nested lists, we should flatten them.
    yield "["
    it = iter(iterable)
    for e in it:
      if isinstance(e, (list, tuple)):
        for f in ProgramSearchAlgolisp._flatten_target_programs(e):
          yield f
      else:
        yield e
    yield "]"

  @staticmethod
  def _parse_json_to_dict(json_line):
    # First parse it through json.
    line_json_dict = json.loads(json_line)

    # The features of interest "text" and "short_tree" are stored as lists in
    # this dictionary -- "short_tree" is a nested list. We flatten and join the
    # lists on space, to return a string in both these cases.

    # Make another dictionary, to return only the features we want.
    return {
        "inputs":
            " ".join(line_json_dict["text"]),
        "targets":
            " ".join([
                i for i in ProgramSearchAlgolisp._flatten_target_programs(
                    line_json_dict["short_tree"])
            ])
    }

  @property
  def is_generate_per_split(self):
    # Return True since we already have the train and the dev set separated out.
    return True

  def maybe_download_dataset(self, tmp_dir, dataset_split):
    """Downloads the appropriate dataset file and returns its path."""
    # Get the dataset url for the split requested.
    url = self.DATA_URLS.get(dataset_split, None)

    # Sanity check.
    if url is None:
      tf.logging.fatal("Unknown dataset_split passed: {}".format(dataset_split))

    # Download the data, if it doesn't already exist.
    return generator_utils.maybe_download(tmp_dir,
                                          self._extract_filename_from_url(url),
                                          url)

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del data_dir

    # Download the data, if it doesn't already exist.
    downloaded_filepath = self.maybe_download_dataset(tmp_dir, dataset_split)

    # Decompress the file and iterate through it.
    with gzip.open(downloaded_filepath, "rb") as data_fp:
      for line in data_fp:
        yield self._parse_json_to_dict(line.strip())
