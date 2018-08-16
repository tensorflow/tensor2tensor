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
"""Base classes for paraphrase generation problems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import io
import json
import os
import zipfile

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow as tf

_MS_COCO_DOWNLOAD_URL = "http://msvocds.blob.core.windows.net/annotations-1-0-3"
_MS_COCO_ZIPPED_FILE = "captions_train-val2014.zip"

_MS_COCO_TRAIN_FILE = "captions_train2014.json"
_MS_COCO_DEV_FILE = "captions_val2014.json"


def create_combination(list_of_sentences):
  """Generates all possible pair combinations for the input list of sentences.

  For example:

  input = ["paraphrase1", "paraphrase2", "paraphrase3"]

  output = [("paraphrase1", "paraphrase2"),
            ("paraphrase1", "paraphrase3"),
            ("paraphrase2", "paraphrase3")]

  Args:
    list_of_sentences: the list of input sentences.
  Returns:
    the list of all possible sentence pairs.
  """
  num_sentences = len(list_of_sentences) - 1
  combinations = []
  for i, _ in enumerate(list_of_sentences):
    if i == num_sentences:
      break
    num_pairs = num_sentences - i
    populated = num_pairs * [list_of_sentences[i]]
    zipped = list(zip(populated, list_of_sentences[i + 1:]))
    combinations += zipped
  return combinations


class ParaphraseGenerationProblem(text_problems.Text2TextProblem):
  """Paraphrase problem."""

  @property
  def bidirectional(self):
    """If set to true, generates data in the following way.

    sentence1 -> sentence2
    sentence2 -> sentence1
    """
    raise NotImplementedError()

  def prepare_data(self, data_dir, tmp_dir, dataset_split):
    raise NotImplementedError()

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    paraphrase_pairs = self.prepare_data(data_dir, tmp_dir, dataset_split)
    for (caption1, caption2) in paraphrase_pairs:
      caption_pairs = [(caption1, caption2)]
      if self.bidirectional:
        caption_pairs += [(caption2, caption1)]
      for caption_pair in caption_pairs:
        yield {
            "inputs": caption_pair[0],
            "targets": caption_pair[1]
        }


class ParaphraseGenerationMsCocoProblem(ParaphraseGenerationProblem):
  """Paraphrase problem."""

  @property
  def is_generate_per_split(self):
    return True

  @property
  def dataset_splits(self):
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 10,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  @property
  def approx_vocab_size(self):
    return 2 ** 13

  def prepare_data(self, data_dir, tmp_dir, dataset_split):
    ms_coco_path = self._maybe_download(tmp_dir, dataset_split)
    captions = self._get_captions(ms_coco_path)
    tf.logging.info("Retrieved %d captions\n" % (len(captions)))
    paraphrase_pairs = []

    tf.logging.info("Generating input combinations...")
    for captions_for_image in captions:
      combinations_of_captions = create_combination(captions_for_image)
      paraphrase_pairs += combinations_of_captions

    tf.logging.info("Created %d combinations pairs." % (len(paraphrase_pairs)))
    return paraphrase_pairs

  def _maybe_download(self, tmp_dir, dataset_split):
    filename = os.path.basename(_MS_COCO_ZIPPED_FILE)
    download_url = os.path.join(_MS_COCO_DOWNLOAD_URL, filename)
    path = generator_utils.maybe_download(tmp_dir, filename, download_url)
    unzip_dir = os.path.join(tmp_dir, filename.strip(".zip"))
    if not tf.gfile.Exists(unzip_dir):
      tf.logging.info("Unzipping data to {}".format(unzip_dir))
      zipfile.ZipFile(path, "r").extractall(unzip_dir)

    if dataset_split == problem.DatasetSplit.TRAIN:
      ms_coco_file = _MS_COCO_TRAIN_FILE
    else:
      ms_coco_file = _MS_COCO_DEV_FILE
    ms_coco_path = os.path.join(unzip_dir, "annotations", ms_coco_file)
    return ms_coco_path

  def _get_captions(self, ms_coco_path):
    caption_file = io.open(ms_coco_path)
    caption_json = json.load(caption_file)
    annotations = caption_json["annotations"]
    captions_for_image = collections.defaultdict(list)

    for annotation in annotations:
      image_id = annotation["image_id"]
      captions_for_image[image_id].append(annotation["caption"])

    captions = list(captions_for_image.values())
    return captions


@registry.register_problem
class ParaphraseGenerationMsCocoProblem2d(
    ParaphraseGenerationMsCocoProblem):

  @property
  def bidirectional(self):
    return True


@registry.register_problem
class ParaphraseGenerationMsCocoProblem1d(
    ParaphraseGenerationMsCocoProblem):

  @property
  def bidirectional(self):
    return False


@registry.register_problem
class ParaphraseGenerationMsCocoProblem2dCharacters(
    ParaphraseGenerationMsCocoProblem2d):

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER


@registry.register_problem
class ParaphraseGenerationMsCocoProblem1dCharacters(
    ParaphraseGenerationMsCocoProblem1d):

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER
