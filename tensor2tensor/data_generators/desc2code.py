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

"""Data generators for the Description2Code OpenAI data-set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import random
import zipfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry

import tensorflow as tf


# End-of-sentence marker.
EOS = text_encoder.EOS_ID

_DATASET_URL = "https://drive.google.com/uc?export=download&id=0Bz3fihKG133ceWNFQTQ5S0xhZUk"
_DATASET_FILENAME = "description2code_current.zip"
_DATASET_PB_PATH = "description2code_current/"

_DESC_DIR_NAME = "description"
_CODE_PY_DIR_NAME = "solutions_python"

_VOCAB_EN_FILENAME = "vocab_desc2code_tok_en"
_VOCAB_PY_FILENAME = "vocab_desc2code_tok_py"

# Struct containing a coding problem (contains the paths to the descriptions
# and code files)
CodingPbInfo = collections.namedtuple("CodingPbInfo", "desc_file, code_files")


class Desc2CodeProblem(problem.Text2TextProblem):
  """Base class for Description2Code problems."""

  @property
  def is_character_level(self):
    return False

  @property
  def num_shards(self):
    return 100

  @property
  def use_subword_tokenizer(self):
    return True


@registry.register_problem("desc2code_py")
class Desc2CodePyProblem(Desc2CodeProblem):
  """Description2Code for python problem."""

  @property
  def targeted_vocab_size(self):
    return 2**13  # 8192

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.PY_TOK

  @property
  def vocab_input_filename(self):
    return "{}.{}".format(_VOCAB_EN_FILENAME, self.targeted_vocab_size)

  @property
  def vocab_target_filename(self):
    return "{}.{}".format(_VOCAB_PY_FILENAME, self.targeted_vocab_size)

  def train_generator(self, data_dir, tmp_dir, train):
    # Called twice: for train and test

    # Get the list of the training samples (coding challenge samples)
    samples = list(generator_samples(tmp_dir))

    # Split between train and dev
    # Suffle to get problems from diverse sources (CodeChef and CodeForces) and
    # dificulties in each set.
    # Need to sort the samples first before shuffling (as walk() isn't
    # deterministic)
    samples.sort(key=lambda x: x.desc_file)  # in-place
    rng = random.Random(7531)  # Local fixed seed
    rng.shuffle(samples)  # in-place

    # Train: 5019/5228 problems
    # Dev: 209/5228 problems
    len_samples = len(samples)
    split = len_samples // 25
    samples = samples[split:] if train else samples[:split]
    tf.logging.info("Number of samples for {}: {}/{}".format(
        "train" if train else "dev",
        len(samples),
        len_samples
    ))

    def generator_samples_content(get_source, get_target):
      source, target = None, None
      # Iterate over the coding samples
      for sample in samples:
        if get_source:
          with tf.gfile.GFile(sample.desc_file, mode="r") as source_file:
            source = source_file.read()

        if get_target:
          # Each challenge can have multiple implementations (or none)
          for code_file in sample.code_files:
            with tf.gfile.GFile(code_file, mode="r") as target_file:
              target = target_file.read()
            yield source, target
        elif sample.code_files:  # Only take the source if a target exists
          yield source, target

    def generator_source():
      for source, _ in generator_samples_content(True, False):
        yield source.strip()

    def generator_target():
      for _, target in generator_samples_content(False, True):
        yield target.strip()

    # Generate vocab for both source and target

    source_vocab = generator_utils.get_or_generate_vocab_inner(
        data_dir=data_dir,
        vocab_filename=self.vocab_input_filename,
        vocab_size=self.targeted_vocab_size,
        generator_fn=generator_source,
    )

    target_vocab = generator_utils.get_or_generate_vocab_inner(
        data_dir=data_dir,
        vocab_filename=self.vocab_target_filename,
        vocab_size=self.targeted_vocab_size,
        generator_fn=generator_target,
    )

    # Yield the training and testing samples
    eos_list = [EOS]
    for source, target in generator_samples_content(True, True):
      source_ints = source_vocab.encode(source.strip()) + eos_list
      target_ints = target_vocab.encode(target.strip()) + eos_list
      yield {
          "inputs": source_ints,
          "targets": target_ints,
      }

  def feature_encoders(self, data_dir):
    source_vocab_filename = os.path.join(data_dir, self.vocab_input_filename)
    target_vocab_filename = os.path.join(data_dir, self.vocab_target_filename)
    source_token = text_encoder.SubwordTextEncoder(source_vocab_filename)
    target_token = text_encoder.SubwordTextEncoder(target_vocab_filename)
    return {
        "inputs": source_token,
        "targets": target_token,
    }


# Utils functions


def generator_samples(tmp_dir):
  """Generator for the dataset samples.

  If not present, download and extract the dataset.

  Args:
    tmp_dir: path to the directory where to download the dataset.

  Yields:
    A CodingPbInfo object containing the next challenge informations.
  """
  # Step1: Download dataset (eventually)
  data_zip_path = generator_utils.maybe_download_from_drive(
      directory=tmp_dir,
      filename=_DATASET_FILENAME,
      url=_DATASET_URL,
  )
  tf.logging.info("Data downloaded in: {}".format(data_zip_path))

  # Step2: Extract dataset
  # We could deduce _DATASET_PB_PATH from the zip file (instead of
  # hardcoded path)
  data_rootdir = os.path.join(tmp_dir, _DATASET_PB_PATH)
  if not tf.gfile.Exists(data_rootdir):
    with zipfile.ZipFile(data_zip_path, "r") as corpus_zip:
      corpus_zip.extractall(tmp_dir)
    # We could remove the extracted __MACOSX folder
    tf.logging.info("Data extracted in: {}".format(tmp_dir))
  else:
    tf.logging.info("Data already extracted in: {}".format(tmp_dir))

  # Step3: Extract the problems list on the extracted folder
  def contains_samples(subdir, dirs, files):  # pylint: disable=unused-argument
    """Check that the folder contains a problem."""
    return (
        _DESC_DIR_NAME in dirs and
        _CODE_PY_DIR_NAME in dirs
    )

  def next_sample(subdir, dirs, files):  # pylint: disable=unused-argument
    """Return the filenames of the problem."""
    # More could be extracted (like the expected inputs/outputs
    # pairs, the problem difficulty, the names of the algorithmic techniques
    # needed)
    desc_file = os.path.join(subdir, _DESC_DIR_NAME, "description.txt")
    code_rootdir = os.path.join(subdir, _CODE_PY_DIR_NAME)
    code_files = [
        f for f in tf.gfile.Glob(os.path.join(code_rootdir, "*.txt"))
    ]
    return CodingPbInfo(
        desc_file=desc_file,
        code_files=code_files
    )

  # The dataset contains problem from two different sources (CodeChef
  # and CodeForces). Due to the limited number of samples, all problems from
  # both sources are merged
  for w in tf.gfile.Walk(data_rootdir):
    if contains_samples(*w):
      yield next_sample(*w)

