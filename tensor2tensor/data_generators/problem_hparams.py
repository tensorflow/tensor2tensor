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
"""Hyperparameters defining different problems.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import modalities  # pylint: disable=unused-import
from tensor2tensor.utils import registry

import tensorflow as tf

# TODO(rsepassi): Merge these problems with their data generators. Currently
# they only implement the hparams.


class AudioTimitProblem(problem.Problem):
  """Base class for TIMIT problems."""

  def example_reading_spec(self):
    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "audio/sample_count": tf.FixedLenFeature((), tf.int64),
        "audio/sample_width": tf.FixedLenFeature((), tf.int64),
        "targets": tf.VarLenFeature(tf.int64),
    }
    return data_fields, None

  def preprocess_example(self, example, mode, hparams):
    example = super(AudioTimitProblem, self).preprocess_example(
        example, mode, hparams)
    # Reshape audio to proper shape
    sample_count = tf.to_int32(example.pop("audio/sample_count"))
    sample_width = tf.to_int32(example.pop("audio/sample_width"))
    channel_count = 1
    example["inputs"] = tf.reshape(example["inputs"],
                                   [sample_count, sample_width, channel_count])
    return example


@registry.register_problem
class AudioTimitCharactersTune(AudioTimitProblem):
  """TIMIT to characters."""

  def feature_encoders(self, _):
    return {
        "inputs": text_encoder.TextEncoder(),
        "targets": text_encoder.ByteTextEncoder(),
    }

  def hparams(self, defaults, model_hparams):
    hp = defaults
    hp.input_modality = {
        "inputs": (registry.Modalities.AUDIO, None),
    }
    hp.target_modality = (registry.Modalities.SYMBOL, 256)


@registry.register_problem
class AudioTimitTokens8kTune(AudioTimitProblem):
  """TIMIT to tokens."""

  @property
  def target_vocab_size(self):
    return 2**13  # 8192

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir,
                                  "vocab.endefr.%d" % self.target_vocab_size)
    subtokenizer = text_encoder.SubwordTextEncoder(vocab_filename)
    return {
        "inputs": text_encoder.TextEncoder(),
        "targets": subtokenizer,
    }

  def hparams(self, defaults, model_hparams):
    hp = defaults
    hp.input_modality = {
        "inputs": (registry.Modalities.AUDIO, None),
    }
    hp.target_modality = (registry.Modalities.SYMBOL,
                          self.get_feature_encoders()["targets"].vocab_size)
    hp.batch_size_multiplier = 256
    hp.loss_multiplier = 2.0
    hp.input_space_id = 13
    hp.target_space_id = 3


@registry.register_problem
class AudioTimitTokens8kTest(AudioTimitTokens8kTune):
  """TIMIT to tokens."""
  pass


@registry.register_problem
class ParsingEnglishPtb8k(problem.Problem):
  """Parsing."""

  @property
  def target_vocab_size(self):
    return 2**13  # 8192

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir,
                                  "vocab.endefr.%d" % self.target_vocab_size)
    subtokenizer = text_encoder.SubwordTextEncoder(vocab_filename)
    return {
        "inputs": subtokenizer,
        "targets": subtokenizer,
    }

  def hparams(self, defaults, model_hparams):
    hp = defaults
    hp.input_modality = {
        "inputs": (registry.Modalities.SYMBOL,
                   self.get_feature_encoders()["inputs"].vocab_size),
    }
    hp.target_modality = (registry.Modalities.SYMBOL,
                          self.get_feature_encoders()["targets"].vocab_size)
    hp.batch_size_multiplier = 256
    hp.loss_multiplier = 2.0
    hp.input_space_id = 3
    hp.target_space_id = 15


@registry.register_problem
class ParsingEnglishPtb16k(problem.Problem):
  """Parsing."""

  @property
  def vocab_prefix(self):
    return "wsj"

  @property
  def inputs_target_vocab_size(self):
    return 2**9  # 512

  @property
  def targets_target_vocab_size(self):
    return 2**14  # 16384

  def feature_encoders(self, data_dir):
    source_vocab_filename = os.path.join(
        data_dir,
        self.vocab_prefix + "_source.vocab.%d" % self.inputs_target_vocab_size)
    target_vocab_filename = os.path.join(
        data_dir,
        self.vocab_prefix + "_target.vocab.%d" % self.targets_target_vocab_size)
    source_subtokenizer = text_encoder.SubwordTextEncoder(source_vocab_filename)
    target_subtokenizer = text_encoder.SubwordTextEncoder(target_vocab_filename)
    return {
        "inputs": source_subtokenizer,
        "targets": target_subtokenizer,
    }

  def hparams(self, defaults, model_hparams):
    hp = defaults
    hp.input_modality = {
        "inputs": (registry.Modalities.SYMBOL,
                   self.get_feature_encoders()["inputs"].vocab_size),
    }
    hp.target_modality = (registry.Modalities.SYMBOL,
                          self.get_feature_encoders()["targets"].vocab_size)
    hp.input_space_id = 3
    hp.target_space_id = 15


class TestProblem(problem.Problem):
  """Test problem."""

  def __init__(self, input_vocab_size, target_vocab_size):
    super(TestProblem, self).__init__(False, False)
    self.input_vocab_size = input_vocab_size
    self.target_vocab_size = target_vocab_size

  def hparams(self, defaults, model_hparams):
    hp = defaults
    hp.input_modality = {
        "inputs": (registry.Modalities.SYMBOL, self.input_vocab_size)
    }
    hp.target_modality = (registry.Modalities.SYMBOL, self.target_vocab_size)


def test_problem_hparams(input_vocab_size=None, target_vocab_size=None):
  """Problem hparams for testing model bodies."""
  p = TestProblem(input_vocab_size, target_vocab_size)
  return p.get_hparams()
