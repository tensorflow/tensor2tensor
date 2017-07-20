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

"""Base class for problem/dataset definitions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.data_generators import generator_utils as utils
from tensor2tensor.data_generators import text_encoder

import tensorflow as tf


class SpaceID(object):
  """Input and target space ids. Add more as needed."""
  # Generic / unknown output space (default)
  GENERIC = 0
  # Image labels
  IMAGE_LABEL = 1
  # English characters
  EN_CHR = 2
  # English tokens
  EN_TOK = 3
  # English bpe tokens
  EN_BPE_TOK = 4
  # French characters
  FR_CHR = 5
  # French tokens
  FR_TOK = 6
  # German characters
  DE_CHR = 7
  # German tokens
  DE_TOK = 8
  # German bpe tokens
  DE_BPE_TOK = 9
  # Digit cipher lexicon 0
  DIGIT_0 = 10
  # Digit cipher lexicon 1
  DIGIT_1 = 11
  # Audio waveform domain
  AUDIO_WAV = 12
  # Audio spectral domain
  AUDIO_SPECTRAL = 13
  # Parse characters
  PARSE_CHR = 14
  # Parse tokens
  PARSE_TOK = 15
  # Chinese tokens
  ZH_TOK = 16
  # Icelandic characters
  ICE_CHAR = 17
  # Icelandic tokens
  ICE_TOK = 18
  # Icelandic parse tokens
  ICE_PARSE_TOK = 19
  # Macedonian tokens
  MK_TOK = 20


class Problem(object):
  """Problem base class. Specifies a T2T problem.

  Problems unify the specification of a problem for data generation, training,
  and inference.

  New problems are specified by the following methods:

  Data generation:
    * generate_data(data_dir, tmp_dir)
        - Generate training and dev datasets into data_dir.
        - Additonal files, e.g. vocabulary files, should also be written to
          data_dir.
        - Downloads and other files can be written to tmp_dir
        - If you have a training and dev generator, you can generate the
          training and dev datasets with
          generator_utils.generate_dataset_and_shuffle.
        - Use the self.training_filepaths and self.dev_filepaths functions to
          get sharded filenames. If shuffled=False, the filenames will contain
          an "unshuffled" suffix; you should then shuffle the data
          shard-by-shard with generator_utils.shuffle_dataset.
        - Allows to specify the number of shards, optionally (can be omitted).
        - Subclasses must override
    * dataset_filename()
        - Base filename for problem.
        - Defaults to registered name (self.name).

  Training:
    * hparams(defaults, model_hparams)
        - Specify the problem hyperparameters (see _default_hparams)
        - Mutate defaults as needed

  Inference:
    * feature_encoders(data_dir)
        - Return a dict of <feature name, TextEncoder> for encoding and decoding
          inference input/output.
        - Defaults to TextEncoder for inputs and targets.
  """

  # ============================================================================
  # BEGIN SUBCLASS INTERFACE
  # ============================================================================

  def generate_data(self, data_dir, tmp_dir, num_shards=None):
    raise NotImplementedError()

  def hparams(self, defaults, model_hparams):
    pass

  def dataset_filename(self):
    return self.name

  def feature_encoders(self, data_dir):
    del data_dir
    return {
        "inputs": text_encoder.TextEncoder(),
        "targets": text_encoder.TextEncoder()
    }

  # ============================================================================
  # END SUBCLASS INTERFACE
  # ============================================================================

  def training_filepaths(self, data_dir, num_shards, shuffled):
    file_basename = self.dataset_filename()
    if not shuffled:
      file_basename += utils.UNSHUFFLED_SUFFIX
    return utils.train_data_filenames(file_basename, data_dir, num_shards)

  def dev_filepaths(self, data_dir, num_shards, shuffled):
    file_basename = self.dataset_filename()
    if not shuffled:
      file_basename += utils.UNSHUFFLED_SUFFIX
    return utils.dev_data_filenames(file_basename, data_dir, num_shards)

  def test_filepaths(self, data_dir, num_shards, shuffled):
    file_basename = self.dataset_filename()
    if not shuffled:
      file_basename += utils.UNSHUFFLED_SUFFIX
    return utils.test_data_filenames(file_basename, data_dir, num_shards)

  def __init__(self, was_reversed=False, was_copy=False):
    """Create a Problem.

    Args:
      was_reversed: bool, whether to reverse inputs and targets.
      was_copy: bool, whether to copy inputs to targets. Can be composed with
        was_reversed so that if both are true, the targets become the inputs,
        which are then copied to targets so that the task is targets->targets.
    """
    self._was_reversed = was_reversed
    self._was_copy = was_copy
    self._encoders = None

  def internal_build_encoders(self, data_dir):
    self._encoders = self.feature_encoders(data_dir)

  def internal_hparams(self, model_hparams):
    """Returns problem_hparams."""
    if self._encoders is None:
      self.internal_build_encoders(model_hparams.data_dir)

    hp = _default_hparams()
    ret = self.hparams(hp, model_hparams)
    if ret is not None:
      raise ValueError("The Problem subclass hparams function should mutate "
                       "the defaults passed in and return None.")

    hp.add_hparam("vocabulary", self._encoders)
    hp.add_hparam("was_reversed", self._was_reversed)
    hp.add_hparam("was_copy", self._was_copy)

    if self._was_reversed:
      _reverse_problem_hparams(hp)
      # TODO(rsepassi): Move this into the cifar10 Problem
      if "image_cifar10" in self.name:
        hp.loss_multiplier = 1.
    if self._was_copy:
      _copy_problem_hparams(hp)
    return hp


def _copy_problem_hparams(p_hparams):
  """Use input modality, vocab, and space id for target."""
  p = p_hparams
  # Duplicate input modality.
  p.target_modality = p.input_modality["inputs"]
  # Duplicate input vocabulary.
  p.vocabulary["targets"] = p.vocabulary["inputs"]
  # Duplicate input space ids.
  p.target_space_id = p.input_space_id
  # Mark that p was reversed.
  p.was_copy = True


def _reverse_problem_hparams(p_hparams):
  """Swap input/output modalities, vocab, and space ids."""
  p = p_hparams

  # Swap modalities.
  input_modality = p.input_modality["inputs"]
  target_modality = p.target_modality
  p.input_modality["inputs"] = target_modality
  p.target_modality = input_modality

  # Swap vocabularies.
  input_vocabulary = p.vocabulary["inputs"]
  target_vocabulary = p.vocabulary["targets"]
  p.vocabulary["inputs"] = target_vocabulary
  p.vocabulary["targets"] = input_vocabulary

  # Swap input/target space ids.
  input_space_id = p.input_space_id
  target_space_id = p.target_space_id
  p.input_space_id = target_space_id
  p.target_space_id = input_space_id

  # Mark that p was reversed.
  p.was_reversed = True


def _default_hparams():
  """A set of basic model hyperparameters."""
  return tf.contrib.training.HParams(
      # Use this parameter to get comparable perplexity numbers with different
      # tokenizations.  This value should be set to the ratio of the number of
      # tokens in the test set according to the tokeization used to the number
      # of tokens in the test set in the "official" tokenization.  For
      # example, if we are using a word-piece based model and we want to
      # compute per-word perplexity, then we set loss_multiplier to the number
      # of wordpieces per word in the test set.
      loss_multiplier=1.0,

      # Use this parameter to allow for larger sequences in the batch. Without
      # the use of this parameter, the size of the inner two dimensions will
      # be used to judge the sequence length.
      batch_size_multiplier=1,

      # To make queues of the right capacity, it's good to know the maximal
      # expected batch size, as it can vary a lot. It only affects performance
      # of input readers and memory use. The defaults should be safe and fast,
      # but decrease if your reader uses a lot of memory and increase if slow.
      max_expected_batch_size_per_shard=64,

      # Modalities used to map from input features to a space compatible with
      # chosen model architecture.  One modality spec (which is a 2-tuple,
      # (modality_full_name, vocab_size)) per feature key. modality_full_name
      # is a string type:name, e.g. class_label:class_label_2d. Leaving off
      # the name uses the default modality for that type (e.g. class_label ==
      # class_label:default).
      input_modality={},

      # Modality used to map from hidden representation to the target space.
      # Specified as a modality spec, a 2-tuple described above.
      target_modality=None,

      # Identifiers used to tell the model which input/target space will be
      # expected. For example, it can tell that we expect French as characters
      # as output, or Spanish as sound. Spaces defined as constants in SpaceID
      # class.
      input_space_id=SpaceID.GENERIC,
      target_space_id=SpaceID.GENERIC)
