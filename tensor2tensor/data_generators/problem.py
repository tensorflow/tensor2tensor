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

"""Base class for problem/dataset definitions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

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
  # Czech tokens
  CS_TOK = 21
  # Czech characters
  CS_CHR = 22
  # Genetic bases (ACTG)
  DNA = 23
  # Real numbers
  REAL = 24
  # Images
  IMAGE = 25
  # Peptide
  PEPTIDE = 26
  # Python
  PY_TOK = 27
  # C++
  CPP_TOK = 28


def preprocess_examples_common(examples, hparams):
  """Preprocessing steps common to all models."""
  if hparams.max_input_seq_length > 0:
    examples["inputs"] = examples["inputs"][:hparams.max_input_seq_length]
  if hparams.max_target_seq_length > 0:
    examples["targets"] = examples["targets"][:hparams.max_target_seq_length]
  if hparams.prepend_inputs_to_targets:
    examples["targets"] = tf.concat(
        [examples["inputs"], [0], examples["targets"]], 0)
  return examples


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
    * example_reading_spec
        - Specify the names and types of the features on disk.
        - Specify tf.contrib.slim.tfexample_decoder
    * preprocess_examples(examples, mode)
        - Preprocess the example feature dict from feature name to Tensor or
          SparseTensor.
        - Used in training, eval, and inference (specified by mode).

  Eval:
    * eval_metrics
        - Specify the set of evaluation metrics for this problem.

  Inference:
    * feature_encoders(data_dir)
        - Return a dict of <feature name, TextEncoder> for encoding and decoding
          inference input/output.
        - Defaults to TextEncoder for inputs and targets.
  """

  # ============================================================================
  # BEGIN SUBCLASS INTERFACE
  # ============================================================================

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
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

  def example_reading_spec(self):
    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets": tf.VarLenFeature(tf.int64)
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)

  def preprocess_examples(self, examples, mode, hparams):
    del mode
    return preprocess_examples_common(examples, hparams)

  def eval_metrics(self):
    return [
        metrics.Metrics.ACC, metrics.Metrics.ACC_TOP5,
        metrics.Metrics.ACC_PER_SEQ, metrics.Metrics.NEG_LOG_PERPLEXITY
    ]

  # ============================================================================
  # END SUBCLASS INTERFACE
  # ============================================================================

  def training_filepaths(self, data_dir, num_shards, shuffled):
    file_basename = self.dataset_filename()
    if not shuffled:
      file_basename += generator_utils.UNSHUFFLED_SUFFIX
    return generator_utils.train_data_filenames(
        file_basename, data_dir, num_shards)

  def dev_filepaths(self, data_dir, num_shards, shuffled):
    file_basename = self.dataset_filename()
    if not shuffled:
      file_basename += generator_utils.UNSHUFFLED_SUFFIX
    return generator_utils.dev_data_filenames(
        file_basename, data_dir, num_shards)

  def test_filepaths(self, data_dir, num_shards, shuffled):
    file_basename = self.dataset_filename()
    if not shuffled:
      file_basename += generator_utils.UNSHUFFLED_SUFFIX
    return generator_utils.test_data_filenames(
        file_basename, data_dir, num_shards)

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
    if self._was_copy:
      _copy_problem_hparams(hp)
    return hp

  def maybe_reverse_features(self, feature_map):
    if not self._was_reversed:
      return
    inputs, targets = feature_map["inputs"], feature_map["targets"]
    feature_map["inputs"], feature_map["targets"] = targets, inputs

  def maybe_copy_features(self, feature_map):
    if not self._was_copy:
      return
    feature_map["targets"] = feature_map["inputs"]


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


class Text2TextProblem(Problem):
  """Base class for text-to-text problems."""

  @property
  def is_character_level(self):
    raise NotImplementedError()

  @property
  def targeted_vocab_size(self):
    raise NotImplementedError()  # Not needed if self.is_character_level.

  def generator(self, data_dir, tmp_dir, is_training):
    """Generator for the training and evaluation data."""
    raise NotImplementedError()

  @property
  def use_train_shards_for_dev(self):
    """If true, we only generate training data and hold out shards for dev."""
    return False

  @property
  def input_space_id(self):
    raise NotImplementedError()

  @property
  def target_space_id(self):
    raise NotImplementedError()

  @property
  def num_shards(self):
    raise NotImplementedError()

  @property
  def num_dev_shards(self):
    return 1

  @property
  def vocab_name(self):
    raise NotImplementedError()

  @property
  def vocab_file(self):
    return "%s.%d" % (self.vocab_name, self.targeted_vocab_size)

  @property
  def use_subword_tokenizer(self):
    raise NotImplementedError()

  @property
  def has_inputs(self):
    return True  # Set to False for language models.

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    train_paths = self.training_filepaths(
        data_dir, self.num_shards, shuffled=False)
    dev_paths = self.dev_filepaths(
        data_dir, self.num_dev_shards, shuffled=False)
    if self.use_train_shards_for_dev:
      all_paths = train_paths + dev_paths
      generator_utils.generate_files(
          self.generator(data_dir, tmp_dir, True), all_paths)
      generator_utils.shuffle_dataset(all_paths)
    generator_utils.generate_dataset_and_shuffle(
        self.generator(data_dir, tmp_dir, True),
        self.training_filepaths(data_dir, self.num_shards, shuffled=False),
        self.generator(data_dir, tmp_dir, False),
        self.dev_filepaths(data_dir, self.num_dev_shards, shuffled=False))

  def feature_encoders(self, data_dir):
    if self.is_character_level:
      encoder = text_encoder.ByteTextEncoder()
    elif self.use_subword_tokenizer:
      vocab_filename = os.path.join(data_dir, self.vocab_file)
      encoder = text_encoder.SubwordTextEncoder(vocab_filename)
    else:
      vocab_filename = os.path.join(data_dir, self.vocab_file)
      encoder = text_encoder.TokenTextEncoder(vocab_filename)
    if self.has_inputs:
      return {"inputs": encoder, "targets": encoder}
    return {"targets": encoder}

  def hparams(self, defaults, unused_model_hparams):
    p = defaults

    if self.has_inputs:
      source_vocab_size = self._encoders["inputs"].vocab_size
      p.input_modality = {"inputs": (registry.Modalities.SYMBOL,
                                     source_vocab_size)}
    target_vocab_size = self._encoders["targets"].vocab_size
    p.target_modality = (registry.Modalities.SYMBOL, target_vocab_size)
    if self.has_inputs:
      p.input_space_id = self.input_space_id
    p.target_space_id = self.target_space_id
    if self.is_character_level:
      p.loss_multiplier = 2.0

  def eval_metrics(self):
    return [
        metrics.Metrics.ACC, metrics.Metrics.ACC_TOP5,
        metrics.Metrics.ACC_PER_SEQ, metrics.Metrics.NEG_LOG_PERPLEXITY,
        metrics.Metrics.APPROX_BLEU, metrics.Metrics.ROUGE_2_F,
        metrics.Metrics.ROUGE_L_F
    ]
