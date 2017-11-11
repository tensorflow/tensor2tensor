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
import collections
import os
import random
# Dependency imports
import six
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
  # Strokes
  STROKES = 29
  # Pickled Python
  PICKLED_PYTHON = 30


def default_model_hparams():
  return tf.contrib.training.HParams(
      max_input_seq_length=0,
      max_target_seq_length=0,
      prepend_mode="none",
      data_dir=None)


def preprocess_example_common(example, hparams, mode):
  """Preprocessing steps common to all models."""
  if hparams.max_input_seq_length > 0:
    example["inputs"] = example["inputs"][:hparams.max_input_seq_length]
  if hparams.max_target_seq_length > 0:
    example["targets"] = example["targets"][:hparams.max_target_seq_length]
  if hparams.prepend_mode != "none":
    if mode == tf.estimator.ModeKeys.PREDICT:
      example["partial_targets"] = tf.concat([example["inputs"], [0]], 0)
    else:
      example["targets"] = tf.concat(
          [example["inputs"], [0], example["targets"]], 0)
  return example


class Problem(object):
  """Problem base class. Specifies a T2T problem.

  Problems unify the specification of a problem for data generation, training,
  and inference.

  New problems are specified by the following methods:

  Data generation:
    * generate_data(data_dir, tmp_dir)
        - Generate training and dev datasets into data_dir.
        - Additional files, e.g. vocabulary files, should also be written to
          data_dir. Vocab files are newline-separated files with each line
          containing a token. The standard convention for the filename is to
          set it to be
                  ${Problem.vocab_name}.${Problem.targeted_vocab_size}
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
    * preprocess_example(example, mode)
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

  def preprocess_example(self, example, mode, hparams):
    return preprocess_example_common(example, hparams, mode)

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
    return generator_utils.train_data_filenames(file_basename, data_dir,
                                                num_shards)

  def dev_filepaths(self, data_dir, num_shards, shuffled):
    file_basename = self.dataset_filename()
    if not shuffled:
      file_basename += generator_utils.UNSHUFFLED_SUFFIX
    return generator_utils.dev_data_filenames(file_basename, data_dir,
                                              num_shards)

  def test_filepaths(self, data_dir, num_shards, shuffled):
    file_basename = self.dataset_filename()
    if not shuffled:
      file_basename += generator_utils.UNSHUFFLED_SUFFIX
    return generator_utils.test_data_filenames(file_basename, data_dir,
                                               num_shards)

  def filepattern(self, data_dir, mode, shard=None):
    """Get filepattern for data files for mode.

    Matches mode to a suffix.
    * TRAIN: train
    * EVAL: dev
    * PREDICT: dev
    * test: test

    Args:
      data_dir: str, data directory.
      mode: tf.estimator.ModeKeys or "test".
      shard: int, if provided, will only read data from the specified shard.

    Returns:
      filepattern str
    """
    path = os.path.join(data_dir, self.dataset_filename())
    shard_str = "-%05d" % shard if shard is not None else ""
    if mode == tf.estimator.ModeKeys.TRAIN:
      suffix = "train"
    elif mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
      suffix = "dev"
    else:
      assert mode == "test"
      suffix = "test"

    return "%s-%s%s*" % (path, suffix, shard_str)

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
    self._hparams = None
    self._feature_info = None

  def get_feature_encoders(self, data_dir=None):
    if self._encoders is None:
      self._encoders = self.feature_encoders(data_dir)
    return self._encoders

  def get_hparams(self, model_hparams=None):
    """Returns problem_hparams."""
    if self._hparams is not None:
      return self._hparams

    if self._encoders is None:
      data_dir = (model_hparams and model_hparams.data_dir) or None
      self.get_feature_encoders(data_dir)

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

    self._hparams = hp
    return self._hparams

  def maybe_reverse_features(self, feature_map):
    if not self._was_reversed:
      return
    inputs, targets = feature_map["inputs"], feature_map["targets"]
    feature_map["inputs"], feature_map["targets"] = targets, inputs

  def maybe_copy_features(self, feature_map):
    if not self._was_copy:
      return
    feature_map["targets"] = feature_map["inputs"]

  def dataset(self,
              mode,
              data_dir=None,
              num_threads=None,
              output_buffer_size=None,
              shuffle_files=None,
              hparams=None,
              preprocess=True,
              dataset_split=None,
              shard=None):
    """Build a Dataset for this problem.

    Args:
      mode: tf.estimator.ModeKeys; determines which files to read from.
      data_dir: directory that contains data files.
      num_threads: int, number of threads to use for decode and preprocess
        Dataset.map calls.
      output_buffer_size: int, how many elements to prefetch in Dataset.map
        calls.
      shuffle_files: whether to shuffle input files. Default behavior (i.e. when
        shuffle_files=None) is to shuffle if mode == TRAIN.
      hparams: tf.contrib.training.HParams; hparams to be passed to
        Problem.preprocess_example and Problem.hparams. If None, will use a
        default set that is a no-op.
      preprocess: bool, whether to map the Dataset through
        Problem.preprocess_example.
      dataset_split: tf.estimator.ModeKeys + ["test"], which split to read data
        from (TRAIN:"-train", EVAL:"-dev", "test":"-test"). Defaults to mode.
      shard: int, if provided, will only read data from the specified shard.

    Returns:
      Dataset containing dict<feature name, Tensor>.
    """
    dataset_split = dataset_split or mode
    assert data_dir

    if hparams is None:
      hparams = default_model_hparams()

    if not hasattr(hparams, "data_dir"):
      hparams.add_hparam("data_dir", data_dir)
    if not hparams.data_dir:
      hparams.data_dir = data_dir
    # Construct the Problem's hparams so that items within it are accessible
    _ = self.get_hparams(hparams)

    data_fields, data_items_to_decoders = self.example_reading_spec()
    if data_items_to_decoders is None:
      data_items_to_decoders = {
          field: tf.contrib.slim.tfexample_decoder.Tensor(field)
          for field in data_fields
      }

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    data_filepattern = self.filepattern(data_dir, dataset_split, shard=shard)
    tf.logging.info("Reading data files from %s", data_filepattern)
    data_files = tf.contrib.slim.parallel_reader.get_data_files(
        data_filepattern)
    if shuffle_files or shuffle_files is None and is_training:
      random.shuffle(data_files)
    dataset = tf.contrib.data.TFRecordDataset(data_files)

    def decode_record(record):
      """Serialized Example to dict of <feature name, Tensor>."""
      decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
          data_fields, data_items_to_decoders)

      decode_items = list(data_items_to_decoders)
      decoded = decoder.decode(record, items=decode_items)
      return dict(zip(decode_items, decoded))

    def _preprocess(example):
      example = self.preprocess_example(example, mode, hparams)
      self.maybe_reverse_features(example)
      self.maybe_copy_features(example)
      return example

    dataset = dataset.map(decode_record, num_threads=num_threads)

    if preprocess:
      dataset = dataset.map(
          _preprocess,
          num_threads=num_threads,
          output_buffer_size=output_buffer_size)

    return dataset

  @property
  def has_inputs(self):
    return "inputs" in self.get_feature_encoders()

  @property
  def feature_info(self):
    """Retrieve dict<feature name, FeatureInfo>.

    Must first call Problem.get_hparams or Problem.dataset to have the problem's
    internal hparams already constructed.

    Returns:
      dict<feature name, FeatureInfo>
    """
    if self._feature_info is not None:
      return self._feature_info

    assert self._hparams is not None

    hp = self.get_hparams()
    input_mods = hp.input_modality
    target_mod = hp.target_modality
    vocabs = hp.vocabulary
    if self.has_inputs:
      in_id = hp.input_space_id
    out_id = hp.target_space_id

    features = collections.defaultdict(FeatureInfo)

    for name, mod_spec in six.iteritems(input_mods):
      mod, vocab_size = mod_spec
      finfo = features[name]
      finfo.modality = mod
      finfo.vocab_size = vocab_size

    mod, vocab_size = target_mod
    features["targets"].modality = mod
    features["targets"].vocab_size = vocab_size

    for name, encoder in six.iteritems(vocabs):
      features[name].encoder = encoder

    if self.has_inputs:
      features["inputs"].space_id = in_id
    features["targets"].space_id = out_id

    self._feature_info = features
    return features


class FeatureInfo(object):

  def __init__(self,
               encoder=None,
               modality=None,
               vocab_size=None,
               space_id=None):
    self.encoder = encoder
    self.modality = modality
    self.vocab_size = vocab_size
    self.space_id = space_id


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
      # tokens in the test set according to the tokenization used to the number
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

      # During inference for autoregressive problems, if the batch_size is 1,
      # the inference will stop when the model predict a text_encoder.EOS_ID
      # token.
      stop_at_eos=False,

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
    """Whether the inputs and targets are sequences of characters."""
    raise NotImplementedError()

  @property
  def targeted_vocab_size(self):
    raise NotImplementedError()  # Not needed if self.is_character_level.

  def generator(self, data_dir, tmp_dir, is_training):
    """Generator for the training and evaluation data.

    Args:
      data_dir: The directory in which to assets, e.g. the vocab file.
      tmp_dir: A scratch directory (if needed).
      is_training: A boolean indicating if we should generate training data
          (True) or dev set data (False).

    Yields:
      dicts with keys "inputs" and "targets", with values being lists of token
      ids.
    """
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
    else:
      generator_utils.generate_dataset_and_shuffle(
          self.generator(data_dir, tmp_dir, True), train_paths,
          self.generator(data_dir, tmp_dir, False), dev_paths)

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
    p.stop_at_eos = int(True)

    if self.has_inputs:
      source_vocab_size = self._encoders["inputs"].vocab_size
      p.input_modality = {
          "inputs": (registry.Modalities.SYMBOL, source_vocab_size)
      }
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
