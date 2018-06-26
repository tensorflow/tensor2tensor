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
"""Base class for problem/dataset definitions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import functools
import os
import random

import six
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import data_reader
from tensor2tensor.utils import metrics
import tensorflow as tf



class DatasetSplit(object):
  TRAIN = tf.estimator.ModeKeys.TRAIN
  EVAL = tf.estimator.ModeKeys.EVAL
  TEST = "test"


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
      split_to_length=0,
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
  if hparams.split_to_length:
    example["targets"] = tf.reshape(example["targets"],
                                    [-1, hparams.split_to_length, 1, 1])
    if len(example) != 1:
      raise ValueError("split_to_length only works for LM problems")
    return tf.data.Dataset.from_tensor_slices(example)
  return example


def _file_num_records_cached(filename):
  """Return the number of TFRecords in a file."""
  # Cache the result, as this is expensive to compute
  if filename in _file_num_records_cache:
    return _file_num_records_cache[filename]
  ret = 0
  for _ in tf.python_io.tf_record_iterator(filename):
    ret += 1
  _file_num_records_cache[filename] = ret
  return ret


_file_num_records_cache = {}


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
                  ${Problem.vocab_filename}.${Problem.targeted_vocab_size}
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

  @property
  def multiprocess_generate(self):
    """Whether to generate the data in multiple parallel processes."""
    return False

  @property
  def num_generate_tasks(self):
    """Needed if multiprocess_generate is True."""
    raise NotImplementedError()

  def prepare_to_generate(self, data_dir, tmp_dir):
    """Prepare to generate data in parallel on different processes.

    This function is called if multiprocess_generate is True.

    Some things that might need to be done once are downloading the data
    if it is not yet downloaded, and building the vocabulary.

    Args:
      data_dir: a string
      tmp_dir: a string
    """
    raise NotImplementedError()

  def hparams(self, defaults, model_hparams):
    pass

  def max_length(self, model_hparams):
    """Maximum sequence length.

    Problems with fixed length should override.

    Args:
      model_hparams: model hyperparameters
    Returns:
      an integer
    """
    return (model_hparams.split_to_length or model_hparams.max_length or
            model_hparams.batch_size)

  def tpu_batch_size_per_shard(self, model_hparams):
    """Batch size in examples per TPU core.

    Args:
      model_hparams: model hyperparameters
    Returns:
      an integer
    """
    if self.batch_size_means_tokens:
      return model_hparams.batch_size // self.max_length(model_hparams)
    else:
      return model_hparams.batch_size

  @property
  def batch_size_means_tokens(self):
    """Do we specify hparams.batch_size in tokens per datashard per batch.

    This is generally done for text problems.

    If False, we assume that batch sizes are specified in examples per
    datashard per batch.

    TODO(noam): we should be more explicit and replace the hyperparameter
    batch size with two hyperparameters:
      hparams.examples_per_batch_per_datashard
      hparams.tokens_per_batch_per_datashard

    Returns:
      a boolean
    """
    return False

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
    """Runtime preprocessing.

    Return a dict or a tf.Data.Datset.from_tensor_slices (if you want each
    example to turn into multiple).

    Args:
      example: dict, features
      mode: tf.estimator.ModeKeys
      hparams: HParams, model hyperparameters

    Returns:
      dict or Dataset
    """
    return preprocess_example_common(example, hparams, mode)

  def eval_metrics(self):
    return [
        metrics.Metrics.ACC, metrics.Metrics.ACC_TOP5,
        metrics.Metrics.ACC_PER_SEQ, metrics.Metrics.NEG_LOG_PERPLEXITY
    ]

  # ============================================================================
  # END SUBCLASS INTERFACE
  # ============================================================================

  def preprocess(self, dataset, mode, hparams):
    """Runtime preprocessing on the whole dataset.

    Return a tf.data.Datset -- the preprocessed version of the given one.
    By default this function calls preprocess_example.

    Args:
      dataset: the Dataset of already decoded but not yet preprocessed features.
      mode: tf.estimator.ModeKeys
      hparams: HParams, model hyperparameters

    Returns:
      a Dataset
    """
    def _preprocess(example):
      examples = self.preprocess_example(example, mode, hparams)
      if not isinstance(examples, tf.data.Dataset):
        examples = tf.data.Dataset.from_tensors(examples)
      return examples

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            _preprocess, sloppy=is_training, cycle_length=8))

    return dataset

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
    * DatasetSplit.TRAIN: train
    * DatasetSplit.EVAL: dev
    * DatasetSplit.TEST: test
    * tf.estimator.ModeKeys.PREDICT: dev

    Args:
      data_dir: str, data directory.
      mode: DatasetSplit
      shard: int, if provided, will only read data from the specified shard.

    Returns:
      filepattern str
    """
    path = os.path.join(data_dir, self.dataset_filename())
    shard_str = "-%05d" % shard if shard is not None else ""
    if mode == DatasetSplit.TRAIN:
      suffix = "train"
    elif mode in [DatasetSplit.EVAL, tf.estimator.ModeKeys.PREDICT]:
      suffix = "dev"
    else:
      assert mode == DatasetSplit.TEST
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
      data_dir = (model_hparams and hasattr(model_hparams, "data_dir") and
                  model_hparams.data_dir) or None
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
    """Reverse features between inputs and targets if the problem is '_rev'."""
    if not self._was_reversed:
      return
    inputs, targets = feature_map["inputs"], feature_map["targets"]
    feature_map["inputs"], feature_map["targets"] = targets, inputs
    if "inputs_segmentation" in feature_map:
      inputs_seg = feature_map["inputs_segmentation"]
      targets_seg = feature_map["targets_segmentation"]
      feature_map["inputs_segmentation"] = targets_seg
      feature_map["targets_segmentation"] = inputs_seg
    if "inputs_position" in feature_map:
      inputs_pos = feature_map["inputs_position"]
      targets_pos = feature_map["targets_position"]
      feature_map["inputs_position"] = targets_pos
      feature_map["targets_position"] = inputs_pos

  def maybe_copy_features(self, feature_map):
    if not self._was_copy:
      return
    feature_map["targets"] = feature_map["inputs"]
    if ("inputs_segmentation" in feature_map and
        "targets_segmentation" not in feature_map):
      feature_map["targets_segmentation"] = feature_map["inputs_segmentation"]
    if ("inputs_position" in feature_map and
        "targets_position" not in feature_map):
      feature_map["targets_position"] = feature_map["inputs_position"]

  def maybe_reverse_and_copy(self, example):
    self.maybe_reverse_features(example)
    self.maybe_copy_features(example)
    return example

  def dataset(self,
              mode,
              data_dir=None,
              num_threads=None,
              output_buffer_size=None,
              shuffle_files=None,
              hparams=None,
              preprocess=True,
              dataset_split=None,
              shard=None,
              partition_id=0,
              num_partitions=1,
              max_records=-1):
    """Build a Dataset for this problem.

    Args:
      mode: tf.estimator.ModeKeys; determines which files to read from.
      data_dir: directory that contains data files.
      num_threads: int, number of threads to use for decode and preprocess
        Dataset.map calls.
      output_buffer_size: int, how many elements to prefetch at end of pipeline.
      shuffle_files: whether to shuffle input files. Default behavior (i.e. when
        shuffle_files=None) is to shuffle if mode == TRAIN.
      hparams: tf.contrib.training.HParams; hparams to be passed to
        Problem.preprocess_example and Problem.hparams. If None, will use a
        default set that is a no-op.
      preprocess: bool, whether to map the Dataset through
        Problem.preprocess_example.
      dataset_split: DatasetSplit, which split to read data
        from (TRAIN:"-train", EVAL:"-dev", "test":"-test"). Defaults to mode.
      shard: int, if provided, will only read data from the specified shard.
      partition_id: integer - which partition of the dataset to read from
      num_partitions: how many partitions in the dataset
      max_records: int, number of records to truncate to.

    Returns:
      Dataset containing dict<feature name, Tensor>.

    Raises:
      ValueError: if num_partitions is greater than the number of data files.
    """
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    shuffle_files = shuffle_files or shuffle_files is None and is_training

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

    data_filepattern = self.filepattern(data_dir, dataset_split, shard=shard)
    tf.logging.info("Reading data files from %s", data_filepattern)
    data_files = sorted(tf.contrib.slim.parallel_reader.get_data_files(
        data_filepattern))

    # Functions used in dataset transforms below. `filenames` can be either a
    # `tf.string` tensor or `tf.data.Dataset` containing one or more filenames.
    def _load_records_and_preprocess(filenames):
      # Load records from file(s) with an 8MiB read buffer.
      dataset = tf.data.TFRecordDataset(filenames, buffer_size=8 * 1024 * 1024)
      # Decode.
      dataset = dataset.map(self.decode_example, num_parallel_calls=num_threads)
      # Preprocess if requested.
      # Note that preprocessing should happen per-file as order may matter.
      if preprocess:
        dataset = self.preprocess(dataset, mode, hparams)
      return dataset

    if len(data_files) < num_partitions:
      raise ValueError(
          "number of data files (%d) must be at least the number of hosts (%d)"
          % (len(data_files), num_partitions))
    data_files = [f for (i, f) in enumerate(data_files)
                  if i % num_partitions == partition_id]
    tf.logging.info(
        "partition: %d num_data_files: %d" % (partition_id, len(data_files)))
    if shuffle_files:
      random.shuffle(data_files)

    dataset = tf.data.Dataset.from_tensor_slices(tf.constant(data_files))
    # Create data-set from files by parsing, pre-processing and interleaving.
    if shuffle_files:
      dataset = dataset.apply(
          tf.contrib.data.parallel_interleave(
              _load_records_and_preprocess, sloppy=True, cycle_length=8))
    else:
      # TFRecordDataset can get filenames as dataset in TF 1.7+.
      # TODO(lukaszkaiser): remove when we require TF 1.7+ in general.
      major, minor = [int(el) for el in tf.__version__.split(".")[:2]]
      filename_dataset_ok = major > 1 or (major == 1 and minor >= 7)
      if filename_dataset_ok:  # We can just pass a Dataset of filenames.
        dataset = _load_records_and_preprocess(dataset)
      else:  # Go file-by-file (can be very slow).
        dataset = None
        for f in data_files:
          f_data = _load_records_and_preprocess(f)
          dataset = f_data if dataset is None else dataset.concatenate(f_data)

    dataset = dataset.map(
        self.maybe_reverse_and_copy, num_parallel_calls=num_threads)
    dataset = dataset.take(max_records)
    if output_buffer_size:
      dataset = dataset.prefetch(output_buffer_size)

    return dataset

  def decode_example(self, serialized_example):
    """Return a dict of Tensors from a serialized tensorflow.Example."""
    data_fields, data_items_to_decoders = self.example_reading_spec()
    # Necessary to rejoin examples in the correct order with the Cloud ML Engine
    # batch prediction API.
    data_fields["batch_prediction_key"] = tf.FixedLenFeature([1], tf.int64, 0)
    if data_items_to_decoders is None:
      data_items_to_decoders = {
          field: tf.contrib.slim.tfexample_decoder.Tensor(field)
          for field in data_fields
      }

    decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
        data_fields, data_items_to_decoders)

    decode_items = list(sorted(data_items_to_decoders))
    decoded = decoder.decode(serialized_example, items=decode_items)
    return dict(zip(decode_items, decoded))

  @property
  def decode_hooks(self):
    """List of functions to be run after full decodes have been produced.

    Returns:
      List of functions. Each function should expect a single argument, an
      instance of decoding.DecodeHookArgs and optionally return a list of
      tf.Summary.Value objects.
    """
    return []

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

  def make_estimator_input_fn(self,
                              mode,
                              hparams,
                              data_dir=None,
                              dataset_kwargs=None):
    """Return input_fn wrapped for Estimator."""

    def estimator_input_fn(params, config):
      return self.input_fn(
          mode,
          hparams,
          data_dir=data_dir,
          params=params,
          config=config,
          dataset_kwargs=dataset_kwargs)

    return estimator_input_fn

  def _dataset_partition(self, mode, config):
    """Which part of the training data to read.

    If there are multiple parallel calls to input_fn (multiple TPU hosts),
    then we want each one to read from a separate partition of the training
    data.

    Args:
      mode: tf.estimator.ModeKeys
      config: RunConfig
    Returns:
      partition_id: an integer
      num_partitions: an integer
    """
    if mode != tf.estimator.ModeKeys.TRAIN or not hasattr(config, "tpu_config"):
      # Reset in the case when using TPU but alternating TRAIN and EVAL.
      self._next_partition_id = 0
      return 0, 1
    if config.tpu_config.per_host_input_for_training:
      num_partitions = max(config.tpu_config.num_shards // 8, 1)
    else:
      num_partitions = config.tpu_config.num_shards
    partition_id = getattr(self, "_next_partition_id", 0)
    self._next_partition_id = partition_id + 1
    tf.logging.info("num_partitions = %d partition_id = %d" %
                    (num_partitions, partition_id))
    assert partition_id < num_partitions
    return partition_id, num_partitions

  def input_fn(self,
               mode,
               hparams,
               data_dir=None,
               params=None,
               config=None,
               dataset_kwargs=None):
    """Builds input pipeline for problem.

    Args:
      mode: tf.estimator.ModeKeys
      hparams: HParams, model hparams
      data_dir: str, data directory; if None, will use hparams.data_dir
      params: dict, may include "batch_size"
      config: RunConfig; should have the data_parallelism attribute if not using
        TPU
      dataset_kwargs: dict, if passed, will pass as kwargs to self.dataset
        method when called

    Returns:
      (features_dict<str name, Tensor feature>, Tensor targets)
    """
    partition_id, num_partitions = self._dataset_partition(mode, config)

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    if config and config.use_tpu:
      num_threads = 64
    else:
      num_threads = 4 if is_training else 1

    max_length = self.max_length(hparams)

    def tpu_valid_size(example):
      return data_reader.example_valid_size(example, hparams.min_length,
                                            max_length)

    def gpu_valid_size(example):
      drop_long_sequences = is_training or hparams.eval_drop_long_sequences
      return data_reader.example_valid_size(example, hparams.min_length,
                                            max_length
                                            if drop_long_sequences else 10**9)

    def define_shapes(example):
      batch_size = config and config.use_tpu and params["batch_size"]
      return standardize_shapes(example, batch_size=batch_size)

    # Read and preprocess
    data_dir = data_dir or (hasattr(hparams, "data_dir") and hparams.data_dir)

    dataset_kwargs = dataset_kwargs or {}
    dataset_kwargs.update({
        "mode": mode,
        "data_dir": data_dir,
        "num_threads": num_threads,
        "hparams": hparams,
        "partition_id": partition_id,
        "num_partitions": num_partitions,
    })

    dataset = self.dataset(**dataset_kwargs)
    if is_training:
      # Repeat and skip a random number of records
      dataset = dataset.repeat()
      data_files = tf.contrib.slim.parallel_reader.get_data_files(
          self.filepattern(data_dir, mode))
      #  In continuous_train_and_eval when switching between train and
      #  eval, this input_fn method gets called multiple times and it
      #  would give you the exact same samples from the last call
      #  (because the Graph seed is set). So this skip gives you some
      #  shuffling.
      dataset = skip_random_fraction(dataset, data_files[0])

    dataset = dataset.map(
        data_reader.cast_ints_to_int32, num_parallel_calls=num_threads)

    if self.batch_size_means_tokens:
      batch_size_means_tokens = True
    else:
      if _are_shapes_fully_defined(dataset.output_shapes):
        batch_size_means_tokens = False
      else:
        tf.logging.warning(
            "Shapes are not fully defined. Assuming batch_size means tokens.")
        batch_size_means_tokens = True

    # Batching
    if not batch_size_means_tokens:
      # Batch size means examples per datashard.
      if config and config.use_tpu:
        # on TPU, we use params["batch_size"], which specifies the number of
        # examples across all datashards
        batch_size = params["batch_size"]
        dataset = dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size))
      else:
        num_shards = config.data_parallelism.n if config else 1
        batch_size = hparams.batch_size * num_shards
        dataset = dataset.batch(batch_size)
    else:
      # batch_size means tokens per datashard
      if config and config.use_tpu:
        dataset = dataset.filter(tpu_valid_size)
        padded_shapes = self._pad_for_tpu(dataset.output_shapes, hparams)
        # on TPU, we use params["batch_size"], which specifies the number of
        # examples across all datashards
        batch_size = params["batch_size"]
        dataset = dataset.apply(
            tf.contrib.data.padded_batch_and_drop_remainder(
                batch_size, padded_shapes))
      else:
        # On GPU, bucket by length
        dataset = dataset.filter(gpu_valid_size)
        shard_multiplier = config.data_parallelism.n if config else 1
        batching_scheme = data_reader.hparams_to_batching_scheme(
            hparams,
            shard_multiplier=shard_multiplier,
            length_multiplier=self.get_hparams().batch_size_multiplier)
        if hparams.use_fixed_batch_size:
          # Here  batch_size really means examples per datashard.
          batching_scheme["batch_sizes"] = [hparams.batch_size]
          batching_scheme["boundaries"] = []
        dataset = data_reader.bucket_by_sequence_length(
            dataset, data_reader.example_length, batching_scheme["boundaries"],
            batching_scheme["batch_sizes"])

        if not is_training:
          batch_multiple = shard_multiplier
          if hparams.use_fixed_batch_size:
            # Make sure the last batch has the same fixed size as the rest.
            batch_multiple *= hparams.batch_size
          if batch_multiple > 1:
            tf.logging.warn(
                "Padding the batch to ensure that remainder eval batches have "
                "a batch size divisible by the number of data shards. This may "
                "lead to incorrect metrics for non-zero-padded features, e.g. "
                "images. Use a single datashard (i.e. 1 GPU) in that case.")
            dataset = dataset.map(
                functools.partial(pad_batch, batch_multiple=batch_multiple),
                num_parallel_calls=num_threads)

    dataset = dataset.map(define_shapes, num_parallel_calls=num_threads)

    def prepare_for_output(example):
      if not config or not config.use_tpu:
        _summarize_features(example,
                            (config and config.data_parallelism.n) or 1)
      if mode == tf.estimator.ModeKeys.PREDICT:
        example["infer_targets"] = example.pop("targets")
        return example
      else:
        return example, example["targets"]

    dataset = dataset.map(prepare_for_output, num_parallel_calls=num_threads)
    dataset = dataset.prefetch(2)

    if mode == tf.estimator.ModeKeys.PREDICT:
      # This is because of a bug in the Estimator that short-circuits prediction
      # if it doesn't see a QueueRunner. DummyQueueRunner implements the
      # minimal expected interface but does nothing.
      tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS,
                           data_reader.DummyQueueRunner())

    return dataset

  def serving_input_fn(self, hparams):
    """Input fn for serving export, starting from serialized example."""
    mode = tf.estimator.ModeKeys.PREDICT
    serialized_example = tf.placeholder(
        dtype=tf.string, shape=[None], name="serialized_example")
    dataset = tf.data.Dataset.from_tensor_slices(serialized_example)
    dataset = dataset.map(self.decode_example)
    dataset = dataset.map(lambda ex: self.preprocess_example(ex, mode, hparams))
    dataset = dataset.map(self.maybe_reverse_and_copy)
    dataset = dataset.map(data_reader.cast_ints_to_int32)
    dataset = dataset.padded_batch(
        tf.shape(serialized_example, out_type=tf.int64)[0],
        dataset.output_shapes)
    dataset = dataset.map(standardize_shapes)
    features = tf.contrib.data.get_single_element(dataset)

    if self.has_inputs:
      features.pop("targets", None)

    return tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors=serialized_example)

  def _pad_for_tpu(self, shapes_dict, hparams):
    """Pads unknown features' dimensions for TPU."""
    max_length = self.max_length(hparams)
    padded_shapes = {}

    def get_filler(specified_max_length):
      if not specified_max_length:
        return max_length
      return min(specified_max_length, max_length)

    inputs_none_filler = get_filler(hparams.max_input_seq_length)
    targets_none_filler = get_filler(hparams.max_target_seq_length)

    def pad_one_shape(shape, none_filler):
      return [
          (dim if dim is not None else none_filler) for dim in shape.as_list()
      ]

    for key, shape in six.iteritems(shapes_dict):
      if key == "inputs":
        padded_shapes[key] = pad_one_shape(shape, inputs_none_filler)
      elif key == "targets":
        padded_shapes[key] = pad_one_shape(shape, targets_none_filler)
      else:
        padded_shapes[key] = pad_one_shape(shape, max_length)
    return padded_shapes


class FeatureInfo(object):
  """Encapsulates information about a feature."""

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


def _are_shapes_fully_defined(shapes_dict):
  for shape in shapes_dict.values():
    if not shape.is_fully_defined():
      return False
  return True


def _summarize_features(features, num_shards=1):
  with tf.name_scope("input_stats"):
    for (k, v) in six.iteritems(features):
      if isinstance(v, tf.Tensor) and v.get_shape().ndims > 1:
        tf.summary.scalar("%s_batch" % k, tf.shape(v)[0] // num_shards)
        tf.summary.scalar("%s_length" % k, tf.shape(v)[1])
        nonpadding = tf.to_float(tf.not_equal(v, 0))
        nonpadding_tokens = tf.reduce_sum(nonpadding)
        tf.summary.scalar("%s_nonpadding_tokens" % k, nonpadding_tokens)
        tf.summary.scalar("%s_nonpadding_fraction" % k,
                          tf.reduce_mean(nonpadding))


def standardize_shapes(features, batch_size=None):
  """Set the right shapes for the features."""

  for fname in ["inputs", "targets"]:
    if fname not in features:
      continue

    f = features[fname]
    while len(f.get_shape()) < 4:
      f = tf.expand_dims(f, axis=-1)

    features[fname] = f

  if batch_size:
    # Ensure batch size is set on all features
    for _, t in six.iteritems(features):
      shape = t.get_shape().as_list()
      shape[0] = batch_size
      t.set_shape(t.get_shape().merge_with(shape))
      # Assert shapes are fully known
      t.get_shape().assert_is_fully_defined()

  return features


def pad_batch(features, batch_multiple):
  """Pad batch dim of features to nearest multiple of batch_multiple."""
  feature = list(features.items())[0][1]
  batch_size = tf.shape(feature)[0]
  mod = batch_size % batch_multiple
  has_mod = tf.cast(tf.cast(mod, tf.bool), tf.int32)
  batch_padding = batch_multiple * has_mod - mod

  padded_features = {}
  for k, feature in features.items():
    rank = len(feature.shape)
    paddings = []
    for _ in range(rank):
      paddings.append([0, 0])
    paddings[0][1] = batch_padding
    padded_feature = tf.pad(feature, paddings)
    padded_features[k] = padded_feature
  return padded_features


def problem_hparams_to_features(problem_hparams):
  input_space_id, target_space_id = 0, 0
  if problem_hparams:
    input_space_id = problem_hparams.input_space_id
    target_space_id = problem_hparams.target_space_id
  return {
      "input_space_id": input_space_id,
      "target_space_id": target_space_id,
  }


def skip_random_fraction(dataset, data_file):
  # Skip a random fraction at the beginning of the stream.  The skip is
  # essential for synchronous highly-parallel training to avoid multiple
  # replicas reading the same data in lock-step.
  num_skip = random.randint(0, _file_num_records_cached(data_file))
  return dataset.skip(num_skip)
