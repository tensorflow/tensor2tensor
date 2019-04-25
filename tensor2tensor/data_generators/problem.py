# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils.hparam import HParams

import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_config



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


class TaskID(object):
  """Problem specific task ids. Add more as needed."""
  # English characters
  EN_CHR = 2
  # English characters sentiment
  EN_CHR_SENT = 3
  # English Premise Hypothesis pair
  EN_PR_HYP = 4
  # English NLI
  EN_NLI = 5
  # COLA
  COLA = 6
  # Enligh Question Context pair
  EN_Q_CONT = 7
  # English similarity task
  EN_SIM = 8
  # English sentence pair
  EN_SENT_PAIR = 9
  # 3 class NLI
  THREE_CL_NLI = 10


def default_model_hparams():
  return HParams(
      max_input_seq_length=0,
      max_target_seq_length=0,
      prepend_mode="none",
      split_to_length=0,
      data_dir=None)


def preprocess_example_common(example, mode, hparams):
  """Preprocessing steps common to all models."""
  if "inputs" in example and hparams.max_input_seq_length > 0:
    example["inputs"] = example["inputs"][:hparams.max_input_seq_length]
  if hparams.prepend_mode != "none":
    if mode == tf.estimator.ModeKeys.PREDICT:
      example["partial_targets"] = tf.concat([example["inputs"], [0]], 0)
    else:
      example["targets"] = tf.concat(
          [example["inputs"], [0], example["targets"]], 0)
  if "targets" in example and hparams.max_target_seq_length > 0:
    example["targets"] = example["targets"][:hparams.max_target_seq_length]
  if hparams.split_to_length:
    new_example = {}
    for k, v in six.iteritems(example):
      if k == "targets" or k == "inputs":
        new_example[k] = tf.reshape(v, [-1, hparams.split_to_length, 1, 1])
      else:
        tf.logging.warning("Dropping feature %s" % k)
    return tf.data.Dataset.from_tensor_slices(new_example)
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
    * preprocess_example(example, mode, hparams)
        - Preprocess the example feature dict from feature name to Tensor or
          SparseTensor.
        - Used in training, eval, and inference (specified by mode).

  Eval:
    * eval_metrics
        - Specify the set of evaluation metrics for this problem.
    * eval_hooks
        - Specify the set of evalueation hooks for this problem.

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

  @property
  def num_training_examples(self):
    """Used when mixing problems - how many examples are in the dataset."""
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
    if self.batch_size_means_tokens and not model_hparams.use_fixed_batch_size:
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

  @property
  def skip_random_fraction_when_training(self):
    """Skip a random number of examples at the beginning of training."""
    # Skip a random fraction at the beginning of the stream.  The skip is
    # essential for synchronous highly-parallel training to avoid multiple
    # replicas reading the same data in lock-step. So keep this true unless
    # you have a very specific setting in which it needs to be turned off.
    return True

  def dataset_filename(self):
    return self.name

  def feature_encoders(self, data_dir):
    del data_dir
    return {
        "inputs": text_encoder.TextEncoder(),
        "targets": text_encoder.TextEncoder()
    }

  def example_reading_spec(self):
    """Define how data is serialized to file and read back.

    Returns:
      data_fields: A dictionary mapping data names to its feature type.
      data_items_to_decoders: A dictionary mapping data names to TF Example
         decoders, to be used when reading back TF examples from disk.
    """
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
    return preprocess_example_common(example, mode, hparams)

  def eval_metrics(self):
    return [
        metrics.Metrics.ACC, metrics.Metrics.ACC_TOP5,
        metrics.Metrics.ACC_PER_SEQ, metrics.Metrics.NEG_LOG_PERPLEXITY
    ]

  def eval_metric_fns(self, model_hparams):
    del model_hparams
    metric_names = self.eval_metrics()
    if not all([m in metrics.METRICS_FNS for m in metric_names]):
      error_str = ("Unrecognized metric. Problem %s specified metrics "
                   "%s. Recognized metrics are %s.")
      raise ValueError(error_str % (self.name,
                                    metric_names,
                                    list(metrics.METRICS_FNS.keys())))
    return {
        metric_name: metrics.METRICS_FNS[metric_name]
        for metric_name in metric_names
    }

  def eval_hooks(self, features, logits, hparams):
    del features, logits, hparams
    return []

  @property
  def task_id(self):
    if self._task_id == -1 and hasattr(self, "global_task_id"):
      self._task_id = self.global_task_id()
    return self._task_id

  def set_task_id(self, new_task_id):
    self._task_id = new_task_id

  # ============================================================================
  # END SUBCLASS INTERFACE
  # ============================================================================

  def preprocess(self, dataset, mode, hparams, interleave=True):
    """Runtime preprocessing on the whole dataset.

    Return a tf.data.Datset -- the preprocessed version of the given one.
    By default this function calls preprocess_example.

    Args:
      dataset: the Dataset of already decoded but not yet preprocessed features.
      mode: tf.estimator.ModeKeys
      hparams: HParams, model hyperparameters
      interleave: bool, whether to use parallel_interleave, which is faster
        but will alter the order of samples non-deterministically, or flat_map,
        which is slower but will preserve the sample order.

    Returns:
      a Dataset
    """
    def _preprocess(example):
      examples = self.preprocess_example(example, mode, hparams)
      if not isinstance(examples, tf.data.Dataset):
        examples = tf.data.Dataset.from_tensors(examples)
      return examples

    if interleave:
      dataset = dataset.apply(
          tf.data.experimental.parallel_interleave(
              _preprocess, sloppy=True, cycle_length=8))
    else:
      dataset = dataset.flat_map(_preprocess)

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

  def data_filepaths(self, split, output_dir, num_shards, shuffled):
    if split == DatasetSplit.TRAIN:
      return self.training_filepaths(output_dir, num_shards, shuffled)
    elif split == DatasetSplit.EVAL:
      return self.dev_filepaths(output_dir, num_shards, shuffled)
    elif split == DatasetSplit.TEST:
      return self.test_filepaths(output_dir, num_shards, shuffled)
    else:
      raise ValueError("Unknown value for split: %s" % split)

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
    self._task_id = -1

  @property
  def was_reversed(self):
    """Whether the problem was reversed."""
    return self._was_reversed

  def get_feature_encoders(self, data_dir=None):
    if self._encoders is None:
      self._encoders = self.feature_encoders(data_dir)
    return self._encoders

  def get_hparams(self, model_hparams=None):
    """Returns problem_hparams."""
    if self._hparams is not None:
      return self._hparams

    if model_hparams is None:
      model_hparams = default_model_hparams()

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
    inputs = feature_map.pop("inputs", None)
    targets = feature_map.pop("targets", None)
    inputs_seg = feature_map.pop("inputs_segmentation", None)
    targets_seg = feature_map.pop("targets_segmentation", None)
    inputs_pos = feature_map.pop("inputs_position", None)
    targets_pos = feature_map.pop("targets_position", None)
    if inputs is not None:
      feature_map["targets"] = inputs
    if targets is not None:
      feature_map["inputs"] = targets
    if inputs_seg is not None:
      feature_map["targets_segmentation"] = inputs_seg
    if targets_seg is not None:
      feature_map["inputs_segmentation"] = targets_seg
    if inputs_pos is not None:
      feature_map["targets_position"] = inputs_pos
    if targets_pos is not None:
      feature_map["inputs_position"] = targets_pos

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
              shuffle_buffer_size=1024,
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
      hparams: HParams; hparams to be passed to
        Problem.preprocess_example and Problem.hparams. If None, will use a
        default set that is a no-op.
      preprocess: bool, whether to map the Dataset through
        Problem.preprocess_example.
      dataset_split: DatasetSplit, which split to read data
        from (TRAIN:"-train", EVAL:"-dev", "test":"-test"). Defaults to mode.
      shard: int, if provided, will only read data from the specified shard.
      partition_id: integer - which partition of the dataset to read from
      num_partitions: how many partitions in the dataset
      shuffle_buffer_size: if shuffle_files is True, this is the buffer size
        used to shuffle records.
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
      """Reads files from a string tensor or a dataset of filenames."""
      # Load records from file(s) with an 8MiB read buffer.
      dataset = tf.data.TFRecordDataset(filenames, buffer_size=8 * 1024 * 1024)
      # Decode.
      dataset = dataset.map(self.decode_example, num_parallel_calls=num_threads)
      # Preprocess if requested.
      # Note that preprocessing should happen per-file as order may matter.
      if preprocess:
        dataset = self.preprocess(dataset, mode, hparams,
                                  interleave=shuffle_files)
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
      mlperf_log.transformer_print(key=mlperf_log.INPUT_ORDER)
      random.shuffle(data_files)

    dataset = tf.data.Dataset.from_tensor_slices(tf.constant(data_files))
    # Create data-set from files by parsing, pre-processing and interleaving.
    if shuffle_files:
      dataset = dataset.apply(
          tf.data.experimental.parallel_interleave(
              _load_records_and_preprocess, sloppy=True, cycle_length=8))
    else:
      dataset = _load_records_and_preprocess(dataset)

    dataset = dataset.map(
        self.maybe_reverse_and_copy, num_parallel_calls=num_threads)
    dataset = dataset.take(max_records)

    ## Shuffle records only for training examples.
    if shuffle_files and is_training:
      dataset = dataset.shuffle(shuffle_buffer_size)
    if hparams.get("pack_dataset", False):
      dataset = generator_utils.pack_dataset(
          dataset, hparams.max_length, keys=["inputs", "targets"],
          use_custom_ops=hparams.get("use_custom_ops", False))
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
    if self.has_inputs:
      in_id = hp.input_space_id
    out_id = hp.target_space_id

    features = collections.defaultdict(FeatureInfo)
    for feature_name, modality_cls in six.iteritems(hp.modality):
      finfo = features[feature_name]
      finfo.modality = modality_cls
      finfo.vocab_size = hp.vocab_size[feature_name]

    vocabs = hp.vocabulary
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
                              force_repeat=False,
                              prevent_repeat=False,
                              dataset_kwargs=None):
    """Return input_fn wrapped for Estimator."""

    def estimator_input_fn(params, config):
      return self.input_fn(
          mode,
          hparams,
          data_dir=data_dir,
          params=params,
          config=config,
          force_repeat=force_repeat,
          prevent_repeat=prevent_repeat,
          dataset_kwargs=dataset_kwargs)

    return estimator_input_fn

  def _dataset_partition(self, mode, config, params):
    """Which part of the training data to read.

    If there are multiple parallel calls to input_fn (multiple TPU hosts),
    then we want each one to read from a separate partition of the training
    data.

    Args:
      mode: tf.estimator.ModeKeys
      config: RunConfig
      params: A dict that contains parameters.
    Returns:
      partition_id: an integer
      num_partitions: an integer
    """
    if mode != tf.estimator.ModeKeys.TRAIN or not hasattr(config, "tpu_config"):
      # Reset in the case when using TPU but alternating TRAIN and EVAL.
      self._next_partition_id = 0
      return 0, 1
    phift = config.tpu_config.per_host_input_for_training
    # This is the mesh-tensorflow case.
    if (hasattr(tpu_config.InputPipelineConfig, "BROADCAST") and
        phift == tpu_config.InputPipelineConfig.BROADCAST):
      return 0, 1
    if phift:
      num_hosts = (params["context"].num_hosts if "context" in params
                   else config.tpu_config.num_shards // 8)
      num_partitions = max(num_hosts, 1)
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
               force_repeat=False,
               prevent_repeat=False,
               dataset_kwargs=None):
    """Builds input pipeline for problem.

    Args:
      mode: tf.estimator.ModeKeys
      hparams: HParams, model hparams
      data_dir: str, data directory; if None, will use hparams.data_dir
      params: dict, may include "batch_size"
      config: RunConfig; should have the data_parallelism attribute if not using
        TPU
      force_repeat: bool, whether to repeat the data even if not training
      prevent_repeat: bool, whether to not repeat when in training mode.
        Overrides force_repeat.
      dataset_kwargs: dict, if passed, will pass as kwargs to self.dataset
        method when called

    Returns:
      (features_dict<str name, Tensor feature>, Tensor targets)
    """
    partition_id, num_partitions = self._dataset_partition(mode, config, params)
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    if config and config.use_tpu:
      num_threads = 64
    else:
      num_threads = data_reader.cpu_count() if is_training else 1
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
    return data_reader.input_fn(
        self.dataset(**dataset_kwargs),
        self.filepattern(data_dir, mode),
        self.skip_random_fraction_when_training,
        self.batch_size_means_tokens,
        self.get_hparams().batch_size_multiplier,
        self.max_length(hparams),
        mode,
        hparams,
        data_dir=data_dir,
        params=params,
        config=config,
        force_repeat=force_repeat,
        prevent_repeat=prevent_repeat)

  @property
  def export_assets(self):
    """Assets to export with the model.

    This property contains a dictionary of assets, such as vocabulary files,
    that should be exported together with the model, or None if no assets
    are needed.
    """

    return None

  def serving_input_fn(self, hparams, decode_hparams=None, use_tpu=False):
    """Input fn for serving export, starting from serialized example."""
    mode = tf.estimator.ModeKeys.PREDICT
    serialized_example = tf.placeholder(
        dtype=tf.string, shape=[None], name="serialized_example")
    dataset = tf.data.Dataset.from_tensor_slices(serialized_example)
    dataset = dataset.map(self.decode_example)
    dataset = dataset.map(lambda ex: self.preprocess_example(ex, mode, hparams))
    dataset = dataset.map(data_reader.cast_ints_to_int32)

    if use_tpu:
      padded_shapes = data_reader.pad_for_tpu(dataset.output_shapes, hparams,
                                              hparams.max_length)
      batch_size = 1 if not decode_hparams else getattr(decode_hparams,
                                                        "batch_size", 1)
      dataset = dataset.padded_batch(
          batch_size, padded_shapes, drop_remainder=False)
      dataset = dataset.map(
          functools.partial(data_reader.pad_batch, batch_multiple=batch_size))
    else:
      dataset = dataset.padded_batch(
          tf.shape(serialized_example, out_type=tf.int64)[0],
          dataset.output_shapes)

    dataset = dataset.map(data_reader.standardize_shapes)
    features = tf.data.experimental.get_single_element(dataset)

    if self.has_inputs:
      features.pop("targets", None)

    return tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors=serialized_example)


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
  p.modality["targets"] = p.modality["inputs"]
  # Duplicate input vocab size.
  p.vocab_size["targets"] = p.vocab_size["inputs"]
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
  # TODO(trandustin): Note this assumes target modalities have feature name
  # 'target', and each intended feature to swap has feature name 'input'.
  # In the future, remove need for this behavior.
  reversed_modality = {}
  for feature_name in p.modality:
    reversed_feature_name = feature_name.replace("target", "input")
    if "target" in feature_name and reversed_feature_name in p.modality:
      reversed_modality[feature_name] = p.modality[reversed_feature_name]
      reversed_modality[reversed_feature_name] = p.modality[feature_name]
    else:
      reversed_modality[feature_name] = p.modality[feature_name]

  p.modality = reversed_modality

  # Swap vocab sizes.
  reversed_vocab_size = {}
  for feature_name in p.vocab_size:
    reversed_feature_name = feature_name.replace("target", "input")
    if "target" in feature_name and reversed_feature_name in p.vocab_size:
      reversed_vocab_size[feature_name] = p.vocab_size[reversed_feature_name]
      reversed_vocab_size[reversed_feature_name] = p.vocab_size[feature_name]
    else:
      reversed_vocab_size[feature_name] = p.vocab_size[feature_name]

  p.vocab_size = reversed_vocab_size

  # Swap vocabularies.
  input_vocabulary = p.vocabulary.pop("inputs", None)
  target_vocabulary = p.vocabulary.pop("targets", None)
  if input_vocabulary is not None:
    p.vocabulary["targets"] = input_vocabulary
  if target_vocabulary is not None:
    p.vocabulary["inputs"] = target_vocabulary

  # Swap input/target space ids.
  input_space_id = p.input_space_id
  target_space_id = p.target_space_id
  if input_space_id is not None:
    p.target_space_id = input_space_id
  else:
    p.target_space_id = SpaceID.GENERIC
  if target_space_id is not None:
    p.input_space_id = target_space_id
  else:
    p.input_space_id = SpaceID.GENERIC

  # Mark that p was reversed.
  p.was_reversed = True


def _default_hparams():
  """A set of basic model hyperparameters."""
  return HParams(
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

      # Modalities used to map from features to a space compatible with
      # chosen model architecture. It comprises key-value pairs of a feature
      # name (str) and its modality type.
      modality={},
      vocab_size={},

      # Identifiers used to tell the model which input/target space will be
      # expected. For example, it can tell that we expect French as characters
      # as output, or Spanish as sound. Spaces defined as constants in SpaceID
      # class.
      input_space_id=SpaceID.GENERIC,
      target_space_id=SpaceID.GENERIC)


def problem_hparams_to_features(problem_hparams):
  input_space_id, target_space_id = 0, 0
  if problem_hparams:
    input_space_id = problem_hparams.input_space_id
    target_space_id = problem_hparams.target_space_id
  return {
      "input_space_id": input_space_id,
      "target_space_id": target_space_id,
  }
