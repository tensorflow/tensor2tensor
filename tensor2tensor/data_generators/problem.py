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
from tensor2tensor.utils import data_reader
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
    example["targets"] = tf.reshape(
        example["targets"], [-1, hparams.split_to_length, 1, 1])
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
    return (
        model_hparams.split_to_length or
        model_hparams.max_length or
        model_hparams.batch_size)

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
              repeat=None,
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
      repeat: whether to repeat the Dataset. Default behavior is to repeat if
        mode == TRAIN.
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
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    repeat = repeat or repeat is None and is_training
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
    dataset = tf.data.Dataset.list_files(data_filepattern)

    if shuffle_files:
      dataset = dataset.shuffle(buffer_size=1024)

    def _load_records(filename):
      return tf.data.TFRecordDataset(filename, buffer_size=16 * 1000 * 1000)

    if hasattr(tf.contrib.data, "parallel_interleave"):
      interleave = lambda ds, fn: ds.apply(  # pylint: disable=g-long-lambda
          tf.contrib.data.parallel_interleave(
              fn, sloppy=is_training, cycle_length=16))
    else:
      interleave = lambda ds, fn: ds.interleave(fn, cycle_length=16)

    dataset = interleave(dataset, _load_records)

    if repeat:
      dataset = dataset.repeat()

    if shuffle_files:
      # Skip a random fraction at the beginning of the stream.  The skip is
      # essential for synchronous highly-parallel training to avoid multiple
      # replicas reading the same data in lock-step.
      data_files = tf.contrib.slim.parallel_reader.get_data_files(
          data_filepattern)
      num_skip = random.randint(0, _file_num_records_cached(data_files[0]))
      dataset = dataset.skip(num_skip)

    def _maybe_reverse_and_copy(example):
      self.maybe_reverse_features(example)
      self.maybe_copy_features(example)
      return example

    def _preprocess(example):
      examples = self.preprocess_example(example, mode, hparams)
      if not isinstance(examples, tf.data.Dataset):
        examples = tf.data.Dataset.from_tensors(examples)
      return examples

    dataset = dataset.map(self.decode_example, num_parallel_calls=num_threads)

    if preprocess:
      dataset = interleave(dataset, _preprocess)

    dataset = dataset.map(
        _maybe_reverse_and_copy, num_parallel_calls=num_threads)

    if output_buffer_size:
      dataset = dataset.prefetch(output_buffer_size)

    return dataset

  def decode_example(self, serialized_example):
    """Return a dict of Tensors from a serialized tensorflow.Example."""
    data_fields, data_items_to_decoders = self.example_reading_spec()
    if data_items_to_decoders is None:
      data_items_to_decoders = {
          field: tf.contrib.slim.tfexample_decoder.Tensor(field)
          for field in data_fields
      }

    decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
        data_fields, data_items_to_decoders)

    decode_items = list(data_items_to_decoders)
    decoded = decoder.decode(serialized_example, items=decode_items)
    return dict(zip(decode_items, decoded))

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

  def make_estimator_input_fn(self, mode, hparams, data_dir=None,
                              dataset_kwargs=None):
    """Return input_fn wrapped for Estimator."""

    def estimator_input_fn(params, config):
      return self.input_fn(mode, hparams, data_dir=data_dir, params=params,
                           config=config, dataset_kwargs=dataset_kwargs)

    return estimator_input_fn

  def input_fn(self, mode, hparams, data_dir=None, params=None, config=None,
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
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    if config.use_tpu:
      num_threads = 32
    else:
      num_threads = 4 if is_training else 1

    max_length = self.max_length(hparams)

    def tpu_valid_size(example):
      return data_reader.example_valid_size(example, hparams.min_length,
                                            max_length)

    def gpu_valid_size(example):
      drop_long_sequences = is_training or hparams.eval_drop_long_sequences
      return data_reader.example_valid_size(
          example,
          hparams.min_length,
          max_length if drop_long_sequences else 10**9)

    def define_shapes(example):
      batch_size = config and config.use_tpu and params["batch_size"]
      return standardize_shapes(example, batch_size=batch_size)

    # Read and preprocess
    data_dir = data_dir or hparams.data_dir

    dataset_kwargs = dataset_kwargs or {}
    dataset_kwargs.update({
        "mode": mode,
        "data_dir": data_dir,
        "num_threads": num_threads,
        "hparams": hparams})

    dataset = self.dataset(**dataset_kwargs)
    dataset = dataset.map(
        data_reader.cast_int64_to_int32, num_parallel_calls=num_threads)
    if is_training:
      dataset = dataset.repeat(None)

    if self.batch_size_means_tokens:
      batch_size_means_tokens = True
    else:
      if _are_shapes_fully_defined(dataset.output_shapes):
        batch_size_means_tokens = False
      else:
        tf.logging.warning(
            "Shapes are not fully defined. Assuming batch_size means tokens. "
            "You should probably override batch_size_means_tokens() "
            "in your problem subclass")
        batch_size_means_tokens = True

    # Batching
    if not batch_size_means_tokens:
      # Batch size means examples per datashard.
      if config and config.use_tpu:
        # on TPU, we use params["batch_size"], which specifies the number of
        # examples across all datashards
        tpu_batch_size = params["batch_size"]
        dataset = dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(tpu_batch_size))
      else:
        num_shards = (config and config.data_parallelism.n) or 1
        dataset = dataset.batch(hparams.batch_size * num_shards)
    else:
      # batch_size means tokens per datashard
      if config and config.use_tpu:
        # On TPU, pad to max_length
        dataset = dataset.filter(tpu_valid_size)
        padded_shapes = _fill_shape_nones(
            dataset.output_shapes, none_filler=max_length)
        # on TPU, we use params["batch_size"], which specifies the number of
        # examples across all datashards
        dataset = dataset.apply(
            tf.contrib.data.padded_batch_and_drop_remainder(
                params["batch_size"], padded_shapes))
      else:
        # On GPU, bucket by length
        dataset = dataset.filter(gpu_valid_size)
        batching_scheme = data_reader.hparams_to_batching_scheme(
            hparams,
            shard_multiplier=(config and config.data_parallelism.n) or 1,
            length_multiplier=self.get_hparams().batch_size_multiplier)
        if hparams.use_fixed_batch_size:
          # Here  batch_size really means examples per datashard.
          batching_scheme["batch_sizes"] = [hparams.batch_size]
          batching_scheme["boundaries"] = []
        dataset = data_reader.bucket_by_sequence_length(
            dataset,
            data_reader.example_length,
            batching_scheme["boundaries"],
            batching_scheme["batch_sizes"])

        if not is_training:
          def _pad_batch(features):
            if not config or config.data_parallelism.n <= 1:
              return features
            tf.logging.warn(
                "Padding the batch to ensure that remainder eval batches have "
                "a batch size divisible by the number of data shards. This may "
                "lead to incorrect metrics for non-zero-padded features, e.g. "
                "images. Use a single datashard (i.e. 1 GPU) in that case.")
            return pad_batch(features, config.data_parallelism.n)

          dataset = dataset.map(_pad_batch, num_parallel_calls=num_threads)

    dataset = dataset.map(define_shapes, num_parallel_calls=num_threads)
    dataset = dataset.prefetch(2)
    features = dataset.make_one_shot_iterator().get_next()
    if not config or not config.use_tpu:
      _summarize_features(features, (config and config.data_parallelism.n) or 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
      features["infer_targets"] = features["targets"]
      features["targets"] = None
      # This is because of a bug in the Estimator that short-circuits prediction
      # if it doesn't see a QueueRunner. DummyQueueRunner implements the
      # minimal expected interface but does nothing.
      tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS,
                           data_reader.DummyQueueRunner())

    return features, features["targets"]

  def serving_input_fn(self, hparams):
    """Input fn for serving export, starting from serialized example."""
    mode = tf.estimator.ModeKeys.PREDICT
    serialized_example = tf.placeholder(
        dtype=tf.string, shape=[None], name="serialized_example")
    dataset = tf.data.Dataset.from_tensor_slices(serialized_example)
    dataset = dataset.map(self.decode_example)
    dataset = dataset.map(lambda ex: self.preprocess_example(ex, mode, hparams))
    dataset = dataset.map(data_reader.cast_int64_to_int32)
    dataset = dataset.padded_batch(1000, dataset.output_shapes)
    dataset = dataset.map(standardize_shapes)
    features = tf.contrib.data.get_single_element(dataset)

    if self.has_inputs:
      features.pop("targets", None)

    return tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors=serialized_example)


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

  @property
  def batch_size_means_tokens(self):
    return True

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
  def packed_length(self):
    """Pack multiple examples into a single example of constant length.

    This is useful for TPU training.  See generator_utils.pack_examples().

    Returns:
      an optional integer
    """
    return None

  def max_length(self, model_hparams):
    """Maximum sequence length."""
    if self.packed_length:
      return self.packed_length
    return super(Text2TextProblem, self).max_length(model_hparams)

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

  def _maybe_pack_examples(self, generator):
    """Helper to generate_data()."""
    if self.packed_length:
      return generator_utils.pack_examples(
          generator, self.has_inputs, self.packed_length,
          chop_long_sequences=not self.has_inputs)
    else:
      return generator

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    train_paths = self.training_filepaths(
        data_dir, self.num_shards, shuffled=False)
    dev_paths = self.dev_filepaths(
        data_dir, self.num_dev_shards, shuffled=False)
    if self.use_train_shards_for_dev:
      all_paths = train_paths + dev_paths
      generator_utils.generate_files(
          self._maybe_pack_examples(self.generator(data_dir, tmp_dir, True)),
          all_paths)
      generator_utils.shuffle_dataset(all_paths)
    else:
      generator_utils.generate_dataset_and_shuffle(
          self._maybe_pack_examples(self.generator(data_dir, tmp_dir, True)),
          train_paths,
          self._maybe_pack_examples(self.generator(data_dir, tmp_dir, False)),
          dev_paths)

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
    if self.packed_length:
      identity = (registry.Modalities.GENERIC, None)
      if self.has_inputs:
        p.input_modality["inputs_segmentation"] = identity
        p.input_modality["inputs_position"] = identity
      p.input_modality["targets_segmentation"] = identity
      p.input_modality["targets_position"] = identity

  def example_reading_spec(self):
    data_fields = {
        "targets": tf.VarLenFeature(tf.int64)
    }
    if self.has_inputs:
      data_fields["inputs"] = tf.VarLenFeature(tf.int64)

    if self.packed_length:
      if self.has_inputs:
        data_fields["inputs_segmentation"] = tf.VarLenFeature(tf.int64)
        data_fields["inputs_position"] = tf.VarLenFeature(tf.int64)
      data_fields["targets_segmentation"] = tf.VarLenFeature(tf.int64)
      data_fields["targets_position"] = tf.VarLenFeature(tf.int64)

    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)

  def eval_metrics(self):
    return [
        metrics.Metrics.ACC, metrics.Metrics.ACC_TOP5,
        metrics.Metrics.ACC_PER_SEQ, metrics.Metrics.NEG_LOG_PERPLEXITY,
        metrics.Metrics.APPROX_BLEU, metrics.Metrics.ROUGE_2_F,
        metrics.Metrics.ROUGE_L_F
    ]


class ChoppedTextProblem(Text2TextProblem):
  """Tokenize and chop text files into fixed-length language-modeling examples.

  The input data is a set of text files, as specified by
  self.train_text_filepaths() and self.dev_text_filepaths().

  The text is tokenized using a SubwordTextEncoder, and
  then split into examples, each of length self.sequence_length().
  """

  def train_text_filepaths(self, tmp_dir):
    """Local filepaths of text files containing training data.

    This function may want to download the files if they do not exist.

    Args:
      tmp_dir: a string
    Returns:
      a list of strings.
    """
    raise NotImplementedError()

  def dev_text_filepaths(self, tmp_dir):
    """Local filepaths of text files containing dev data.

    This function may want to download the files if they do not exist.

    Args:
      tmp_dir: a string
    Returns:
      a list of strings.
    """
    raise NotImplementedError()

  @property
  def sequence_length(self):
    """Length of each example (in tokens)."""
    raise NotImplementedError()

  def max_length(self, model_hparams):
    return model_hparams.split_to_length or self.sequence_length

  @property
  def is_character_level(self):
    return False

  def text_filepaths_for_task(self, tmp_dir, task_id):
    """List of input filepaths for a particular training or dev shard.

    Args:
      tmp_dir: a string
      task_id: an integer less than self.num_shards
    Returns:
      a list of tuples (filepath, start_pos, num_bytes)
    """
    assert task_id >= 0
    assert task_id < self.num_train_shards + self.num_dev_shards
    if task_id < self.num_train_shards:
      return [f for i, f in enumerate(self.train_text_filepaths(tmp_dir))
              if i % self.num_train_shards == task_id]
    else:
      return [f for i, f in enumerate(self.dev_text_filepaths(tmp_dir))
              if i % self.num_dev_shards == task_id - self.num_train_shards]

  def filepath_to_unicode_strings(self, filepath):
    """Read text out of an input file.

    The default just reads the text, converts to unicode and yields one
    unicode string.

    Subclasses can override this function in order to preprocess, and can
    yield any number of strings.

    Args:
      filepath: a string
    Yields:
      unicode strings.
    """
    f = tf.gfile.Open(filepath)
    b = f.read()
    yield to_unicode_ignore_erros(b)

  def file_generator(self,
                     filepaths,
                     max_chars_per_file=None,
                     max_chars_total=None):
    """Read complete text of input files and yield unicode strings.

    By default, one unicode string is produced per file, but this is
    not guaranteed, since subclasses can override
    filepath_to_unicode_strings().

    max_chars_per_file and max_chars_total can also be specified, in which
    case some strings may be truncated or dropped to limit the total
    amount of output.

    Args:
      filepaths: a list of strings
      max_chars_per_file: an optional integer
      max_chars_total: an optional integer
    Yields:
      unicode strings
    """
    chars_total = 0
    for fname in filepaths:
      chars_this_file = 0
      tf.logging.info("reading file %s" % fname)
      for text in self.filepath_to_unicode_strings(fname):
        if (max_chars_per_file and chars_this_file + len(text)
            > max_chars_per_file):
          text = text[:max_chars_per_file - chars_this_file]
        if max_chars_total and chars_total + len(text) > max_chars_total:
          text = text[:max_chars_total - chars_total]
        chars_total += len(text)
        chars_this_file += len(text)
        if text:
          yield text
        if max_chars_total and chars_total >= max_chars_total:
          return
        if max_chars_per_file and chars_this_file >= max_chars_per_file:
          break

  def example_generator(self, encoder, tmp_dir, task_id):
    """Generator for examples.

    Args:
      encoder: a TextEncoder
      tmp_dir: a string
      task_id: an integer
    Yields:
      feature dictionaries
    """
    filepaths = self.text_filepaths_for_task(tmp_dir, task_id)
    if task_id >= self.num_train_shards:
      # this is dev data - limit the total length.
      max_chars_per_file = self.max_dev_chars // (
          self.num_dev_shards * len(filepaths))
    else:
      max_chars_per_file = None
    tokens = []
    for ftext in self.file_generator(
        filepaths, max_chars_per_file=max_chars_per_file):
      tokens.extend(encoder.encode(ftext))
      pos = 0
      while pos + self.sequence_length <= len(tokens):
        yield {"inputs": [0], "targets": tokens[pos:pos + self.sequence_length]}
        pos += self.sequence_length
      if pos > 0:
        tokens = tokens[pos:]
    if self.remainder_policy == "pad":
      if tokens:
        targets = tokens + [0] * (self.sequence_length - len(tokens))
        yield {"inputs": [0], "targets": targets}
    else:
      assert self.remainder_policy == "drop"

  @property
  def remainder_policy(self):
    """What to do with leftover tokens.

    Returns:
      a string - either "pad" or  "drop".
    """
    return "pad"

  def prepare_to_generate(self, data_dir, tmp_dir):
    """Make sure that the data is prepared and the vocab is generated."""
    self.get_or_generate_vocab(data_dir, tmp_dir)
    self.train_text_filepaths(tmp_dir)
    self.dev_text_filepaths(tmp_dir)

  def get_or_generate_vocab(self, data_dir, tmp_dir):
    return generator_utils.get_or_generate_vocab_inner(
        data_dir, self.vocab_file, self.targeted_vocab_size,
        self.file_generator(
            self.train_text_filepaths(tmp_dir),
            max_chars_total=self.max_chars_for_vocab))

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    """Generates training/dev data.

    Args:
      data_dir: a string
      tmp_dir: a string
      task_id: an optional integer
    Returns:
      shard or shards for which data was generated.
    """
    tf.logging.info("generate_data task_id=%s" % task_id)
    encoder = self.get_or_generate_vocab(data_dir, tmp_dir)
    assert task_id >= 0 and task_id < self.num_generate_tasks
    if task_id < self.num_train_shards:
      out_file = self.training_filepaths(
          data_dir, self.num_train_shards, shuffled=False)[task_id]
    else:
      out_file = self.dev_filepaths(
          data_dir, self.num_dev_shards,
          shuffled=False)[task_id - self.num_train_shards]
    generator_utils.generate_files(
        self.example_generator(encoder, tmp_dir, task_id), [out_file])
    generator_utils.shuffle_dataset([out_file])

  @property
  def max_chars_for_vocab(self):
    """Number of characters of training data to use for generating vocab."""
    return 10 ** 7

  @property
  def target_space_id(self):
    return SpaceID.EN_TOK

  @property
  def num_train_shards(self):
    return 100

  @property
  def num_dev_shards(self):
    return 1

  @property
  def max_dev_chars(self):
    """Limit dev set to at most this many characters (default 10M)."""
    return 10 ** 7

  @property
  def multiprocess_generate(self):
    return True

  @property
  def num_generate_tasks(self):
    return self.num_train_shards + self.num_dev_shards

  @property
  def vocab_name(self):
    raise NotImplementedError()

  @property
  def use_subword_tokenizer(self):
    return True

  @property
  def has_inputs(self):
    return False

  def eval_metrics(self):
    return [
        metrics.Metrics.ACC, metrics.Metrics.NEG_LOG_PERPLEXITY
    ]


def to_unicode_ignore_erros(s):
  return (unicode(s, "utf-8", errors="ignore") if six.PY2 else
          s.decode("utf-8", "ignore"))


def _are_shapes_fully_defined(shapes_dict):
  for shape in shapes_dict.values():
    if not shape.is_fully_defined():
      return False
  return True


def _fill_shape_nones(shapes_dict, none_filler=None):
  padded_shapes = {}
  for key, shape in six.iteritems(shapes_dict):
    padded_shapes[key] = [
        (dim if dim is not None else none_filler) for dim in shape.as_list()
    ]
  return padded_shapes


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
