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

"""Base classes for text-based Problems.

* Text2TextProblem: input=text, target=text.
* Text2ClassProblem: input=text, target=class.
* Text2RealProblem: input=text, target=float.
* Text2SelfProblem (for language modeling): target=text
* QuestionAndContext2TextProblem: input=text, context=text, target=text.

The Text2TextTmpDir problem allows you to train without defining a problem. It
expects you to format your data in a particular way and put it in tmp_dir. See
its docstring.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


class VocabType(object):
  """Available text vocabularies."""
  CHARACTER = "character"
  SUBWORD = "subwords"
  TOKEN = "tokens"


class Text2TextProblem(problem.Problem):
  """Base class for text-to-text problems.

  Subclasses only must override `generate_samples` and `is_generate_per_split`.
  See the "Subclass interface" code block below to see what else subclasses can
  override.
  """

  # START: Subclass interface
  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 100,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  @property
  def is_generate_per_split(self):
    """A single call to `generate_samples` generates for all `dataset_splits`.

    Set to True if you already have distinct subsets of data for each dataset
    split specified in `self.dataset_splits`. `self.generate_samples` will be
    called once for each split.

    Set to False if you have a unified dataset that you'd like to have split out
    into training and evaluation data automatically. `self.generate_samples`
    will be called only once and the data will be sharded across the dataset
    splits specified in `self.dataset_splits`.

    Returns:
      bool
    """
    raise NotImplementedError()

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Generate samples of input text and target text pairs.

    Each yielded dict will be made into a single example. The values should be
    raw text. The Problem will generate a vocabulary and encode the raw text as
    integers as part of the data generation process.

    This method is typically called once per split in `self.dataset_splits`
    unless `self.is_generate_per_split=False`.

    Args:
      data_dir: final data directory. Typically only used in this method to copy
        over user-supplied vocab files (for example, if vocab_type ==
        VocabType.TOKEN).
      tmp_dir: temporary directory that you can use for downloading and scratch.
      dataset_split: problem.DatasetSplit, which data split to generate samples
        for (for example, training and evaluation).

    Yields:
      {"inputs": text, "targets": text}
    """
    raise NotImplementedError()

  @property
  def vocab_type(self):
    """What kind of vocabulary to use.

    `VocabType`s:
      * `SUBWORD`: `SubwordTextEncoder`, an invertible wordpiece vocabulary.
        Must provide `self.approx_vocab_size`. Generates the vocabulary based on
        the training data. To limit the number of samples the vocab generation
        looks at, override `self.max_samples_for_vocab`. Recommended and
        default.
      * `CHARACTER`: `ByteTextEncoder`, encode raw bytes.
      * `TOKEN`: `TokenTextEncoder`, vocabulary based on a file. Must provide a
        vocabulary file yourself (`TokenTextEncoder.store_to_file`) because one
        will not be generated for you. The vocab file should be stored in
        `data_dir/` with the name specified by `self.vocab_filename`.

    Returns:
      VocabType constant
    """
    return VocabType.SUBWORD

  @property
  def approx_vocab_size(self):
    """Approximate vocab size to generate. Only for VocabType.SUBWORD."""
    return 2**15  # ~32k

  @property
  def additional_reserved_tokens(self):
    """Additional reserved tokens. Only for VocabType.SUBWORD.

    Returns:
      List of str tokens that will get vocab ids 2+ (0 and 1 are reserved for
      padding and end-of-string).
    """
    return []

  @property
  def oov_token(self):
    """Out of vocabulary token. Only for VocabType.TOKEN."""
    return None

  @property
  def max_samples_for_vocab(self):
    """How many samples from `generate_samples` to look at for vocab generation.

    Only applies if self.vocab_type == VocabType.SUBWORD.

    If None, look at all training samples.

    Returns:
      None or int.
    """
    return None

  @property
  def packed_length(self):
    """Pack multiple examples into a single example of constant length.

    This is useful for TPU training to reduce the fraction of padding tokens.
    See generator_utils.pack_examples.

    Returns:
      None or int
    """
    return None

  @property
  def packed_spacing(self):
    """If this is a packed dataset, how much padding to insert between examples.

    Returns:
      int
    """
    return 0

  # END: Subclass interface

  @property
  def has_inputs(self):
    return True

  def max_length(self, model_hparams):
    return (self.packed_length or
            super(Text2TextProblem, self).max_length(model_hparams))

  def feature_encoders(self, data_dir):
    encoder = self.get_or_create_vocab(data_dir, None, force_get=True)
    encoders = {"targets": encoder}
    if self.has_inputs:
      encoders["inputs"] = encoder
    return encoders

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    for i, sample in enumerate(
        self.generate_samples(data_dir, tmp_dir, problem.DatasetSplit.TRAIN)):
      if self.has_inputs:
        yield sample["inputs"]
      yield sample["targets"]
      if self.max_samples_for_vocab and (i + 1) >= self.max_samples_for_vocab:
        break

  @property
  def vocab_filename(self):
    other_problem = self.use_vocab_from_other_problem
    if other_problem:
      return other_problem.vocab_filename
    if self.vocab_type == VocabType.SUBWORD:
      return "vocab.%s.%d.%s" % (self.dataset_filename(),
                                 self.approx_vocab_size,
                                 VocabType.SUBWORD)
    else:
      return "vocab.%s.%s" % (self.dataset_filename(), VocabType.TOKEN)

  @property
  def use_vocab_from_other_problem(self):
    """Optional - use the vocabulary from a different problem.

    TODO(noam): problems should override this method instead of overriding
    vocab_filename(), so as to generate the correct vocabulary. Fix everywhere.

    Returns:
       a Text2TextProblem instance or None
    """
    return None

  def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
    if self.vocab_type == VocabType.CHARACTER:
      encoder = text_encoder.ByteTextEncoder()
    elif self.vocab_type == VocabType.SUBWORD:
      if force_get:
        vocab_filepath = os.path.join(data_dir, self.vocab_filename)
        encoder = text_encoder.SubwordTextEncoder(vocab_filepath)
      else:
        other_problem = self.use_vocab_from_other_problem
        if other_problem:
          return other_problem.get_or_create_vocab(data_dir, tmp_dir, force_get)
        encoder = generator_utils.get_or_generate_vocab_inner(
            data_dir, self.vocab_filename, self.approx_vocab_size,
            self.generate_text_for_vocab(data_dir, tmp_dir),
            max_subtoken_length=self.max_subtoken_length,
            reserved_tokens=(
                text_encoder.RESERVED_TOKENS + self.additional_reserved_tokens))
    elif self.vocab_type == VocabType.TOKEN:
      vocab_filename = os.path.join(data_dir, self.vocab_filename)
      encoder = text_encoder.TokenTextEncoder(vocab_filename,
                                              replace_oov=self.oov_token)
    else:
      raise ValueError(
          "Unrecognized VocabType: %s" % str(self.vocab_type))
    return encoder

  def _pack_fn(self):
    """For packed datasets, returns a function to pack examples.

    Returns:
      None or a function from list of TFRecords to list of TFRecords
    """
    if not self.packed_length:
      return None
    def my_fn(records):
      """Function from list of TFRecords to list of TFRecords."""
      examples = []
      for record in records:
        x = tf.train.Example()
        x.ParseFromString(record)
        example_dict = {}
        if self.has_inputs:
          example_dict["inputs"] = [
              int(i) for i in x.features.feature["inputs"].int64_list.value]
        example_dict["targets"] = [
            int(i) for i in x.features.feature["targets"].int64_list.value]
        examples.append(example_dict)
      examples = list(self._maybe_pack_examples(examples))
      return [
          generator_utils.to_example(x).SerializeToString() for x in examples]
    return my_fn

  def _maybe_pack_examples(self, generator):
    """Wraps generator with packer if self.packed_length."""
    if not self.packed_length:
      return generator
    return generator_utils.pack_examples(
        generator,
        self.has_inputs,
        self.packed_length,
        spacing=self.packed_spacing,
        chop_long_sequences=not self.has_inputs)

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    if dataset_split == problem.DatasetSplit.TRAIN:
      mlperf_log.transformer_print(key=mlperf_log.PREPROC_TOKENIZE_TRAINING)
    elif dataset_split == problem.DatasetSplit.EVAL:
      mlperf_log.transformer_print(key=mlperf_log.PREPROC_TOKENIZE_EVAL)

    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    encoder = self.get_or_create_vocab(data_dir, tmp_dir)
    return text2text_generate_encoded(generator, encoder,
                                      has_inputs=self.has_inputs,
                                      inputs_prefix=self.inputs_prefix,
                                      targets_prefix=self.targets_prefix)

  @property
  def max_subtoken_length(self):
    """Maximum subtoken length when generating vocab.

    SubwordTextEncoder vocabulary building is quadratic-time wrt this variable,
    setting it to None uses the length of the longest token in the corpus.

    Returns:
      an integer or None
    """
    return 200

  @property
  def batch_size_means_tokens(self):
    return True

  @property
  def already_shuffled(self):
    return False

  @property
  def inputs_prefix(self):
    """String to prepend to inputs before tokenization."""
    return ""

  @property
  def targets_prefix(self):
    """String to prepend to targets before tokenization."""
    return ""

  def generate_data(self, data_dir, tmp_dir, task_id=-1):

    filepath_fns = {
        problem.DatasetSplit.TRAIN: self.training_filepaths,
        problem.DatasetSplit.EVAL: self.dev_filepaths,
        problem.DatasetSplit.TEST: self.test_filepaths,
    }

    split_paths = [(split["split"], filepath_fns[split["split"]](
        data_dir, split["shards"], shuffled=self.already_shuffled))
                   for split in self.dataset_splits]
    all_paths = []
    for _, paths in split_paths:
      all_paths.extend(paths)

    if self.is_generate_per_split:
      for split, paths in split_paths:
        generator_utils.generate_files(
            self.generate_encoded_samples(data_dir, tmp_dir, split), paths)
    else:
      generator_utils.generate_files(
          self.generate_encoded_samples(
              data_dir, tmp_dir, problem.DatasetSplit.TRAIN), all_paths)

    generator_utils.shuffle_dataset(all_paths, extra_fn=self._pack_fn())

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.stop_at_eos = int(True)

    p.modality = {"targets": modalities.ModalityType.SYMBOL}
    p.vocab_size = {"targets": self._encoders["targets"].vocab_size}
    if self.has_inputs:
      p.modality["inputs"] = modalities.ModalityType.SYMBOL
      p.vocab_size["inputs"] = self._encoders["inputs"].vocab_size
    if self.vocab_type == VocabType.CHARACTER:
      p.loss_multiplier = 2.0

    if self.packed_length:
      if self.has_inputs:
        p.modality["inputs_segmentation"] = modalities.ModalityType.IDENTITY
        p.modality["inputs_position"] = modalities.ModalityType.IDENTITY
        p.vocab_size["inputs_segmentation"] = None
        p.vocab_size["inputs_position"] = None
      p.modality["targets_segmentation"] = modalities.ModalityType.IDENTITY
      p.modality["targets_position"] = modalities.ModalityType.IDENTITY
      p.vocab_size["targets_segmentation"] = None
      p.vocab_size["targets_position"] = None

  def example_reading_spec(self):
    data_fields = {"targets": tf.VarLenFeature(tf.int64)}
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


class QuestionAndContext2TextProblem(Text2TextProblem):
  """Problems consisting of inputs, context, and a target.

  Variant of Text2TextProblem that includes a "context" feature in addition to
  "inputs" and "targets."
  """
  QUESTION_SEPARATOR = "<EOQ>"
  QUESTION_SEPARATOR_ID = 2

  @property
  def additional_reserved_tokens(self):
    return [self.QUESTION_SEPARATOR]

  def feature_encoders(self, data_dir):
    encoders = (super(QuestionAndContext2TextProblem, self)
                .feature_encoders(data_dir))
    encoders["context"] = encoders["inputs"]
    return encoders

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    for i, sample in enumerate(
        self.generate_samples(data_dir, tmp_dir, problem.DatasetSplit.TRAIN)):
      yield sample["inputs"]
      yield sample["context"]
      yield sample["targets"]
      if self.max_samples_for_vocab and (i + 1) >= self.max_samples_for_vocab:
        break

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    generator = super(
        QuestionAndContext2TextProblem, self).generate_encoded_samples(
            data_dir, tmp_dir, dataset_split)
    vocab = self.feature_encoders(data_dir)["context"]
    for sample in generator:
      context = vocab.encode(sample["context"])
      context.append(text_encoder.EOS_ID)
      sample["context"] = context
      yield sample

  def hparams(self, defaults, unused_model_hparams):
    (super(QuestionAndContext2TextProblem, self)
     .hparams(defaults, unused_model_hparams))
    p = defaults
    p.modality["context"] = modalities.ModalityType.SYMBOL
    p.vocab_size["context"] = self._encoders["context"].vocab_size
    if self.packed_length:
      raise NotImplementedError("QuestionAndContext2Text does not "
                                "support packed_length")

  def example_reading_spec(self):
    data_fields, data_items_to_decoders = (super(QuestionAndContext2TextProblem,
                                                 self)
                                           .example_reading_spec())
    data_fields["context"] = tf.VarLenFeature(tf.int64)
    return (data_fields, data_items_to_decoders)


class Text2SelfProblem(Text2TextProblem):
  """Language modeling problems base class.

  See Text2TextProblem for subclass interface.
  """

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Generate samples of text.

    Args:
      data_dir: final data directory. Typically only used in this method to copy
        over user-supplied vocab files (for example, if vocab_type ==
        VocabType.TOKEN).
      tmp_dir: temporary directory that you can use for downloading and scratch.
      dataset_split: problem.DatasetSplit, which data split to generate samples
        for (for example, training and evaluation).

    Yields:
      Sample: dict<str feature_name, str text>: for language modeling problems
        (i.e. Text2SelfProblems), this generator should yield dicts with only
        the "targets" key.
    """
    raise NotImplementedError()

  @property
  def has_inputs(self):
    return False


class Text2ClassProblem(Text2TextProblem):
  """Base class for text classification problems."""

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Generate samples of text and label pairs.

    Each yielded dict will be a single example. The inputs should be raw text.
    The label should be an int in [0, self.num_classes).

    Args:
      data_dir: final data directory. Typically only used in this method to copy
        over user-supplied vocab files (for example, if vocab_type ==
        VocabType.TOKEN).
      tmp_dir: temporary directory that you can use for downloading and scratch.
      dataset_split: problem.DatasetSplit, which data split to generate samples
        for (for example, training and evaluation).

    Yields:
      {"inputs": text, "label": int}
    """
    raise NotImplementedError()

  # START: Additional subclass interface
  @property
  def num_classes(self):
    """The number of classes."""
    raise NotImplementedError()

  def class_labels(self, data_dir):
    """String representation of the classes."""
    del data_dir
    return ["ID_%d" % i for i in range(self.num_classes)]

  # END: Additional subclass interface

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    for i, sample in enumerate(
        self.generate_samples(data_dir, tmp_dir, problem.DatasetSplit.TRAIN)):
      yield sample["inputs"]
      if self.max_samples_for_vocab and (i + 1) >= self.max_samples_for_vocab:
        break

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    encoder = self.get_or_create_vocab(data_dir, tmp_dir)
    for sample in generator:
      inputs = encoder.encode(sample["inputs"])
      inputs.append(text_encoder.EOS_ID)
      label = sample["label"]
      yield {"inputs": inputs, "targets": [label]}

  def feature_encoders(self, data_dir):
    encoder = self.get_or_create_vocab(data_dir, None, force_get=True)

    return {
        "inputs": encoder,
        "targets": text_encoder.ClassLabelEncoder(self.class_labels(data_dir))
    }

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.modality = {"inputs": modalities.ModalityType.SYMBOL,
                  "targets": modalities.ModalityType.CLASS_LABEL}
    p.vocab_size = {"inputs": self._encoders["inputs"].vocab_size,
                    "targets": self.num_classes}

  def example_reading_spec(self):
    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets": tf.FixedLenFeature([1], tf.int64),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)


class TextConcat2ClassProblem(Text2ClassProblem):
  """Base class for text classification problems with multiple inputs.

  For problems where there are multiple input sentences and we wish to concat
  these inputs with a special delimiter. See, for example, NLI tasks.
  """
  CONCAT_TOKEN = "$"

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    for i, sample in enumerate(
        self.generate_samples(data_dir, tmp_dir, problem.DatasetSplit.TRAIN)):
      for inp in sample["inputs"]:
        yield inp
        if self.max_samples_for_vocab and (i + 1) >= self.max_samples_for_vocab:
          break

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    encoder = self.get_or_create_vocab(data_dir, tmp_dir)
    for sample in generator:
      inputs = []
      for idx, inp in enumerate(sample["inputs"]):
        inputs += encoder.encode(inp)
        if idx < len(sample["inputs"]) - 1:
          inputs.append(encoder.encode(self.CONCAT_TOKEN)[0])
      inputs.append(text_encoder.EOS_ID)
      label = sample["label"]
      yield {"inputs": inputs, "targets": [label]}


class Text2RealProblem(Text2TextProblem):
  """Base class for text regression problems with one or more tasks.

    Suitable for text-based problems where targets are continuous, real values.
    When ntasks = 1, each text example is mapped to a single scalar value. When
    ntasks > 1, each text example is mapped to a 1-d vector of length ntasks.
  """

  @property
  def ntasks(self):
    """Set to n > 1 for multitask regression."""
    return 1

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Generate samples of text and real-valued target pairs.

    Each yielded dict will be a single example. The inputs should be raw text.
    The target should be a list containing ntasks floats.
    Args:
      data_dir: final data directory. Typically only used in this method to copy
        over user-supplied vocab files (for example, if vocab_type ==
        VocabType.TOKEN).
      tmp_dir: temporary directory that you can use for downloading and scratch.
      dataset_split: problem.DatasetSplit, which data split to generate samples
        for (for example, training and evaluation).
    Yields:
      {"inputs": text, "targets": [x1, x2, ..., xN]} where N is ntasks
    """
    raise NotImplementedError()

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    for i, sample in enumerate(
        self.generate_samples(data_dir, tmp_dir, problem.DatasetSplit.TRAIN)):
      yield sample["inputs"]
      if self.max_samples_for_vocab and (i + 1) >= self.max_samples_for_vocab:
        break

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    encoder = self.get_or_create_vocab(data_dir, tmp_dir)
    for sample in generator:
      inputs = encoder.encode(sample["inputs"])
      inputs.append(text_encoder.EOS_ID)
      yield {"inputs": inputs, "targets": sample["targets"]}

  def feature_encoders(self, data_dir):
    encoder = self.get_or_create_vocab(data_dir, None, force_get=True)

    return {
        "inputs": encoder,
        "targets": text_encoder.RealEncoder(),
    }

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.modality = {
        "inputs": modalities.ModalityType.SYMBOL,
        "targets": modalities.ModalityType.REAL_L2_LOSS,
    }
    p.vocab_size = {
        "inputs": self._encoders["inputs"].vocab_size,
        "targets": self.ntasks
    }
    p.target_space_id = problem.SpaceID.REAL
    p.add_hparam("regression_targets", True)

  def max_length(self, model_hparams):
    return model_hparams.batch_size * self.ntasks

  def preprocess_example(self, example, unused_mode, unused_hparams):
    example = problem.preprocess_example_common(example, unused_mode,
                                                unused_hparams)
    example["targets"] = tf.reshape(example["targets"], [1, 1, self.ntasks])
    return example

  def example_reading_spec(self):
    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets": tf.FixedLenFeature([self.ntasks], tf.float32),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)

  def eval_metrics(self):
    metrics_list = [metrics.Metrics.RMSE]
    if self.ntasks == 1:
      metrics_list.append(metrics.Metrics.PEARSON)
    return metrics_list


def txt_line_iterator(txt_path):
  """Iterate through lines of file."""
  with tf.gfile.Open(txt_path) as f:
    for line in f:
      yield line.strip()


def txt_and_label_iterator(txt_path):
  """Iterate through lines of file."""
  problem_pattern_without_vocab_size = re.compile("(.*)\tExtra_Label: (.*)")
  with tf.gfile.Open(txt_path) as f:
    for line in f:
      results = problem_pattern_without_vocab_size.search(line.strip())
      try:
        line = results.group(1)
        extra_label = int(results.group(2))
      except AttributeError:
        raise ValueError(
            "Please provide the file in the right format, with each line having"
            " the following format:\n<word_1 word_2 ... word_n>\\t"
            "Extra_Label:\\s<int_label>"
        )
      yield [line, extra_label]


def text2text_txt_iterator(source_txt_path, target_txt_path):
  """Yield dicts for Text2TextProblem.generate_samples from lines of files."""
  for inputs, targets in zip(
      txt_line_iterator(source_txt_path), txt_line_iterator(target_txt_path)):
    yield {"inputs": inputs, "targets": targets}


def text2text_txt_iterator_with_label(source_txt_path, target_txt_path):
  """Yield dicts for Text2TextProblem.generate_samples from lines of files."""
  for inputs, (targets, extra_label) in zip(
      txt_line_iterator(source_txt_path),
      txt_and_label_iterator(target_txt_path)):
    yield {"inputs": inputs, "targets": targets, "extra_label": [extra_label]}


def text2text_txt_iterator_with_index(source_txt_path, target_txt_path):
  """Yield dicts for Text2TextProblem.generate_samples from lines of files."""
  for (idx, (inputs, targets)) in enumerate(zip(
      txt_line_iterator(source_txt_path),
      txt_line_iterator(target_txt_path))):
    yield {"inputs": inputs, "targets": targets, "idx": [idx]}


def text2text_distill_iterator(source_txt_path, target_txt_path,
                               distill_txt_path):
  """Yield dicts for Text2TextProblem.generate_samples from lines of files."""
  for inputs, targets, dist_targets in zip(
      txt_line_iterator(source_txt_path), txt_line_iterator(target_txt_path),
      txt_line_iterator(distill_txt_path)):
    yield {"inputs": inputs, "targets": targets, "dist_targets": dist_targets}


def text2self_txt_iterator(txt_path):
  for line in txt_line_iterator(txt_path):
    yield {"targets": line}


def text2class_txt_iterator(source_txt_path, label_txt_path, class_strs=None):
  """Yield dicts for Text2ClassProblem.generate_samples from lines of files.

  Args:
    source_txt_path: txt file with record per line.
    label_txt_path: txt file with label per line, either as int or str. If
      string, must provide class_strs.
    class_strs: list<str> of class label names. Must be in correct order (i.e.
      ["a", "b", "c"] means that "a" will get class ID 0, "b" ID 1, etc.).

  Yields:
    {"inputs": inputs, "label": label}
  """
  if class_strs:
    class_strs = dict([(s, i) for i, s in enumerate(class_strs)])
  for inputs, label in zip(
      txt_line_iterator(source_txt_path), txt_line_iterator(label_txt_path)):
    label = label.strip()
    if class_strs:
      label = class_strs[label]
    else:
      label = int(label)
    yield {"inputs": inputs, "label": label}


def text2real_txt_iterator(source_txt_path, target_txt_path):
  """Yield dicts for Text2RealProblem.generate_samples from lines of files.

  Args:
    source_txt_path: txt file with record per line.
    target_txt_path: txt file with float (or space-separated float list for
      multitask) per line.
  Yields:
    {"inputs": inputs, "targets": targets}
  """
  for inputs, targets in zip(
      txt_line_iterator(source_txt_path), txt_line_iterator(target_txt_path)):
    targets = [float(x) for x in targets.split(" ")]
    yield {"inputs": inputs, "targets": targets}


def text2text_txt_tab_iterator(txt_path):
  """Yield dicts for Text2TextProblem.generate_samples from lines of txt_path.

  Args:
    txt_path: path to txt file with a record per line, source and target
      are tab-separated.

  Yields:
    {"inputs": inputs, "targets": targets}
  """
  for line in txt_line_iterator(txt_path):
    if line and "\t" in line:
      parts = line.split("\t", 1)
      inputs, targets = parts[:2]
      yield {"inputs": inputs.strip(), "targets": targets.strip()}


def text2text_generate_encoded(sample_generator,
                               vocab,
                               targets_vocab=None,
                               has_inputs=True,
                               inputs_prefix="",
                               targets_prefix=""):
  """Encode Text2Text samples from the generator with the vocab."""
  targets_vocab = targets_vocab or vocab
  for sample in sample_generator:
    if has_inputs:
      sample["inputs"] = vocab.encode(inputs_prefix + sample["inputs"])
      sample["inputs"].append(text_encoder.EOS_ID)
    sample["targets"] = targets_vocab.encode(targets_prefix + sample["targets"])
    sample["targets"].append(text_encoder.EOS_ID)
    yield sample


@registry.register_problem
class Text2textTmpdir(Text2TextProblem):
  """Allows training a Text2TextProblem without defining a subclass.

  Put your training and evaluation data into the following files in tmp_dir,
  with 1 record per line:

  * inputs.train.txt
  * targets.train.txt
  * inputs.eval.txt
  * targets.eval.txt
  """
  TRAIN_FILES = ("inputs.train.txt", "targets.train.txt")
  EVAL_FILES = ("inputs.eval.txt", "targets.eval.txt")

  def is_generate_per_split(self):
    return True

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del data_dir
    is_training = dataset_split == problem.DatasetSplit.TRAIN
    files = self.TRAIN_FILES if is_training else self.EVAL_FILES
    files = [os.path.join(self._tmp_dir_override or tmp_dir, f) for f in files]
    inputs_file, targets_file = files
    return text2text_txt_iterator(inputs_file, targets_file)

  @property
  def _tmp_dir_override(self):
    return None


class Text2TextRemotedir(Text2textTmpdir):
  """Text2TextProblem from files in a remote directory.

  SRC_REMOTE_DIR should be a remote directory, e.g. a GCS bucket (gs://...),
  that contains the following files, 1 record per line:

    * inputs.train.txt
    * targets.train.txt
    * inputs.eval.txt
    * targets.eval.txt

  """
  # Override in subclass.
  SRC_REMOTE_DIR = None

  @property
  def _tmp_dir_override(self):
    assert self.SRC_REMOTE_DIR
    return self.SRC_REMOTE_DIR


@registry.register_problem
class Text2textTmpdirTokens(Text2textTmpdir):
  """Allows training a token-based variant of Text2textTmpdir.

  Put your training and evaluation data into the following files in tmp_dir,
  with 1 record per line along with a vocabulary file with 1 token per line
  (you can leave out PAD, EOS, and UNK as those will be automatically added)

  * inputs.train.txt
  * targets.train.txt
  * inputs.eval.txt
  * targets.eval.txt
  * vocab.txt
  """

  @property
  def vocab_type(self):
    return VocabType.TOKEN

  @property
  def oov_token(self):
    return "<UNK>"

  def _generate_vocab(self, tmp_dir):
    vocab_list = [self.oov_token]
    user_vocab_file = os.path.join(tmp_dir, "vocab.txt")
    with tf.gfile.GFile(user_vocab_file, "r") as vocab_file:
      for line in vocab_file:
        token = line.strip()
        vocab_list.append(token)
    token_encoder = text_encoder.TokenTextEncoder(None, vocab_list=vocab_list)
    return token_encoder

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    vocab_filepath = os.path.join(data_dir, self.vocab_filename)
    if not tf.gfile.Exists(vocab_filepath):
      token_encoder = self._generate_vocab(tmp_dir)
      token_encoder.store_to_file(vocab_filepath)
    return super(Text2textTmpdirTokens, self).generate_samples(data_dir,
                                                               tmp_dir,
                                                               dataset_split)


class ChoppedTextProblem(Text2SelfProblem):
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
      return [
          f for i, f in enumerate(self.train_text_filepaths(tmp_dir))
          if i % self.num_train_shards == task_id
      ]
    else:
      return [
          f for i, f in enumerate(self.dev_text_filepaths(tmp_dir))
          if i % self.num_dev_shards == task_id - self.num_train_shards
      ]

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
    yield text_encoder.to_unicode_ignore_errors(b)

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
        if (max_chars_per_file and
            chars_this_file + len(text) > max_chars_per_file):
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
        yield {"targets": tokens[pos:pos + self.sequence_length]}
        pos += self.sequence_length
      if pos > 0:
        tokens = tokens[pos:]
    if self.remainder_policy == "pad":
      if tokens:
        targets = tokens + [0] * (self.sequence_length - len(tokens))
        yield {"targets": targets}
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
    self.get_or_create_vocab(data_dir, tmp_dir)
    self.train_text_filepaths(tmp_dir)
    self.dev_text_filepaths(tmp_dir)

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    return self.file_generator(
        self.train_text_filepaths(tmp_dir),
        max_chars_total=self.max_chars_for_vocab)

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
    encoder = self.get_or_create_vocab(data_dir, tmp_dir)
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
    return 10**7

  @property
  def num_train_shards(self):
    return self.dataset_splits[0]["shards"]

  @property
  def num_dev_shards(self):
    return self.dataset_splits[1]["shards"]

  @property
  def max_dev_chars(self):
    """Limit dev set to at most this many characters (default 10M)."""
    return 10**7

  @property
  def multiprocess_generate(self):
    return True

  @property
  def num_generate_tasks(self):
    return self.num_train_shards + self.num_dev_shards

  def eval_metrics(self):
    return [metrics.Metrics.ACC, metrics.Metrics.NEG_LOG_PERPLEXITY]


class DistributedText2TextProblem(Text2TextProblem):
  """Base class for text-to-text problems for large-datasets.

  Text2TextProblem doesn't support data generation in a distributed manner.

  Use DistributedText2TextProblem if you have a sharded dataset(s) and want to
  create tf.Examples from them in a distributed manner.

  Every task will write to one output shard and will read from specific input
  shards.

  Subclasses should override `generate_samples`, `input_dataset_files`
  and `is_generate_per_split` as described below.

  Users need to generate the vocabulary before generating data.
  See tensor2tensor/bin/build_vocab.py.
  """

  # START: Subclass interface

  def generate_samples(self, data_dir, tmp_dir, dataset_split, input_files):
    """Generate samples of input text and target text pairs.

    Subclasses should generate the samples using only files from `input_files`.

    Please see Text2TextProblem.generate_samples for a fuller explanation.

    Args:
      data_dir: final data directory.
      tmp_dir: temporary directory that you can use for downloading and scratch.
      dataset_split: problem.DatasetSplit, which data split to generate samples
        for (for example, training and evaluation).
      input_files: Generate samples using only these input dataset files.

    Yields:
      {"inputs": text, "targets": text}
    """
    raise NotImplementedError()

  def input_files(self, dataset_split=problem.DatasetSplit.TRAIN):
    """The input files of the input dataset.

    If you don't have a separate dev/test split then returning []
    suffices for dataset_split != problem.DatasetSplit.TRAIN

    Args:
      dataset_split: The split for which to return the input files for.

    Returns:
      list of strings: The files for the supplied datasplit
    """

    raise NotImplementedError()

  # END: Subclass interface

  @property
  def num_output_shards(self):
    # Returns the total number of output shards.
    num_output_shards = 0
    for split in self.dataset_splits:
      num_output_shards += split["shards"]
    return num_output_shards

  @property
  def split_to_input_filenames(self):
    # Dictionary of dataset split to input dataset filenames.
    split_to_input_filenames = {}
    num_input_files = 0
    if not self.is_generate_per_split:
      # We just have a single input dataset file.
      split_to_input_filenames[problem.DatasetSplit.TRAIN] = (
          self.input_files(problem.DatasetSplit.TRAIN))
      num_input_files += len(
          split_to_input_filenames[problem.DatasetSplit.TRAIN])
    else:
      # We have separate input dataset files.
      for dataset_split in self.dataset_splits:
        split = dataset_split["split"]
        split_to_input_filenames[split] = self.input_files(split)
        num_input_files += len(split_to_input_filenames[split])

    # Number of input files >= number of output files. So that every task should
    # have some work to do!
    assert num_input_files >= self.num_output_shards

    return split_to_input_filenames

  def _task_id_to_output_split(self, task_id):
    # Takes a task_id and returns a tuple of
    # (split of the dataset to operate on, number of shards in that split,
    # offset of this task from the first task to operate on that split)
    num_output_shards = 0
    for dataset_split in self.dataset_splits:
      num_output_shards += dataset_split["shards"]
      if task_id < num_output_shards:
        return (dataset_split["split"], dataset_split["shards"],
                (task_id - num_output_shards + dataset_split["shards"]))

  def _task_id_to_output_file(self, data_dir, task_id):
    # Returns the output filename that this task will write.

    dataset_split, shards, offset = self._task_id_to_output_split(task_id)

    filepath_fns = {
        problem.DatasetSplit.TRAIN: self.training_filepaths,
        problem.DatasetSplit.EVAL: self.dev_filepaths,
        problem.DatasetSplit.TEST: self.test_filepaths,
    }

    return filepath_fns[dataset_split](data_dir, shards, False)[offset]

  @staticmethod
  def _divide_equally(input_files, num_tasks, task_id):
    # There are num_tasks total tasks, we need to divide these
    # input files among them equally and return the slice that task_id should
    # read from.
    task_load, remainder = divmod(len(input_files), num_tasks)

    # This is the slice of almost equal sized chunks of files for a task_id to
    # handle -- this distributes the excess remainder tasks among the first
    # "remainder" task_ids.

    # The extra min(task_id, remainder) in the end comes from assigning the
    # remainder of the tasks to task_ids [0, remainder), so we need to advance
    # the start by how many ever remainder tasks already assigned.
    start_idx = task_id * task_load + min(task_id, remainder)

    # This will handle atleast `task_load` files, plus an extra one if `task_id`
    # is still less than remainder.
    num_elements = task_load + int(task_id < remainder)

    return input_files[start_idx : start_idx + num_elements]

  def _task_id_to_input_files(self, task_id):
    # Returns a list of input files that this task should read and process.

    if not self.is_generate_per_split:
      # We just have one unified input dataset to handle, so all tasks will read
      # from the TRAIN dataset.
      input_files = self.split_to_input_filenames[problem.DatasetSplit.TRAIN]

      return self._divide_equally(input_files, self.num_output_shards, task_id)

    # self.is_generate_per_split is True.
    dataset_split, num_shards, offset = self._task_id_to_output_split(task_id)
    input_files = self.split_to_input_filenames[dataset_split]
    return self._divide_equally(input_files, num_shards, offset)

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    # We need to override this because we'll be reading from specific files
    # instead

    # What files should we read for creating the vocabulary?
    input_files_for_vocab = []
    if self.is_generate_per_split:
      input_files_for_vocab = (
          self.split_to_input_filenames[problem.DatasetSplit.TRAIN])
    else:
      # We need to compute the 'train' shards from the whole input.
      # Go over all task_ids that output training data, collect their input
      # files.
      for task_id in range(self.num_output_shards):
        split, _, _ = self._task_id_to_output_split(task_id)
        if split == problem.DatasetSplit.TRAIN:
          input_files_for_vocab.extend(self._task_id_to_input_files(task_id))

    # Generate samples only from the above generated files.
    for i, sample in enumerate(
        self.generate_samples(data_dir, tmp_dir, problem.DatasetSplit.TRAIN,
                              input_files_for_vocab)):
      if self.has_inputs:
        yield sample["inputs"]
      yield sample["targets"]
      if self.max_samples_for_vocab and (i + 1) >= self.max_samples_for_vocab:
        break

  def generate_encoded_samples(self,
                               data_dir,
                               tmp_dir,
                               dataset_split,
                               input_files):
    # Since this is a distributed problem, we don't want every task to create
    # its own vocabulary, so we assume that the dictionary is already created
    # for example by using build_vocab.py
    vocab_filepath = os.path.join(data_dir, self.vocab_filename)
    if not tf.gfile.Exists(vocab_filepath):
      raise ValueError("Vocab file: %s doesn't exist, please use "
                       "build_vocab.py to create one." % vocab_filepath)
    encoder = self.get_or_create_vocab(data_dir, tmp_dir, force_get=True)
    generator = self.generate_samples(data_dir, tmp_dir, dataset_split,
                                      input_files)
    return text2text_generate_encoded(
        generator, encoder, has_inputs=self.has_inputs,
        inputs_prefix=self.inputs_prefix,
        targets_prefix=self.targets_prefix)

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    # task_id should be in [0, self.num_output_shards)
    assert (0 <= task_id) and (task_id < self.num_output_shards)

    # A task_id is only supposed to write only one output shard, it can operate
    # over multiple *input* shards.
    input_files = self._task_id_to_input_files(task_id)
    output_file = self._task_id_to_output_file(data_dir, task_id)

    # Which output split is this task writing to?
    split, _, _ = self._task_id_to_output_split(task_id)

    # Actually generate examples.
    generator_utils.generate_files(
        self.generate_encoded_samples(
            data_dir, tmp_dir, split, input_files),
        [output_file])

    # Shuffle the output.
    generator_utils.shuffle_dataset([output_file], extra_fn=self._pack_fn())
