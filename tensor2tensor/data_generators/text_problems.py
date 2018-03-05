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

"""Base classes for text-based Problems.

* Text2TextProblem: input=text, target=text.
* Text2ClassProblem: input=text, target=class.
* Text2SelfProblem (for language modeling): target=text

The Text2TextTmpDir problem allows you to train without defining a problem. It
expects you to format your data in a particular way and put it in tmp_dir. See
its docstring.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf


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
    if self.vocab_type == VocabType.SUBWORD:
      return "vocab.%s.%d.%s" % (self.dataset_filename(),
                                 self.approx_vocab_size,
                                 VocabType.SUBWORD)
    else:
      return "vocab.%s.%s" % (self.dataset_filename(), VocabType.TOKEN)

  def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
    if self.vocab_type == VocabType.CHARACTER:
      encoder = text_encoder.ByteTextEncoder()
    elif self.vocab_type == VocabType.SUBWORD:
      if force_get:
        vocab_filepath = os.path.join(data_dir, self.vocab_filename)
        encoder = text_encoder.SubwordTextEncoder(vocab_filepath)
      else:
        encoder = generator_utils.get_or_generate_vocab_inner(
            data_dir, self.vocab_filename, self.approx_vocab_size,
            self.generate_text_for_vocab(data_dir, tmp_dir))
    elif self.vocab_type == VocabType.TOKEN:
      vocab_filename = os.path.join(data_dir, self.vocab_filename)
      encoder = text_encoder.TokenTextEncoder(vocab_filename)
    else:
      raise ValueError("Unrecognized VocabType")
    return encoder

  def _maybe_pack_examples(self, generator):
    """Wraps generator with packer if self.packed_length."""
    if not self.packed_length:
      return generator
    return generator_utils.pack_examples(
        generator,
        self.has_inputs,
        self.packed_length,
        chop_long_sequences=not self.has_inputs)

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    encoder = self.get_or_create_vocab(data_dir, tmp_dir)
    return text2text_generate_encoded(generator, encoder,
                                      has_inputs=self.has_inputs)

  @property
  def batch_size_means_tokens(self):
    return True

  def generate_data(self, data_dir, tmp_dir, task_id=-1):

    filepath_fns = {
        problem.DatasetSplit.TRAIN: self.training_filepaths,
        problem.DatasetSplit.EVAL: self.dev_filepaths,
        problem.DatasetSplit.TEST: self.test_filepaths,
    }

    split_paths = [(split["split"], filepath_fns[split["split"]](
        data_dir, split["shards"], shuffled=False))
                   for split in self.dataset_splits]
    all_paths = []
    for _, paths in split_paths:
      all_paths.extend(paths)

    if self.is_generate_per_split:
      for split, paths in split_paths:
        generator_utils.generate_files(
            self._maybe_pack_examples(
                self.generate_encoded_samples(data_dir, tmp_dir, split)), paths)
    else:
      generator_utils.generate_files(
          self._maybe_pack_examples(
              self.generate_encoded_samples(
                  data_dir, tmp_dir, problem.DatasetSplit.TRAIN)), all_paths)

    generator_utils.shuffle_dataset(all_paths)

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
    if self.vocab_type == VocabType.CHARACTER:
      p.loss_multiplier = 2.0

    if self.packed_length:
      identity = (registry.Modalities.GENERIC, None)
      if self.has_inputs:
        p.input_modality["inputs_segmentation"] = identity
        p.input_modality["inputs_position"] = identity
      p.input_modality["targets_segmentation"] = identity
      p.input_modality["targets_position"] = identity

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
        "targets": text_encoder.ClassLabelEncoder(self.class_labels)
    }

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    source_vocab_size = self._encoders["inputs"].vocab_size
    p.input_modality = {
        "inputs": (registry.Modalities.SYMBOL, source_vocab_size)
    }
    p.target_modality = (registry.Modalities.CLASS_LABEL, self.num_classes)

  def example_reading_spec(self):
    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets": tf.FixedLenFeature([1], tf.int64),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)


def txt_line_iterator(txt_path):
  """Iterate through lines of file."""
  with tf.gfile.Open(txt_path) as f:
    for line in f:
      yield line.strip()


def text2text_txt_iterator(source_txt_path, target_txt_path):
  """Yield dicts for Text2TextProblem.generate_samples from lines of files."""
  for inputs, targets in zip(
      txt_line_iterator(source_txt_path), txt_line_iterator(target_txt_path)):
    yield {"inputs": inputs, "targets": targets}


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
                               has_inputs=True):
  """Encode Text2Text samples from the generator with the vocab."""
  targets_vocab = targets_vocab or vocab
  for sample in sample_generator:
    if has_inputs:
      sample["inputs"] = vocab.encode(sample["inputs"])
      sample["inputs"].append(text_encoder.EOS_ID)
    sample["targets"] = targets_vocab.encode(sample["targets"])
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
    files = [os.path.join(tmp_dir, f) for f in files]
    inputs_file, targets_file = files
    return text2text_txt_iterator(inputs_file, targets_file)


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
    yield text_encoder.to_unicode_ignore_erros(b)

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
