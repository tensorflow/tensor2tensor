# Copyright 2017 Google Inc.
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

# Dependency imports

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import modality

import tensorflow as tf


def default_problem_hparams():
  """A set of basic model hyperparameters."""
  return tf.contrib.training.HParams(
      # Use this parameter to get comparable perplexity numbers with different
      # tokenizations.  This value should be set to the ratio of the number of
      # tokens in the test set according to the tokeization used to the number
      # of tokens in the test set in the "official" tokenization.  For example,
      # if we are using a word-piece based model and we want to compute
      # per-word perplexity, then we set loss_multiplier to the number of
      # wordpieces per word in the test set.
      loss_multiplier=1.0,

      # Use this parameter to allow for larger sequences in the batch. Without
      # the use of this parameter, the size of the inner two dimensions will be
      # used to judge the sequence length.
      batch_size_multiplier=1,

      # To make queues of the right capacity, it's good to know the maximal
      # expected batch size, as it can vary a lot. It only affects performance
      # of input readers and memory use. The defaults should be safe and fast,
      # but decrease if your reader uses a lot of memory and increase if slow.
      max_expected_batch_size_per_shard=64,

      # Modalities used to map from input features to a space compatible with
      # chosen model architecture.  One modality per feature key.
      input_modality={},

      # Modality used to map from hidden representation to the target space.
      target_modality=None,

      # Identifiers used to tell the model which input/target space will be
      # expected. For example, it can tell that we expect French as characters
      # as output, or Spanish as sound. An integer with the following semantics:
      #   0: Generic / unknown output space (default)
      #   1: Image labels
      #   2: English characters
      #   3: English tokens
      #   4: English bpe tokens
      #   5: French characters
      #   6: French tokens
      #   7: German characters
      #   8: German tokens
      #   9: German bpe tokens
      #   10: Digit cipher lexicon 0
      #   11: Digit cipher lexicon 1
      #   12: Audio waveform domain
      #   13: Audio spectral domain
      #   14: Parse characters
      #   15: Parse tokens
      # Add more above if needed.
      input_space_id=0,
      target_space_id=0,

      # Vocabulary per feature key.
      #   a vocabulary converts to/from human-readable strings.
      # E.g. {"inputs": text_encoder.ByteTextEncoder(),
      #       "targets": wordpiece.WordpieceVocab("vocab_filename.txt")}
      vocabulary={
          "inputs": text_encoder.TextEncoder(),
          "targets": text_encoder.TextEncoder()
      },

      # This is a marker to keep track if the problem was reversed or copied.
      # Only set automatically, do not override the default.
      #
      # These tags can be combined in order to perform copies of the input or
      # the targets. For instance `problem_copy` will copy the inputs, but
      # `problem_rev_copy` will copy the targets.
      was_reversed=False,
      was_copy=False,)


def parse_problem_name(problem_name):
  """Determines if problem_name specifies a copy and/or reversal.

  Args:
    problem_name: A string containing a single problem name from FLAGS.problems.

  Returns:
    base_name: A string with the base problem name.
    was_reversed: A boolean.
    was_copy: A boolean.
  """
  # Recursively strip tags until we reach a base name.
  if len(problem_name) > 4 and problem_name[-4:] == "_rev":
    base, _, was_copy = parse_problem_name(problem_name[:-4])
    return base, True, was_copy
  elif len(problem_name) > 5 and problem_name[-5:] == "_copy":
    base, was_reversed, _ = parse_problem_name(problem_name[:-5])
    return base, was_reversed, True
  else:
    return problem_name, False, False


def problem_hparams(problem_name, model_hparams):
  """Generate problem hyperparameters based on problem name.

  Args:
    problem_name: a string
    model_hparams: a tf.contrib.training.HParams

  Returns:
    a tf.contrib.training.HParams

  Raises:
    ValueError: if problem_name is unknown.
  """
  base_name, was_reversed, was_copy = parse_problem_name(problem_name)
  if base_name not in _problem_hparams_map:
    map_str = "\n* ".join(_problem_hparams_map.keys())
    error_msg = "%s not in the supported set of problems:\n%s" % (base_name,
                                                                  map_str)
    raise ValueError(error_msg)
  p = _problem_hparams_map.get(base_name)(model_hparams)
  if was_reversed:
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
  if was_copy:
    # Duplicate input modality.
    p.target_modality = p.input_modality["inputs"]
    # Duplicate input vocabulary.
    p.vocabulary["targets"] = p.vocabulary["inputs"]
    # Duplicate input space ids.
    p.target_space_id = p.input_space_id
    # Mark that p was reversed.
    p.was_copy = True
  return p


def test_problem_hparams(model_hparams, input_vocab_size, target_vocab_size):
  """Problem hparams for testing model bodies."""
  p = default_problem_hparams()
  p.input_modality = {
      "inputs": modality.SymbolModality(model_hparams, input_vocab_size)
  }
  p.target_modality = modality.SymbolModality(model_hparams, target_vocab_size)
  p.vocabulary = {
      "inputs": text_encoder.TextEncoder(),
      "targets": text_encoder.TextEncoder()
  }
  return p


def algorithmic(vocab_size, model_hparams):
  """Default parameters for algorithmic tasks."""
  p = default_problem_hparams()
  p.input_modality = {
      "inputs": modality.SymbolModality(model_hparams, vocab_size)
  }
  p.target_modality = modality.SymbolModality(model_hparams, vocab_size)
  p.vocabulary = {
      "inputs": text_encoder.TextEncoder(num_reserved_ids=1),
      "targets": text_encoder.TextEncoder(num_reserved_ids=1),
  }
  p.input_space_id = 10
  p.target_space_id = 11
  return p


def audio_timit_characters(model_hparams):
  """English audio transcription benchmark."""
  p = default_problem_hparams()
  p.input_modality = {
      "inputs": modality.AudioModality(model_hparams),
  }
  p.target_modality = modality.SymbolModality(model_hparams, 256)
  p.vocabulary = {
      "inputs": text_encoder.TextEncoder(),
      "targets": text_encoder.ByteTextEncoder(),
  }
  p.batch_size_multiplier = 256
  p.loss_multiplier = 2.0
  p.input_space_id = 12
  p.target_space_id = 2
  return p


def audio_timit_tokens(model_hparams, wrong_vocab_size):
  """English audio transcription benchmark.

  Args:
    model_hparams: a tf.contrib.training.HParams
    wrong_vocab_size: a number used in the filename indicating the approximate
      vocabulary size.  This is not to be confused with the actual vocabulary
      size.
  Returns:
    a tf.contrib.training.HParams
  """
  p = default_problem_hparams()
  # This vocab file must be present within the data directory.
  vocab_filename = os.path.join(model_hparams.data_dir,
                                "tokens.vocab.%d" % wrong_vocab_size)
  subtokenizer = text_encoder.SubwordTextEncoder(vocab_filename)
  p.input_modality = {
      "inputs": modality.AudioModality(model_hparams),
  }
  p.target_modality = modality.SymbolModality(model_hparams,
                                              subtokenizer.vocab_size)
  p.vocabulary = {
      "inputs": text_encoder.TextEncoder(),
      "targets": subtokenizer,
  }
  p.batch_size_multiplier = 256
  p.loss_multiplier = 2.0
  p.input_space_id = 13
  p.target_space_id = 3
  return p


def audio_wsj_characters(model_hparams):
  """English audio transcription benchmark."""
  p = default_problem_hparams()
  p.input_modality = {
      "inputs": modality.AudioSpectralModality(model_hparams),
  }
  p.target_modality = modality.SymbolModality(model_hparams, 256)
  p.vocabulary = {
      "inputs": text_encoder.TextEncoder(),
      "targets": text_encoder.ByteTextEncoder(),
  }
  p.batch_size_multiplier = 512
  p.loss_multiplier = 2.0
  p.input_space_id = 13
  p.target_space_id = 2
  return p


def audio_wsj_tokens(model_hparams, wrong_vocab_size):
  """English audio transcription benchmark.

  Args:
    model_hparams: a tf.contrib.training.HParams
    wrong_vocab_size: a number used in the filename indicating the approximate
      vocabulary size.  This is not to be confused with the actual vocabulary
      size.
  Returns:
    a tf.contrib.training.HParams
  """
  p = default_problem_hparams()
  # This vocab file must be present within the data directory.
  vocab_filename = os.path.join(model_hparams.data_dir,
                                "tokens.vocab.%d" % wrong_vocab_size)
  subtokenizer = text_encoder.SubwordTextEncoder(vocab_filename)
  p.input_modality = {
      "inputs": modality.AudioModality(model_hparams),
  }
  p.target_modality = modality.SymbolModality(model_hparams,
                                              subtokenizer.vocab_size)
  p.vocabulary = {
      "inputs": text_encoder.TextEncoder(),
      "targets": subtokenizer,
  }
  p.batch_size_multiplier = 512
  p.loss_multiplier = 2.0
  p.input_space_id = 12
  p.target_space_id = 3
  return p


def lm1b_16k(model_hparams):
  """Billion-word language-modeling benchmark, 16k subtoken vocabulary."""
  p = default_problem_hparams()
  p.perplexity_exponent = 1.184206
  p.input_modality = {}
  p.target_modality = modality.SymbolModality(model_hparams, 16384)
  p.vocabulary = {
      "targets":
          text_encoder.SubwordTextEncoder(
              os.path.join(model_hparams.data_dir,
                           "lm1b_16k.subword_text_encoder"))
  }
  p.target_space_id = 3
  return p


def lm1b_64k(model_hparams):
  """Billion-word language-modeling benchmark, 64k subtoken vocabulary."""
  p = default_problem_hparams()
  p.perplexity_exponent = 1.067068
  p.input_modality = {}
  p.target_modality = modality.SymbolModality(model_hparams, 65536)
  p.vocabulary = {
      "targets":
          text_encoder.SubwordTextEncoder(
              os.path.join(model_hparams.data_dir,
                           "lm1b_64k.subword_text_encoder"))
  }
  p.target_space_id = 3
  return p


def wmt_enfr_characters(model_hparams):
  """English to French translation benchmark."""
  p = default_problem_hparams()
  p.input_modality = {"inputs": modality.SymbolModality(model_hparams, 256)}
  p.target_modality = modality.SymbolModality(model_hparams, 256)
  p.vocabulary = {
      "inputs": text_encoder.ByteTextEncoder(),
      "targets": text_encoder.ByteTextEncoder(),
  }
  p.loss_multiplier = 2.0
  p.input_space_id = 2
  p.target_space_id = 5
  return p


def wmt_enfr_tokens(model_hparams, wrong_vocab_size):
  """English to French translation benchmark.

  Args:
    model_hparams: a tf.contrib.training.HParams
    wrong_vocab_size: a number used in the filename indicating the approximate
      vocabulary size.  This is not to be confused with the actual vocabulary
      size.
  Returns:
    a tf.contrib.training.HParams
  """
  p = default_problem_hparams()
  # This vocab file must be present within the data directory.
  vocab_filename = os.path.join(model_hparams.data_dir,
                                "tokens.vocab.%d" % wrong_vocab_size)
  subtokenizer = text_encoder.SubwordTextEncoder(vocab_filename)
  p.input_modality = {
      "inputs": modality.SymbolModality(model_hparams, subtokenizer.vocab_size)
  }
  p.target_modality = modality.SymbolModality(model_hparams,
                                              subtokenizer.vocab_size)
  p.vocabulary = {
      "inputs": subtokenizer,
      "targets": subtokenizer,
  }
  p.input_space_id = 3
  p.target_space_id = 6
  return p


def wmt_ende_bpe32k(model_hparams):
  """English to German translation benchmark."""
  p = default_problem_hparams()
  # single modality object enables embedding sharing between inputs and target
  # when model_hparams.shared_source_target_embedding is True.
  vocab_size = 40960
  m = modality.SymbolModality(model_hparams, vocab_size)
  p.input_modality = {"inputs": m}
  p.target_modality = m
  # This vocab file must be present within the data directory.
  vocab_filename = os.path.join(model_hparams.data_dir, "vocab.bpe.32000")
  p.vocabulary = {
      "inputs": text_encoder.TokenTextEncoder(vocab_filename=vocab_filename),
      "targets": text_encoder.TokenTextEncoder(vocab_filename=vocab_filename),
  }
  p.loss_multiplier = 1.4
  p.input_space_id = 4
  p.target_space_id = 9
  return p


def wmt_ende_characters(model_hparams):
  """English to German translation benchmark."""
  p = default_problem_hparams()
  p.input_modality = {"inputs": modality.SymbolModality(model_hparams, 256)}
  p.target_modality = modality.SymbolModality(model_hparams, 256)
  p.vocabulary = {
      "inputs": text_encoder.ByteTextEncoder(),
      "targets": text_encoder.ByteTextEncoder(),
  }
  p.loss_multiplier = 2.0
  p.input_space_id = 2
  p.target_space_id = 7
  return p


def wmt_ende_tokens(model_hparams, wrong_vocab_size):
  """English to German translation benchmark."""
  p = default_problem_hparams()
  # This vocab file must be present within the data directory.
  vocab_filename = os.path.join(model_hparams.data_dir,
                                "tokens.vocab.%d" % wrong_vocab_size)
  subtokenizer = text_encoder.SubwordTextEncoder(vocab_filename)
  p.input_modality = {
      "inputs": modality.SymbolModality(model_hparams, subtokenizer.vocab_size)
  }
  p.target_modality = modality.SymbolModality(model_hparams,
                                              subtokenizer.vocab_size)
  p.vocabulary = {
      "inputs": subtokenizer,
      "targets": subtokenizer,
  }
  p.input_space_id = 3
  p.target_space_id = 8
  return p


def wmt_ende_v2(model_hparams, vocab_size):
  """English to German translation benchmark with separate vocabularies."""
  p = default_problem_hparams()
  # These vocab files must be present within the data directory.
  source_vocab_filename = os.path.join(model_hparams.data_dir,
                                       "wmt_ende_v2.en.vocab.%d" % vocab_size)
  target_vocab_filename = os.path.join(model_hparams.data_dir,
                                       "wmt_ende_v2.de.vocab.%d" % vocab_size)
  p.input_modality = {
      "inputs": modality.SymbolModality(model_hparams, vocab_size)
  }
  p.target_modality = modality.SymbolModality(model_hparams, vocab_size)
  p.vocabulary = {
      "inputs": text_encoder.SubwordTextEncoder(source_vocab_filename),
      "targets": text_encoder.SubwordTextEncoder(target_vocab_filename),
  }
  p.input_space_id = 3
  p.target_space_id = 8
  return p


def wmt_concat(model_hparams, wrong_vocab_size):
  """English to German translation benchmark."""
  p = default_problem_hparams()
  # This vocab file must be present within the data directory.
  vocab_filename = os.path.join(model_hparams.data_dir,
                                "tokens.vocab.%d" % wrong_vocab_size)
  subtokenizer = text_encoder.SubwordTextEncoder(vocab_filename)
  vocab_size = subtokenizer.vocab_size
  p.input_modality = {}
  p.target_modality = modality.SymbolModality(model_hparams, vocab_size)
  p.vocabulary = {"targets": subtokenizer}
  return p


def wmt_parsing_characters(model_hparams):
  """English to parse tree translation benchmark."""
  p = default_problem_hparams()
  p.input_modality = {"inputs": modality.SymbolModality(model_hparams, 256)}
  p.target_modality = modality.SymbolModality(model_hparams, 256)
  p.vocabulary = {
      "inputs": text_encoder.ByteTextEncoder(),
      "targets": text_encoder.ByteTextEncoder(),
  }
  p.loss_multiplier = 2.0
  p.input_space_id = 2
  p.target_space_id = 14
  return p


def wmt_parsing_tokens(model_hparams, wrong_vocab_size):
  """English to parse tree translation benchmark.

  Args:
    model_hparams: a tf.contrib.training.HParams
    wrong_vocab_size: a number used in the filename indicating the approximate
      vocabulary size.  This is not to be confused with the actual vocabulary
      size.
  Returns:
    a tf.contrib.training.HParams
  """
  p = default_problem_hparams()
  # This vocab file must be present within the data directory.
  vocab_filename = os.path.join(model_hparams.data_dir,
                                "tokens.vocab.%d" % wrong_vocab_size)
  subtokenizer = text_encoder.SubwordTextEncoder(vocab_filename)
  p.input_modality = {
      "inputs": modality.SymbolModality(model_hparams, subtokenizer.vocab_size)
  }
  p.target_modality = modality.SymbolModality(model_hparams,
                                              subtokenizer.vocab_size)
  p.vocabulary = {
      "inputs": subtokenizer,
      "targets": subtokenizer,
  }
  p.input_space_id = 3
  p.target_space_id = 15
  return p


def wsj_parsing_tokens(model_hparams, wrong_source_vocab_size,
                       wrong_target_vocab_size):
  """English to parse tree translation benchmark.

  Args:
    model_hparams: a tf.contrib.training.HParams
    wrong_source_vocab_size: a number used in the filename indicating the
      approximate vocabulary size.  This is not to be confused with the actual
      vocabulary size.
    wrong_target_vocab_size: a number used in the filename indicating the
      approximate target vocabulary size. This is not to be confused with the
      actual target vocabulary size.
  Returns:
    a tf.contrib.training.HParams
  """
  p = default_problem_hparams()
  # This vocab file must be present within the data directory.
  source_vocab_filename = os.path.join(
      model_hparams.data_dir,
      "wsj_source.tokens.vocab.%d" % wrong_source_vocab_size)
  target_vocab_filename = os.path.join(
      model_hparams.data_dir,
      "wsj_target.tokens.vocab.%d" % wrong_target_vocab_size)
  source_subtokenizer = text_encoder.SubwordTextEncoder(
      source_vocab_filename)
  target_subtokenizer = text_encoder.SubwordTextEncoder(
      target_vocab_filename)
  p.input_modality = {
      "inputs": modality.SymbolModality(model_hparams,
                                        source_subtokenizer.vocab_size)
  }
  p.target_modality = modality.SymbolModality(model_hparams,
                                              target_subtokenizer.vocab_size)
  p.vocabulary = {
      "inputs": source_subtokenizer,
      "targets": target_subtokenizer,
  }
  p.input_space_id = 3
  p.target_space_id = 15
  return p


def image_cifar10(model_hparams):
  """CIFAR-10."""
  p = default_problem_hparams()
  p.input_modality = {"inputs": modality.SmallImageModality(model_hparams)}
  p.target_modality = modality.ClassLabelModality(model_hparams, 10)
  p.batch_size_multiplier = 4
  p.max_expected_batch_size_per_shard = 8
  p.loss_multiplier = 3.0
  p.input_space_id = 1
  p.target_space_id = 1
  return p


def image_mnist(model_hparams):
  """MNIST."""
  p = default_problem_hparams()
  p.input_modality = {"inputs": modality.SymbolModality(model_hparams, 256)}
  p.target_modality = modality.ClassLabelModality(model_hparams, 10)
  p.batch_size_multiplier = 4
  p.max_expected_batch_size_per_shard = 8
  p.loss_multiplier = 3.0
  p.input_space_id = 1
  p.target_space_id = 1
  return p


def image_imagenet(model_hparams):
  """ImageNet."""
  p = default_problem_hparams()
  p.input_modality = {
      "inputs": modality.ImageModality(model_hparams),
  }
  p.target_modality = modality.ClassLabelModality(
      model_hparams, 1000, is2d=model_hparams.imagenet_use_2d)
  p.batch_size_multiplier = 256
  p.max_expected_batch_size_per_shard = 2
  p.loss_multiplier = 0.7
  p.input_space_id = 1
  p.target_space_id = 1
  return p


def image_mscoco_characters(model_hparams):
  """COCO image captioning with captions as characters."""
  p = default_problem_hparams()
  p.input_modality = {"inputs": modality.ImageModality(model_hparams)}
  p.target_modality = modality.SymbolModality(model_hparams, 256)
  p.vocabulary = {
      "inputs": text_encoder.TextEncoder(),
      "targets": text_encoder.ByteTextEncoder(),
  }
  p.batch_size_multiplier = 128
  p.max_expected_batch_size_per_shard = 2
  p.loss_multiplier = 2.0
  p.input_space_id = 1
  p.target_space_id = 2
  return p


def image_mscoco_tokens(model_hparams, vocab_count):
  """COCO image captioning with captions as tokens."""
  p = default_problem_hparams()
  p.input_modality = {"inputs": modality.ImageModality(model_hparams)}
  # This vocab file must be present within the data directory.
  vocab_filename = os.path.join(model_hparams.data_dir,
                                "tokens.vocab.%d" % vocab_count)
  subtokenizer = text_encoder.SubwordTextEncoder(vocab_filename)
  p.target_modality = modality.SymbolModality(model_hparams,
                                              subtokenizer.vocab_size)
  p.vocabulary = {
      "inputs": text_encoder.TextEncoder(),
      "targets": subtokenizer,
  }
  p.batch_size_multiplier = 256
  p.max_expected_batch_size_per_shard = 2
  p.input_space_id = 1
  p.target_space_id = 3
  return p


# Dictionary of named hyperparameter settings for various problems.
# This is only accessed through the problem_hparams function below.
_problem_hparams_map = {
    "algorithmic_addition_binary40": lambda p: algorithmic(3, p),
    "algorithmic_addition_decimal40": lambda p: algorithmic(11, p),
    "algorithmic_identity_binary40": lambda p: algorithmic(3, p),
    "algorithmic_identity_decimal40": lambda p: algorithmic(11, p),
    "algorithmic_multiplication_binary40": lambda p: algorithmic(3, p),
    "algorithmic_multiplication_decimal40": lambda p: algorithmic(11, p),
    "algorithmic_reverse_binary40": lambda p: algorithmic(3, p),
    "algorithmic_reverse_decimal40": lambda p: algorithmic(11, p),
    "algorithmic_shift_decimal40": lambda p: algorithmic(21, p),
    "audio_timit_characters_tune": audio_timit_characters,
    "audio_timit_characters_test": audio_timit_characters,
    "audio_timit_tokens_8k_tune": lambda p: audio_timit_tokens(p, 2**13),
    "audio_timit_tokens_8k_test": lambda p: audio_timit_tokens(p, 2**13),
    "audio_wsj_characters_tune": audio_wsj_characters,
    "audio_wsj_characters_test": audio_wsj_characters,
    "audio_wsj_tokens_8k_tune": lambda p: audio_wsj_tokens(p, 2**13),
    "audio_wsj_tokens_8k_test": lambda p: audio_wsj_tokens(p, 2**13),
    "lm1b_16k": lm1b_16k,
    "lm1b_64k": lm1b_64k,
    "wmt_parsing_characters": wmt_parsing_characters,
    "wmt_parsing_tokens_8k": lambda p: wmt_parsing_tokens(p, 2**13),
    "wsj_parsing_tokens_16k": lambda p: wsj_parsing_tokens(p, 2**14, 2**9),
    "wsj_parsing_tokens_32k": lambda p: wsj_parsing_tokens(p, 2**15, 2**9),
    "wmt_enfr_characters": wmt_enfr_characters,
    "wmt_enfr_tokens_8k": lambda p: wmt_enfr_tokens(p, 2**13),
    "wmt_enfr_tokens_32k": lambda p: wmt_enfr_tokens(p, 2**15),
    "wmt_enfr_tokens_32k_shuffled": lambda p: wmt_enfr_tokens(p, 2**15),
    "wmt_enfr_tokens_32k_combined": lambda p: wmt_enfr_tokens(p, 2**15),
    "wmt_enfr_tokens_128k": lambda p: wmt_enfr_tokens(p, 2**17),
    # bytes per subtoken: 3.267350
    "wmt_ende_concat_8k": lambda p: wmt_concat(p, 2**13),
    # bytes per subtoken: 4.236272
    "wmt_ende_concat_32k": lambda p: wmt_concat(p, 2**15),
    "wmt_ende_characters": wmt_ende_characters,
    "wmt_ende_tokens_8k": lambda p: wmt_ende_tokens(p, 2**13),
    "wmt_ende_tokens_32k": lambda p: wmt_ende_tokens(p, 2**15),
    "wmt_ende_tokens_128k": lambda p: wmt_ende_tokens(p, 2**17),
    # bytes per subtoken: 4.59291664162
    "wmt_ende_bpe32k": wmt_ende_bpe32k,
    "wmt_ende_bpe32k_shuffled": wmt_ende_bpe32k,
    "wmt_ende_bpe32k_combined": wmt_ende_bpe32k,
    "wmt_ende_bpe32k_160": wmt_ende_bpe32k,
    "wmt_ende_v2_32k_combined": lambda p: wmt_ende_v2(p, 2**15),
    "wmt_ende_v2_16k_combined": lambda p: wmt_ende_v2(p, 2**14),
    "image_cifar10_tune": image_cifar10,
    "image_cifar10_test": image_cifar10,
    "image_mnist_tune": image_mnist,
    "image_mnist_test": image_mnist,
    "image_mscoco_characters_tune": image_mscoco_characters,
    "image_mscoco_characters_test": image_mscoco_characters,
    "image_mscoco_tokens_8k_tune": lambda p: image_mscoco_tokens(p, 2**13),
    "image_mscoco_tokens_8k_test": lambda p: image_mscoco_tokens(p, 2**13),
    "image_mscoco_tokens_32k_tune": lambda p: image_mscoco_tokens(p, 2**15),
    "image_mscoco_tokens_32k_test": lambda p: image_mscoco_tokens(p, 2**15),
    "image_mscoco_tokens_128k_tune": lambda p: image_mscoco_tokens(p, 2**17),
    "image_mscoco_tokens_128k_test": lambda p: image_mscoco_tokens(p, 2**17),
    "image_imagenet": image_imagenet,
}
