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

"""Data generators for WMT data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import text_encoder

import tensorflow as tf


def character_generator(source_path, target_path, eos=None):
  """Generator for sequence-to-sequence tasks that just uses characters.

  This generator assumes the files at source_path and target_path have
  the same number of lines and yields dictionaries of "inputs" and "targets"
  where inputs are characters from the source lines converted to integers,
  and targets are characters from the target lines, also converted to integers.

  Args:
    source_path: path to the file with source sentences.
    target_path: path to the file with target sentences.
    eos: integer to append at the end of each sequence (default: None).

  Yields:
    A dictionary {"inputs": source-line, "targets": target-line} where
    the lines are integer lists converted from characters in the file lines.
  """
  eos_list = [] if eos is None else [eos]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      while source and target:
        source_ints = [ord(c) for c in source.strip()] + eos_list
        target_ints = [ord(c) for c in target.strip()] + eos_list
        yield {"inputs": source_ints, "targets": target_ints}
        source, target = source_file.readline(), target_file.readline()


def token_generator(source_path, target_path, token_vocab, eos=None):
  """Generator for sequence-to-sequence tasks that uses tokens.

  This generator assumes the files at source_path and target_path have
  the same number of lines and yields dictionaries of "inputs" and "targets"
  where inputs are token ids from the " "-split source (and target, resp.) lines
  converted to integers using the token_map.

  Args:
    source_path: path to the file with source sentences.
    target_path: path to the file with target sentences.
    token_vocab: text_encoder.TextEncoder object.
    eos: integer to append at the end of each sequence (default: None).

  Yields:
    A dictionary {"inputs": source-line, "targets": target-line} where
    the lines are integer lists converted from tokens in the file lines.
  """
  eos_list = [] if eos is None else [eos]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      while source and target:
        source_ints = token_vocab.encode(source.strip()) + eos_list
        target_ints = token_vocab.encode(target.strip()) + eos_list
        yield {"inputs": source_ints, "targets": target_ints}
        source, target = source_file.readline(), target_file.readline()


def _get_wmt_ende_dataset(directory, filename):
  """Extract the WMT en-de corpus `filename` to directory unless it's there."""
  train_path = os.path.join(directory, filename)
  if not (tf.gfile.Exists(train_path + ".de") and
          tf.gfile.Exists(train_path + ".en")):
    # We expect that this file has been downloaded from:
    # https://drive.google.com/open?id=0B_bZck-ksdkpM25jRUN2X2UxMm8 and placed
    # in `directory`.
    corpus_file = os.path.join(directory, "wmt16_en_de.tar.gz")
    with tarfile.open(corpus_file, "r:gz") as corpus_tar:
      corpus_tar.extractall(directory)
  return train_path


def ende_bpe_token_generator(tmp_dir, train):
  """Instance of token generator for the WMT en->de task, training set."""
  dataset_path = ("train.tok.clean.bpe.32000"
                  if train else "newstest2013.tok.bpe.32000")
  train_path = _get_wmt_ende_dataset(tmp_dir, dataset_path)
  token_path = os.path.join(tmp_dir, "vocab.bpe.32000")
  token_vocab = text_encoder.TokenTextEncoder(vocab_filename=token_path)
  return token_generator(train_path + ".en", train_path + ".de", token_vocab, 1)


_ENDE_TRAIN_DATASETS = [
    [
        "http://data.statmt.org/wmt16/translation-task/training-parallel-nc-v11.tgz",  # pylint: disable=line-too-long
        ("training-parallel-nc-v11/news-commentary-v11.de-en.en",
         "training-parallel-nc-v11/news-commentary-v11.de-en.de")
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
        ("commoncrawl.de-en.en", "commoncrawl.de-en.de")
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
        ("training/europarl-v7.de-en.en", "training/europarl-v7.de-en.de")
    ],
]
_ENDE_TEST_DATASETS = [
    [
        "http://data.statmt.org/wmt16/translation-task/dev.tgz",
        ("dev/newstest2013.en", "dev/newstest2013.de")
    ],
]

_ENFR_TRAIN_DATASETS = [
    [
        "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
        ("commoncrawl.fr-en.en", "commoncrawl.fr-en.fr")
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
        ("training/europarl-v7.fr-en.en", "training/europarl-v7.fr-en.fr")
    ],
    [
        "http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz",
        ("training/news-commentary-v9.fr-en.en",
         "training/news-commentary-v9.fr-en.fr")
    ],
    [
        "http://www.statmt.org/wmt10/training-giga-fren.tar",
        ("giga-fren.release2.fixed.en.gz", "giga-fren.release2.fixed.fr.gz")
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-un.tgz",
        ("un/undoc.2000.fr-en.en", "un/undoc.2000.fr-en.fr")
    ],
]
_ENFR_TEST_DATASETS = [
    [
        "http://data.statmt.org/wmt16/translation-task/dev.tgz",
        ("dev/newstest2013.en", "dev/newstest2013.fr")
    ],
]


def _compile_data(tmp_dir, datasets, filename):
  """Concatenate all `datasets` and save to `filename`."""
  filename = os.path.join(tmp_dir, filename)
  lang1_lines, lang2_lines = [], []
  for dataset in datasets:
    url = dataset[0]
    compressed_filename = os.path.basename(url)
    compressed_filepath = os.path.join(tmp_dir, compressed_filename)

    lang1_filename, lang2_filename = dataset[1]
    lang1_filepath = os.path.join(tmp_dir, lang1_filename)
    lang2_filepath = os.path.join(tmp_dir, lang2_filename)

    if not os.path.exists(compressed_filepath):
      generator_utils.maybe_download(tmp_dir, compressed_filename, url)
    if not os.path.exists(lang1_filepath) or not os.path.exists(lang2_filepath):
      mode = "r:gz" if "gz" in compressed_filepath else "r"
      with tarfile.open(compressed_filepath, mode) as corpus_tar:
        corpus_tar.extractall(tmp_dir)
    if ".gz" in lang1_filepath:
      new_filepath = lang1_filepath.strip(".gz")
      generator_utils.gunzip_file(lang1_filepath, new_filepath)
      lang1_filepath = new_filepath
    if ".gz" in lang2_filepath:
      new_filepath = lang2_filepath.strip(".gz")
      generator_utils.gunzip_file(lang2_filepath, new_filepath)
      lang2_filepath = new_filepath
    with tf.gfile.GFile(lang1_filepath, mode="r") as lang1_file:
      with tf.gfile.GFile(lang2_filepath, mode="r") as lang2_file:
        lang1_file_lines = lang1_file.readlines()
        lang2_file_lines = lang2_file.readlines()
        assert len(lang1_file_lines) == len(lang2_file_lines), lang1_filepath
        lang1_lines.extend(lang1_file_lines)
        lang2_lines.extend(lang2_file_lines)

  write_chunk_size = 10000
  assert len(lang1_lines) == len(lang2_lines)
  with tf.gfile.GFile(filename + ".lang1", mode="w") as lang1_file:
    i = 0
    while i <= len(lang1_lines):
      for line in lang1_lines[i * write_chunk_size:(i + 1) * write_chunk_size]:
        lang1_file.write(line)
      i += 1
    for line in lang1_lines[i * write_chunk_size:]:
      lang1_file.write(line)
  with tf.gfile.GFile(filename + ".lang2", mode="w") as lang2_file:
    i = 0
    while i <= len(lang2_lines):
      for line in lang2_lines[i * write_chunk_size:(i + 1) * write_chunk_size]:
        lang2_file.write(line)
      i += 1
    for line in lang2_lines[i * write_chunk_size:]:
      lang2_file.write(line)
  return filename


def ende_wordpiece_token_generator(tmp_dir, train, vocab_size):
  symbolizer_vocab = generator_utils.get_or_generate_vocab(
      tmp_dir, "tokens.vocab.%d" % vocab_size, vocab_size)
  datasets = _ENDE_TRAIN_DATASETS if train else _ENDE_TEST_DATASETS
  tag = "train" if train else "dev"
  data_path = _compile_data(tmp_dir, datasets, "wmt_ende_tok_%s" % tag)
  return token_generator(data_path + ".lang1", data_path + ".lang2",
                         symbolizer_vocab, 1)


def ende_character_generator(tmp_dir, train):
  datasets = _ENDE_TRAIN_DATASETS if train else _ENDE_TEST_DATASETS
  tag = "train" if train else "dev"
  data_path = _compile_data(tmp_dir, datasets, "wmt_ende_chr_%s" % tag)
  return character_generator(data_path + ".lang1", data_path + ".lang2", 1)


def enfr_wordpiece_token_generator(tmp_dir, train, vocab_size):
  """Instance of token generator for the WMT en->fr task."""
  symbolizer_vocab = generator_utils.get_or_generate_vocab(
      tmp_dir, "tokens.vocab.%d" % vocab_size, vocab_size)
  datasets = _ENFR_TRAIN_DATASETS if train else _ENFR_TEST_DATASETS
  tag = "train" if train else "dev"
  data_path = _compile_data(tmp_dir, datasets, "wmt_enfr_tok_%s" % tag)
  return token_generator(data_path + ".lang1", data_path + ".lang2",
                         symbolizer_vocab, 1)


def enfr_character_generator(tmp_dir, train):
  """Instance of character generator for the WMT en->fr task."""
  datasets = _ENFR_TRAIN_DATASETS if train else _ENFR_TEST_DATASETS
  tag = "train" if train else "dev"
  data_path = _compile_data(tmp_dir, datasets, "wmt_enfr_chr_%s" % tag)
  return character_generator(data_path + ".lang1", data_path + ".lang2", 1)


def parsing_character_generator(tmp_dir, train):
  filename = "parsing_%s" % ("train" if train else "dev")
  text_filepath = os.path.join(tmp_dir, filename + ".text")
  tags_filepath = os.path.join(tmp_dir, filename + ".tags")
  return character_generator(text_filepath, tags_filepath, 1)


def parsing_token_generator(tmp_dir, train, vocab_size):
  symbolizer_vocab = generator_utils.get_or_generate_vocab(
      tmp_dir, "tokens.vocab.%d" % vocab_size, vocab_size)
  filename = "parsing_%s" % ("train" if train else "dev")
  text_filepath = os.path.join(tmp_dir, filename + ".text")
  tags_filepath = os.path.join(tmp_dir, filename + ".tags")
  return token_generator(text_filepath, tags_filepath, symbolizer_vocab, 1)
