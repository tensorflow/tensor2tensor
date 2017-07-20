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

"""Data generators for WMT data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import wsj_parsing
from tensor2tensor.utils import registry

import tensorflow as tf

tf.flags.DEFINE_string("ende_bpe_path", "", "Path to BPE files in tmp_dir."
                       "Download from https://drive.google.com/open?"
                       "id=0B_bZck-ksdkpM25jRUN2X2UxMm8")

FLAGS = tf.flags.FLAGS


# End-of-sentence marker.
EOS = text_encoder.EOS_ID


def _default_token_feature_encoders(data_dir, target_vocab_size):
  vocab_filename = os.path.join(data_dir,
                                "vocab.endefr.%d" % target_vocab_size)
  subtokenizer = text_encoder.SubwordTextEncoder(vocab_filename)
  return {
      "inputs": subtokenizer,
      "targets": subtokenizer,
  }


def _default_character_feature_encoders():
  return {
      "inputs": text_encoder.ByteTextEncoder(),
      "targets": text_encoder.ByteTextEncoder(),
  }


class WMTProblem(problem.Problem):
  """Base class for WMT problems."""

  @property
  def is_character_level(self):
    return False

  @property
  def targeted_vocab_size(self):
    raise NotImplementedError()  # Not needed if self.is_character_level.

  @property
  def train_generator(self):
    """Generator; takes data_dir, tmp_dir, is_training, targeted_vocab_size."""
    raise NotImplementedError()

  @property
  def dev_generator(self):
    return self.train_generator

  @property
  def input_space_id(self):
    raise NotImplementedError()

  @property
  def target_space_id(self):
    raise NotImplementedError()

  @property
  def num_shards(self):
    return 100

  def generate_data(self, data_dir, tmp_dir, num_shards=None):
    if num_shards is None:
      num_shards = self.num_shards
    if self.is_character_level:
      generator_utils.generate_dataset_and_shuffle(
          self.train_generator(tmp_dir, True),
          self.training_filepaths(data_dir, num_shards, shuffled=False),
          self.dev_generator(tmp_dir, False),
          self.dev_filepaths(data_dir, 1, shuffled=False))
    else:
      generator_utils.generate_dataset_and_shuffle(
          self.train_generator(data_dir, tmp_dir, True,
                               self.targeted_vocab_size),
          self.training_filepaths(data_dir, num_shards, shuffled=False),
          self.dev_generator(data_dir, tmp_dir, False,
                             self.targeted_vocab_size),
          self.dev_filepaths(data_dir, 1, shuffled=False))

  def feature_encoders(self, data_dir):
    if self.is_character_level:
      return _default_character_feature_encoders()
    return _default_token_feature_encoders(data_dir, self.targeted_vocab_size)

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    if self.is_character_level:
      source_vocab_size = 256
      target_vocab_size = 256
    else:
      source_vocab_size = self._encoders["inputs"].vocab_size
      target_vocab_size = self._encoders["targets"].vocab_size
    p.input_modality = {"inputs": (registry.Modalities.SYMBOL,
                                   source_vocab_size)}
    p.target_modality = (registry.Modalities.SYMBOL, target_vocab_size)
    p.input_space_id = self.input_space_id
    p.target_space_id = self.target_space_id
    if self.is_character_level:
      p.loss_multiplier = 2.0


# Generic generators used later for multiple problems.


def character_generator(source_path, target_path, character_vocab, eos=None):
  """Generator for sequence-to-sequence tasks that just uses characters.

  This generator assumes the files at source_path and target_path have
  the same number of lines and yields dictionaries of "inputs" and "targets"
  where inputs are characters from the source lines converted to integers,
  and targets are characters from the target lines, also converted to integers.

  Args:
    source_path: path to the file with source sentences.
    target_path: path to the file with target sentences.
    character_vocab: a TextEncoder to encode the characters.
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
        source_ints = character_vocab.encode(source.strip()) + eos_list
        target_ints = character_vocab.encode(target.strip()) + eos_list
        yield {"inputs": source_ints, "targets": target_ints}
        source, target = source_file.readline(), target_file.readline()


def tabbed_generator(source_path, source_vocab, target_vocab, eos=None):
  r"""Generator for sequence-to-sequence tasks using tabbed files.

  Tokens are derived from text files where each line contains both
  a source and a target string. The two strings are separated by a tab
  character ('\t'). It yields dictionaries of "inputs" and "targets" where
  inputs are characters from the source lines converted to integers, and
  targets are characters from the target lines, also converted to integers.

  Args:
    source_path: path to the file with source and target sentences.
    source_vocab: a SunwordTextEncoder to encode the source string.
    target_vocab: a SunwordTextEncoder to encode the target string.
    eos: integer to append at the end of each sequence (default: None).

  Yields:
    A dictionary {"inputs": source-line, "targets": target-line} where
    the lines are integer lists converted from characters in the file lines.
  """
  eos_list = [] if eos is None else [eos]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    for line in source_file:
      if line and "\t" in line:
        parts = line.split("\t", maxsplit=1)
        source, target = parts[0].strip(), parts[1].strip()
        source_ints = source_vocab.encode(source) + eos_list
        target_ints = target_vocab.encode(target) + eos_list
        yield {"inputs": source_ints, "targets": target_ints}


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


def bi_vocabs_token_generator(source_path,
                              target_path,
                              source_token_vocab,
                              target_token_vocab,
                              eos=None):
  """Generator for sequence-to-sequence tasks that uses tokens.

  This generator assumes the files at source_path and target_path have
  the same number of lines and yields dictionaries of "inputs" and "targets"
  where inputs are token ids from the " "-split source (and target, resp.) lines
  converted to integers using the token_map.

  Args:
    source_path: path to the file with source sentences.
    target_path: path to the file with target sentences.
    source_token_vocab: text_encoder.TextEncoder object.
    target_token_vocab: text_encoder.TextEncoder object.
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
        source_ints = source_token_vocab.encode(source.strip()) + eos_list
        target_ints = target_token_vocab.encode(target.strip()) + eos_list
        yield {"inputs": source_ints, "targets": target_ints}
        source, target = source_file.readline(), target_file.readline()


# Data-set URLs.


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

_ZHEN_TRAIN_DATASETS = [[("http://data.statmt.org/wmt17/translation-task/"
                          "training-parallel-nc-v12.tgz"),
                         ("training/news-commentary-v12.zh-en.zh",
                          "training/news-commentary-v12.zh-en.en")]]

_ZHEN_TEST_DATASETS = [[
    "http://data.statmt.org/wmt17/translation-task/dev.tgz",
    ("dev/newsdev2017-zhen-src.zh", "dev/newsdev2017-zhen-ref.en")
]]

# For Macedonian-English the SETimes corpus
# from http://nlp.ffzg.hr/resources/corpora/setimes/ is used.
# The original dataset has 207,777 parallel sentences.
# For training the first 205,777 sentences are used.
_MKEN_TRAIN_DATASETS = [[
    "https://github.com/stefan-it/nmt-mk-en/raw/master/data/setimes.mk-en.train.tgz",  # pylint: disable=line-too-long
    ("train.mk", "train.en")
]]

# For development 1000 parallel sentences are used.
_MKEN_TEST_DATASETS = [[
    "https://github.com/stefan-it/nmt-mk-en/raw/master/data/setimes.mk-en.dev.tgz",  # pylint: disable=line-too-long
    ("dev.mk", "dev.en")
]]


# Generators.


def _get_wmt_ende_dataset(directory, filename):
  """Extract the WMT en-de corpus `filename` to directory unless it's there."""
  train_path = os.path.join(directory, filename)
  if not (tf.gfile.Exists(train_path + ".de") and
          tf.gfile.Exists(train_path + ".en")):
    # We expect that this file has been downloaded from:
    # https://drive.google.com/open?id=0B_bZck-ksdkpM25jRUN2X2UxMm8 and placed
    # in `directory`.
    corpus_file = os.path.join(directory, FLAGS.ende_bpe_path)
    with tarfile.open(corpus_file, "r:gz") as corpus_tar:
      corpus_tar.extractall(directory)
  return train_path


def ende_bpe_token_generator(data_dir, tmp_dir, train):
  """Instance of token generator for the WMT en->de task, training set."""
  dataset_path = ("train.tok.clean.bpe.32000"
                  if train else "newstest2013.tok.bpe.32000")
  train_path = _get_wmt_ende_dataset(tmp_dir, dataset_path)
  token_tmp_path = os.path.join(tmp_dir, "vocab.bpe.32000")
  token_path = os.path.join(data_dir, "vocab.bpe.32000")
  tf.gfile.Copy(token_tmp_path, token_path, overwrite=True)
  token_vocab = text_encoder.TokenTextEncoder(vocab_filename=token_path)
  return token_generator(train_path + ".en", train_path + ".de", token_vocab,
                         EOS)


def _compile_data(tmp_dir, datasets, filename):
  """Concatenate all `datasets` and save to `filename`."""
  filename = os.path.join(tmp_dir, filename)
  with tf.gfile.GFile(filename + ".lang1", mode="w") as lang1_resfile:
    with tf.gfile.GFile(filename + ".lang2", mode="w") as lang2_resfile:
      for dataset in datasets:
        url = dataset[0]
        compressed_filename = os.path.basename(url)
        compressed_filepath = os.path.join(tmp_dir, compressed_filename)

        lang1_filename, lang2_filename = dataset[1]
        lang1_filepath = os.path.join(tmp_dir, lang1_filename)
        lang2_filepath = os.path.join(tmp_dir, lang2_filename)

        if not os.path.exists(compressed_filepath):
          generator_utils.maybe_download(tmp_dir, compressed_filename, url)
        if not (os.path.exists(lang1_filepath) and
                os.path.exists(lang2_filepath)):
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
            line1, line2 = lang1_file.readline(), lang2_file.readline()
            while line1 or line2:
              lang1_resfile.write(line1.strip() + "\n")
              lang2_resfile.write(line2.strip() + "\n")
              line1, line2 = lang1_file.readline(), lang2_file.readline()

  return filename


def ende_wordpiece_token_generator(data_dir, tmp_dir, train, vocab_size):
  symbolizer_vocab = generator_utils.get_or_generate_vocab(
      data_dir, tmp_dir, "vocab.endefr.%d" % vocab_size, vocab_size)
  datasets = _ENDE_TRAIN_DATASETS if train else _ENDE_TEST_DATASETS
  tag = "train" if train else "dev"
  data_path = _compile_data(tmp_dir, datasets, "wmt_ende_tok_%s" % tag)
  return token_generator(data_path + ".lang1", data_path + ".lang2",
                         symbolizer_vocab, EOS)


@registry.register_problem("wmt_ende_tokens_8k")
class WMTEnDeTokens8k(WMTProblem):
  """Problem spec for WMT En-De translation."""

  @property
  def targeted_vocab_size(self):
    return 2**13  # 8192

  @property
  def train_generator(self):
    return ende_wordpiece_token_generator

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.DE_TOK


@registry.register_problem("wmt_ende_tokens_32k")
class WMTEnDeTokens32k(WMTEnDeTokens8k):

  @property
  def targeted_vocab_size(self):
    return 2**15  # 32768


def ende_character_generator(tmp_dir, train):
  character_vocab = text_encoder.ByteTextEncoder()
  datasets = _ENDE_TRAIN_DATASETS if train else _ENDE_TEST_DATASETS
  tag = "train" if train else "dev"
  data_path = _compile_data(tmp_dir, datasets, "wmt_ende_chr_%s" % tag)
  return character_generator(data_path + ".lang1", data_path + ".lang2",
                             character_vocab, EOS)


@registry.register_problem("wmt_ende_characters")
class WMTEnDeCharacters(WMTProblem):
  """Problem spec for WMT En-De translation."""

  @property
  def is_character_level(self):
    return True

  @property
  def train_generator(self):
    return ende_character_generator

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def target_space_id(self):
    return problem.SpaceID.DE_CHR


def zhen_wordpiece_token_bigenerator(data_dir, tmp_dir, train,
                                     source_vocab_size, target_vocab_size):
  """Wordpiece generator for the WMT'17 zh-en dataset."""
  datasets = _ZHEN_TRAIN_DATASETS if train else _ZHEN_TEST_DATASETS
  source_datasets = [[item[0], [item[1][0]]] for item in _ZHEN_TRAIN_DATASETS]
  target_datasets = [[item[0], [item[1][1]]] for item in _ZHEN_TRAIN_DATASETS]
  source_vocab = generator_utils.get_or_generate_vocab(
      data_dir, tmp_dir, "vocab.zh.%d" % source_vocab_size,
      source_vocab_size, source_datasets)
  target_vocab = generator_utils.get_or_generate_vocab(
      data_dir, tmp_dir, "vocab.en.%d" % target_vocab_size,
      target_vocab_size, target_datasets)
  tag = "train" if train else "dev"
  data_path = _compile_data(tmp_dir, datasets, "wmt_zhen_tok_%s" % tag)
  return bi_vocabs_token_generator(data_path + ".lang1", data_path + ".lang2",
                                   source_vocab, target_vocab, EOS)


def zhen_wordpiece_token_generator(data_dir, tmp_dir, train, vocab_size):
  return zhen_wordpiece_token_bigenerator(data_dir, tmp_dir, train,
                                          vocab_size, vocab_size)


@registry.register_problem("wmt_zhen_tokens_8k")
class WMTZhEnTokens8k(WMTProblem):
  """Problem spec for WMT Zh-En translation."""

  @property
  def targeted_vocab_size(self):
    return 2**13  # 8192

  @property
  def train_generator(self):
    return zhen_wordpiece_token_generator

  @property
  def input_space_id(self):
    return problem.SpaceID.ZH_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_TOK

  def feature_encoders(self, data_dir):
    vocab_size = self.targeted_vocab_size
    source_vocab_filename = os.path.join(data_dir,
                                         "vocab.zh.%d" % vocab_size)
    target_vocab_filename = os.path.join(data_dir,
                                         "vocab.en.%d" % vocab_size)
    source_token = text_encoder.SubwordTextEncoder(source_vocab_filename)
    target_token = text_encoder.SubwordTextEncoder(target_vocab_filename)
    return {
        "inputs": source_token,
        "targets": target_token,
    }


@registry.register_problem("wmt_zhen_tokens_32k")
class WMTZhEnTokens32k(WMTZhEnTokens8k):

  @property
  def targeted_vocab_size(self):
    return 2**15  # 32768


def enfr_wordpiece_token_generator(data_dir, tmp_dir, train, vocab_size):
  """Instance of token generator for the WMT en->fr task."""
  symbolizer_vocab = generator_utils.get_or_generate_vocab(
      data_dir, tmp_dir, "vocab.endefr.%d" % vocab_size, vocab_size)
  datasets = _ENFR_TRAIN_DATASETS if train else _ENFR_TEST_DATASETS
  tag = "train" if train else "dev"
  data_path = _compile_data(tmp_dir, datasets, "wmt_enfr_tok_%s" % tag)
  return token_generator(data_path + ".lang1", data_path + ".lang2",
                         symbolizer_vocab, EOS)


@registry.register_problem("wmt_enfr_tokens_8k")
class WMTEnFrTokens8k(WMTProblem):
  """Problem spec for WMT En-Fr translation."""

  @property
  def targeted_vocab_size(self):
    return 2**13  # 8192

  @property
  def train_generator(self):
    return enfr_wordpiece_token_generator

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.FR_TOK


@registry.register_problem("wmt_enfr_tokens_32k")
class WMTEnFrTokens32k(WMTEnFrTokens8k):

  @property
  def targeted_vocab_size(self):
    return 2**15  # 32768


def enfr_character_generator(tmp_dir, train):
  """Instance of character generator for the WMT en->fr task."""
  character_vocab = text_encoder.ByteTextEncoder()
  datasets = _ENFR_TRAIN_DATASETS if train else _ENFR_TEST_DATASETS
  tag = "train" if train else "dev"
  data_path = _compile_data(tmp_dir, datasets, "wmt_enfr_chr_%s" % tag)
  return character_generator(data_path + ".lang1", data_path + ".lang2",
                             character_vocab, EOS)


@registry.register_problem("wmt_enfr_characters")
class WMTEnFrCharacters(WMTProblem):
  """Problem spec for WMT En-Fr translation."""

  @property
  def is_character_level(self):
    return True

  @property
  def train_generator(self):
    return enfr_character_generator

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def target_space_id(self):
    return problem.SpaceID.FR_CHR


def mken_wordpiece_token_generator(data_dir, tmp_dir, train, vocab_size):
  """Wordpiece generator for the SETimes Mk-En dataset."""
  datasets = _MKEN_TRAIN_DATASETS if train else _MKEN_TEST_DATASETS
  source_datasets = [[item[0], [item[1][0]]] for item in _MKEN_TRAIN_DATASETS]
  target_datasets = [[item[0], [item[1][1]]] for item in _MKEN_TRAIN_DATASETS]
  symbolizer_vocab = generator_utils.get_or_generate_vocab(
      data_dir, tmp_dir, "vocab.mken.%d" % vocab_size, vocab_size,
      source_datasets + target_datasets)
  tag = "train" if train else "dev"
  data_path = _compile_data(tmp_dir, datasets, "setimes_mken_tok_%s" % tag)
  return token_generator(data_path + ".lang1", data_path + ".lang2",
                         symbolizer_vocab, EOS)


@registry.register_problem("setimes_mken_tokens_32k")
class SETimesMkEnTokens32k(WMTProblem):
  """Problem spec for SETimes Mk-En translation."""

  @property
  def targeted_vocab_size(self):
    return 2**15  # 32768

  @property
  def train_generator(self):
    return mken_wordpiece_token_generator

  @property
  def input_space_id(self):
    return problem.SpaceID.MK_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_TOK


def parsing_character_generator(tmp_dir, train):
  character_vocab = text_encoder.ByteTextEncoder()
  filename = "parsing_%s" % ("train" if train else "dev")
  text_filepath = os.path.join(tmp_dir, filename + ".text")
  tags_filepath = os.path.join(tmp_dir, filename + ".tags")
  return character_generator(text_filepath, tags_filepath, character_vocab, EOS)


def tabbed_parsing_token_generator(data_dir, tmp_dir, train, prefix,
                                   source_vocab_size, target_vocab_size):
  """Generate source and target data from a single file."""
  source_vocab = generator_utils.get_or_generate_tabbed_vocab(
      data_dir, tmp_dir, "parsing_train.pairs", 0,
      prefix + "_source.vocab.%d" % source_vocab_size, source_vocab_size)
  target_vocab = generator_utils.get_or_generate_tabbed_vocab(
      data_dir, tmp_dir, "parsing_train.pairs", 1,
      prefix + "_target.vocab.%d" % target_vocab_size, target_vocab_size)
  filename = "parsing_%s" % ("train" if train else "dev")
  pair_filepath = os.path.join(tmp_dir, filename + ".pairs")
  return tabbed_generator(pair_filepath, source_vocab, target_vocab, EOS)


def tabbed_parsing_character_generator(tmp_dir, train):
  """Generate source and target data from a single file."""
  character_vocab = text_encoder.ByteTextEncoder()
  filename = "parsing_%s" % ("train" if train else "dev")
  pair_filepath = os.path.join(tmp_dir, filename + ".pairs")
  return tabbed_generator(pair_filepath, character_vocab, character_vocab, EOS)


def parsing_token_generator(data_dir, tmp_dir, train, vocab_size):
  symbolizer_vocab = generator_utils.get_or_generate_vocab(
      data_dir, tmp_dir, "vocab.endefr.%d" % vocab_size, vocab_size)
  filename = "%s_%s.trees" % (FLAGS.parsing_path, "train" if train else "dev")
  tree_filepath = os.path.join(tmp_dir, filename)
  return wsj_parsing.token_generator(tree_filepath, symbolizer_vocab,
                                     symbolizer_vocab, EOS)
