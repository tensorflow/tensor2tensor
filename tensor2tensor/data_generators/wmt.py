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

"""Data generators for translation data-sets."""

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

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID


class TranslateProblem(problem.Text2TextProblem):
  """Base class for translation problems."""

  @property
  def is_character_level(self):
    return False

  @property
  def num_shards(self):
    return 100

  @property
  def vocab_name(self):
    return "vocab.endefr"

  @property
  def use_subword_tokenizer(self):
    return True


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
    source_vocab: a SubwordTextEncoder to encode the source string.
    target_vocab: a SubwordTextEncoder to encode the target string.
    eos: integer to append at the end of each sequence (default: None).

  Yields:
    A dictionary {"inputs": source-line, "targets": target-line} where
    the lines are integer lists converted from characters in the file lines.
  """
  eos_list = [] if eos is None else [eos]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    for line in source_file:
      if line and "\t" in line:
        parts = line.split("\t", 1)
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
        "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz",  # pylint: disable=line-too-long
        ("training/news-commentary-v12.de-en.en",
         "training/news-commentary-v12.de-en.de")
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
        "http://data.statmt.org/wmt17/translation-task/dev.tgz",
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
        "http://data.statmt.org/wmt17/translation-task/dev.tgz",
        ("dev/newstest2013.en", "dev/newstest2013.fr")
    ],
]

_ZHEN_TRAIN_DATASETS = [[("http://data.statmt.org/wmt17/translation-task/"
                          "training-parallel-nc-v12.tgz"),
                         ("training/news-commentary-v12.zh-en.zh",
                          "training/news-commentary-v12.zh-en.en")]]

_ZHEN_TEST_DATASETS = [[
    "http://data.statmt.org/wmt17/translation-task/dev.tgz",
    ("dev/newsdev2017-zhen-src.zh.sgm", "dev/newsdev2017-zhen-ref.en.sgm")
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

# English-Czech datasets
_ENCS_TRAIN_DATASETS = [
    [
        ("https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/"
         "11234/1-1458/data-plaintext-format.tar"),
        ("tsv", 3, 2, "data.plaintext-format/*train.gz")
    ],
    [
        "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz",  # pylint: disable=line-too-long
        ("training/news-commentary-v12.cs-en.en",
         "training/news-commentary-v12.cs-en.cs")
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
        ("commoncrawl.cs-en.en", "commoncrawl.cs-en.cs")
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
        ("training/europarl-v7.cs-en.en", "training/europarl-v7.cs-en.cs")
    ],
]
_ENCS_TEST_DATASETS = [
    [
        "http://data.statmt.org/wmt17/translation-task/dev.tgz",
        ("dev/newstest2013.en", "dev/newstest2013.cs")
    ],
]

# Generators.


def _get_wmt_ende_bpe_dataset(directory, filename):
  """Extract the WMT en-de corpus `filename` to directory unless it's there."""
  train_path = os.path.join(directory, filename)
  if not (tf.gfile.Exists(train_path + ".de") and
          tf.gfile.Exists(train_path + ".en")):
    url = ("https://drive.google.com/uc?export=download&id="
           "0B_bZck-ksdkpM25jRUN2X2UxMm8")
    corpus_file = generator_utils.maybe_download_from_drive(
        directory, "wmt16_en_de.tar.gz", url)
    with tarfile.open(corpus_file, "r:gz") as corpus_tar:
      corpus_tar.extractall(directory)
  return train_path


@registry.register_problem
class TranslateEndeWmtBpe32k(TranslateProblem):
  """Problem spec for WMT En-De translation, BPE version."""

  @property
  def targeted_vocab_size(self):
    return 32000

  @property
  def vocab_name(self):
    return "vocab.bpe"

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    encoder = text_encoder.TokenTextEncoder(vocab_filename, replace_oov="UNK")
    return {"inputs": encoder, "targets": encoder}

  def generator(self, data_dir, tmp_dir, train):
    """Instance of token generator for the WMT en->de task, training set."""
    dataset_path = ("train.tok.clean.bpe.32000"
                    if train else "newstest2013.tok.bpe.32000")
    train_path = _get_wmt_ende_bpe_dataset(tmp_dir, dataset_path)
    token_tmp_path = os.path.join(tmp_dir, self.vocab_file)
    token_path = os.path.join(data_dir, self.vocab_file)
    tf.gfile.Copy(token_tmp_path, token_path, overwrite=True)
    with tf.gfile.GFile(token_path, mode="a") as f:
      f.write("UNK\n")  # Add UNK to the vocab.
    token_vocab = text_encoder.TokenTextEncoder(token_path, replace_oov="UNK")
    return token_generator(train_path + ".en", train_path + ".de", token_vocab,
                           EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_BPE_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.DE_BPE_TOK


def _preprocess_sgm(line, is_sgm):
  """Preprocessing to strip tags in SGM files."""
  if not is_sgm:
    return line
  # In SGM files, remove <srcset ...>, <p>, <doc ...> lines.
  if line.startswith("<srcset") or line.startswith("</srcset"):
    return ""
  if line.startswith("<doc") or line.startswith("</doc"):
    return ""
  if line.startswith("<p>") or line.startswith("</p>"):
    return ""
  # Strip <seg> tags.
  line = line.strip()
  if line.startswith("<seg") and line.endswith("</seg>"):
    i = line.index(">")
    return line[i + 1:-6]  # Strip first <seg ...> and last </seg>.


def _compile_data(tmp_dir, datasets, filename):
  """Concatenate all `datasets` and save to `filename`."""
  filename = os.path.join(tmp_dir, filename)
  with tf.gfile.GFile(filename + ".lang1", mode="w") as lang1_resfile:
    with tf.gfile.GFile(filename + ".lang2", mode="w") as lang2_resfile:
      for dataset in datasets:
        url = dataset[0]
        compressed_filename = os.path.basename(url)
        compressed_filepath = os.path.join(tmp_dir, compressed_filename)

        if dataset[1][0] == "tsv":
          _, src_column, trg_column, glob_pattern = dataset[1]
          filenames = tf.gfile.Glob(os.path.join(tmp_dir, glob_pattern))
          if not filenames:
            # Capture *.tgz and *.tar.gz too.
            mode = "r:gz" if compressed_filepath.endswith("gz") else "r"
            with tarfile.open(compressed_filepath, mode) as corpus_tar:
              corpus_tar.extractall(tmp_dir)
            filenames = tf.gfile.Glob(os.path.join(tmp_dir, glob_pattern))
          for tsv_filename in filenames:
            if tsv_filename.endswith(".gz"):
              new_filename = tsv_filename.strip(".gz")
              generator_utils.gunzip_file(tsv_filename, new_filename)
              tsv_filename = new_filename
            with tf.gfile.GFile(tsv_filename, mode="r") as tsv_file:
              for line in tsv_file:
                if line and "\t" in line:
                  parts = line.split("\t")
                  source, target = parts[src_column], parts[trg_column]
                  lang1_resfile.write(source.strip() + "\n")
                  lang2_resfile.write(target.strip() + "\n")
        else:
          lang1_filename, lang2_filename = dataset[1]
          lang1_filepath = os.path.join(tmp_dir, lang1_filename)
          lang2_filepath = os.path.join(tmp_dir, lang2_filename)
          is_sgm = (lang1_filename.endswith("sgm") and
                    lang2_filename.endswith("sgm"))

          if not (os.path.exists(lang1_filepath) and
                  os.path.exists(lang2_filepath)):
            # For .tar.gz and .tgz files, we read compressed.
            mode = "r:gz" if compressed_filepath.endswith("gz") else "r"
            with tarfile.open(compressed_filepath, mode) as corpus_tar:
              corpus_tar.extractall(tmp_dir)
          if lang1_filepath.endswith(".gz"):
            new_filepath = lang1_filepath.strip(".gz")
            generator_utils.gunzip_file(lang1_filepath, new_filepath)
            lang1_filepath = new_filepath
          if lang2_filepath.endswith(".gz"):
            new_filepath = lang2_filepath.strip(".gz")
            generator_utils.gunzip_file(lang2_filepath, new_filepath)
            lang2_filepath = new_filepath
          with tf.gfile.GFile(lang1_filepath, mode="r") as lang1_file:
            with tf.gfile.GFile(lang2_filepath, mode="r") as lang2_file:
              line1, line2 = lang1_file.readline(), lang2_file.readline()
              while line1 or line2:
                line1res = _preprocess_sgm(line1, is_sgm)
                line2res = _preprocess_sgm(line2, is_sgm)
                if line1res or line2res:
                  lang1_resfile.write(line1res.strip() + "\n")
                  lang2_resfile.write(line2res.strip() + "\n")
                line1, line2 = lang1_file.readline(), lang2_file.readline()

  return filename


@registry.register_problem
class TranslateEndeWmt8k(TranslateProblem):
  """Problem spec for WMT En-De translation."""

  @property
  def targeted_vocab_size(self):
    return 2**13  # 8192

  def generator(self, data_dir, tmp_dir, train):
    symbolizer_vocab = generator_utils.get_or_generate_vocab(
        data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size)
    datasets = _ENDE_TRAIN_DATASETS if train else _ENDE_TEST_DATASETS
    tag = "train" if train else "dev"
    data_path = _compile_data(tmp_dir, datasets, "wmt_ende_tok_%s" % tag)
    return token_generator(data_path + ".lang1", data_path + ".lang2",
                           symbolizer_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.DE_TOK


@registry.register_problem
class TranslateEndeWmt32k(TranslateEndeWmt8k):

  @property
  def targeted_vocab_size(self):
    return 2**15  # 32768


@registry.register_problem
class TranslateEndeWmtCharacters(TranslateProblem):
  """Problem spec for WMT En-De translation."""

  @property
  def is_character_level(self):
    return True

  def generator(self, _, tmp_dir, train):
    character_vocab = text_encoder.ByteTextEncoder()
    datasets = _ENDE_TRAIN_DATASETS if train else _ENDE_TEST_DATASETS
    tag = "train" if train else "dev"
    data_path = _compile_data(tmp_dir, datasets, "wmt_ende_chr_%s" % tag)
    return character_generator(data_path + ".lang1", data_path + ".lang2",
                               character_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def target_space_id(self):
    return problem.SpaceID.DE_CHR


@registry.register_problem
class TranslateEnzhWmt8k(TranslateProblem):
  """Problem spec for WMT Zh-En translation."""

  @property
  def targeted_vocab_size(self):
    return 2**13  # 8192

  @property
  def num_shards(self):
    return 10  # This is a small dataset.

  @property
  def source_vocab_name(self):
    return "vocab.zhen-zh.%d" % self.targeted_vocab_size

  @property
  def target_vocab_name(self):
    return "vocab.zhen-en.%d" % self.targeted_vocab_size

  def generator(self, data_dir, tmp_dir, train):
    datasets = _ZHEN_TRAIN_DATASETS if train else _ZHEN_TEST_DATASETS
    source_datasets = [[item[0], [item[1][0]]] for item in _ZHEN_TRAIN_DATASETS]
    target_datasets = [[item[0], [item[1][1]]] for item in _ZHEN_TRAIN_DATASETS]
    source_vocab = generator_utils.get_or_generate_vocab(
        data_dir, tmp_dir, self.source_vocab_name, self.targeted_vocab_size,
        source_datasets)
    target_vocab = generator_utils.get_or_generate_vocab(
        data_dir, tmp_dir, self.target_vocab_name, self.targeted_vocab_size,
        target_datasets)
    tag = "train" if train else "dev"
    data_path = _compile_data(tmp_dir, datasets, "wmt_zhen_tok_%s" % tag)
    # We generate English->X data by convention, to train reverse translation
    # just add the "_rev" suffix to the problem name, e.g., like this.
    #   --problems=translate_enzh_wmt8k_rev
    return bi_vocabs_token_generator(data_path + ".lang2", data_path + ".lang1",
                                     source_vocab, target_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.ZH_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_TOK

  def feature_encoders(self, data_dir):
    source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
    target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
    source_token = text_encoder.SubwordTextEncoder(source_vocab_filename)
    target_token = text_encoder.SubwordTextEncoder(target_vocab_filename)
    return {
        "inputs": source_token,
        "targets": target_token,
    }


@registry.register_problem
class TranslateEnfrWmt8k(TranslateProblem):
  """Problem spec for WMT En-Fr translation."""

  @property
  def targeted_vocab_size(self):
    return 2**13  # 8192

  def generator(self, data_dir, tmp_dir, train):
    symbolizer_vocab = generator_utils.get_or_generate_vocab(
        data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size)
    datasets = _ENFR_TRAIN_DATASETS if train else _ENFR_TEST_DATASETS
    tag = "train" if train else "dev"
    data_path = _compile_data(tmp_dir, datasets, "wmt_enfr_tok_%s" % tag)
    return token_generator(data_path + ".lang1", data_path + ".lang2",
                           symbolizer_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.FR_TOK


@registry.register_problem
class TranslateEnfrWmt32k(TranslateEnfrWmt8k):

  @property
  def targeted_vocab_size(self):
    return 2**15  # 32768


@registry.register_problem
class TranslateEnfrWmtCharacters(TranslateProblem):
  """Problem spec for WMT En-Fr translation."""

  @property
  def is_character_level(self):
    return True

  def generator(self, data_dir, tmp_dir, train):
    character_vocab = text_encoder.ByteTextEncoder()
    datasets = _ENFR_TRAIN_DATASETS if train else _ENFR_TEST_DATASETS
    tag = "train" if train else "dev"
    data_path = _compile_data(tmp_dir, datasets, "wmt_enfr_chr_%s" % tag)
    return character_generator(data_path + ".lang1", data_path + ".lang2",
                               character_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def target_space_id(self):
    return problem.SpaceID.FR_CHR


@registry.register_problem
class TranslateEnmkSetimes32k(TranslateProblem):
  """Problem spec for SETimes Mk-En translation."""

  @property
  def targeted_vocab_size(self):
    return 2**15  # 32768

  @property
  def vocab_name(self):
    return "vocab.mken"

  def generator(self, data_dir, tmp_dir, train):
    datasets = _MKEN_TRAIN_DATASETS if train else _MKEN_TEST_DATASETS
    source_datasets = [[item[0], [item[1][0]]] for item in datasets]
    target_datasets = [[item[0], [item[1][1]]] for item in datasets]
    symbolizer_vocab = generator_utils.get_or_generate_vocab(
        data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size,
        source_datasets + target_datasets)
    tag = "train" if train else "dev"
    data_path = _compile_data(tmp_dir, datasets, "setimes_mken_tok_%s" % tag)
    # We generate English->X data by convention, to train reverse translation
    # just add the "_rev" suffix to the problem name, e.g., like this.
    #   --problems=translate_enmk_setimes32k_rev
    return token_generator(data_path + ".lang2", data_path + ".lang1",
                           symbolizer_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.MK_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_TOK


@registry.register_problem
class TranslateEncsWmt32k(TranslateProblem):
  """Problem spec for WMT English-Czech translation."""

  @property
  def targeted_vocab_size(self):
    return 2**15  # 32768

  @property
  def vocab_name(self):
    return "vocab.encs"

  def generator(self, data_dir, tmp_dir, train):
    datasets = _ENCS_TRAIN_DATASETS if train else _ENCS_TEST_DATASETS
    tag = "train" if train else "dev"
    vocab_datasets = []
    data_path = _compile_data(tmp_dir, datasets, "wmt_encs_tok_%s" % tag)
    # CzEng contains 100 gz files with tab-separated columns, so let's expect
    # it is the first dataset in datasets and use the newly created *.lang{1,2}
    # files for vocab construction.
    if datasets[0][0].endswith("data-plaintext-format.tar"):
      vocab_datasets.append([datasets[0][0], ["wmt_encs_tok_%s.lang1" % tag,
                                              "wmt_encs_tok_%s.lang2" % tag]])
      datasets = datasets[1:]
    vocab_datasets += [[item[0], [item[1][0], item[1][1]]] for item in datasets]
    symbolizer_vocab = generator_utils.get_or_generate_vocab(
        data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size,
        vocab_datasets)
    return token_generator(data_path + ".lang1", data_path + ".lang2",
                           symbolizer_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.CS_TOK


@registry.register_problem
class TranslateEncsWmtCharacters(TranslateProblem):
  """Problem spec for WMT En-Cs character-based translation."""

  @property
  def is_character_level(self):
    return True

  def generator(self, data_dir, tmp_dir, train):
    character_vocab = text_encoder.ByteTextEncoder()
    datasets = _ENCS_TRAIN_DATASETS if train else _ENCS_TEST_DATASETS
    tag = "train" if train else "dev"
    data_path = _compile_data(tmp_dir, datasets, "wmt_encs_chr_%s" % tag)
    return character_generator(data_path + ".lang1", data_path + ".lang2",
                               character_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def target_space_id(self):
    return problem.SpaceID.CS_CHR


def parsing_token_generator(data_dir, tmp_dir, train, vocab_size):
  symbolizer_vocab = generator_utils.get_or_generate_vocab(
      data_dir, tmp_dir, "vocab.endefr.%d" % vocab_size, vocab_size)
  filename = "%s_%s.trees" % (FLAGS.parsing_path, "train" if train else "dev")
  tree_filepath = os.path.join(tmp_dir, filename)
  return wsj_parsing.token_generator(tree_filepath, symbolizer_vocab,
                                     symbolizer_vocab, EOS)
