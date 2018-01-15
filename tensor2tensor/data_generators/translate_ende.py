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
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

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
class TranslateEndeWmtBpe32k(translate.TranslateProblem):
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
    with tf.gfile.GFile(token_path, mode="r") as f:
      vocab_data = "<pad>\n<EOS>\n" + f.read() + "UNK\n"
    with tf.gfile.GFile(token_path, mode="w") as f:
      f.write(vocab_data)
    token_vocab = text_encoder.TokenTextEncoder(token_path, replace_oov="UNK")
    return translate.token_generator(train_path + ".en", train_path + ".de",
                                     token_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_BPE_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.DE_BPE_TOK


@registry.register_problem
class TranslateEndeWmt8k(translate.TranslateProblem):
  """Problem spec for WMT En-De translation."""

  @property
  def targeted_vocab_size(self):
    return 2**13  # 8192

  @property
  def vocab_name(self):
    return "vocab.ende"

  def generator(self, data_dir, tmp_dir, train):
    symbolizer_vocab = generator_utils.get_or_generate_vocab(
        data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size,
        _ENDE_TRAIN_DATASETS)
    datasets = _ENDE_TRAIN_DATASETS if train else _ENDE_TEST_DATASETS
    tag = "train" if train else "dev"
    data_path = translate.compile_data(tmp_dir, datasets,
                                       "wmt_ende_tok_%s" % tag)
    return translate.token_generator(data_path + ".lang1", data_path + ".lang2",
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
class TranslateEndeWmt32kPacked(TranslateEndeWmt32k):

  @property
  def packed_length(self):
    return 256


@registry.register_problem
class TranslateEndeWmtCharacters(translate.TranslateProblem):
  """Problem spec for WMT En-De translation."""

  @property
  def is_character_level(self):
    return True

  @property
  def vocab_name(self):
    return "vocab.ende"

  def generator(self, _, tmp_dir, train):
    character_vocab = text_encoder.ByteTextEncoder()
    datasets = _ENDE_TRAIN_DATASETS if train else _ENDE_TEST_DATASETS
    tag = "train" if train else "dev"
    data_path = translate.compile_data(tmp_dir, datasets,
                                       "wmt_ende_chr_%s" % tag)
    return translate.character_generator(
        data_path + ".lang1", data_path + ".lang2", character_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def target_space_id(self):
    return problem.SpaceID.DE_CHR
