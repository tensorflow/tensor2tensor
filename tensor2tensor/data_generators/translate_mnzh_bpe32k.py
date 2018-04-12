#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Jack on 04/02/2018

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
class TranslateMnzhBpe32k(translate.TranslateProblem):
  """Problem spec for WMT En-De translation, BPE version."""

  @property
  def targeted_vocab_size(self):
    return 32000

  @property
  def vocab_name(self):
    return "vocab.bpe"

  def source_vocab_name(self):
    return "vocab.32k.mn.txt"

  def target_vocab_name(self):
      return "vocab.32k.zh.txt"

  def feature_encoders(self, data_dir):
    source_vocab_filename = os.path.join(data_dir, self.source_vocab_name())
    target_vocab_filename = os.path.join(data_dir, self.target_vocab_name())
    source_encoder = text_encoder.TokenTextEncoder(source_vocab_filename, replace_oov="UNK")
    target_encoder = text_encoder.TokenTextEncoder(target_vocab_filename, replace_oov="UNK")
    return {"inputs": source_encoder, "targets": target_encoder}

  def generator(self, data_dir, tmp_dir, train):
    """Instance of token generator for the mn->zh task, training set."""
    dataset_path = ("train.32k"
                    if train else "valid.32k")
    train_path = os.path.join(data_dir, dataset_path)

    source_token_path = os.path.join(data_dir, self.source_vocab_name())
    target_token_path = os.path.join(data_dir, self.target_vocab_name())
    for token_path in [source_token_path, target_token_path]:
        with tf.gfile.GFile(token_path, mode="r") as f:
          vocab_data = "<pad>\n<EOS>\n" + f.read() + "UNK\n"
        with tf.gfile.GFile(token_path, mode="w") as f:
          f.write(vocab_data)
    source_token_vocab = text_encoder.TokenTextEncoder(source_token_path, replace_oov="UNK")
    target_token_vocab = text_encoder.TokenTextEncoder(source_token_path, replace_oov="UNK")
    return translate.token_generator(train_path + ".mn", train_path + ".zh",
                                     source_token_vocab, target_token_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.MN_BPE_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.ZH_BPE_TOK
