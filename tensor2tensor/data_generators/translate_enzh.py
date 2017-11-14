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

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

# This is far from being the real WMT17 task - only toyset here
# you need to register to get UN data and CWT data. Also, by convention,
# this is EN to ZH - use translate_enzh_wmt8k_rev for ZH to EN task
_ENZH_TRAIN_DATASETS = [[("http://data.statmt.org/wmt17/translation-task/"
                          "training-parallel-nc-v12.tgz"),
                         ("training/news-commentary-v12.zh-en.en",
                          "training/news-commentary-v12.zh-en.zh")]]

_ENZH_TEST_DATASETS = [[
    "http://data.statmt.org/wmt17/translation-task/dev.tgz",
    ("dev/newsdev2017-zhen-src.en.sgm", "dev/newsdev2017-zhen-ref.zh.sgm")
]]


@registry.register_problem
class TranslateEnzhWmt8k(translate.TranslateProblem):
  """Problem spec for WMT En-Zh translation."""

  @property
  def targeted_vocab_size(self):
    return 2**13  # 8192

  @property
  def num_shards(self):
    return 10  # This is a small dataset.

  @property
  def source_vocab_name(self):
    return "vocab.enzh-en.%d" % self.targeted_vocab_size

  @property
  def target_vocab_name(self):
    return "vocab.enzh-zh.%d" % self.targeted_vocab_size

  def generator(self, data_dir, tmp_dir, train):
    datasets = _ENZH_TRAIN_DATASETS if train else _ENZH_TEST_DATASETS
    source_datasets = [[item[0], [item[1][0]]] for item in _ENZH_TRAIN_DATASETS]
    target_datasets = [[item[0], [item[1][1]]] for item in _ENZH_TRAIN_DATASETS]
    source_vocab = generator_utils.get_or_generate_vocab(
        data_dir, tmp_dir, self.source_vocab_name, self.targeted_vocab_size,
        source_datasets)
    target_vocab = generator_utils.get_or_generate_vocab(
        data_dir, tmp_dir, self.target_vocab_name, self.targeted_vocab_size,
        target_datasets)
    tag = "train" if train else "dev"
    data_path = translate.compile_data(tmp_dir, datasets,
                                       "wmt_enzh_tok_%s" % tag)
    return translate.bi_vocabs_token_generator(data_path + ".lang1",
                                               data_path + ".lang2",
                                               source_vocab, target_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.ZH_TOK

  def feature_encoders(self, data_dir):
    source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
    target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
    source_token = text_encoder.SubwordTextEncoder(source_vocab_filename)
    target_token = text_encoder.SubwordTextEncoder(target_vocab_filename)
    return {
        "inputs": source_token,
        "targets": target_token,
    }
