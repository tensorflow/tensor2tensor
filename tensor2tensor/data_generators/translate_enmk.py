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


@registry.register_problem
class TranslateEnmkSetimes32k(translate.TranslateProblem):
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
    data_path = translate.compile_data(tmp_dir, datasets,
                                       "setimes_mken_tok_%s" % tag)
    # We generate English->X data by convention, to train reverse translation
    # just add the "_rev" suffix to the problem name, e.g., like this.
    #   --problems=translate_enmk_setimes32k_rev
    return translate.token_generator(data_path + ".lang2", data_path + ".lang1",
                                     symbolizer_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.MK_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_TOK
