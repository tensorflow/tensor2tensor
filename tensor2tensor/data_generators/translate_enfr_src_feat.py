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

"""
Data generators for translation data-sets with multiple source features.

Problem, model and source feature input modalities are all defined
in data_generator/translate_source_features.py

In order to run data generation, training and inference, here are
the requirements:

PROBLEM=translate_enfr_src_feat (defined below)
MODEL=transformer_src_features
HPARAMS=transformer_sfeats_hparams (defined below)

The last hparams set is defined in the current file and can be
based on any hparams defined for the transformer model.

There can be an arbitrary number of source features. These are
provided in a separate file, where words are separated by space
and the word features by TranslateEnfrSrcFeat.sfeat_delimiter.

Example with parts-of-speech and lemmas as source features ('|'
as delimiter):
source file: I hated cats .
sfeats file: Pronoun|I Verb-Past|hate Noun-Plural|cat Punctuation|.

Since all features must be synchronized at word level, the inputs
(and outputs) must be already tokenized. Words (i.e. the first feature)
are then split into subword units and source features are multiplied
in order to keep the same unique sentence length. To add subword tags
marking the beginning, inside and end of a word, set
TranslateEnfrSrcFeat.use_subword_tags() below to True (and give a
size to this last feature embedding in "source_feature_sizes").

For inference, the file containing the features must be given in
a separate argument for t2t-decoder:

--source_feature_file=/path/to/my/source_feature_file
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import translate_source_features
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry

import tensorflow as tf


_ENFR_TRAIN_DATA = [
    [
        "https://dl.dropboxusercontent.com/s/d8i4qdrm01s5wef/data_onmt.tgz",
        ("baseline-1M_train.clean.en.tok",
         "baseline-1M_train.clean.fr.tok"),
        "baseline-1M_train.clean.en.sft"
    ],
]
_ENFR_DEV_DATA = [
    [
        "https://dl.dropboxusercontent.com/s/d8i4qdrm01s5wef/data_onmt.tgz",
        ("baseline-1M_valid.clean.en.tok",
         "baseline-1M_valid.clean.fr.tok"),
        "baseline-1M_valid.clean.en.sft"
    ],
]


@registry.register_hparams
def transformer_sfeats_hparams():
  # define initial transformer hparams here
  hp = transformer.transformer_base()
  #hp = transformer.transformer_big()

  # feature vector size setting
  # the order of the features is the same
  # as in the source feature file. All
  # sizes are separated by ':'
  hp.add_hparam("source_feature_sizes", "16:56:8")
  # set encoder hidden size
  ehs = sum([int(size) for size in hp.source_feature_sizes.split(':')])
  ehs += hp.hidden_size
  hp.add_hparam("enc_hidden_size", ehs)
  return hp


class VocabType(object):
  """Available text vocabularies."""
  CHARACTER = "character"
  SUBWORD = "subwords"
  TOKEN = "tokens"


@registry.register_problem
class TranslateEnfrSrcFeat(translate_source_features.SourceFeatureProblem):

  @property
  def approx_vocab_size(self):
    return 2**15 

  @property
  def vocab_filename(self):
    return "vocab.enfr.%d" % self.approx_vocab_size
    
  @property
  def sfeat_delimiter(self):
    r"""Source feature delimiter in feature file"""
    return '|'

  @property
  def use_subword_tags(self):
    r"""use subword tags: these will be generated
    when the source words are subword encoded.
    This source feature is the last one among
    all other features and its vector size must
    be set in hparams (source_feature_sizes).
    """
    return True

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

  def vocab_sfeat_filenames(self, f_id: int):
    r"""One vocab per feature type"""
    return "vocab.enfr.sfeat.%d" % f_id

  def vocab_data_files(self):
    return _ENFR_TRAIN_DATA
    
  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    datasets = _ENFR_TRAIN_DATA if train else _ENFR_DEV_DATA
    return datasets

