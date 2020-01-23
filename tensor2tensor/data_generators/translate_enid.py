# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""Data generators for En-Id translation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

_REPO = "https://github.com/prasastoadi/parallel-corpora-en-id/raw/master/"

# IWSLT17 :
# 109335 sentences
# https://wit3.fbk.eu/mt.php?release=2017-01-more
# PANL-BPPT :
# 24024 sentences
# http://www.panl10n.net/english/outputs/Indonesia/BPPT/0902/BPPTIndToEngCorpusHalfM.zip # pylint: disable=line-too-long
_ENID_TRAIN_DATASETS = [
    [
        _REPO + "IWSLT17.train.en-id.tgz",
        ("IWSLT17.train.en-id.en", "IWSLT17.train.en-id.id")
    ],
    [
        _REPO + "PANL-BPPT-ECO-EN-ID-150Kw.tgz",
        ("PANL-BPPT-ECO-EN-150Kw.txt", "PANL-BPPT-ECO-ID-150Kw.txt")
    ],
    [
        _REPO + "PANL-BPPT-INT-EN-ID-150Kw.tgz",
        ("PANL-BPPT-INT-EN-150Kw.txt", "PANL-BPPT-INT-ID-150Kw.txt")
    ],
    [
        _REPO + "PANL-BPPT-SCI-EN-ID-100Kw.tgz",
        ("PANL-BPPT-SCI-EN-100Kw.txt", "PANL-BPPT-SCI-ID-100Kw.txt")
    ],
    [
        _REPO + "PANL-BPPT-SPO-EN-ID-100Kw.tgz",
        ("PANL-BPPT-SPO-EN-100Kw.txt", "PANL-BPPT-SPO-ID-100Kw.txt")
    ],
]

# IWSLT17 :
# 1478 sentences
# https://wit3.fbk.eu/mt.php?release=2017-01-more
_ENID_TEST_DATASETS = [
    [
        _REPO + "IWSLT17.TED.tst2017plus.en-id.tgz",
        ("IWSLT17.TED.tst2017plus.en-id.en",
         "IWSLT17.TED.tst2017plus.en-id.id")
    ]
]


@registry.register_problem
class TranslateEnidIwslt32k(translate.TranslateProblem):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENID_TRAIN_DATASETS if train else _ENID_TEST_DATASETS
