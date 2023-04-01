# coding=utf-8
# Copyright 2023 The Tensor2Tensor Authors.
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

"""Data generators for English-Czech backtranslation NMT data-sets.

To use this problem you need to provide backtranslated (synthetic) data to
tmp_dir (cs_mono_{en,cs}.txt{0,1,2} - each file of a similar size to the
authentic training data).
You can either translate the monolingual data yourself or you can download
"csmono" data from CzEng2.0 (http://ufal.mff.cuni.cz/czeng, registration needed)
which comes with synthetic translations into English using a
backtranslation-trained model, thus the final model will be using
"iterated" backtranslation.

To get the best results out of the Block-Backtranslation
(where blocks of synthetic and authentic training data are concatenated
without shuffling), you should use checkpoint averaging (see t2t-avg-all).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import translate_encs
from tensor2tensor.utils import registry


@registry.register_problem
class TranslateEncsCubbitt(translate_encs.TranslateEncsWmt32k):
  """Problem spec for English-Czech CUBBITT (CUni Block-Backtranslation-Improved Transformer Translation)."""

  @property
  def use_vocab_from_other_problem(self):
    return translate_encs.TranslateEncsWmt32k()

  @property
  def already_shuffled(self):
    return True

  @property
  def skip_random_fraction_when_training(self):
    return False

  @property
  def backtranslate_data_filenames(self):
    """List of pairs of files with matched back-translated data."""
    # Files must be placed in tmp_dir, each similar size to authentic data.
    return [("cs_mono_en.txt%d" % i, "cs_mono_cs.txt%d" % i) for i in [0, 1, 2]]

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 1,  # Use just 1 shard so as to not mix data.
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    datasets = self.source_data_files(dataset_split)
    tag = "train" if dataset_split == problem.DatasetSplit.TRAIN else "dev"
    data_path = translate.compile_data(
        tmp_dir, datasets, "%s-compiled-%s" % (self.name, tag))
    # For eval, use authentic data.
    if dataset_split != problem.DatasetSplit.TRAIN:
      for example in text_problems.text2text_txt_iterator(
          data_path + ".lang1", data_path + ".lang2"):
        yield example
    else:  # For training, mix synthetic and authentic data as follows.
      for (file1, file2) in self.backtranslate_data_filenames:
        path1 = os.path.join(tmp_dir, file1)
        path2 = os.path.join(tmp_dir, file2)
        # Synthetic data first.
        for example in text_problems.text2text_txt_iterator(path1, path2):
          yield example
        # Now authentic data.
        for example in text_problems.text2text_txt_iterator(
            data_path + ".lang1", data_path + ".lang2"):
          yield example
