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

"""Data generators for LM1B and IMDb combined data-set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import imdb
from tensor2tensor.data_generators import lm1b
from tensor2tensor.data_generators import multi_problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry


@registry.register_problem
class LanguagemodelLm1bSentimentIMDB(multi_problem.MultiProblem):
  """LM1b and IMDb mixed problem class for multitask learning."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(LanguagemodelLm1bSentimentIMDB, self).__init__(was_reversed, was_copy)
    self.task_list.append(lm1b.LanguagemodelLm1bCharacters())
    self.task_list.append(imdb.SentimentIMDBCharacters())

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER
