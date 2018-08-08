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
"""Data generators for LM1B and MNLI combined datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import lm1b
from tensor2tensor.data_generators import multi_problem
from tensor2tensor.data_generators import multinli
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry


@registry.register_problem
class LanguagemodelLm1bMultiNLISubwords(multi_problem.MultiProblem):
  """LM1b and MNLI mixed problem class for multitask learning."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(LanguagemodelLm1bMultiNLISubwords, self).__init__(
        was_reversed, was_copy)
    self.task_list.append(lm1b.LanguagemodelLm1b32k())
    self.task_list.append(multinli.MultiNLISharedVocab())

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD


@registry.register_problem
class LanguagemodelLm1bMultiNLI(multi_problem.MultiProblem):
  """LM1b and MNLI mixed problem class for multitask learning."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(LanguagemodelLm1bMultiNLI, self).__init__(was_reversed, was_copy)
    self.task_list.append(lm1b.LanguagemodelLm1bCharacters())
    self.task_list.append(multinli.MultiNLICharacters())

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER
