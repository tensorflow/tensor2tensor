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

"""Data generators for Wiki LM and MNLI combined datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import cnn_dailymail
from tensor2tensor.data_generators import multi_problem
from tensor2tensor.data_generators import multinli
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate_enfr
from tensor2tensor.data_generators import wiki_lm
from tensor2tensor.utils import registry


@registry.register_problem
class LanguagemodelEnWikiLMMultiNLISubwords(multi_problem.MultiProblem):
  """Wiki LM and MNLI mixed problem class."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(LanguagemodelEnWikiLMMultiNLISubwords, self).__init__(
        was_reversed, was_copy)
    self.task_list.append(wiki_lm.LanguagemodelEnWiki32k())
    self.task_list.append(multinli.MultiNLIWikiLMSharedVocab())

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD


@registry.register_problem
class LanguagemodelEnWikiLMMultiNLISubwords64k(multi_problem.MultiProblem):
  """Wiki LM and MNLI mixed problem class."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(LanguagemodelEnWikiLMMultiNLISubwords64k, self).__init__(
        was_reversed, was_copy)
    self.task_list.append(wiki_lm.LanguagemodelEnWiki64k())
    self.task_list.append(multinli.MultiNLIWikiLMSharedVocab64k())

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD


@registry.register_problem
class LanguagemodelEnWikiLMShortMultiNLISubwords64k(multi_problem.MultiProblem):
  """Wiki LM and MNLI mixed problem class."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(LanguagemodelEnWikiLMShortMultiNLISubwords64k, self).__init__(
        was_reversed, was_copy)
    self.task_list.append(wiki_lm.LanguagemodelEnWiki64kShorter())
    self.task_list.append(multinli.MultiNLIWikiLMSharedVocab64k())

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD


@registry.register_problem
class LanguagemodelEnWikiLMSummarizeCnndmSubwords(multi_problem.MultiProblem):
  """Wiki LM and CNN/DM summarization mixed problem class."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(LanguagemodelEnWikiLMSummarizeCnndmSubwords, self).__init__(
        was_reversed, was_copy)
    self.task_list.append(wiki_lm.LanguagemodelEnWiki32k())
    self.task_list.append(
        cnn_dailymail.SummarizeCnnDailymailWikiLMSharedVocab())

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD


@registry.register_problem
class LanguagemodelEnWikiLMSummarizeCnndmSubwords64k(
    multi_problem.MultiProblem):
  """Wiki LM and CNN/DM summarization mixed problem class."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(LanguagemodelEnWikiLMSummarizeCnndmSubwords64k, self).__init__(
        was_reversed, was_copy)
    self.task_list.append(wiki_lm.LanguagemodelEnWiki64k())
    self.task_list.append(
        cnn_dailymail.SummarizeCnnDailymailWikiLMSharedVocab64k())

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD


@registry.register_problem
class LanguagemodelMultiWikiTranslateFr(multi_problem.MultiProblem):
  """Wiki multi-lingual LM and En-Fr translation."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(LanguagemodelMultiWikiTranslateFr, self).__init__(
        was_reversed, was_copy)
    self.task_list.append(wiki_lm.LanguagemodelDeEnFrRoWiki64k())
    self.task_list.append(translate_enfr.TranslateEnfrWmtMulti64k())

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD
