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

"""Data generators for Wiki LM and MNLI combined datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import cnn_dailymail
from tensor2tensor.data_generators import multi_problem
from tensor2tensor.data_generators import multi_problem_v2
from tensor2tensor.data_generators import multinli
from tensor2tensor.data_generators import squad
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate_ende
from tensor2tensor.data_generators import translate_enfr
from tensor2tensor.data_generators import translate_enro
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
class LanguagemodelEnWikiLMMultiNLISubwordsV2(
    multi_problem_v2.MultiText2TextProblem):
  """Wiki LM and MNLI mixed problem class."""

  def __init__(self, was_reversed=False, was_copy=False):
    problems = [
        wiki_lm.LanguagemodelEnWiki32k(),
        multinli.MultiNLIWikiLMSharedVocab(),
    ]
    schedule = multi_problem_v2.constant_schedule([0.5, 0.5])
    super(LanguagemodelEnWikiLMMultiNLISubwordsV2, self).__init__(
        problems, schedule, was_reversed=was_reversed, was_copy=was_copy)

  @property
  def has_inputs(self):
    return False

  @property
  def use_vocab_from_other_problem(self):
    return wiki_lm.LanguagemodelEnWiki32k()

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD


@registry.register_problem
class LanguagemodelMultiWikiTranslatePacked1k(
    multi_problem_v2.MultiText2TextProblem):
  """Wiki-LM, Translation, MNLI, SQUAD mixed problem class."""

  def __init__(self, was_reversed=False, was_copy=False):
    problems = []
    rates = []
    for rate, also_reverse, cls in self.problems_and_rates:
      for r in [False, True] if also_reverse else [False]:
        problems.append(cls(was_reversed=r))
        rates.append(rate)
    pmf = multi_problem_v2.epoch_rates_to_pmf(problems, epoch_rates=rates)
    schedule = multi_problem_v2.constant_schedule(pmf)
    super(LanguagemodelMultiWikiTranslatePacked1k, self).__init__(
        problems, schedule, was_reversed=was_reversed, was_copy=was_copy)

  @property
  def problems_and_rates(self):
    """Returns a list of (weight, also_reverse, problem_class) triples."""
    return [
        (1.0, True, wiki_lm.LanguagemodelDeEnFrRoWiki64kFitbPacked1k),
        (1.0, True, translate_ende.TranslateEndeWmtMulti64kPacked1k),
        (1.0, True, translate_enfr.TranslateEnfrWmtMulti64kPacked1k),
        (1.0, True, translate_enro.TranslateEnroWmtMultiTiny64kPacked1k),
        (1.0, True, cnn_dailymail.SummarizeCnnDailymailMulti64kPacked1k),
        (1.0, False, multinli.MultiNLIText2textMulti64kPacked1k),
        (1.0, False, squad.SquadText2textMulti64kPacked1k),
    ]

  @property
  def has_inputs(self):
    return True

  @property
  def use_vocab_from_other_problem(self):
    return wiki_lm.LanguagemodelDeEnFrRoWiki64k()

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD

  @property
  def packed_length(self):
    return 1024


@registry.register_problem
class LanguagemodelMultiWikiTranslatePacked1kV2(
    LanguagemodelMultiWikiTranslatePacked1k):
  """Higher rates for rarer problems."""

  @property
  def problems_and_rates(self):
    """Returns a list of (weight, also_reverse, problem_class) triples."""
    return [
        (1.0, True, wiki_lm.LanguagemodelDeEnFrRoWiki64kFitbPacked1k),
        (3.0, True, translate_ende.TranslateEndeWmtMulti64kPacked1k),
        (1.0, True, translate_enfr.TranslateEnfrWmtMulti64kPacked1k),
        (100.0, True, translate_enro.TranslateEnroWmtMultiTiny64kPacked1k),
        (1.0, True, cnn_dailymail.SummarizeCnnDailymailMulti64kPacked1k),
        (10.0, False, multinli.MultiNLIText2textMulti64kPacked1k),
        (10.0, False, squad.SquadText2textMulti64kPacked1k),
    ]


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


@registry.register_problem
class LanguagemodelMultiWikiTranslate(multi_problem.MultiProblem):
  """Wiki multi-lingual LM and multiple translations."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(LanguagemodelMultiWikiTranslate, self).__init__(
        was_reversed, was_copy)
    self.task_list.append(wiki_lm.LanguagemodelDeEnFrRoWiki64k())
    self.task_list.append(translate_ende.TranslateEndeWmtMulti64k())
    self.task_list.append(translate_enfr.TranslateEnfrWmtMulti64k())
    self.task_list.append(translate_enro.TranslateEnroWmtMultiTiny64k())
    self.task_list.append(translate_ende.TranslateEndeWmtMulti64k(
        was_reversed=True))
    self.task_list.append(translate_enfr.TranslateEnfrWmtMulti64k(
        was_reversed=True))
    self.task_list.append(translate_enro.TranslateEnroWmtMultiTiny64k(
        was_reversed=True))
    self.task_list.append(
        cnn_dailymail.SummarizeCnnDailymailWikiLMMultiVocab64k())
    self.task_list.append(multinli.MultiNLIWikiLMMultiVocab64k())
    self.task_list.append(squad.SquadConcatMulti64k())

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD


@registry.register_problem
class LanguagemodelEnWikiLMSummarizeFrac1CnndmSubwords64k(
    multi_problem.MultiProblem):
  """Wiki LM and CNN/DM summarization mixed problem class."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(LanguagemodelEnWikiLMSummarizeFrac1CnndmSubwords64k, self).__init__(
        was_reversed, was_copy)
    self.task_list.append(wiki_lm.LanguagemodelEnWiki64k())
    self.task_list.append(
        cnn_dailymail.SummarizeFrac1CnnDailymailWikiLMSharedVocab64k())

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD


@registry.register_problem
class LanguagemodelEnWikiLMSummarizeFrac2CnndmSubwords64k(
    multi_problem.MultiProblem):
  """Wiki LM and CNN/DM summarization mixed problem class."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(LanguagemodelEnWikiLMSummarizeFrac2CnndmSubwords64k, self).__init__(
        was_reversed, was_copy)
    self.task_list.append(wiki_lm.LanguagemodelEnWiki64k())
    self.task_list.append(
        cnn_dailymail.SummarizeFrac2CnnDailymailWikiLMSharedVocab64k())

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD


@registry.register_problem
class LanguagemodelEnWikiLMSummarizeFrac5CnndmSubwords64k(
    multi_problem.MultiProblem):
  """Wiki LM and CNN/DM summarization mixed problem class."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(LanguagemodelEnWikiLMSummarizeFrac5CnndmSubwords64k, self).__init__(
        was_reversed, was_copy)
    self.task_list.append(wiki_lm.LanguagemodelEnWiki64k())
    self.task_list.append(
        cnn_dailymail.SummarizeFrac5CnnDailymailWikiLMSharedVocab64k())

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD


@registry.register_problem
class LanguagemodelEnWikiLMSummarizeFrac10CnndmSubwords64k(
    multi_problem.MultiProblem):
  """Wiki LM and CNN/DM summarization mixed problem class."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(LanguagemodelEnWikiLMSummarizeFrac10CnndmSubwords64k, self).__init__(
        was_reversed, was_copy)
    self.task_list.append(wiki_lm.LanguagemodelEnWiki64k())
    self.task_list.append(
        cnn_dailymail.SummarizeFrac10CnnDailymailWikiLMSharedVocab64k())

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD


@registry.register_problem
class LanguagemodelEnWikiLMSummarizeFrac20CnndmSubwords64k(
    multi_problem.MultiProblem):
  """Wiki LM and CNN/DM summarization mixed problem class."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(LanguagemodelEnWikiLMSummarizeFrac20CnndmSubwords64k, self).__init__(
        was_reversed, was_copy)
    self.task_list.append(wiki_lm.LanguagemodelEnWiki64k())
    self.task_list.append(
        cnn_dailymail.SummarizeFrac20CnnDailymailWikiLMSharedVocab64k())

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD


@registry.register_problem
class LanguagemodelEnWikiLMSummarizeFrac50CnndmSubwords64k(
    multi_problem.MultiProblem):
  """Wiki LM and CNN/DM summarization mixed problem class."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(LanguagemodelEnWikiLMSummarizeFrac50CnndmSubwords64k, self).__init__(
        was_reversed, was_copy)
    self.task_list.append(wiki_lm.LanguagemodelEnWiki64k())
    self.task_list.append(
        cnn_dailymail.SummarizeFrac50CnnDailymailWikiLMSharedVocab64k())

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD


@registry.register_problem
class LanguagemodelEnWikiLMSquadConcatSubwords(multi_problem.MultiProblem):
  """Wiki LM and MNLI mixed problem class."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(LanguagemodelEnWikiLMSquadConcatSubwords, self).__init__(
        was_reversed, was_copy)
    self.task_list.append(wiki_lm.LanguagemodelEnWiki32k())
    self.task_list.append(multinli.SquadConcatSharedVocab())

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD
