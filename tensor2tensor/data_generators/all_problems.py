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

"""Imports for problem modules."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import six
from six.moves import range  # pylint: disable=redefined-builtin

MODULES = [
    "tensor2tensor.data_generators.algorithmic",
    "tensor2tensor.data_generators.algorithmic_math",
    "tensor2tensor.data_generators.algorithmic_math_deepmind",
    "tensor2tensor.data_generators.algorithmic_math_two_variables",
    "tensor2tensor.data_generators.allen_brain",
    "tensor2tensor.data_generators.audio",
    "tensor2tensor.data_generators.babi_qa",
    "tensor2tensor.data_generators.bair_robot_pushing",
    "tensor2tensor.data_generators.celeba",
    "tensor2tensor.data_generators.celebahq",
    "tensor2tensor.data_generators.cifar",
    "tensor2tensor.data_generators.cipher",
    "tensor2tensor.data_generators.cnn_dailymail",
    "tensor2tensor.data_generators.cola",
    "tensor2tensor.data_generators.common_voice",
    "tensor2tensor.data_generators.desc2code",
    "tensor2tensor.data_generators.dialog_cornell",
    "tensor2tensor.data_generators.dialog_dailydialog",
    "tensor2tensor.data_generators.dialog_opensubtitles",
    "tensor2tensor.data_generators.dialog_personachat",
    "tensor2tensor.data_generators.enwik8",
    "tensor2tensor.data_generators.fsns",
    "tensor2tensor.data_generators.function_docstring",
    "tensor2tensor.data_generators.gene_expression",
    "tensor2tensor.data_generators.google_robot_pushing",
    "tensor2tensor.data_generators.gym_env",
    "tensor2tensor.data_generators.ice_parsing",
    "tensor2tensor.data_generators.imagenet",
    "tensor2tensor.data_generators.image_lsun",
    "tensor2tensor.data_generators.imdb",
    "tensor2tensor.data_generators.lambada",
    "tensor2tensor.data_generators.librispeech",
    "tensor2tensor.data_generators.lm1b",
    "tensor2tensor.data_generators.lm1b_imdb",
    "tensor2tensor.data_generators.lm1b_mnli",
    "tensor2tensor.data_generators.mnist",
    "tensor2tensor.data_generators.moving_mnist",
    "tensor2tensor.data_generators.mrpc",
    "tensor2tensor.data_generators.mscoco",
    "tensor2tensor.data_generators.multinli",
    "tensor2tensor.data_generators.paraphrase_ms_coco",
    "tensor2tensor.data_generators.program_search",
    "tensor2tensor.data_generators.ocr",
    "tensor2tensor.data_generators.pointer_generator_word",
    "tensor2tensor.data_generators.problem_hparams",
    "tensor2tensor.data_generators.ptb",
    "tensor2tensor.data_generators.qnli",
    "tensor2tensor.data_generators.quora_qpairs",
    "tensor2tensor.data_generators.rte",
    "tensor2tensor.data_generators.scitail",
    "tensor2tensor.data_generators.snli",
    "tensor2tensor.data_generators.stanford_nli",
    "tensor2tensor.data_generators.style_transfer",
    "tensor2tensor.data_generators.squad",
    "tensor2tensor.data_generators.sst_binary",
    "tensor2tensor.data_generators.subject_verb_agreement",
    "tensor2tensor.data_generators.timeseries",
    "tensor2tensor.data_generators.transduction_problems",
    "tensor2tensor.data_generators.translate_encs",
    "tensor2tensor.data_generators.translate_ende",
    "tensor2tensor.data_generators.translate_enes",
    "tensor2tensor.data_generators.translate_enet",
    "tensor2tensor.data_generators.translate_enfr",
    "tensor2tensor.data_generators.translate_enid",
    "tensor2tensor.data_generators.translate_enmk",
    "tensor2tensor.data_generators.translate_envi",
    "tensor2tensor.data_generators.translate_enzh",
    "tensor2tensor.data_generators.video_generated",
    "tensor2tensor.data_generators.vqa",
    "tensor2tensor.data_generators.wiki",
    "tensor2tensor.data_generators.wiki_lm",
    "tensor2tensor.data_generators.wiki_revision",
    "tensor2tensor.data_generators.wiki_multi_problems",
    "tensor2tensor.data_generators.wikisum.wikisum",
    "tensor2tensor.data_generators.wikitext103",
    "tensor2tensor.data_generators.wsj_parsing",
    "tensor2tensor.data_generators.wnli",
    "tensor2tensor.data_generators.yelp_polarity",
    "tensor2tensor.data_generators.yelp_full",
    "tensor2tensor.envs.mujoco_problems",
    "tensor2tensor.envs.tic_tac_toe_env_problem",
]
ALL_MODULES = list(MODULES)



def _is_import_err_msg(err_str, module):
  parts = module.split(".")
  suffixes = [".".join(parts[i:]) for i in range(len(parts))]
  return err_str in (
      ["No module named %s" % suffix for suffix in suffixes] +
      ["No module named '%s'" % suffix for suffix in suffixes])


def _handle_errors(errors):
  """Log out and possibly reraise errors during import."""
  if not errors:
    return
  log_all = True  # pylint: disable=unused-variable
  err_msg = "T2T: skipped importing {num_missing} data_generators modules."
  print(err_msg.format(num_missing=len(errors)))
  for module, err in errors:
    err_str = str(err)
    if log_all:
      print("Did not import module: %s; Cause: %s" % (module, err_str))
    if not _is_import_err_msg(err_str, module):
      print("From module %s" % module)
      raise err


def import_modules(modules):
  errors = []
  for module in modules:
    try:
      importlib.import_module(module)
    except ImportError as error:
      errors.append((module, error))
  _handle_errors(errors)
