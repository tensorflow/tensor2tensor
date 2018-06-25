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
"""Imports for problem modules."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import six

MODULES = [
    "tensor2tensor.data_generators.algorithmic",
    "tensor2tensor.data_generators.algorithmic_math",
    "tensor2tensor.data_generators.audio",
    "tensor2tensor.data_generators.babi_qa",
    "tensor2tensor.data_generators.bair_robot_pushing",
    "tensor2tensor.data_generators.celeba",
    "tensor2tensor.data_generators.cifar",
    "tensor2tensor.data_generators.cipher",
    "tensor2tensor.data_generators.cnn_dailymail",
    "tensor2tensor.data_generators.common_voice",
    "tensor2tensor.data_generators.desc2code",
    "tensor2tensor.data_generators.fsns",
    "tensor2tensor.data_generators.gene_expression",
    "tensor2tensor.data_generators.google_robot_pushing",
    "tensor2tensor.data_generators.gym_problems",
    "tensor2tensor.data_generators.ice_parsing",
    "tensor2tensor.data_generators.imagenet",
    "tensor2tensor.data_generators.imdb",
    "tensor2tensor.data_generators.lambada",
    "tensor2tensor.data_generators.librispeech",
    "tensor2tensor.data_generators.lm1b",
    "tensor2tensor.data_generators.mnist",
    "tensor2tensor.data_generators.mscoco",
    "tensor2tensor.data_generators.multinli",
    "tensor2tensor.data_generators.program_search",
    "tensor2tensor.data_generators.ocr",
    "tensor2tensor.data_generators.problem_hparams",
    "tensor2tensor.data_generators.ptb",
    "tensor2tensor.data_generators.snli",
    "tensor2tensor.data_generators.style_transfer",
    "tensor2tensor.data_generators.squad",
    "tensor2tensor.data_generators.subject_verb_agreement",
    "tensor2tensor.data_generators.timeseries",
    "tensor2tensor.data_generators.translate_encs",
    "tensor2tensor.data_generators.translate_ende",
    "tensor2tensor.data_generators.translate_enet",
    "tensor2tensor.data_generators.translate_enfr",
    "tensor2tensor.data_generators.translate_enid",
    "tensor2tensor.data_generators.translate_enmk",
    "tensor2tensor.data_generators.translate_envi",
    "tensor2tensor.data_generators.translate_enzh",
    "tensor2tensor.data_generators.twentybn",
    "tensor2tensor.data_generators.video_generated",
    "tensor2tensor.data_generators.wiki",
    "tensor2tensor.data_generators.wikisum.wikisum",
    "tensor2tensor.data_generators.wikitext103",
    "tensor2tensor.data_generators.wsj_parsing",
]
ALL_MODULES = list(MODULES)



def _py_err_msg(module):
  if six.PY2:
    msg = "No module named %s" % module.split(".")[-1]
  else:
    msg = "No module named '%s'" % module
  return msg


def _handle_errors(errors):
  """Log out and possibly reraise errors during import."""
  if not errors:
    return
  log_all = True  # pylint: disable=unused-variable
  err_msg = "Skipped importing {num_missing} data_generators modules."
  print(err_msg.format(num_missing=len(errors)))
  for module, err in errors:
    err_str = str(err)
    if err_str != _py_err_msg(module):
      print("From module %s" % module)
      raise err
    if log_all:
      print("Did not import module: %s; Cause: %s" % (module, err_str))


def import_modules(modules):
  errors = []
  for module in modules:
    try:
      importlib.import_module(module)
    except ImportError as error:
      errors.append((module, error))
  _handle_errors(errors)
