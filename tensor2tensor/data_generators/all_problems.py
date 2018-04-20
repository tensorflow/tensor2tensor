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


modules = [
    "tensor2tensor.data_generators.algorithmic",
    "tensor2tensor.data_generators.algorithmic_math",
    "tensor2tensor.data_generators.audio",
    "tensor2tensor.data_generators.celeba",
    "tensor2tensor.data_generators.cifar",
    "tensor2tensor.data_generators.cipher",
    "tensor2tensor.data_generators.cnn_dailymail",
    "tensor2tensor.data_generators.desc2code",
    "tensor2tensor.data_generators.fsns",
    "tensor2tensor.data_generators.gene_expression",
    "tensor2tensor.data_generators.gym",
    "tensor2tensor.data_generators.ice_parsing",
    "tensor2tensor.data_generators.imagenet",
    "tensor2tensor.data_generators.imdb",
    "tensor2tensor.data_generators.librispeech",
    "tensor2tensor.data_generators.lm1b",
    "tensor2tensor.data_generators.mnist",
    "tensor2tensor.data_generators.mscoco",
    "tensor2tensor.data_generators.multinli",
    "tensor2tensor.data_generators.ocr",
    "tensor2tensor.data_generators.problem_hparams",
    "tensor2tensor.data_generators.ptb",
    "tensor2tensor.data_generators.snli",
    "tensor2tensor.data_generators.squad",
    "tensor2tensor.data_generators.translate_encs",
    "tensor2tensor.data_generators.translate_ende",
    "tensor2tensor.data_generators.translate_enfr",
    "tensor2tensor.data_generators.translate_enmk",
    "tensor2tensor.data_generators.translate_envi",
    "tensor2tensor.data_generators.translate_enzh",
    "tensor2tensor.data_generators.twentybn",
    "tensor2tensor.data_generators.wiki",
    "tensor2tensor.data_generators.wsj_parsing",
]


for module in modules:
  try:
    importlib.import_module(module)
  except ImportError as error:
    print("Did not import module: %s; Cause: %s" % (module, str(error)))
