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

"""Access T2T Problems."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import all_problems
from tensor2tensor.utils import registry


def problem(name):
  return registry.problem(name)


def available():
  return registry.list_base_problems()


all_problems.import_modules(all_problems.ALL_MODULES)
