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
"""Github function to text similatrity problems."""

import os

from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model


@registry.register_model
class SimilarityTransformer(t2t_model.T2TModel):
  """Similarity scores between functions and docstrings."""

  def body(self, features):
    # TODO(sanyamkapoor): need to fill this with Transformer encoder/decoder
    # and loss calculation
    raise NotImplementedError


@registry.register_problem
class GithubFunctionDocstring(text_problems.Text2TextProblem):
  """The problem of similarity between Python function and docstring."""

  @property
  def is_generate_per_split(self):
    return False

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Returns the generator of {"inputs": [text], "targets": [text]} dict."""

    functions_file_path = os.path.join(
        data_dir, '{}.function'.format(dataset_split))
    docstrings_file_path = os.path.join(
        data_dir, '{}.docstring'.format(dataset_split))

    return text_problems.text2text_txt_iterator(
        functions_file_path, docstrings_file_path)
