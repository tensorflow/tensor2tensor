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
"""Github function/text similatrity problems."""
import csv
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry


# There are 10 splits of the data as CSV files.
_DATA_BASE_URL = "https://storage.googleapis.com/kubeflow-examples/t2t-code-search/data"
_GITHUB_FUNCTION_DOCSTRING_FILES = [
    [
        "{}/pairs-0000{}-of-00010.csv".format(_DATA_BASE_URL, i),
        "pairs-0000{}-of-00010.csv".format(i),
    ]
    for i in range(10)
]


@registry.register_problem
class GithubFunctionDocstring(translate.TranslateProblem):
  """This class defines the problem of finding similarity between Python
  function and docstring"""

  @property
  def is_generate_per_split(self):
    return False

  @property
  def approx_vocab_size(self):
    return 2**13

  def source_data_files(self, dataset_split):
    # TODO(sanyamkapoor): separate train/eval data set.
    return _GITHUB_FUNCTION_DOCSTRING_FILES

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Returns a generator to return {"inputs": [text], "targets": [text]}."""

    pair_csv_files = [
        generator_utils.maybe_download(data_dir, filename, uri)
        for uri, filename in self.source_data_files(dataset_split)
    ]

    for pairs_file in pair_csv_files:
      with open(pairs_file, "r") as csv_file:
        pairs_reader = csv.reader(csv_file)
        for row in pairs_reader:
          function_tokens, docstring_tokens = row[-2:]
          yield {"inputs": docstring_tokens, "targets": function_tokens}

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    for sample in self.generate_samples(data_dir, tmp_dir, None):
      yield sample["inputs"]
      yield sample["targets"]

  def eval_metrics(self):
    return [
        metrics.Metrics.ACC
    ]
