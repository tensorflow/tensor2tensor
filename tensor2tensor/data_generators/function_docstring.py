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

"""Github function/text similatrity problems."""
import csv
from six import StringIO
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry
import tensorflow.compat.v1 as tf


@registry.register_problem
class GithubFunctionDocstring(text_problems.Text2TextProblem):
  """Function and Docstring similarity Problem.

  This problem contains the data consisting of function
  and docstring pairs as CSV files. The files are structured
  such that they contain two columns without headers containing
  the docstring tokens and function tokens. The delimiter is
  ",".
  """

  NUM_SHARDS = 100

  @property
  def base_url(self):
    return "gs://kubeflow-examples/t2t-code-search/raw_data"

  @property
  def pair_files_list(self):
    files = []
    for i in range(self.NUM_SHARDS):
      files.append([
          "{}/func-doc-pairs-{:05}-of-{:05}.csv".format(self.base_url, i,
                                                        self.NUM_SHARDS),
          ("func-doc-pairs-{:05}-of-{:05}.csv".format(i, self.NUM_SHARDS),)
      ])
    return files

  @property
  def is_generate_per_split(self):
    return False

  @property
  def approx_vocab_size(self):
    return 2**13

  @property
  def max_samples_for_vocab(self):
    # FIXME(sanyamkapoor): This exists to handle memory explosion.
    return int(3.5e5)

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """A generator to return data samples.Returns the data generator to return.


    Args:
      data_dir: A string representing the data directory.
      tmp_dir: A string representing the temporary directory and is
              used to download files if not already available.
      dataset_split: Train, Test or Eval.

    Yields:
      Each element yielded is of a Python dict of the form
        {"inputs": "STRING", "targets": "STRING"}
    """

    # TODO(sanyamkapoor): Manually separate train/eval data set.
    csv_file_names = self.pair_files_list
    csv_files = [
        generator_utils.maybe_download(tmp_dir, file_list[0], uri)
        for uri, file_list in csv_file_names
    ]

    for pairs_file in csv_files:
      tf.logging.debug("Reading {}".format(pairs_file))
      with open(pairs_file, "r") as csv_file:
        for line in csv_file:
          reader = csv.reader(StringIO(line))
          for docstring_tokens, function_tokens in reader:
            yield {
                "inputs": docstring_tokens,
                "targets": function_tokens,
                "embed_code": [0],
            }

  def preprocess_example(self, example, mode, unused_hparams):
    if mode != tf.estimator.ModeKeys.TRAIN:
      example["embed_code"] = [0]
    return example

  def eval_metrics(self):
    return [
        metrics.Metrics.ACC
    ]
