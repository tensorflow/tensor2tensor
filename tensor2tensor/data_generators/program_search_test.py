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

"""Tests for tensor2tensor.data_generators.program_search."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import shutil
import tempfile

from builtins import bytes  # pylint: disable=redefined-builtin
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import program_search

import tensorflow.compat.v1 as tf


class ProgramSearchAlgolispStub(program_search.ProgramSearchAlgolisp):
  """Stub of ProgramSearchAlgolisp that stubs out maybe_download_dataset.

  The maybe_download_dataset writes one predetermined example in a zip file
  self.n number of times and returns the file path.
  """

  EXAMPLE = ('{"funcs": [], "tests": [{"output": 0, "input": {"a": 5}}, '
             '{"output": 1, "input": {"a": 20}}, {"output": 2, "input": '
             '{"a": 28}}, {"output": 1, "input": {"a": 13}}, {"output": 1, '
             '"input": {"a": 27}}, {"output": 1, "input": {"a": 13}}, '
             '{"output": 1, "input": {"a": 20}}, {"output": 0, '
             '"input": {"a": 8}}, {"output": 0, "input": {"a": 8}}, '
             '{"output": 0, "input": {"a": 4}}], "short_tree": ["invoke1", '
             '["lambda1", ["if", ["==", ["len", ["digits", "arg1"]], "1"], "0",'
             ' ["+", "1", ["self", ["reduce", ["digits", "arg1"], "0", '
             '"+"]]]]], "a"], "tags": [], "text": ["given", "a", "number", "a",'
             ' ",", "find", "how", "many", "times", "you", "can", "replace", '
             '"a", "with", "sum", "of", "its", "digits", "before", "it", '
             '"becomes", "a", "single", "digit", "number"], "return_type": '
             '"int", "args": {"a": "int"}, "nodes": ["l1_recursive_digits"]}')

  EXAMPLE_INPUT = ('given a number a , find how many times you can replace a '
                   'with sum of its digits before it becomes a single digit '
                   'number')

  EXAMPLE_TARGET = ('[ invoke1 [ lambda1 [ if [ == [ len [ digits arg1 ] ] 1 ]'
                    ' 0 [ + 1 [ self [ reduce [ digits arg1 ] 0 + ] ] ] ] ] a '
                    ']')

  N = 10

  def maybe_download_dataset(self, tmp_dir, dataset_split):
    (_, data_file) = tempfile.mkstemp(
        suffix='.gz', prefix=str(dataset_split) + '-', dir=tmp_dir)

    with gzip.open(data_file, 'wb') as gz_file:
      content = '\n'.join([self.EXAMPLE] * self.N)
      gz_file.write(bytes(content, 'utf-8'))
    return data_file


class ProgramSearchAlgolispTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    # Setup the temp directory tree.
    cls.tmp_dir = tf.test.get_temp_dir()
    shutil.rmtree(cls.tmp_dir)
    os.mkdir(cls.tmp_dir)

  @classmethod
  def tearDownClass(cls):
    # Cleanup the temp directory tree.
    shutil.rmtree(cls.tmp_dir)

  def testEndToEnd(self):
    # End-to-end test, the stub problem class creates a .gz file with nps_stub.N
    # example and we check if we're able to process it correctly.
    nps_stub = ProgramSearchAlgolispStub()
    num = 0
    for example in nps_stub.generate_samples(None, self.tmp_dir,
                                             problem.DatasetSplit.TRAIN):

      # Only one example in 'file', so this is OK.
      self.assertEqual(example['inputs'],
                       ProgramSearchAlgolispStub.EXAMPLE_INPUT)

      self.assertEqual(example['targets'],
                       ProgramSearchAlgolispStub.EXAMPLE_TARGET)

      num += 1

    # assert that we have as many examples as there are in the file.
    self.assertEqual(num, nps_stub.N)


if __name__ == '__main__':
  tf.test.main()
