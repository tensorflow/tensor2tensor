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

"""Translate generators test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tarfile
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate

import tensorflow.compat.v1 as tf


class TranslateTest(tf.test.TestCase):
  DATASETS = [
      ["data1.tgz", ("train1.en", "train1.de")],
      ["data2.tgz", ("train2.en", "train2.de")],
      ["data3.tgz", ("train3.en", "train3.de")],
  ]

  @classmethod
  def setUpClass(cls):
    tmp_dir = tf.test.get_temp_dir()
    compressed_dir = os.path.join(tmp_dir, "compressed")
    shutil.rmtree(tmp_dir)
    tf.gfile.MakeDirs(compressed_dir)

    en_data = [str(i) for i in range(10, 40)]
    de_data = [str(i) for i in range(100, 130)]
    data = list(zip(en_data, de_data))

    for i, dataset in enumerate(cls.DATASETS):
      tar_file = dataset[0]
      en_file, de_file = [
          os.path.join(compressed_dir, name) for name in dataset[1]
      ]
      with tf.gfile.Open(en_file, "w") as en_f:
        with tf.gfile.Open(de_file, "w") as de_f:
          start = i * 10
          end = start + 10
          for en_line, de_line in data[start:end]:
            en_f.write(en_line)
            en_f.write("\n")
            de_f.write(de_line)
            de_f.write("\n")

      with tarfile.open(os.path.join(tmp_dir, tar_file), "w:gz") as tar_f:
        tar_f.add(en_file, os.path.basename(en_file))
        tar_f.add(de_file, os.path.basename(de_file))

    cls.tmp_dir = tmp_dir
    cls.data = data

  def testCompileData(self):
    filename = "out"
    filepath = os.path.join(self.tmp_dir, filename)
    translate.compile_data(self.tmp_dir, self.DATASETS, filename)

    count = 0
    for i, example in enumerate(
        text_problems.text2text_txt_iterator(filepath + ".lang1",
                                             filepath + ".lang2")):
      expected = self.data[i]
      self.assertEqual(list(expected), [example["inputs"], example["targets"]])
      count += 1
    self.assertEqual(count, len(self.data))


if __name__ == "__main__":
  tf.test.main()
