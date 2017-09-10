# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

"""Generator utilities test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import io
import os
import tempfile

# Dependency imports

from builtins import bytes  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import generator_utils

import tensorflow as tf


class GeneratorUtilsTest(tf.test.TestCase):

  def testGenerateFiles(self):
    tmp_dir = self.get_temp_dir()
    (_, tmp_file_path) = tempfile.mkstemp(dir=tmp_dir)
    tmp_file_name = os.path.basename(tmp_file_path)

    # Generate a trivial file and assert the file exists.
    def test_generator():
      yield {"inputs": [1], "target": [1]}

    filenames = generator_utils.train_data_filenames(tmp_file_name, tmp_dir, 1)
    generator_utils.generate_files(test_generator(), filenames)
    self.assertTrue(tf.gfile.Exists(tmp_file_path + "-train-00000-of-00001"))

    # Clean up.
    os.remove(tmp_file_path + "-train-00000-of-00001")
    os.remove(tmp_file_path)

  def testMaybeDownload(self):
    tmp_dir = self.get_temp_dir()
    (_, tmp_file_path) = tempfile.mkstemp(dir=tmp_dir)
    tmp_file_name = os.path.basename(tmp_file_path)

    # Download Google index to the temporary file.http.
    res_path = generator_utils.maybe_download(tmp_dir, tmp_file_name + ".http",
                                              "http://google.com")
    self.assertEqual(res_path, tmp_file_path + ".http")

    # Clean up.
    os.remove(tmp_file_path + ".http")
    os.remove(tmp_file_path)

  def testMaybeDownloadFromDrive(self):
    tmp_dir = self.get_temp_dir()
    (_, tmp_file_path) = tempfile.mkstemp(dir=tmp_dir)
    tmp_file_name = os.path.basename(tmp_file_path)

    # Download Google index to the temporary file.http.
    res_path = generator_utils.maybe_download_from_drive(
        tmp_dir, tmp_file_name + ".http", "http://drive.google.com")
    self.assertEqual(res_path, tmp_file_path + ".http")

    # Clean up.
    os.remove(tmp_file_path + ".http")
    os.remove(tmp_file_path)

  def testGunzipFile(self):
    tmp_dir = self.get_temp_dir()
    (_, tmp_file_path) = tempfile.mkstemp(dir=tmp_dir)

    # Create a test zip file and unzip it.
    with gzip.open(tmp_file_path + ".gz", "wb") as gz_file:
      gz_file.write(bytes("test line", "utf-8"))
    generator_utils.gunzip_file(tmp_file_path + ".gz", tmp_file_path + ".txt")

    # Check that the unzipped result is as expected.
    lines = []
    for line in io.open(tmp_file_path + ".txt", "rb"):
      lines.append(line.decode("utf-8").strip())
    self.assertEqual(len(lines), 1)
    self.assertEqual(lines[0], "test line")

    # Clean up.
    os.remove(tmp_file_path + ".gz")
    os.remove(tmp_file_path + ".txt")
    os.remove(tmp_file_path)

  def testGetOrGenerateTxtVocab(self):
    data_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    test_file = os.path.join(self.get_temp_dir(), "test.txt")
    with tf.gfile.Open(test_file, "w") as outfile:
      outfile.write("a b c\n")
      outfile.write("d e f\n")
    # Create a vocab over the test file.
    vocab1 = generator_utils.get_or_generate_txt_vocab(
        data_dir, "test.voc", 20, test_file)
    self.assertTrue(tf.gfile.Exists(os.path.join(data_dir, "test.voc")))
    self.assertIsNotNone(vocab1)

    # Append a new line to the test file which would change the vocab if
    # the vocab were not being read from file.
    with tf.gfile.Open(test_file, "a") as outfile:
      outfile.write("g h i\n")
    vocab2 = generator_utils.get_or_generate_txt_vocab(
        data_dir, "test.voc", 20, test_file)
    self.assertTrue(tf.gfile.Exists(os.path.join(data_dir, "test.voc")))
    self.assertIsNotNone(vocab2)
    self.assertEqual(vocab1.dump(), vocab2.dump())

if __name__ == "__main__":
  tf.test.main()
