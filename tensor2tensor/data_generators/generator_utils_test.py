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

"""Generator utilities test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import io
import os
import tempfile
from builtins import bytes  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import generator_utils

import tensorflow.compat.v1 as tf


INPUTS = (
    (1, 2, 3),
    (4, 5,),
    (6,),
)
TARGETS = (
    (10,),
    (20, 30, 40),
    (50, 60,),
)
INPUTS_PACKED = (
    (1, 2, 3, 4, 5),
    (6, 0, 0, 0, 0),
)
INPUTS_SEGMENTATION = (
    (1, 1, 1, 2, 2),
    (1, 0, 0, 0, 0),
)
INPUTS_POSITION = (
    (0, 1, 2, 0, 1),
    (0, 0, 0, 0, 0),
)
TARGETS_PACKED = (
    (10, 20, 30, 40, 0),
    (50, 60, 0, 0, 0),
)
TARGETS_SEGMENTATION = (
    (1, 2, 2, 2, 0),
    (1, 1, 0, 0, 0),
)
TARGETS_POSITION = (
    (0, 0, 1, 2, 0),
    (0, 1, 0, 0, 0),
)


def example_generator():
  for i, t in zip(INPUTS, TARGETS):
    yield {"inputs": list(i), "targets": list(t)}


def trim_right(x):
  x = {k: list(v) for k, v in x.items()}
  while all(x.values()) and not any(i[-1] for i in x.values()):
    _ = [i.pop() for i in x.values()]
  return x


def reference_packing(trim_fn=None):
  no_trim = lambda x: {k: list(v) for k, v in x.items()}
  trim_fn = trim_fn or no_trim
  outputs = [INPUTS_PACKED, INPUTS_POSITION, INPUTS_SEGMENTATION,
             TARGETS_PACKED, TARGETS_POSITION, TARGETS_SEGMENTATION]
  for i, i_pos, i_seg, t, t_pos, t_seg in zip(*outputs):
    output = trim_fn({"inputs": i, "inputs_position": i_pos,
                      "inputs_segmentation": i_seg})
    output.update(trim_fn({"targets": t, "targets_position": t_pos,
                           "targets_segmentation": t_seg}))
    yield output


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

  def testPacking(self):
    packed = generator_utils.pack_examples(
        example_generator(), has_inputs=True, packed_length=5, queue_size=2,
        spacing=0)
    for example, reference in zip(packed, reference_packing(trim_right)):
      self.assertAllEqual(set(example.keys()), set(reference.keys()))
      for k in reference:
        self.assertAllEqual(example[k], reference[k])

  def testDatasetPacking(self):
    dataset = tf.data.Dataset.from_generator(
        example_generator,
        output_types={"inputs": tf.int64, "targets": tf.int64},
        output_shapes={"inputs": tf.TensorShape((None,)),
                       "targets": tf.TensorShape((None,))}
    )
    dataset = generator_utils.pack_dataset(
        dataset, length=5, keys=("inputs", "targets"), use_custom_ops=False)

    with tf.Session().as_default() as sess:
      batch = dataset.make_one_shot_iterator().get_next()
      for reference in reference_packing():
        example = sess.run(batch)
        self.assertAllEqual(set(example.keys()), set(reference.keys()))
        for k in reference:
          self.assertAllEqual(example[k], reference[k])


if __name__ == "__main__":
  tf.test.main()
