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
"""Text problems test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
from tensor2tensor.data_generators import problem as problem_lib
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems

import tensorflow as tf


class Test1(text_problems.Text2textTmpdir):

  @property
  def name(self):
    # name is normally provided by register_problem, but this problem is not
    # registered, so we provide one here to avoid inheriting the parent class's
    # name.
    return "test1"

  @property
  def approx_vocab_size(self):
    return 3

  @property
  def dataset_splits(self):
    return [{
        "split": problem_lib.DatasetSplit.TRAIN,
        "shards": 1,
    }, {
        "split": problem_lib.DatasetSplit.EVAL,
        "shards": 1,
    }]


class TextProblems(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.tmp_dir = tf.test.get_temp_dir()
    shutil.rmtree(cls.tmp_dir)
    os.mkdir(cls.tmp_dir)

    cls.inputs = [
        "Hello world",
        "Goodbye world",
    ]
    cls.targets = [
        "Hola mundo",
        "Adios mundo",
    ]
    cls.labels = [2, 3]
    cls.labels_strs = ["c", "d"]

    cls.inputs_file = os.path.join(cls.tmp_dir, "inputs.train.txt")
    cls.targets_file = os.path.join(cls.tmp_dir, "targets.train.txt")
    cls.labels_file = os.path.join(cls.tmp_dir, "labels.train.txt")
    cls.labels_str_file = os.path.join(cls.tmp_dir, "labels_str.train.txt")
    data = [(cls.inputs, cls.inputs_file), (cls.targets, cls.targets_file),
            (cls.labels, cls.labels_file), (cls.labels_strs,
                                            cls.labels_str_file)]

    for lines, filename in data:
      with tf.gfile.Open(filename, "w") as f:
        for line in lines:
          f.write(str(line))
          f.write("\n")

    cls.tabbed_file = os.path.join(cls.tmp_dir, "tabbed.train.txt")
    with tf.gfile.Open(cls.tabbed_file, "w") as f:
      for inputs, targets in zip(cls.inputs, cls.targets):
        f.write("%s\t%s\n" % (inputs, targets))

    tf.gfile.Copy(cls.inputs_file, os.path.join(cls.tmp_dir, "inputs.eval.txt"))
    tf.gfile.Copy(cls.targets_file, os.path.join(cls.tmp_dir,
                                                 "targets.eval.txt"))

  def testTxtLineIterator(self):
    lines = [line for line in text_problems.txt_line_iterator(self.inputs_file)]
    self.assertEqual(lines, self.inputs)

  def testText2TextTxtIterator(self):
    inputs = []
    targets = []
    for entry in text_problems.text2text_txt_iterator(self.inputs_file,
                                                      self.targets_file):
      inputs.append(entry["inputs"])
      targets.append(entry["targets"])
    self.assertEqual(inputs, self.inputs)
    self.assertEqual(targets, self.targets)

  def testText2SelfTxtIterator(self):
    targets = [
        entry["targets"]
        for entry in text_problems.text2self_txt_iterator(self.targets_file)
    ]
    self.assertEqual(targets, self.targets)

  def testText2ClassTxtIterator(self):
    inputs = []
    labels = []
    for entry in text_problems.text2class_txt_iterator(self.inputs_file,
                                                       self.labels_file):
      inputs.append(entry["inputs"])
      labels.append(entry["label"])
    self.assertEqual(inputs, self.inputs)
    self.assertEqual(labels, self.labels)

  def testText2ClassTxtIteratorWithStrs(self):
    inputs = []
    labels = []
    for entry in text_problems.text2class_txt_iterator(
        self.inputs_file, self.labels_str_file, class_strs=["a", "b", "c",
                                                            "d"]):
      inputs.append(entry["inputs"])
      labels.append(entry["label"])
    self.assertEqual(inputs, self.inputs)
    self.assertEqual(labels, self.labels)

  def testText2TextTxtTabIterator(self):
    inputs = []
    targets = []
    for entry in text_problems.text2text_txt_tab_iterator(self.tabbed_file):
      inputs.append(entry["inputs"])
      targets.append(entry["targets"])
    self.assertEqual(inputs, self.inputs)
    self.assertEqual(targets, self.targets)

  def testText2TextTmpDir(self):
    problem = Test1()
    problem.generate_data(self.tmp_dir, self.tmp_dir)
    vocab_file = os.path.join(self.tmp_dir, "vocab.test1.3.subwords")
    train_file = os.path.join(self.tmp_dir, "test1-train-00000-of-00001")
    eval_file = os.path.join(self.tmp_dir, "test1-dev-00000-of-00001")
    self.assertTrue(tf.gfile.Exists(vocab_file))
    self.assertTrue(tf.gfile.Exists(train_file))
    self.assertTrue(tf.gfile.Exists(eval_file))

    dataset = problem.dataset(tf.estimator.ModeKeys.TRAIN, self.tmp_dir)
    features = dataset.make_one_shot_iterator().get_next()

    examples = []
    exhausted = False
    with self.test_session() as sess:
      examples.append(sess.run(features))
      examples.append(sess.run(features))
      try:
        sess.run(features)
      except tf.errors.OutOfRangeError:
        exhausted = True

    self.assertTrue(exhausted)
    self.assertEqual(2, len(examples))

    self.assertNotEqual(
        list(examples[0]["inputs"]), list(examples[1]["inputs"]))

    example = examples[0]
    encoder = text_encoder.SubwordTextEncoder(vocab_file)
    inputs_encoded = list(example["inputs"])
    inputs_encoded.pop()  # rm EOS
    self.assertTrue(encoder.decode(inputs_encoded) in self.inputs)
    targets_encoded = list(example["targets"])
    targets_encoded.pop()  # rm EOS
    self.assertTrue(encoder.decode(targets_encoded) in self.targets)


if __name__ == "__main__":
  tf.test.main()
