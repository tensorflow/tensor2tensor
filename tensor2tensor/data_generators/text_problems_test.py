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

"""Text problems test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
from tensor2tensor.data_generators import problem as problem_lib
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems

import tensorflow.compat.v1 as tf


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

    cls.targets_regr = [[1.23, 2.34], [4.56, 5.67]]
    cls.targets_regr_file = os.path.join(cls.tmp_dir, "targets_regr.train.txt")
    with tf.gfile.Open(cls.targets_regr_file, "w") as f:
      for targets in cls.targets_regr:
        f.write(" ".join([str(x) for x in targets]) + "\n")

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

  def testText2RealTxtIterator(self):
    inputs = []
    targets = []
    for entry in text_problems.text2real_txt_iterator(self.inputs_file,
                                                      self.targets_regr_file):
      inputs.append(entry["inputs"])
      targets.append(entry["targets"])
    self.assertEqual(inputs, self.inputs)
    self.assertEqual(targets, self.targets_regr)

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


class FakeDistributedProblem(text_problems.DistributedText2TextProblem):

  def __init__(self):
    self.name = "fake_distributed_problem"
    # Call the base class ctor.
    super(FakeDistributedProblem, self).__init__()

  def generate_samples(self, data_dir, tmp_dir, dataset_split, input_files):
    # Read all lines from all the input_files and return the same word as input
    # and target.
    for input_file in input_files:
      with tf.gfile.Open(input_file, "r") as f:
        for line in f.read().strip().split("\n"):
          yield {"inputs": line.strip(), "targets": line.strip()}

  @property
  def is_generate_per_split(self):
    return True

  @property
  def dataset_splits(self):
    return [{
        "split": problem_lib.DatasetSplit.TRAIN,
        "shards": 2,
    }, {
        "split": problem_lib.DatasetSplit.EVAL,
        "shards": 3,
    }, {
        "split": problem_lib.DatasetSplit.TEST,
        "shards": 4,
    }]

  def input_files(self, dataset_split=problem_lib.DatasetSplit.TRAIN):
    if dataset_split == problem_lib.DatasetSplit.TRAIN:
      return self.train_files
    elif dataset_split == problem_lib.DatasetSplit.EVAL:
      return self.dev_files
    return self.test_files

  @classmethod
  def setup_for_test(cls):
    # First setup the temp train, dev, test files and then call the ctor.
    cls.tmp_dir = tf.test.get_temp_dir()
    shutil.rmtree(cls.tmp_dir)
    os.mkdir(cls.tmp_dir)

    # Write 25 train files, 5 dev files, 11 test files.
    train_pattern = os.path.join(cls.tmp_dir, "train-%05d-of-00025")
    dev_pattern = os.path.join(cls.tmp_dir, "dev-%05d-of-00005")
    test_pattern = os.path.join(cls.tmp_dir, "test-%05d-of-00011")
    cls.train_files, cls.dev_files, cls.test_files = [], [], []
    for i in range(25):
      cls.train_files.append(train_pattern % i)
      with tf.gfile.Open(cls.train_files[-1], "w") as f:
        f.write("train_%d\n" % i)
    for i in range(5):
      cls.dev_files.append(dev_pattern % i)
      with tf.gfile.Open(cls.dev_files[-1], "w") as f:
        f.write("dev_%d\n" % i)
    for i in range(11):
      cls.test_files.append(test_pattern % i)
      with tf.gfile.Open(cls.test_files[-1], "w") as f:
        f.write("test_%d\n" % i)


class FakeDistributedProblemNotPerSplit(FakeDistributedProblem):

  @property
  def is_generate_per_split(self):
    return False


class DistributedText2TextProblemsTest(tf.test.TestCase):

  def setUp(self):
    FakeDistributedProblem.setup_for_test()

  def testOutputSharding(self):
    problem = FakeDistributedProblemNotPerSplit()

    # self.dataset_split is 2, 3, 4
    # So:
    # num output shards = 2 + 3 + 4 = 9
    # task_ids will be in range = [0, 9)

    expected_split_shard_and_offset = [
        (problem_lib.DatasetSplit.TRAIN, 2, 0),
        (problem_lib.DatasetSplit.TRAIN, 2, 1),
        (problem_lib.DatasetSplit.EVAL, 3, 0),
        (problem_lib.DatasetSplit.EVAL, 3, 1),
        (problem_lib.DatasetSplit.EVAL, 3, 2),
        (problem_lib.DatasetSplit.TEST, 4, 0),
        (problem_lib.DatasetSplit.TEST, 4, 1),
        (problem_lib.DatasetSplit.TEST, 4, 2),
        (problem_lib.DatasetSplit.TEST, 4, 3),
    ]

    expected_output_filenames = [
        "/tmp/fake_distributed_problem-unshuffled-train-00000-of-00002",
        "/tmp/fake_distributed_problem-unshuffled-train-00001-of-00002",
        "/tmp/fake_distributed_problem-unshuffled-dev-00000-of-00003",
        "/tmp/fake_distributed_problem-unshuffled-dev-00001-of-00003",
        "/tmp/fake_distributed_problem-unshuffled-dev-00002-of-00003",
        "/tmp/fake_distributed_problem-unshuffled-test-00000-of-00004",
        "/tmp/fake_distributed_problem-unshuffled-test-00001-of-00004",
        "/tmp/fake_distributed_problem-unshuffled-test-00002-of-00004",
        "/tmp/fake_distributed_problem-unshuffled-test-00003-of-00004"
    ]

    actual_split_shard_and_offset = []
    actual_output_filenames = []
    for task_id in range(9):
      actual_split_shard_and_offset.append(
          problem._task_id_to_output_split(task_id))
      actual_output_filenames.append(
          problem._task_id_to_output_file("/tmp", task_id))

    self.assertSequenceEqual(expected_split_shard_and_offset,
                             actual_split_shard_and_offset)

    self.assertSequenceEqual(expected_output_filenames, actual_output_filenames)

  def testInputShardingNoGeneratePerSplit(self):
    # 25 input shards (train only, is_generate_per_split = False).
    # 9 output tasks in all (2 + 3 + 4), so
    #
    # Division should be like:
    # task_id 0 -> 0, 1, 2
    # task_id 1 -> 3, 4, 5
    # ...
    # task_id 6 -> 18, 19, 20
    # task_id 7 -> 21, 22
    # task_id 8 -> 23, 24

    # tasks 0 to 6
    expected_input_file_sharding = [[
        "train-%05d-of-00025" % j for j in [i, i + 1, i + 2]
    ] for i in range(0, 20, 3)]
    # tasks 7 and 8
    expected_input_file_sharding.extend(
        [["train-%05d-of-00025" % i for i in [21, 22]],
         ["train-%05d-of-00025" % i for i in [23, 24]]])

    problem = FakeDistributedProblemNotPerSplit()

    list_input_files = []
    for task_id in range(9):
      input_files = problem._task_id_to_input_files(task_id)
      list_input_files.append(
          [os.path.basename(input_file) for input_file in input_files])

    self.assertSequenceEqual(expected_input_file_sharding, list_input_files)

  def testInputShardingWithGeneratePerSplit(self):
    # 25, 5, 11 train, dev, test input shards
    # 9 output tasks in all (2 + 3 + 4), so
    #
    # Division should be like:
    #
    # Train
    # task_id 0 -> 0, .. 12
    # task_id 1 -> 13 .. 24
    #
    # Dev
    # task_id 2 -> 0, 1
    # task_id 3 -> 2, 3,
    # task_id 4 -> 4
    #
    # Test
    # task_id 5 -> 0, 1, 2
    # task_id 6 -> 3, 4, 5
    # task_id 7 -> 6, 7, 8
    # task_id 8 -> 9, 10

    expected_input_file_sharding = [
        ["train-%05d-of-00025" % i for i in range(13)],      # task_id 0
        ["train-%05d-of-00025" % i for i in range(13, 25)],  # task_id 1
        ["dev-%05d-of-00005" % i for i in [0, 1]],           # task_id 2
        ["dev-%05d-of-00005" % i for i in [2, 3]],           # task_id 3
        ["dev-%05d-of-00005" % i for i in [4]],              # task_id 4
        ["test-%05d-of-00011" % i for i in [0, 1, 2]],       # task_id 5
        ["test-%05d-of-00011" % i for i in [3, 4, 5]],       # task_id 6
        ["test-%05d-of-00011" % i for i in [6, 7, 8]],       # task_id 7
        ["test-%05d-of-00011" % i for i in [9, 10]],         # task_id 8
    ]

    problem = FakeDistributedProblem()

    list_input_files = []
    for task_id in range(9):
      input_files = problem._task_id_to_input_files(task_id)
      list_input_files.append(
          [os.path.basename(input_file) for input_file in input_files])

    self.assertSequenceEqual(expected_input_file_sharding, list_input_files)

  def testVocabularyIsAllTrain(self):
    problem = FakeDistributedProblem()

    tmp_dir = problem.tmp_dir

    for text in problem.generate_text_for_vocab(tmp_dir, tmp_dir):
      # All the vocabulary is coming from training input shards.
      self.assertTrue("train_" in text, "train is not in %s" % text)


if __name__ == "__main__":
  tf.test.main()
