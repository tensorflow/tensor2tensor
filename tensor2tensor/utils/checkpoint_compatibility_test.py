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

"""Test for checkpoint compatibility."""
# The checkpoint in test_data/transformer_test_ckpt is generated with the OSS
# release.
# t2t-trainer \
#   --model=transformer \
#   --hparams_set=transformer_test \
#   --problem=translate_ende_wmt8k \
#   --data_dir=~/t2t/data \
#   --output_dir=/tmp/t2t_train \
#   --train_steps=1 \
#   --eval_steps=1

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from six.moves import range
from tensor2tensor import models  # pylint: disable=unused-import
from tensor2tensor import problems  # pylint: disable=unused-import
from tensor2tensor.utils import data_reader
from tensor2tensor.utils import trainer_lib

import tensorflow.compat.v1 as tf


def get_data_dir():
  pkg = os.path.abspath(__file__)
  pkg, _ = os.path.split(pkg)
  pkg, _ = os.path.split(pkg)
  return os.path.join(pkg, "test_data")


_DATA_DIR = get_data_dir()
_CKPT_DIR = os.path.join(_DATA_DIR, "transformer_test_ckpt")


class CheckpointCompatibilityTest(tf.test.TestCase):
  BATCH_SIZE = 3

  def testCompatibility(self):
    model = "transformer"
    hp_set = "transformer_test"
    problem_name = "translate_ende_wmt8k"

    hp = trainer_lib.create_hparams(
        hp_set, data_dir=_DATA_DIR, problem_name=problem_name)
    run_config = trainer_lib.create_run_config(model, model_dir=_CKPT_DIR)
    estimator = trainer_lib.create_estimator(model, hp, run_config)

    for prediction in estimator.predict(self.input_fn):
      self.assertEqual(prediction["outputs"].dtype, np.int32)

  def input_fn(self):
    types = {"inputs": tf.int32}
    shapes = {"inputs": tf.TensorShape([None])}
    dataset = tf.data.Dataset.from_generator(self.input_generator, types,
                                             shapes)
    dataset = dataset.padded_batch(self.BATCH_SIZE, shapes)
    dataset = dataset.map(data_reader.standardize_shapes)
    features = dataset.make_one_shot_iterator().get_next()
    return features

  def input_generator(self):
    for _ in range(self.BATCH_SIZE):
      vals = np.random.randint(
          1, 100, size=np.random.randint(20), dtype=np.int32)
      yield {"inputs": vals}


if __name__ == "__main__":
  tf.test.main()
