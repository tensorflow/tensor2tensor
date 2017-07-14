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

"""Tests for trainer_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.data_generators import algorithmic
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_utils as utils  # pylint: disable=unused-import

import tensorflow as tf

FLAGS = tf.flags.FLAGS


@registry.register_problem
class TinyAlgo(algorithmic.AlgorithmicIdentityBinary40):

  def generate_data(self, data_dir, _):
    generator_utils.generate_files(
        algorithmic.identity_generator(self.num_symbols, 40, 100000),
        self.training_filepaths(data_dir, 1, shuffled=True), 100)
    generator_utils.generate_files(
        algorithmic.identity_generator(self.num_symbols, 400, 10000),
        self.dev_filepaths(data_dir, 1, shuffled=True), 100)


@registry.register_hparams
def transformer_test():
  hparams = transformer.transformer_base()
  hparams.batch_size = 10
  hparams.hidden_size = 10
  hparams.num_hidden_layers = 1
  hparams.num_heads = 2
  hparams.max_length = 16
  return hparams


class TrainerUtilsTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    # Generate a small test dataset
    FLAGS.problems = "tiny_algo"
    TrainerUtilsTest.data_dir = tf.test.get_temp_dir()
    registry.problem(FLAGS.problems).generate_data(TrainerUtilsTest.data_dir,
                                                   None)

  def testModelsImported(self):
    models = registry.list_models()
    self.assertTrue("baseline_lstm_seq2seq" in models)

  def testHParamsImported(self):
    hparams = registry.list_hparams()
    self.assertTrue("transformer_base" in hparams)

  def testSingleStep(self):
    model_name = "transformer"
    FLAGS.hparams_set = "transformer_test"
    exp = utils.create_experiment(
        output_dir=tf.test.get_temp_dir(),
        data_dir=TrainerUtilsTest.data_dir,
        model_name=model_name,
        train_steps=1,
        eval_steps=1)
    exp.test()


if __name__ == "__main__":
  tf.test.main()
