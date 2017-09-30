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

"""Tests for trainer_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

# Dependency imports

from tensor2tensor.data_generators import algorithmic
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.models import transformer
from tensor2tensor.utils import model_builder
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_utils

import tensorflow as tf

flags = tf.flags
FLAGS = tf.flags.FLAGS

flags.DEFINE_string("schedule", "train_and_evaluate", "")
flags.DEFINE_integer("eval_steps", 10, "Number of steps in evaluation.")
flags.DEFINE_string("master", "", "Address of TensorFlow master.")
flags.DEFINE_string("output_dir", "", "Base output directory for run.")


@registry.register_problem
class TinyAlgo(algorithmic.AlgorithmicIdentityBinary40):

  def generate_data(self, data_dir, _):
    identity_problem = algorithmic.AlgorithmicIdentityBinary40()
    generator_utils.generate_files(
        identity_problem.generator(self.num_symbols, 40, 100000),
        self.training_filepaths(data_dir, 1, shuffled=True), 100)
    generator_utils.generate_files(
        identity_problem.generator(self.num_symbols, 400, 10000),
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
    tmp_dir = tf.test.get_temp_dir()
    shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)

    # Generate a small test dataset
    FLAGS.problems = "tiny_algo"
    TrainerUtilsTest.data_dir = tmp_dir
    registry.problem(FLAGS.problems).generate_data(TrainerUtilsTest.data_dir,
                                                   None)

  def testModelsImported(self):
    models = registry.list_models()
    self.assertTrue("lstm_seq2seq" in models)

  def testHParamsImported(self):
    hparams = registry.list_hparams()
    self.assertTrue("transformer_base" in hparams)

  def testSingleStep(self):
    model_name = "transformer"
    data_dir = TrainerUtilsTest.data_dir
    hparams = trainer_utils.create_hparams("transformer_test", data_dir)
    trainer_utils.add_problem_hparams(hparams, FLAGS.problems)
    exp = trainer_utils.create_experiment(
        data_dir=data_dir,
        model_name=model_name,
        train_steps=1,
        eval_steps=1,
        hparams=hparams,
        run_config=trainer_utils.create_run_config(
            output_dir=tf.test.get_temp_dir()))
    exp.test()

  def testSingleEvalStepRawSession(self):
    """Illustrate how to run a T2T model in a raw session."""

    # Set model name, hparams, problems as would be set on command line.
    model_name = "transformer"
    FLAGS.hparams_set = "transformer_test"
    FLAGS.problems = "tiny_algo"
    data_dir = "/tmp"  # Used only when a vocab file or such like is needed.

    # Create the problem object, hparams, placeholders, features dict.
    encoders = registry.problem(FLAGS.problems).feature_encoders(data_dir)
    hparams = trainer_utils.create_hparams(FLAGS.hparams_set, data_dir)
    trainer_utils.add_problem_hparams(hparams, FLAGS.problems)
    inputs_ph = tf.placeholder(dtype=tf.int32)  # Just length dimension.
    batch_inputs = tf.reshape(inputs_ph, [1, -1, 1, 1])  # Make it 4D.
    # In INFER mode targets can be None.
    targets_ph = tf.placeholder(dtype=tf.int32)  # Just length dimension.
    batch_targets = tf.reshape(targets_ph, [1, -1, 1, 1])  # Make it 4D.
    features = {
        "inputs": batch_inputs,
        "targets": batch_targets,
        "problem_choice": 0,  # We run on the first problem here.
        "input_space_id": hparams.problems[0].input_space_id,
        "target_space_id": hparams.problems[0].target_space_id
    }

    # Now set a mode and create the graph by invoking model_fn.
    mode = tf.estimator.ModeKeys.EVAL
    estimator_spec = model_builder.model_fn(
        model_name, features, mode, hparams, problem_names=[FLAGS.problems])
    predictions_dict = estimator_spec.predictions
    predictions = tf.squeeze(  # These are not images, axis=2,3 are not needed.
        predictions_dict["predictions"],
        axis=[2, 3])

    # Having the graph, let's run it on some data.
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      inputs = "0 1 0"
      targets = "0 1 0"
      # Encode from raw string to numpy input array using problem encoders.
      inputs_numpy = encoders["inputs"].encode(inputs)
      targets_numpy = encoders["targets"].encode(targets)
      # Feed the encoded inputs and targets and run session.
      feed = {inputs_ph: inputs_numpy, targets_ph: targets_numpy}
      np_predictions = sess.run(predictions, feed)
      # Check that the result has the correct shape: batch x length x vocab_size
      #   where, for us, batch = 1, length = 3, vocab_size = 4.
      self.assertEqual(np_predictions.shape, (1, 3, 4))


if __name__ == "__main__":
  tf.test.main()
