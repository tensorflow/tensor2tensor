# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

"""Tests for trainer_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensor2tensor.data_generators import algorithmic
from tensor2tensor.models import transformer  # pylint: disable=unused-import
from tensor2tensor.utils import data_reader
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
import tensorflow as tf


class TrainerLibTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    algorithmic.TinyAlgo.setup_for_test()

  def testExperiment(self):
    exp_fn = trainer_lib.create_experiment_fn(
        "transformer",
        "tiny_algo",
        algorithmic.TinyAlgo.data_dir,
        train_steps=1,
        eval_steps=1,
        min_eval_frequency=1,
        use_tpu=False)
    run_config = trainer_lib.create_run_config(
        model_name="transformer",
        model_dir=algorithmic.TinyAlgo.data_dir,
        num_gpus=0,
        use_tpu=False)
    hparams = registry.hparams("transformer_tiny_tpu")
    exp = exp_fn(run_config, hparams)
    exp.test()

  def testExperimentWithClass(self):
    exp_fn = trainer_lib.create_experiment_fn(
        "transformer",
        algorithmic.TinyAlgo(),
        algorithmic.TinyAlgo.data_dir,
        train_steps=1,
        eval_steps=1,
        min_eval_frequency=1,
        use_tpu=False)
    run_config = trainer_lib.create_run_config(
        model_name="transformer",
        model_dir=algorithmic.TinyAlgo.data_dir,
        num_gpus=0,
        use_tpu=False)
    hparams = registry.hparams("transformer_tiny_tpu")
    exp = exp_fn(run_config, hparams)
    exp.test()

  def testModel(self):
    # HParams
    hparams = trainer_lib.create_hparams(
        "transformer_tiny", data_dir=algorithmic.TinyAlgo.data_dir,
        problem_name="tiny_algo")

    # Dataset
    problem = hparams.problem
    dataset = problem.dataset(tf.estimator.ModeKeys.TRAIN,
                              algorithmic.TinyAlgo.data_dir)
    dataset = dataset.repeat(None).padded_batch(10, dataset.output_shapes)
    features = dataset.make_one_shot_iterator().get_next()
    features = data_reader.standardize_shapes(features)

    # Model
    model = registry.model("transformer")(hparams, tf.estimator.ModeKeys.TRAIN)
    logits, losses = model(features)

    self.assertTrue("training" in losses)
    loss = losses["training"]

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      logits_val, loss_val = sess.run([logits, loss])
      logits_shape = list(logits_val.shape)
      logits_shape[1] = None
      self.assertAllEqual(logits_shape, [10, None, 1, 1, 4])
      self.assertEqual(loss_val.shape, tuple())

  def testMultipleTargetModalities(self):
    # Use existing hparams and override target modality.
    hparams = trainer_lib.create_hparams(
        "transformer_tiny", data_dir=algorithmic.TinyAlgo.data_dir,
        problem_name="tiny_algo")
    # Manually turn off sharing. It is not currently supported for multitargets.
    hparams.shared_embedding_and_softmax_weights = 0  # pylint: disable=line-too-long
    hparams.problem_hparams.modality = {
        "targets": hparams.problem_hparams.modality["targets"],
        "targets_A": hparams.problem_hparams.modality["targets"],
        "targets_B": hparams.problem_hparams.modality["targets"],
    }
    hparams.problem_hparams.vocab_size = {
        "targets": hparams.problem_hparams.vocab_size["targets"],
        "targets_A": hparams.problem_hparams.vocab_size["targets"],
        "targets_B": hparams.problem_hparams.vocab_size["targets"],
    }
    hparams.problem._hparams = hparams.problem_hparams

    # Dataset
    problem = hparams.problem
    dataset = problem.dataset(tf.estimator.ModeKeys.TRAIN,
                              algorithmic.TinyAlgo.data_dir)
    dataset = dataset.repeat(None).padded_batch(10, dataset.output_shapes)
    features = dataset.make_one_shot_iterator().get_next()
    features = data_reader.standardize_shapes(features)
    features["targets_A"] = features["targets_B"] = features["targets"]

    # Model
    model = registry.model("transformer")(hparams, tf.estimator.ModeKeys.TRAIN)

    def body(args, mb=model.body):
      out = mb(args)
      return {"targets": out, "targets_A": out, "targets_B": out}

    model.body = body

    logits, losses = model(features)

    self.assertTrue("training" in losses)
    loss = losses["training"]

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run([logits, loss])

  def testCreateHparams(self):
    # Get json_path
    pkg, _ = os.path.split(__file__)
    pkg, _ = os.path.split(pkg)
    json_path = os.path.join(
        pkg, "test_data", "transformer_test_ckpt", "hparams.json")

    # Create hparams
    hparams = trainer_lib.create_hparams("transformer_big", "hidden_size=1",
                                         hparams_path=json_path)
    self.assertEqual(2, hparams.num_hidden_layers)  # from json
    self.assertEqual(1, hparams.hidden_size)  # from hparams_overrides_str

    # Compare with base hparams
    base_hparams = trainer_lib.create_hparams("transformer_big")
    self.assertEqual(len(base_hparams.values()), len(hparams.values()))


if __name__ == "__main__":
  tf.test.main()
