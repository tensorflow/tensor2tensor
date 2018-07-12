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
"""Adversarially attack a model."""

import os

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import problem as problem_lib  # pylint: disable=unused-import
from tensor2tensor.utils import adv_attack_utils
from tensor2tensor.utils import cloud_mlengine
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

# See flags.py for additional command-line flags.
flags.DEFINE_string("attack_params_set", None,
                    "Which attack parameters to use.")
flags.DEFINE_boolean(
    "ignore_incorrect", False, "Ignore examples that are "
    "incorrectly classified to begin with.")


def create_attack_params():
  return registry.attack_params(FLAGS.attack_params_set)


def create_attack(attack):
  return registry.attacks(attack)


def main(argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  trainer_lib.set_random_seed(FLAGS.random_seed)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
  t2t_trainer.maybe_log_registry_and_exit()


  if FLAGS.cloud_mlengine:
    cloud_mlengine.launch()
    return

  if FLAGS.generate_data:
    t2t_trainer.generate_data()

  if cloud_mlengine.job_dir():
    FLAGS.output_dir = cloud_mlengine.job_dir()

  if argv:
    t2t_trainer.set_hparams_from_args(argv[1:])
  hparams = t2t_trainer.create_hparams()
  trainer_lib.add_problem_hparams(hparams, FLAGS.problem)
  attack_params = create_attack_params()
  attack_params.add_hparam("eps", 0.0)

  config = t2t_trainer.create_run_config(hparams)
  params = {"batch_size": hparams.batch_size}

  # add "_rev" as a hack to avoid image standardization
  problem = registry.problem(FLAGS.problem + "_rev")
  input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.EVAL, hparams)
  dataset = input_fn(params, config).repeat()
  features, _ = dataset.make_one_shot_iterator().get_next()
  inputs, labels = features["targets"], features["inputs"]
  inputs = tf.to_float(inputs)
  labels = tf.squeeze(labels)

  sess = tf.Session()

  model_fn = t2t_model.T2TModel.make_estimator_model_fn(
      FLAGS.model, hparams, use_tpu=FLAGS.use_tpu)
  ch_model = adv_attack_utils.T2TAttackModel(model_fn, params, config)

  acc_mask = None
  probs = ch_model.get_probs(inputs)
  if FLAGS.ignore_incorrect:
    preds = tf.argmax(probs, -1)
    preds = tf.squeeze(preds)
    acc_mask = tf.to_float(tf.equal(labels, preds))
  one_hot_labels = tf.one_hot(labels, probs.shape[-1])

  attack = create_attack(attack_params.attack)(ch_model, sess=sess)

  # Restore weights
  saver = tf.train.Saver()
  checkpoint_path = os.path.expanduser(FLAGS.output_dir or
                                       FLAGS.checkpoint_path)
  saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

  # reuse variables
  tf.get_variable_scope().reuse_variables()

  def compute_accuracy(x, labels, mask):
    preds = ch_model.get_probs(x)
    preds = tf.squeeze(preds)
    preds = tf.argmax(preds, -1, output_type=labels.dtype)
    _, acc_update_op = tf.metrics.accuracy(
        labels=labels, predictions=preds, weights=mask)
    sess.run(tf.initialize_local_variables())
    for _ in range(FLAGS.eval_steps):
      acc = sess.run(acc_update_op)
    return acc

  acc = compute_accuracy(inputs, labels, acc_mask)
  epsilon_acc_pairs = [(0.0, acc)]
  for epsilon in attack_params.attack_epsilons:
    attack_params.eps = epsilon
    adv_x = attack.generate(inputs, y=one_hot_labels, **attack_params.values())
    acc = compute_accuracy(adv_x, labels, acc_mask)
    epsilon_acc_pairs.append((epsilon, acc))

  for epsilon, acc in epsilon_acc_pairs:
    tf.logging.info("Accuracy @ eps=%f: %f" % (epsilon, acc))


if __name__ == "__main__":
  tf.app.run()
