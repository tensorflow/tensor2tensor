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

r"""Prune T2TModels using some heuristic.

This supports a very common form of pruning known as magnitude-based pruning.
It ranks individual weights or units according to their magnitudes and zeros
out the smallest k% of weights, effectively removing them from the graph.

Example run:
- train a resnet on cifar10:
    bin/t2t_trainer.py --problem=image_cifar10 --hparams_set=resnet_cifar_32 \
      --model=resnet

- evaluate different pruning percentages using weight-level pruning:
    bin/t2t_prune.py --pruning_params_set=resnet_weight --problem=image_cifar10\
      --hparams_set=resnet_cifar_32 --model=resnet
"""

import os

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import problem as problem_lib  # pylint: disable=unused-import
from tensor2tensor.utils import pruning_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir

import tensorflow.compat.v1 as tf

flags = tf.flags
FLAGS = flags.FLAGS

# See flags.py for additional command-line flags.
flags.DEFINE_string("pruning_params_set", None,
                    "Which pruning parameters to use.")


def create_pruning_params():
  return registry.pruning_params(FLAGS.pruning_params_set)


def create_pruning_strategy(name):
  return registry.pruning_strategy(name)


def main(argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  trainer_lib.set_random_seed(FLAGS.random_seed)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
  t2t_trainer.maybe_log_registry_and_exit()


  if FLAGS.generate_data:
    t2t_trainer.generate_data()

  if argv:
    t2t_trainer.set_hparams_from_args(argv[1:])
  hparams = t2t_trainer.create_hparams()
  trainer_lib.add_problem_hparams(hparams, FLAGS.problem)
  pruning_params = create_pruning_params()
  pruning_strategy = create_pruning_strategy(pruning_params.strategy)

  config = t2t_trainer.create_run_config(hparams)
  params = {"batch_size": hparams.batch_size}

  # add "_rev" as a hack to avoid image standardization
  problem = registry.problem(FLAGS.problem)
  input_fn = problem.make_estimator_input_fn(tf.estimator.ModeKeys.EVAL,
                                             hparams)
  dataset = input_fn(params, config).repeat()
  features, labels = dataset.make_one_shot_iterator().get_next()

  sess = tf.Session()

  model_fn = t2t_model.T2TModel.make_estimator_model_fn(
      FLAGS.model, hparams, use_tpu=FLAGS.use_tpu)
  spec = model_fn(
      features,
      labels,
      tf.estimator.ModeKeys.EVAL,
      params=hparams,
      config=config)

  # Restore weights
  saver = tf.train.Saver()
  checkpoint_path = os.path.expanduser(FLAGS.output_dir or
                                       FLAGS.checkpoint_path)
  saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

  def eval_model():
    preds = spec.predictions["predictions"]
    preds = tf.argmax(preds, -1, output_type=labels.dtype)
    _, acc_update_op = tf.metrics.accuracy(labels=labels, predictions=preds)
    sess.run(tf.initialize_local_variables())
    for _ in range(FLAGS.eval_steps):
      acc = sess.run(acc_update_op)
    return acc

  pruning_utils.sparsify(sess, eval_model, pruning_strategy, pruning_params)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
