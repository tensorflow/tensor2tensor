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

"""Train on TPU."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# Dependency imports

from tensor2tensor import models  # pylint: disable=unused-import
from tensor2tensor import problems as problems_lib  # pylint: disable=unused-import
from tensor2tensor.tpu import tpu_trainer_lib
from tensor2tensor.utils import decoding
from tensor2tensor.utils import flags as t2t_flags  # pylint: disable=unused-import
from tensor2tensor.utils import registry
from tensor2tensor.utils import usr_dir

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

# See flags.py for additional command-line flags.
flags.DEFINE_string("t2t_usr_dir", "",
                    "Path to a Python module that will be imported. The "
                    "__init__.py file should include the necessary imports. "
                    "The imported files should contain registrations, "
                    "e.g. @registry.register_model calls, that will then be "
                    "available to the t2t-trainer.")
flags.DEFINE_integer("tpu_num_shards", 8, "Number of tpu shards.")
flags.DEFINE_integer("iterations_per_loop", 1000,
                     "Number of iterations in a TPU training loop.")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU.")

# To maintain compatibility with some internal libs, we guard against these flag
# definitions possibly erroring. Apologies for the ugliness.
try:
  flags.DEFINE_string("master", "", "Address of TensorFlow master.")
  flags.DEFINE_string("output_dir", "", "Base output directory for run.")
  flags.DEFINE_string("schedule", "continuous_train_and_eval",
                      "Method of Experiment to run.")
  flags.DEFINE_integer("eval_steps", 200, "Number of steps in evaluation.")
except:  # pylint: disable=bare-except
  pass


def get_problem_name():
  problems = FLAGS.problems.split("-")
  assert len(problems) == 1
  return problems[0]


def create_hparams():
  return tpu_trainer_lib.create_hparams(FLAGS.hparams_set, FLAGS.hparams)


def create_experiment_fn():
  use_validation_monitor = (FLAGS.schedule in
                            ["train_and_evaluate", "continuous_train_and_eval"]
                            and FLAGS.local_eval_frequency)
  return tpu_trainer_lib.create_experiment_fn(
      FLAGS.model,
      get_problem_name(),
      os.path.expanduser(FLAGS.data_dir),
      FLAGS.train_steps,
      FLAGS.eval_steps,
      FLAGS.local_eval_frequency,
      FLAGS.schedule,
      export=FLAGS.export_saved_model,
      decode_hparams=decoding.decode_hparams(FLAGS.decode_hparams),
      use_tfdbg=FLAGS.tfdbg,
      use_dbgprofile=FLAGS.dbgprofile,
      use_validation_monitor=use_validation_monitor,
      eval_early_stopping_steps=FLAGS.eval_early_stopping_steps,
      eval_early_stopping_metric=FLAGS.eval_early_stopping_metric,
      eval_early_stopping_metric_minimize=FLAGS.
      eval_early_stopping_metric_minimize,
      use_tpu=FLAGS.use_tpu)


def create_run_config(hp):
  return tpu_trainer_lib.create_run_config(
      model_dir=os.path.expanduser(FLAGS.output_dir),
      master=FLAGS.master,
      iterations_per_loop=FLAGS.iterations_per_loop,
      num_shards=FLAGS.tpu_num_shards,
      log_device_placement=FLAGS.log_device_placement,
      save_checkpoints_steps=max(FLAGS.iterations_per_loop,
                                 FLAGS.local_eval_frequency),
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
      num_gpus=FLAGS.worker_gpu,
      gpu_order=FLAGS.gpu_order,
      shard_to_cpu=FLAGS.locally_shard_to_cpu,
      num_async_replicas=FLAGS.worker_replicas,
      gpu_mem_fraction=FLAGS.worker_gpu_memory_fraction,
      enable_graph_rewriter=FLAGS.experimental_optimize_placement,
      use_tpu=FLAGS.use_tpu,
      schedule=FLAGS.schedule,
      no_data_parallelism=hp.no_data_parallelism,
      daisy_chain_variables=hp.daisy_chain_variables,
      ps_replicas=FLAGS.ps_replicas,
      ps_job=FLAGS.ps_job,
      ps_gpu=FLAGS.ps_gpu,
      sync=FLAGS.sync,
      worker_id=FLAGS.worker_id,
      worker_job=FLAGS.worker_job)


def log_registry():
  if FLAGS.registry_help:
    tf.logging.info(registry.help_string())
    sys.exit(0)


def execute_schedule(exp):
  if not hasattr(exp, FLAGS.schedule):
    raise ValueError(
        "Experiment has no method %s, from --schedule" % FLAGS.schedule)
  getattr(exp, FLAGS.schedule)()


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.set_random_seed(123)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
  log_registry()

  hparams = create_hparams()
  run_config = create_run_config(hparams)

  exp_fn = create_experiment_fn()
  exp = exp_fn(run_config, hparams)
  execute_schedule(exp)


if __name__ == "__main__":
  tf.app.run()
