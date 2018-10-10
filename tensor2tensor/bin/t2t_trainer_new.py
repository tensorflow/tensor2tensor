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

"""Train and evaluate.

Rewrite of t2t_trainer, introducing funtion run() that takes some parameters as
arguments instead of flags. Also takes problem and hparams as instances instead
of names in registry.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import sys

import six

from tensor2tensor import models  # pylint: disable=unused-import
from tensor2tensor import problems as problems_lib  # pylint: disable=unused-import
from tensor2tensor.bin import t2t_trainer  # pylint: disable=unused-import
from tensor2tensor.data_generators import problem  # pylint: disable=unused-import
from tensor2tensor.utils import cloud_mlengine
from tensor2tensor.utils import decoding
from tensor2tensor.utils import flags as t2t_flags  # pylint: disable=unused-import
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib_new
from tensor2tensor.utils import usr_dir
import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_config

flags = tf.flags
FLAGS = flags.FLAGS


def set_hparams_from_args(args):
  """Set hparams overrides from unparsed args list."""
  if not args:
    return

  hp_prefix = "--hp_"
  tf.logging.info("Found unparsed command-line arguments. Checking if any "
                  "start with %s and interpreting those as hparams "
                  "settings.", hp_prefix)

  pairs = []
  i = 0
  while i < len(args):
    arg = args[i]
    if arg.startswith(hp_prefix):
      pairs.append((arg[len(hp_prefix):], args[i+1]))
      i += 2
    else:
      tf.logging.warn("Found unknown flag: %s", arg)
      i += 1

  as_hparams = ",".join(["%s=%s" % (key, val) for key, val in pairs])
  if FLAGS.hparams:
    as_hparams = "," + as_hparams
  FLAGS.hparams += as_hparams


def create_hparams(hparams_set, hparams):
  if FLAGS.use_tpu and "tpu" not in hparams_set:
    tf.logging.warn("Not all hyperparameter sets work on TPU. "
                    "Prefer hparams_sets with a '_tpu' suffix, "
                    "e.g. transformer_tpu, if available for your model.")
  return trainer_lib_new.create_hparams(hparams_set, hparams)


def create_experiment_fn(**params):
  return trainer_lib_new.create_experiment_fn(
      model_name=params["model"],
      problem=params["problem"],
      data_dir=os.path.expanduser(params["data_dir"]),
      train_steps=params["train_steps"],
      eval_steps=params["eval_steps"],
      min_eval_frequency=params["local_eval_frequency"],
      schedule=FLAGS.schedule,
      eval_throttle_seconds=FLAGS.eval_throttle_seconds,
      export=FLAGS.export_saved_model,
      decode_hparams=decoding.decode_hparams(FLAGS.decode_hparams),
      use_tfdbg=FLAGS.tfdbg,
      use_dbgprofile=FLAGS.dbgprofile,
      eval_early_stopping_steps=FLAGS.eval_early_stopping_steps,
      eval_early_stopping_metric=FLAGS.eval_early_stopping_metric,
      eval_early_stopping_metric_delta=FLAGS.eval_early_stopping_metric_delta,
      eval_early_stopping_metric_minimize=FLAGS
      .eval_early_stopping_metric_minimize,
      use_tpu=FLAGS.use_tpu,
      use_tpu_estimator=FLAGS.use_tpu_estimator,
      use_xla=FLAGS.xla_compile,
      warm_start_from=FLAGS.warm_start_from,
      decode_from_file=FLAGS.decode_from_file,
      decode_to_file=FLAGS.decode_to_file,
      decode_reference=FLAGS.decode_reference,
      std_server_protocol=FLAGS.std_server_protocol)


def create_run_config(hp, **params):
  """Create a run config.

  Args:
    hp: model hyperparameters
    **params: other parameters passed to run().

  Returns:
    a run config
  """
  save_ckpt_steps = max(
      FLAGS.iterations_per_loop, params["local_eval_frequency"]
  )
  save_ckpt_secs = FLAGS.save_checkpoints_secs or None
  if save_ckpt_secs:
    save_ckpt_steps = None
  assert params["output_dir"] or FLAGS.checkpoint_path
  tpu_config_extra_kwargs = {}

  if getattr(hp, "mtf_mode", False):
    save_ckpt_steps = None  # Disable the default saver
    save_ckpt_secs = None  # Disable the default saver
    tpu_config_extra_kwargs = {
        "num_cores_per_replica": 1,
        "per_host_input_for_training": tpu_config.InputPipelineConfig.BROADCAST,
    }

  # the various custom getters we have written do not play well together yet.
  # TODO(noam): ask rsepassi for help here.
  daisy_chain_variables = (
      hp.daisy_chain_variables and
      hp.activation_dtype == "float32" and
      hp.weight_dtype == "float32")
  return trainer_lib_new.create_run_config(
      model_dir=os.path.expanduser(params["output_dir"]),
      master=FLAGS.master,
      iterations_per_loop=FLAGS.iterations_per_loop,
      num_shards=FLAGS.tpu_num_shards,
      log_device_placement=FLAGS.log_device_placement,
      save_checkpoints_steps=save_ckpt_steps,
      save_checkpoints_secs=save_ckpt_secs,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
      num_gpus=FLAGS.worker_gpu,
      gpu_order=FLAGS.gpu_order,
      shard_to_cpu=FLAGS.locally_shard_to_cpu,
      num_async_replicas=FLAGS.worker_replicas,
      gpu_mem_fraction=FLAGS.worker_gpu_memory_fraction,
      enable_graph_rewriter=FLAGS.enable_graph_rewriter,
      use_tpu=FLAGS.use_tpu,
      use_tpu_estimator=FLAGS.use_tpu_estimator,
      schedule=FLAGS.schedule,
      no_data_parallelism=hp.no_data_parallelism,
      daisy_chain_variables=daisy_chain_variables,
      ps_replicas=FLAGS.ps_replicas,
      ps_job=FLAGS.ps_job,
      ps_gpu=FLAGS.ps_gpu,
      sync=FLAGS.sync,
      worker_id=FLAGS.worker_id,
      worker_job=FLAGS.worker_job,
      random_seed=FLAGS.random_seed,
      tpu_infeed_sleep_secs=FLAGS.tpu_infeed_sleep_secs,
      inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
      log_step_count_steps=FLAGS.log_step_count_steps,
      intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads,
      tpu_config_extra_kwargs=tpu_config_extra_kwargs,
      cloud_tpu_name=FLAGS.cloud_tpu_name)


def generate_data(data_dir, tmp_dir, problem):  # pylint: disable=redefined-outer-name
  # Generate data if requested.
  data_dir = os.path.expanduser(data_dir)
  tmp_dir = os.path.expanduser(tmp_dir)
  tf.gfile.MakeDirs(data_dir)
  tf.gfile.MakeDirs(tmp_dir)

  tf.logging.info("Generating data for %s" % problem.name)
  problem.generate_data(data_dir, tmp_dir)


@contextlib.contextmanager
def profile_context():
  if FLAGS.profile:
    with tf.contrib.tfprof.ProfileContext(
        "t2tprof", trace_steps=range(100), dump_steps=range(100)) as pctx:
      opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
      pctx.add_auto_profiling("op", opts, range(100))
      yield
  else:
    yield


def maybe_log_registry_and_exit():
  if FLAGS.registry_help:
    tf.logging.info(registry.help_string())
    sys.exit(0)


def is_chief():
  schedules = ["train", "train_and_evaluate", "continuous_train_and_eval"]
  return FLAGS.worker_id == 0 and FLAGS.schedule in schedules


def save_metadata(hparams, output_dir):
  """Saves FLAGS and hparams to output_dir."""
  output_dir = os.path.expanduser(output_dir)
  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

  # Save FLAGS in txt file
  # This currently does not consider overrides passed to run().
  if hasattr(FLAGS, "flags_into_string"):
    flags_str = FLAGS.flags_into_string()
    t2t_flags_str = "\n".join([
        "--%s=%s" % (f.name, f.value)
        for f in FLAGS.flags_by_module_dict()["tensor2tensor.utils.flags"]
    ])
  else:
    flags_dict = FLAGS.__dict__["__flags"]
    flags_str = "\n".join(
        ["--%s=%s" % (name, str(f)) for (name, f) in flags_dict.items()])
    t2t_flags_str = None

  flags_txt = os.path.join(output_dir, "flags.txt")
  with tf.gfile.Open(flags_txt, "w") as f:
    f.write(flags_str)

  if t2t_flags_str:
    t2t_flags_txt = os.path.join(output_dir, "flags_t2t.txt")
    with tf.gfile.Open(t2t_flags_txt, "w") as f:
      f.write(t2t_flags_str)

  # Save hparams as hparams.json
  hparams_fname = os.path.join(output_dir, "hparams.json")
  with tf.gfile.Open(hparams_fname, "w") as f:
    f.write(hparams.to_json(indent=0, sort_keys=True))


def execute_schedule(exp):
  if not hasattr(exp, FLAGS.schedule):
    raise ValueError(
        "Experiment has no method %s, from --schedule" % FLAGS.schedule)
  with profile_context():
    getattr(exp, FLAGS.schedule)()


def run_std_server():
  exp = trainer_lib_new.T2TExperiment(*([None] * 5))
  exp.run_std_server()


def run(**params):  # pylint: disable=redefined-outer-name
  expected_params = {
      "data_dir", "eval_steps", "hparams", "hparams_set",
      "local_eval_frequency", "model", "output_dir", "problem",
      "train_steps"
  }

  # Check for unexpected params.
  for param_name in params:
    if param_name not in expected_params:
      raise ValueError("Unexpected parameter {}.".format(param_name))

  # Substitute missing params from FLAGS.
  for param_name in expected_params:
    if param_name not in params:
      params[param_name] = getattr(FLAGS, param_name)

  if isinstance(params["hparams"], six.string_types):
    params["hparams"] = create_hparams(params["hparams_set"], params["hparams"])
  if isinstance(params["problem"], six.string_types):
    params["problem"] = registry.problem(params["problem"])

  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.schedule == "run_std_server":
    run_std_server()
  trainer_lib_new.set_random_seed(FLAGS.random_seed)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
  maybe_log_registry_and_exit()

  if FLAGS.cloud_mlengine:
    cloud_mlengine.launch()
    return

  if FLAGS.generate_data:
    generate_data(params["data_dir"], FLAGS.tmp_dir, problem)

  if cloud_mlengine.job_dir():
    params["output_dir"] = cloud_mlengine.job_dir()

  exp_fn = create_experiment_fn(**params)
  exp = exp_fn(
      create_run_config(params["hparams"], **params), params["hparams"]
  )
  if is_chief():
    save_metadata(params["hparams"], params["output_dir"])
  execute_schedule(exp)


def main(argv):
  if argv:
    set_hparams_from_args(argv[1:])

  run()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
