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

"""Common command-line flags."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool("registry_help", False,
                  "If True, logs the contents of the registry and exits.")
flags.DEFINE_bool("tfdbg", False,
                  "If True, use the TF debugger CLI on train/eval.")
flags.DEFINE_bool("export_saved_model", False,
                  "DEPRECATED - see serving/export.py.")
flags.DEFINE_bool("dbgprofile", False,
                  "If True, record the timeline for chrome://tracing/.")
flags.DEFINE_string("model", None, "Which model to use.")
flags.DEFINE_string("hparams_set", None, "Which parameters to use.")
flags.DEFINE_string("hparams_range", None, "Parameters range.")
flags.DEFINE_string("hparams", "",
                    "A comma-separated list of `name=value` hyperparameter "
                    "values. This flag is used to override hyperparameter "
                    "settings either when manually selecting hyperparameters "
                    "or when using Vizier. If a hyperparameter setting is "
                    "specified by this flag then it must be a valid "
                    "hyperparameter name for the model.")
flags.DEFINE_string("problem", None, "Problem name.")

# data_dir is a common flag name - catch conflicts and define it once.
try:
  flags.DEFINE_string("data_dir", None, "Directory with training data.")
except:  # pylint: disable=bare-except
  pass

flags.DEFINE_integer("train_steps", 250000,
                     "The number of steps to run training for.")
flags.DEFINE_string("eval_early_stopping_metric", "loss",
                    "If --eval_early_stopping_steps is not None, then stop "
                    "when --eval_early_stopping_metric has not decreased for "
                    "--eval_early_stopping_steps")
flags.DEFINE_float("eval_early_stopping_metric_delta", 0.1,
                   "Delta determining whether metric has plateaued.")
flags.DEFINE_integer("eval_early_stopping_steps", None,
                     "If --eval_early_stopping_steps is not None, then stop "
                     "when --eval_early_stopping_metric has not decreased for "
                     "--eval_early_stopping_steps")
flags.DEFINE_bool("eval_early_stopping_metric_minimize", True,
                  "Whether to check for the early stopping metric going down "
                  "or up.")
flags.DEFINE_integer("eval_timeout_mins", 240,
                     "The maximum amount of time to wait to wait between "
                     "checkpoints. Set -1 to wait indefinitely.")
flags.DEFINE_bool("eval_run_autoregressive", False,
                  "Run eval autoregressively where we condition on previous"
                  "generated output instead of the actual target.")
flags.DEFINE_bool("eval_use_test_set", False,
                  "Whether to use the '-test' data for EVAL (and PREDICT).")
flags.DEFINE_integer("keep_checkpoint_max", 20,
                     "How many recent checkpoints to keep.")
flags.DEFINE_bool("enable_graph_rewriter", False,
                  "Enable graph optimizations that are not on by default.")
flags.DEFINE_integer("keep_checkpoint_every_n_hours", 10000,
                     "Number of hours between each checkpoint to be saved. "
                     "The default value 10,000 hours effectively disables it.")
flags.DEFINE_integer("save_checkpoints_secs", 0,
                     "Save checkpoints every this many seconds. "
                     "Default=0 means save checkpoints each x steps where x "
                     "is max(iterations_per_loop, local_eval_frequency).")
flags.DEFINE_bool("log_device_placement", False,
                  "Whether to log device placement.")
flags.DEFINE_string("warm_start_from", None, "Warm start from checkpoint.")

# Distributed training flags
flags.DEFINE_integer("local_eval_frequency", 1000,
                     "Save checkpoints and run evaluation every N steps during "
                     "local training.")
flags.DEFINE_integer("eval_throttle_seconds", 600,
                     "Do not re-evaluate unless the last evaluation was started"
                     " at least this many seconds ago.")
flags.DEFINE_bool("sync", False, "Sync compute on PS.")
flags.DEFINE_string("worker_job", "/job:localhost", "name of worker job")
flags.DEFINE_integer("worker_gpu", 1, "How many GPUs to use.")
flags.DEFINE_integer("worker_replicas", 1, "How many workers to use.")
flags.DEFINE_integer("worker_id", 0, "Which worker task are we.")
flags.DEFINE_float("worker_gpu_memory_fraction", 0.95,
                   "Fraction of GPU memory to allocate.")
flags.DEFINE_integer("ps_gpu", 0, "How many GPUs to use per ps.")
flags.DEFINE_string("gpu_order", "", "Optional order for daisy-chaining GPUs."
                    " e.g. \"1 3 2 4\"")
flags.DEFINE_string("ps_job", "/job:ps", "name of ps job")
flags.DEFINE_integer("ps_replicas", 0, "How many ps replicas.")

# Decoding flags
flags.DEFINE_string("decode_hparams", "",
                    "Comma-separated list of name=value pairs to control "
                    "decode behavior. See decoding.decode_hparams for "
                    "defaults.")
flags.DEFINE_string("decode_from_file", "",
                    "Path to the source file for decoding, used by "
                    "continuous_decode_from_file.")
flags.DEFINE_string("decode_to_file", "",
                    "Path to the decoded file generated by decoding, used by "
                    "continuous_decode_from_file.")
flags.DEFINE_string("decode_reference", "",
                    "Path to the reference file for decoding, used by "
                    "continuous_decode_from_file to compute BLEU score.")
