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

"""Output command line arguments and json-encoded TF_CONFIGs.

Usage:

`t2t-make-tf-configs --masters="server1:1234" --ps="server3:2134,server4:2334"`

Outputs 1 line per job to stdout, first the masters, then the parameter servers.
Each line has the TF_CONFIG, then a tab, then the command line flags for that
job.

If there is a single master, it will have the `--sync` flag.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tensorflow.compat.v1 as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("masters", "", "Comma-separated list of master addresses")
flags.DEFINE_string("ps", "", "Comma-separated list of ps addresses")


def main(_):
  if not (FLAGS.masters and FLAGS.ps):
    raise ValueError("Must provide --masters and --ps")

  masters = FLAGS.masters.split(",")
  ps = FLAGS.ps.split(",")

  is_sync = len(masters) == 1
  if is_sync:
    print("Assuming SYNC distributed training with a single master and %d "
          "workers" % len(ps))
    cluster = {"ps": ps, "master": masters}
  else:
    print("Assuming ASYNC distributed training with %d workers and %d "
          "parameter servers" % (len(masters), len(ps)))
    cluster = {"ps": ps, "chief": [masters[0]], "worker": masters[1:]}

  # Trainer configs
  for idx, addr in enumerate(masters):
    cmd_line_flags = [
        "--master=grpc://%s" % addr,
        "--ps_replicas=%d" % len(ps),
        "--worker_replicas=%d" % len(masters),
        "--worker_gpu=%d" % (0 if is_sync else 1),
        "--worker_id=%d" % idx,
        "--ps_gpu=%d" % (1 if is_sync else 0),
        "--sync" if is_sync else "",
        "--schedule=train",
    ]
    if is_sync:
      task_type = "master"
      cmd_line_flags.append("--worker_job='/job:master'")
    else:
      if idx == 0:
        task_type = "chief"
        idx = 0
        cmd_line_flags.append("--worker_job='/job:chief'")
      else:
        task_type = "worker"
        idx -= 1
        cmd_line_flags.append("--worker_job='/job:worker'")

    tf_config = json.dumps({
        "cluster": cluster,
        "task": {
            "type": task_type,
            "index": idx
        },
        "environment": "cloud",
    })
    cmd_line_flags = " ".join(cmd_line_flags)
    print("'%s'\t%s" % (tf_config, cmd_line_flags))

  # Std server configs
  for idx, addr in enumerate(ps):
    tf_config = json.dumps({
        "cluster": cluster,
        "task": {
            "type": "ps",
            "index": idx
        },
        "environment": "cloud",
    })
    cmd_line_flags = "--schedule=run_std_server"
    print("'%s'\t%s" % (tf_config, cmd_line_flags))


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
