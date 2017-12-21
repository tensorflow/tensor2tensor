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

# Dependency imports

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("masters", "", "Comma-separated list of master addresses")
flags.DEFINE_string("ps", "", "Comma-separated list of ps addresses")


def main(_):
  if not (FLAGS.masters and FLAGS.ps):
    raise ValueError("Must provide --masters and --ps")

  masters = FLAGS.masters.split(",")
  ps = FLAGS.ps.split(",")

  cluster = {"ps": ps, "master": masters}

  for task_type, jobs in (("master", masters), ("ps", ps)):
    for idx, job in enumerate(jobs):
      if task_type == "master":
        cmd_line_flags = " ".join([
            "--master=grpc://%s" % job,
            "--ps_replicas=%d" % len(ps),
            "--worker_replicas=%d" % len(masters),
            "--worker_gpu=1",
            "--worker_id=%d" % idx,
            "--worker_job='/job:master'",
            "--ps_gpu=1",
            "--schedule=train",
            "--sync" if len(masters) == 1 else "",
        ])
      else:
        cmd_line_flags = " ".join([
            "--master=grpc://%s" % job,
            "--schedule=run_std_server",
        ])

      tf_config = json.dumps({
          "cluster": cluster,
          "task": {
              "type": task_type,
              "index": idx
          },
          "environment": "cloud",
      })
      print("'%s'\t%s" % (tf_config, cmd_line_flags))


if __name__ == "__main__":
  tf.app.run()
