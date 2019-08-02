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

"""Simple client binary that talks to remote envs, for debugging."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
from absl import app
from absl import flags
import numpy as np  # pylint: disable=unused-import
from tensor2tensor import envs  # pylint: disable=unused-import
from tensor2tensor.envs import client_env
from tensor2tensor.envs import env_problem_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("server_bns", "", "Server's BNS.")
flags.DEFINE_integer("replicas", 0, "Number of replicas in the server.")


def main(argv):
  del argv

  if FLAGS.replicas == 0:
    env = client_env.ClientEnv(FLAGS.server_bns)
    pdb.set_trace()
    env.close()
    return

  # Replicated server.
  per_env_kwargs = [{
      "remote_env_address": os.path.join(FLAGS.server_bns, str(replica))
  } for replica in range(FLAGS.replicas)]
  env = env_problem_utils.make_env(
      batch_size=FLAGS.replicas,
      env_problem_name="ClientEnv-v0",
      resize=False,
      parallelism=FLAGS.replicas,
      per_env_kwargs=per_env_kwargs)

  pdb.set_trace()

  env.close()


if __name__ == "__main__":
  app.run(main)
