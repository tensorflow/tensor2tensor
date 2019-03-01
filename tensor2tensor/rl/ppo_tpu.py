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

"""PPO on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver

from tensor2tensor.models.research.rl import PolicyBase
from tensor2tensor.rl import ppo
from tensor2tensor.rl import tf_new_collect
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib

import tensorflow as tf

from gym.spaces import Discrete
from munch import Munch


FLAGS = tf.flags.FLAGS


EPOCH_LENGTH = 50
BATCH_SIZE = 16
HISTORY = 4
TPU_NAME = "ng-tpu-01"


def main(_):
  batch_env = tf_new_collect.NewSimulatedBatchEnv(
      BATCH_SIZE,
      "next_frame_basic_stochastic_discrete",
      trainer_lib.create_hparams("next_frame_basic_stochastic_discrete_long")
  )
  batch_env = tf_new_collect.NewStackWrapper(batch_env, HISTORY)

  ppo_hparams = trainer_lib.create_hparams("ppo_original_params")
  ppo_hparams.policy_network = "feed_forward_cnn_small_categorical_policy"
  action_space = Discrete(2)
  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    with tf.variable_scope("collect", reuse=tf.AUTO_REUSE):
      memory = tf_new_collect.new_define_collect(
          batch_env, ppo_hparams, action_space, force_beginning_resets=True
      )
    ppo_summary = memory
    ppo_summary = ppo.define_ppo_epoch(
        memory, ppo_hparams, action_space, batch_env.batch_size
    )
  ppo_summary = tpu.rewrite(lambda: ppo_summary)
  tpu_grpc_url = TPUClusterResolver(tpu=[TPU_NAME]).get_master()
  with tf.Session(tpu_grpc_url) as sess:
    sess.run(tpu.initialize_system())
    sess.run(tf.global_variables_initializer())
    summary = sess.run(ppo_summary)
    print(summary)
    sess.run(tpu.shutdown_system())


if __name__ == "__main__":
  tf.app.run()
