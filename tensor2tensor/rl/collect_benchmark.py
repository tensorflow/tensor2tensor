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

"""Collect benchmark."""

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

import functools
from gym.spaces import Discrete
from munch import Munch
from time import time


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("tpu", "", "Which TPU to use.")
tf.flags.DEFINE_string(
    "output_dir", "gs://tpu_atari_training_data/test", "Output dir."
)


HISTORY = 4


def make_model_fn(batch_size, epoch_length, num_tpus):
  def model_fn(features, labels, mode, params):
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
      batch_env = tf_new_collect.NewSimulatedBatchEnv(
          batch_size,
          "next_frame_basic_stochastic_discrete",
          trainer_lib.create_hparams("next_frame_basic_stochastic_discrete_long")
      )
      batch_env = tf_new_collect.NewStackWrapper(batch_env, HISTORY)
  
    ppo_hparams = trainer_lib.create_hparams("ppo_original_params")
    ppo_hparams.policy_network = "feed_forward_cnn_small_categorical_policy"
    ppo_hparams.epoch_length = epoch_length
    action_space = Discrete(2)
    with tf.variable_scope(
        "collect_{}_{}_{}".format(num_tpus, batch_size, epoch_length)
    ):
      memory = tf_new_collect.new_define_collect(
          batch_env, ppo_hparams, action_space, force_beginning_resets=True
      )
    # Summaries don't work when training on TPU; remove them.
    tf.get_default_graph().get_collection_ref(tf.GraphKeys.SUMMARIES)[:] = []
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT,
        predictions={"x": memory[1]},
    )
  return model_fn


def input_fn(params):
  return tf.data.Dataset.from_tensors(tf.zeros([16]))


def run(sess, run_config, num_tpus, batch_size, epoch_length):
  model_fn = make_model_fn(batch_size, epoch_length, num_tpus)
  estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=bool(FLAGS.tpu),
      train_batch_size=batch_size,
      eval_batch_size=batch_size,
      predict_batch_size=batch_size,
      params={},
      config=run_config,
  )
  return list(estimator.predict(input_fn=input_fn))


def main(_):
  if FLAGS.tpu:
    resolver = TPUClusterResolver(tpu=[FLAGS.tpu])
    target = resolver.get_master()
  else:
    target = ""
  with tf.Session(target) as sess:
    if FLAGS.tpu:
      topology = sess.run(tpu.initialize_system())
    else:
      topology = None

    run_config = tf.contrib.tpu.RunConfig(
        cluster=resolver,
        model_dir=FLAGS.output_dir,
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True
        ),
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=1,
            num_shards=8,
        ),
    )

    run(sess, run_config, num_tpus=8, batch_size=16, epoch_length=50)

    if FLAGS.tpu:
      sess.run(tpu.shutdown_system())


if __name__ == "__main__":
  tf.app.run()
