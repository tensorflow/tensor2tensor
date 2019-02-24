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

r"""Computes the reward prediction confusion matrix given checkpoints and data.

  Usage:
  reward_confusion \
  --problem="gym_pong_deterministic-v4_random" \
  --model="next_frame_sv2p" \
  --hparams_set="next_frame_sv2p" \
  --output_dir=$CHECKPOINT_DIRECTORY \
  --data_dir=$DATA_DIRECTORY \

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensor2tensor.bin.t2t_decoder import create_hparams
from tensor2tensor.data_generators import problem  # pylint: disable=unused-import
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS


def print_confusion_matrix(title, cm):
  print("=" * 30)
  print(title)
  print("=" * 30)
  print(cm)
  print("=" * 30)
  print()


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  trainer_lib.set_random_seed(FLAGS.random_seed)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)

  # Create hparams
  hparams = create_hparams()
  hparams.force_full_predict = True
  batch_size = hparams.batch_size

  # Iterating over dev/test partition of the data.
  # Change the data partition if necessary.
  dataset = registry.problem(FLAGS.problem).dataset(
      tf.estimator.ModeKeys.PREDICT,
      shuffle_files=False,
      hparams=hparams)

  dataset = dataset.batch(batch_size, drop_remainder=True)
  data = dataset.make_one_shot_iterator().get_next()
  input_data = dict((k, data[k]) for k in data.keys() if k.startswith("input"))

  # Creat model
  model_cls = registry.model(FLAGS.model)
  model = model_cls(hparams, tf.estimator.ModeKeys.PREDICT)
  prediction_ops = model.infer(input_data)

  # Confusion Matrix
  nr = hparams.problem.num_rewards
  cm_per_frame = np.zeros((nr, nr), dtype=np.uint64)
  cm_next_frame = np.zeros((nr, nr), dtype=np.uint64)

  saver = tf.train.Saver()
  with tf.train.SingularMonitoredSession() as sess:
    # Load latest checkpoint
    ckpt = tf.train.get_checkpoint_state(FLAGS.output_dir).model_checkpoint_path
    saver.restore(sess.raw_session(), ckpt)

    counter = 0
    while not sess.should_stop():
      counter += 1
      if counter % 1 == 0:
        print(counter)

      # Predict next frames
      rew_pd, rew_gt = sess.run(
          [prediction_ops["target_reward"], data["target_reward"]])

      for i in range(batch_size):
        cm_next_frame[rew_gt[i, 0, 0], rew_pd[i, 0, 0]] += 1
        for gt, pd in zip(rew_gt[i], rew_pd[i]):
          cm_per_frame[gt, pd] += 1

  print_confusion_matrix("Per-frame Confusion Matrix", cm_per_frame)
  print_confusion_matrix("Next-frame Confusion Matrix", cm_next_frame)

if __name__ == "__main__":
  tf.app.run()
