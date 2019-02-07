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

r"""Generates gifs out of a video checkpoint.

  Usage:
  prediction2gif \
  --problem="gym_pong_deterministic-v4_random" \
  --model="next_frame_sv2p" \
  --hparams_set="next_frame_sv2p" \
  --output_dir=$CHECKPOINT_DIRECTORY \
  --data_dir=$DATA_DIRECTORY \
  --output_gif=$USER/out.gif \

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import matplotlib as mpl
import numpy as np
from queue import Queue

from tensor2tensor.bin import t2t_trainer  # pylint: disable=unused-import
from tensor2tensor.layers import common_video
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir

import tensorflow as tf

mpl.use("Agg")
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("num_steps", 100, "Number of prediction steps.")
flags.DEFINE_integer("fps", 10, "Generated gif FPS.")
flags.DEFINE_string("output_gif", None, "Output path to save the gif.")


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  trainer_lib.set_random_seed(FLAGS.random_seed)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)

  # Create hparams
  hparams = trainer_lib.create_hparams(
      FLAGS.hparams_set,
      FLAGS.hparams,
      data_dir=os.path.expanduser(FLAGS.data_dir),
      problem_name=FLAGS.problem)
  hparams.force_full_predict = True
  hparams.scheduled_sampling_k = -1

  # Params
  num_agents = 1  # TODO(mbz): fix the code for more agents
  num_steps = FLAGS.num_steps
  if hasattr(hparams.problem, "num_actions"):
    num_actions = hparams.problem.num_actions
  else:
    num_actions = None
  frame_shape = hparams.problem.frame_shape
  resized_frame = hparams.preprocess_resize_frames is not None
  if resized_frame:
    frame_shape = hparams.preprocess_resize_frames
    frame_shape += [hparams.problem.num_channels]

  dataset = registry.problem(FLAGS.problem).dataset(
      tf.estimator.ModeKeys.TRAIN,
      shuffle_files=True,
      data_dir=os.path.expanduser(FLAGS.data_dir),
      hparams=hparams)

  dataset = dataset.batch(num_agents, drop_remainder=True)
  data = dataset.make_one_shot_iterator().get_next()
  # Setup input placeholders
  input_size = [num_agents, hparams.video_num_input_frames]
  if num_actions is None:
    placeholders = {
        "inputs": tf.placeholder(tf.float32, input_size + frame_shape)
    }
  else:
    placeholders = {
        "inputs": tf.placeholder(tf.float32, input_size + frame_shape),
        "input_action": tf.placeholder(tf.int64, input_size + [1]),
        "input_reward": tf.placeholder(tf.int64, input_size + [1]),
        "reset_internal_states": tf.placeholder(tf.float32, []),
    }
  # Create model.
  model_cls = registry.model(FLAGS.model)
  model = model_cls(hparams, tf.estimator.ModeKeys.PREDICT)
  prediction_ops = model.infer(placeholders)

  states_q = Queue(maxsize=hparams.video_num_input_frames)
  actions_q = Queue(maxsize=hparams.video_num_input_frames)
  rewards_q = Queue(maxsize=hparams.video_num_input_frames)
  if num_actions is not None:
    all_qs = [states_q, actions_q, rewards_q]
  else:
    all_qs = [states_q]

  writer = common_video.WholeVideoWriter(
      fps=FLAGS.fps, output_path=FLAGS.output_gif)

  saver = tf.train.Saver(tf.trainable_variables())
  with tf.train.SingularMonitoredSession() as sess:
    # Load latest checkpoint
    ckpt = tf.train.get_checkpoint_state(FLAGS.output_dir).model_checkpoint_path
    saver.restore(sess.raw_session(), ckpt)

    # get init frames from the dataset
    data_np = sess.run(data)

    frames = np.split(data_np["inputs"], hparams.video_num_input_frames, 1)
    for frame in frames:
      frame = np.squeeze(frame, 1)
      states_q.put(frame)
      writer.write(frame[0].astype(np.uint8))

    if num_actions is not None:
      actions = np.split(data_np["input_action"],
                         hparams.video_num_input_frames, 1)
      for action in actions:
        actions_q.put(np.squeeze(action, 1))

      rewards = np.split(data_np["input_reward"],
                         hparams.video_num_input_frames, 1)
      for reward in rewards:
        rewards_q.put(np.squeeze(reward, 1))

    for step in range(num_steps):
      print(">>>>>>> ", step)

      if num_actions is not None:
        random_actions = np.random.randint(num_actions-1)
        random_actions = np.expand_dims(random_actions, 0)
        random_actions = np.tile(random_actions, (num_agents, 1))

        # Shape inputs and targets
        inputs, input_action, input_reward = (
            np.stack(list(q.queue), axis=1) for q in all_qs)
      else:
        assert len(all_qs) == 1
        q = all_qs[0]
        elems = list(q.queue)
        # Need to adjust shapes sometimes.
        for i, e in enumerate(elems):
          if len(e.shape) < 4:
            elems[i] = np.expand_dims(e, axis=0)
        inputs = np.stack(elems, axis=1)

      # Predict next frames
      if num_actions is None:
        feed = {placeholders["inputs"]: inputs}
      else:
        feed = {
            placeholders["inputs"]: inputs,
            placeholders["input_action"]: input_action,
            placeholders["input_reward"]: input_reward,
            placeholders["reset_internal_states"]: float(step == 0),
        }
      predictions = sess.run(prediction_ops, feed_dict=feed)

      if num_actions is None:
        predicted_states = predictions[:, 0]
      else:
        predicted_states = predictions["targets"][:, 0]
        predicted_reward = predictions["target_reward"][:, 0]

      # Update queues
      if num_actions is None:
        new_data = (predicted_states)
      else:
        new_data = (predicted_states, random_actions, predicted_reward)
      for q, d in zip(all_qs, new_data):
        q.get()
        q.put(d.copy())

      writer.write(np.round(predicted_states[0]).astype(np.uint8))

    writer.finish_to_disk()

if __name__ == "__main__":
  tf.app.run()
