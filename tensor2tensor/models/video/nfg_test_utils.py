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

"""Testing utils for next_frame_glow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import numpy as np
from tensor2tensor.data_generators import video_generated  # pylint: disable=unused-import
from tensor2tensor.models.video import next_frame_glow
from tensor2tensor.utils import registry
import tensorflow.compat.v1 as tf
MODES = tf.estimator.ModeKeys


# TODO(mechcoder): Refactor or merge tests with the other next_frame_tests when
# this moves to a public version.
def fill_hparams(hparams, in_frames, out_frames, gen_mode="conditional",
                 latent_dist_encoder="pointwise", pretrain_steps=-1,
                 num_train_frames=-1, cond_first_frame=False,
                 apply_dilations=False, activation="relu"):
  """Set next_frame_glow hparams."""
  hparams.latent_activation = activation
  hparams.latent_apply_dilations = apply_dilations
  hparams.video_num_input_frames = in_frames
  hparams.video_num_target_frames = out_frames
  hparams.latent_dist_encoder = latent_dist_encoder
  hparams.gen_mode = gen_mode
  hparams.pretrain_steps = pretrain_steps
  hparams.num_train_frames = num_train_frames
  hparams.cond_first_frame = cond_first_frame
  if latent_dist_encoder in ["conv_net", "conv3d_net"]:
    hparams.num_cond_latents = in_frames
  else:
    hparams.num_cond_latents = 1
  problem = registry.problem("video_stochastic_shapes10k")
  p_hparams = problem.get_hparams(hparams)
  hparams.problem = problem
  hparams.problem_hparams = p_hparams
  hparams.tiny_mode = True
  hparams.reward_prediction = False
  hparams.latent_architecture = "glow_resnet"
  hparams.latent_encoder_depth = 2
  hparams.latent_pre_output_channels = 32
  if (hparams.gen_mode == "conditional" and
      hparams.latent_dist_encoder == "pointwise"):
    hparams.batch_size = 16
    hparams.init_batch_size = 16
  else:
    hparams.batch_size = 16
    hparams.init_batch_size = 32
  hparams.affine_coupling_width = 32
  hparams.depth = 5
  hparams.n_levels = 2
  return hparams


def fill_infer_targets(x):
  x["infer_targets"] = tf.identity(x["targets"])
  return x


def create_basic_features(hparams):
  dataset = hparams.problem.dataset(MODES.TRAIN, hparams=hparams)
  dataset = dataset.batch(hparams.batch_size)
  dataset = dataset.map(fill_infer_targets)
  return dataset.make_one_shot_iterator().get_next()


class NextFrameGlowTest(tf.test.TestCase):
  """Utils for testing next_frame_glow."""

  def should_run_session(self, hparams):
    # dilated conv-3d not available on CPU.
    return tf.test.is_gpu_available() or not hparams.latent_apply_dilations

  def checkAllConds(self, conds_array, num_total_frames, hparams):
    if hparams.cond_first_frame:
      self.assertEqual(conds_array, [True]*(num_total_frames + 1))
    elif hparams.pretrain_steps > -1:
      self.assertEqual(conds_array, [False]*num_total_frames)
    elif hparams.latent_dist_encoder != "pointwise":
      self.assertEqual(conds_array, [True]*num_total_frames)

  def RunModel(self, model, train_op, hparams, features, num_frames,
               model_path=None):
    exp_num_frames = num_frames + int(hparams.cond_first_frame)
    if hparams.gen_mode == "conditional":
      self.assertLen(model.all_top_latents, exp_num_frames)
      self.assertLen(model.all_level_latents, exp_num_frames)

    with tf.Session() as session:

      if model_path is not None:
        saver = tf.train.Saver()

      session.run(tf.global_variables_initializer())

      # Run initialization.
      init_op = tf.get_collection("glow_init_op")
      session.run(init_op)

      loss, top_conds = session.run([train_op["training"], model._all_conds])  # pylint: disable=protected-access
      self.checkAllConds(top_conds, num_frames, hparams)

      if model_path is not None:
        saver.save(session, model_path)

      # Check that one forward-propagation does not NaN, i.e
      # initialization etc works as expected.
      self.assertTrue(loss > 0.0 and loss < 10.0)

  def GlowTrainAndDecode(self, in_frames=1, out_frames=1,
                         latent_dist_encoder="pointwise",
                         gen_mode="conditional", pretrain_steps=-1,
                         num_train_frames=-1, cond_first_frame=False,
                         apply_dilations=False, activation="relu"):
    """Test 1 forward pass and sampling gives reasonable results."""
    if num_train_frames == -1:
      total_frames = in_frames + out_frames
    else:
      total_frames = num_train_frames

    curr_dir = tempfile.mkdtemp()
    model_path = os.path.join(curr_dir, "model")

    # Training pipeline
    with tf.Graph().as_default():
      hparams = next_frame_glow.next_frame_glow_hparams()
      hparams = fill_hparams(hparams, in_frames, out_frames,
                             gen_mode, latent_dist_encoder, pretrain_steps,
                             num_train_frames, cond_first_frame,
                             apply_dilations, activation)
      features = create_basic_features(hparams)
      model = next_frame_glow.NextFrameGlow(hparams, MODES.TRAIN)
      _, train_op = model(features)
      if self.should_run_session(hparams):
        self.RunModel(model, train_op, hparams, features, total_frames,
                      model_path)

    # Inference pipeline
    with tf.Graph().as_default():
      hparams = next_frame_glow.next_frame_glow_hparams()
      if hparams.gen_mode == "unconditional":
        hparams.video_num_target_frames = 1
      hparams = fill_hparams(hparams, in_frames, out_frames,
                             gen_mode, latent_dist_encoder, pretrain_steps,
                             num_train_frames, cond_first_frame,
                             apply_dilations, activation)
      features = create_basic_features(hparams)
      model = next_frame_glow.NextFrameGlow(
          hparams, tf.estimator.ModeKeys.PREDICT)
      predictions = model.infer(features)
      outputs = predictions["outputs"]
      model_path = os.path.join(curr_dir, "model")

      if self.should_run_session(hparams):
        with tf.Session() as session:
          saver = tf.train.Saver()
          saver.restore(session, model_path)
          outputs_np = session.run(outputs)
          self.assertEqual(outputs_np.shape, (16, out_frames, 64, 64, 3))
          self.assertTrue(np.all(outputs_np <= 255))
          self.assertTrue(np.all(outputs_np >= 0))


if __name__ == "__main__":
  tf.test.main()
