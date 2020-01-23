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

"""Test when the latent-network encoder is a conv3d net."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from tensor2tensor.models.video import nfg_test_utils
import tensorflow.compat.v1 as tf

conv3d_net_hparams = (
    ("conv3d_net", 2, 2, "conv3d_net", "conditional", -1, 3),
    ("conv3d_net_gatu", 2, 2, "conv3d_net", "conditional", -1, 3, False, False,
     "gatu"),
    ("conv3d_dil", 2, 2, "conv3d_net", "conditional", -1, -1, False, True),)


class NextFrameGlowConv3DTest(nfg_test_utils.NextFrameGlowTest,
                              parameterized.TestCase):

  @parameterized.named_parameters(*conv3d_net_hparams)
  def testGlowTrainAndDecode(self, in_frames=1, out_frames=1,
                             latent_dist_encoder="pointwise",
                             gen_mode="conditional", pretrain_steps=-1,
                             num_train_frames=-1, cond_first_frame=False,
                             apply_dilations=False, activation="relu"):
    self.GlowTrainAndDecode(
        in_frames=in_frames, out_frames=out_frames,
        latent_dist_encoder=latent_dist_encoder, gen_mode=gen_mode,
        pretrain_steps=pretrain_steps, num_train_frames=num_train_frames,
        cond_first_frame=cond_first_frame, apply_dilations=apply_dilations,
        activation=activation)


if __name__ == "__main__":
  tf.test.main()
