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

"""Test when the latent-network encoder is a 2-D conv."""

from absl.testing import parameterized
from tensor2tensor.models.video import nfg_test_utils
import tensorflow.compat.v1 as tf

conv_net_hparams = (
    ("in_3_out_2_conv", 3, 1, "conv_net", "conditional"),
    ("conv_net_cond_first", 2, 2, "conv_net", "conditional", -1, 3, True),)


class NextFrameGlowConvTest(nfg_test_utils.NextFrameGlowTest,
                            parameterized.TestCase):

  @parameterized.named_parameters(*conv_net_hparams)
  def testGlowTrainAndDecode(self, in_frames=1, out_frames=1,
                             latent_dist_encoder="pointwise",
                             gen_mode="conditional", pretrain_steps=-1,
                             num_train_frames=-1, cond_first_frame=False):
    self.GlowTrainAndDecode(
        in_frames=in_frames, out_frames=out_frames, gen_mode=gen_mode,
        latent_dist_encoder=latent_dist_encoder,
        pretrain_steps=pretrain_steps, num_train_frames=num_train_frames,
        cond_first_frame=cond_first_frame)


if __name__ == "__main__":
  tf.test.main()
