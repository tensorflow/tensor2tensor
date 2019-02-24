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

"""Basic tests for SAVP model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.models.video import savp
from tensor2tensor.models.video import savp_params
from tensor2tensor.models.video import tests_utils


import tensorflow as tf


class NextFrameTest(tests_utils.BaseNextFrameTest):

  def testSavpVAE(self):
    savp_hparams = savp_params.next_frame_savp()
    savp_hparams.use_vae = True
    savp_hparams.use_gan = False
    self.TestOnVariousInputOutputSizes(
        savp_hparams, savp.NextFrameSAVP, 1)
    self.TestOnVariousUpSampleLayers(
        savp_hparams, savp.NextFrameSAVP, 1)

  def testSavpGAN(self):
    hparams = savp_params.next_frame_savp()
    hparams.use_gan = True
    hparams.use_vae = False
    self.TestVideoModel(7, 5, hparams, savp.NextFrameSAVP, 1)

    hparams.gan_optimization = "sequential"
    self.TestVideoModel(7, 5, hparams, savp.NextFrameSAVP, 1)

  def testSavpGANVAE(self):
    hparams = savp_params.next_frame_savp()
    hparams.use_vae = True
    hparams.use_gan = True
    self.TestVideoModel(7, 5, hparams, savp.NextFrameSAVP, 1)

  def testInvalidVAEGANCombinations(self):
    hparams = savp_params.next_frame_savp()
    hparams.use_gan = False
    hparams.use_vae = False
    self.assertRaises(ValueError, self.TestVideoModel,
                      7, 5, hparams, savp.NextFrameSAVP, 1)

if __name__ == "__main__":
  tf.test.main()
