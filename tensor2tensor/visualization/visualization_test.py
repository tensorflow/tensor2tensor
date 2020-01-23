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

"""Tests for visualization library.

    IF ANY OF THESE TESTS BREAK PLEASE UPDATE THE CODE IN THE VIZ NOTEBOOK
******************************************************************************

Any fixes you have to make to this test or visualization.py to fix this test
might have to be reflected in the visualization notebook, for example if the
name of the hparams_set changes.

If you need help testing the changes please contact llion@.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor.utils import trainer_lib
from tensor2tensor.visualization import visualization
import tensorflow.compat.v1 as tf


def get_data_dir():
  pkg, _ = os.path.split(__file__)
  pkg, _ = os.path.split(pkg)
  return os.path.join(pkg, 'test_data')


problem_name = 'translate_ende_wmt32k'
model_name = 'transformer'
hparams_set = 'transformer_tiny'


class VisualizationTest(tf.test.TestCase):

  def setUp(self):
    super(VisualizationTest, self).setUp()
    self.data_dir = get_data_dir()

  def test_build_model_greedy(self):
    inputs, targets, outputs, _ = visualization.build_model(
        hparams_set, model_name, self.data_dir, problem_name, beam_size=1)

    self.assertAllEqual((1, None, 1, 1), inputs.shape.as_list())
    self.assertAllEqual((1, None, 1, 1), targets.shape.as_list())
    self.assertAllEqual((None, None), outputs.shape.as_list())

  def test_build_model_beam(self):
    inputs, targets, outputs, _ = visualization.build_model(
        hparams_set, model_name, self.data_dir, problem_name, beam_size=8)

    self.assertAllEqual((1, None, 1, 1), inputs.shape.as_list())
    self.assertAllEqual((1, None, 1, 1), targets.shape.as_list())
    self.assertAllEqual((None, None), outputs.shape.as_list())

  def test_get_vis_data_from_string(self):
    visualizer = visualization.AttentionVisualizer(
        hparams_set, model_name, self.data_dir, problem_name, beam_size=8)

    input_sentence = 'I have two dogs.'
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      _, inp_text, out_text, att_mats = (
          visualizer.get_vis_data_from_string(sess, input_sentence))

    self.assertAllEqual(
        [u'I_', u'have_', u'two_', u'dogs_', u'._', u'<EOS>'], inp_text)

    hparams = trainer_lib.create_hparams(
        hparams_set, data_dir=self.data_dir, problem_name=problem_name)

    enc_atts, dec_atts, encdec_atts = att_mats

    self.assertAllEqual(hparams.num_hidden_layers, len(enc_atts))

    enc_atts = enc_atts[0]
    dec_atts = dec_atts[0]
    encdec_atts = encdec_atts[0]

    batch_size = 1
    num_heads = hparams.num_heads
    inp_len = len(inp_text)
    out_len = len(out_text)

    self.assertAllEqual(
        (batch_size, num_heads, inp_len, inp_len), enc_atts.shape)
    self.assertAllEqual(
        (batch_size, num_heads, out_len, out_len), dec_atts.shape)
    self.assertAllEqual(
        (batch_size, num_heads, out_len, inp_len), encdec_atts.shape)

if __name__ == '__main__':
  tf.test.main()
