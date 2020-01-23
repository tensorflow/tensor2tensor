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

"""Tests for tensor2tensor.utils.misc_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.utils import hparam
from tensor2tensor.utils import misc_utils
import tensorflow.compat.v1 as tf


class MiscUtilsTest(tf.test.TestCase):

  def test_camelcase_to_snakecase(self):
    self.assertEqual("typical_camel_case",
                     misc_utils.camelcase_to_snakecase("TypicalCamelCase"))
    self.assertEqual("numbers_fuse2gether",
                     misc_utils.camelcase_to_snakecase("NumbersFuse2gether"))
    self.assertEqual("numbers_fuse2_gether",
                     misc_utils.camelcase_to_snakecase("NumbersFuse2Gether"))
    self.assertEqual("lstm_seq2_seq",
                     misc_utils.camelcase_to_snakecase("LSTMSeq2Seq"))
    self.assertEqual("starts_lower",
                     misc_utils.camelcase_to_snakecase("startsLower"))
    self.assertEqual("starts_lower_caps",
                     misc_utils.camelcase_to_snakecase("startsLowerCAPS"))
    self.assertEqual("caps_fuse_together",
                     misc_utils.camelcase_to_snakecase("CapsFUSETogether"))
    self.assertEqual("startscap",
                     misc_utils.camelcase_to_snakecase("Startscap"))
    self.assertEqual("s_tartscap",
                     misc_utils.camelcase_to_snakecase("STartscap"))

  def test_snakecase_to_camelcase(self):
    self.assertEqual("TypicalCamelCase",
                     misc_utils.snakecase_to_camelcase("typical_camel_case"))
    self.assertEqual("NumbersFuse2gether",
                     misc_utils.snakecase_to_camelcase("numbers_fuse2gether"))
    self.assertEqual("NumbersFuse2Gether",
                     misc_utils.snakecase_to_camelcase("numbers_fuse2_gether"))
    self.assertEqual("LstmSeq2Seq",
                     misc_utils.snakecase_to_camelcase("lstm_seq2_seq"))

  def test_pprint_hparams(self):
    hparams = hparam.HParams(
        int_=1, str_="str", bool_=True, float_=1.1, list_int=[1, 2], none=None)

    # pylint: disable=g-inconsistent-quotes
    expected_string = r"""
{'bool_': True,
 'float_': 1.1,
 'int_': 1,
 'list_int': [1,
              2],
 'none': None,
 'str_': 'str'}"""
    # pylint: enable=g-inconsistent-quotes

    self.assertEqual(expected_string, misc_utils.pprint_hparams(hparams))

if __name__ == "__main__":
  tf.test.main()
