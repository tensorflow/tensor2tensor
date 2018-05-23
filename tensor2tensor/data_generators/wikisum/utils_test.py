# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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
"""Tests for tensor2tensor.data_generators.wikisum.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensor2tensor.data_generators.wikisum import utils

import tensorflow as tf

pkg_dir, _ = os.path.split(__file__)
_TESTDATA = os.path.join(pkg_dir, "test_data")


def _get_testdata(filename):
  with tf.gfile.Open(os.path.join(_TESTDATA, filename)) as f:
    return f.read()


class UtilsTest(tf.test.TestCase):

  def test_filter_paragraph(self):
    for bad in tf.gfile.Glob(os.path.join(_TESTDATA, "para_bad*.txt")):
      for p in _get_testdata(bad).split("\n"):
        self.assertTrue(utils.filter_paragraph(p),
                        msg="Didn't filter %s" % p)
    for good in tf.gfile.Glob(os.path.join(_TESTDATA, "para_good*.txt")):
      for p in _get_testdata(good).split("\n"):
        p = _get_testdata(good)
      self.assertFalse(utils.filter_paragraph(p), msg="Filtered %s" % p)


if __name__ == "__main__":
  tf.test.main()
