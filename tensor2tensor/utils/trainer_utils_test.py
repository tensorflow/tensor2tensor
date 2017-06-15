# Copyright 2017 Google Inc.
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

"""Tests for trainer_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_utils as utils  # pylint: disable=unused-import

import tensorflow as tf


class TrainerUtilsTest(tf.test.TestCase):

  def testModelsImported(self):
    models = registry.list_models()
    self.assertTrue("baseline_lstm_seq2seq" in models)

  def testHParamsImported(self):
    hparams = registry.list_hparams()
    self.assertTrue("transformer_base" in hparams)


if __name__ == "__main__":
  tf.test.main()
