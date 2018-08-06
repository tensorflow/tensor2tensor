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
"""Hook to run tf.GraphKeys.UPDATE_OPS."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class UpdateOpsHook(tf.train.SessionRunHook):
  """Hook to run assign_ops."""

  def before_run(self, run_context):
    del run_context
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    return tf.train.SessionRunArgs(update_ops)
