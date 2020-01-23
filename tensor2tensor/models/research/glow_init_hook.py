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

"""Hook to run glow initialization on a larger batch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


class GlowInitHook(tf.train.SessionRunHook):
  """
  Hook that runs data-dependent initialization once before the first step.

  The init op is stored in the tf collection glow_init_op. Look at the
  "body" in glow.py for more details.
  """

  def after_create_session(self, session, coord):
    del coord
    global_step = session.run(tf.train.get_global_step())
    if global_step == 0:
      ddi = tf.get_collection("glow_init_op")
      # In-case of a multi-GPU system, this just runs the first op in the
      # collection.
      if ddi:
        session.run(ddi[0])
