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

"""Restore hooks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensor2tensor.utils import contrib
import tensorflow.compat.v1 as tf


class RestoreHook(tf.train.SessionRunHook):
  """Restore variables from a checkpoint path."""

  def __init__(self, checkpoint_path="", new_model_scope="", old_model_scope="",
               include=None, exclude=None):
    self._checkpoint_path = checkpoint_path
    self._new_model_scope = new_model_scope
    self._old_model_scope = old_model_scope
    self._include = include
    self._exclude = exclude

  def begin(self):
    """Load variables from checkpoint.

    New model variables have the following name foramt:
    new_model_scope/old_model_scope/xxx/xxx:0 To find the map of
    name to variable, need to strip the new_model_scope and then
    match the old_model_scope and remove the suffix :0.

    """
    variables_to_restore = contrib.framework().get_variables_to_restore(
        include=self._include, exclude=self._exclude)
    # remove new_model_scope from variable name prefix
    assignment_map = {variable.name[len(self._new_model_scope):]: variable
                      for variable in variables_to_restore
                      if variable.name.startswith(self._new_model_scope)}
    # remove :0 from variable name suffix
    assignment_map = {name.split(":")[0]: variable
                      for name, variable in six.iteritems(assignment_map)
                      if name.startswith(self._old_model_scope)}
    self._assignment_map = assignment_map

    tf.logging.info("restoring %d variables from checkpoint %s"%(
        len(assignment_map), self._checkpoint_path))
    tf.train.init_from_checkpoint(self._checkpoint_path, self._assignment_map)
