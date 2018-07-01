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
"""Restore hooks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

from tensor2tensor.data_generators import generator_utils
import tensorflow as tf


class RestoreHook(tf.train.SessionRunHook):
  """Restore variables from a checkpoint path."""

  def __init__(self, checkpoint_path, new_model_scope="", old_model_scope="",
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
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(
        include=self._include, exclude=self._exclude)
    # remove new_model_scope from variable name prefix
    assignment_map = {variable.name[len(self._new_model_scope):]: variable
                      for variable in variables_to_restore
                      if variable.name.startswith(self._new_model_scope)}
    # remove :0 from variable name suffix
    assignment_map = {name.split(":")[0]: variable
                      for name, variable in assignment_map.iteritems()
                      if name.startswith(self._old_model_scope)}
    self._assignment_map = assignment_map

    tf.logging.info("restoring variables from checkpoint %s"%(
        self._checkpoint_path))
    tf.train.init_from_checkpoint(self._checkpoint_path, self._assignment_map)


class RestoreResnetHook(RestoreHook):
  """Restore Resnet models given scopes."""

  _RESNET_URL = "http://download.tensorflow.org/models/{}_2017_04_14.tar.gz"

  def __init__(self, new_model_scope="", include=None, exclude=None,
               old_model_scope="resnet_v2_152/", model_dir="/tmp"):
    model_name = old_model_scope[:-1]
    checkpoint_path = self.get_model(model_name, model_dir)
    super(RestoreResnetHook, self).__init__(
        checkpoint_path, new_model_scope, old_model_scope, include, exclude)

  def get_model(self, model_name, model_dir):
    """Download the model given model name and extract it to a directory."""
    resnet_url = self._RESNET_URL.format(model_name)
    model_filename = "{}.tar.gz".format(model_name)
    ckpt_filename = "{}.ckpt".format(model_name)

    path = generator_utils.maybe_download(model_dir, model_filename, resnet_url)
    with tarfile.open(path, "r:gz") as modeltar:
      modeltar.extractall(model_dir)
    return os.path.join(model_dir, ckpt_filename)
