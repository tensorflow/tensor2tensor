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
"""Utility to load code from an external user-supplied directory."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os
import sys
import tensorflow as tf


INTERNAL_USR_DIR_PACKAGE = "t2t_usr_dir_internal"


def import_usr_dir(usr_dir):
  """Import module at usr_dir, if provided."""
  if not usr_dir:
    return
  if usr_dir == INTERNAL_USR_DIR_PACKAGE:
    # The package has been installed with pip under this name for Cloud ML
    # Engine so just import it.
    importlib.import_module(INTERNAL_USR_DIR_PACKAGE)
    return

  dir_path = os.path.abspath(os.path.expanduser(usr_dir).rstrip("/"))
  containing_dir, module_name = os.path.split(dir_path)
  tf.logging.info("Importing user module %s from path %s", module_name,
                  containing_dir)
  sys.path.insert(0, containing_dir)
  importlib.import_module(module_name)
  sys.path.pop(0)
