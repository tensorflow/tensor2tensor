# Copyright 2017 The Tensor2Tensor Authors.
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

"""Utility to load code from an external directory supplied by user."""

import os
import sys
import importlib
import tensorflow as tf


def import_usr_dir(usr_dir):
  """Import user module, if provided."""
  if not usr_dir:
    return
  dir_path = os.path.expanduser(usr_dir)
  if dir_path[-1] == "/":
    dir_path = dir_path[:-1]
  containing_dir, module_name = os.path.split(dir_path)
  tf.logging.info("Importing user module %s from path %s", module_name,
                  containing_dir)
  sys.path.insert(0, containing_dir)
  importlib.import_module(module_name)
  sys.path.pop(0)
