# coding=utf-8
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

"""Utility to load code from an external user-supplied directory."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os
import sys

from tensor2tensor.utils import registry

# Dependency imports

import tensorflow as tf

from gcloud.gcs import fhfile


def import_usr_dir(usr_dir):
  """Import module at usr_dir, if provided."""
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


#Fathom
def fix_paths_for_workspace(FLAGS):
  """Update FLAGs to using workspace directories"""
  FLAGS.output_dir = fhfile.get_workspace_path(FLAGS.output_dir)
  FLAGS.data_dir = fhfile.get_workspace_path(os.path.expanduser(FLAGS.data_dir))

  problem_name = get_problem_name()
  problem = registry.problem(problem_name)
  for flag, _ in problem.file_flags_for_export_with_model().items():
    curr_val = FLAGS.__getattr__(flag)
    new_val = fhfile.get_workspace_path(curr_val)
    FLAGS.__setattr__(flag, new_val)


def get_problem_name(problems):
  problems = problems.split("-")
  assert len(problems) == 1
  return problems[0]
