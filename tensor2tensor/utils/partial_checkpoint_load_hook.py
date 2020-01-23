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

"""Hook to partially load a checkpoint."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


class PartialCheckpointLoad(tf.train.SessionRunHook):
  """Partially load train_variables from a checkpoint.

  Hook used to load each variable saved in checkpoint into the graph. It
  will ignore any additional variables present in the graph that are not
  saved in the checkpoint. (Note: The loaded variables include ADAM/training
  variables, if they exist in the checkpoint)
  Can perform mapping if the base scopename for graph variables is different
  from the checkpoint variables.
  """

  def __init__(self, hook_context, chk_scopename, graph_scopename):
    """Initialize the hook with chkp directory and scopenames.

    Args:
      hook_context: HookContext object containing hparams.
      chk_scopename: Base scopename of variables in the checkpoint being loaded
      graph_scopename: Base scopename of variables in current graph
    """
    self.checkpoint_path = hook_context.hparams.partial_load_checkpoint
    self.chk_scopename = chk_scopename
    self.graph_scopename = graph_scopename

  def begin(self):
    # TODO(karishmamalkan): Add logging for when variables are loaded
    variable_references = {var.name: var for var in tf.all_variables()}
    variable_mappings = {}
    vars_in_chk = tf.train.list_variables(self.checkpoint_path)
    for (var, _) in vars_in_chk:
      variable_mappings[var] = variable_references[
          var.replace(self.chk_scopename, self.graph_scopename) + ":0"]
    tf.train.init_from_checkpoint(self.checkpoint_path, variable_mappings)
