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

"""T2T HParams handling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from tensor2tensor.data_generators import problem as problem_lib
from tensor2tensor.utils import hparam
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


def copy_hparams(hparams):
  hp_vals = hparams.values()
  new_hparams = hparam.HParams(**hp_vals)
  other_attrs = ["problem", "problem_hparams"]
  for attr in other_attrs:
    attr_val = getattr(hparams, attr, None)
    if attr_val is not None:
      setattr(new_hparams, attr, attr_val)
  return new_hparams


def create_hparams(hparams_set,
                   hparams_overrides_str="",
                   data_dir=None,
                   problem_name=None,
                   hparams_path=None):
  """Create HParams with data_dir and problem hparams, if kwargs provided."""
  hparams = registry.hparams(hparams_set)
  if hparams_path and tf.gfile.Exists(hparams_path):
    hparams = create_hparams_from_json(hparams_path, hparams)
  if data_dir:
    hparams.add_hparam("data_dir", data_dir)
  if hparams_overrides_str:
    tf.logging.info("Overriding hparams in %s with %s", hparams_set,
                    hparams_overrides_str)
    hparams = hparams.parse(hparams_overrides_str)
  if problem_name:
    add_problem_hparams(hparams, problem_name)
  return hparams


def create_hparams_from_json(json_path, hparams=None):
  """Loading hparams from json; can also start from hparams if specified."""
  tf.logging.info("Loading hparams from existing json %s" % json_path)
  with tf.gfile.Open(json_path, "r") as f:
    hparams_values = json.load(f)
    # Prevent certain keys from overwriting the passed-in hparams.
    # TODO(trandustin): Remove this hack after registries are available to avoid
    # saving them as functions.
    if hparams:
      hparams_values.pop("bottom", None)
      hparams_values.pop("loss", None)
      hparams_values.pop("name", None)
      hparams_values.pop("top", None)
      hparams_values.pop("weights_fn", None)
    new_hparams = hparam.HParams(**hparams_values)
    # Some keys are in new_hparams but not hparams, so we need to be more
    #   careful than simply using parse_json() from HParams
    if hparams:  # hparams specified, so update values from json
      for key in sorted(new_hparams.values().keys()):
        if hasattr(hparams, key):  # Overlapped keys
          value = getattr(hparams, key)
          new_value = getattr(new_hparams, key)
          if value != new_value:  # Different values
            tf.logging.info("Overwrite key %s: %s -> %s" % (
                key, value, new_value))
            setattr(hparams, key, new_value)
    else:
      hparams = new_hparams

  return hparams


def add_problem_hparams(hparams, problem_name_or_instance):
  """Add problem hparams for the problems."""
  if isinstance(problem_name_or_instance, problem_lib.Problem):
    problem = problem_name_or_instance
  else:
    problem = registry.problem(problem_name_or_instance)
  p_hparams = problem.get_hparams(hparams)
  hparams.problem = problem
  hparams.problem_hparams = p_hparams
