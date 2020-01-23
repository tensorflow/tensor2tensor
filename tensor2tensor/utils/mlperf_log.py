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

# Copyright 2018 MLBenchmark Group. All Rights Reserved.
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
# ==============================================================================
"""Convenience function for logging compliance tags to stdout.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import json
import logging
import os
import re
import sys
import time
import uuid

# pylint: disable=wildcard-import,unused-wildcard-import
from tensor2tensor.utils.mlperf_tags import *
# pylint: enable=wildcard-import,unused-wildcard-import


ROOT_DIR_GNMT = None

# Set by imagenet_main.py
ROOT_DIR_RESNET = None

# Set by transformer_main.py and process_data.py
ROOT_DIR_TRANSFORMER = None


PATTERN = re.compile("[a-zA-Z0-9]+")

LOG_FILE = os.getenv("COMPLIANCE_FILE")
# create logger with 'mlperf_compliance'
LOGGER = logging.getLogger("mlperf_compliance")
LOGGER.setLevel(logging.DEBUG)

_STREAM_HANDLER = logging.StreamHandler(stream=sys.stdout)
_STREAM_HANDLER.setLevel(logging.INFO)
LOGGER.addHandler(_STREAM_HANDLER)

if LOG_FILE:
  _FILE_HANDLER = logging.FileHandler(LOG_FILE)
  _FILE_HANDLER.setLevel(logging.DEBUG)
  LOGGER.addHandler(_FILE_HANDLER)
else:
  _STREAM_HANDLER.setLevel(logging.DEBUG)


def get_mode(hparams):
  """Returns whether we should do MLPerf logging."""
  return "mlperf_mode" in hparams and hparams.mlperf_mode


def get_caller(stack_index=2, root_dir=None):
  # pylint: disable=g-doc-args
  """Returns file.py:lineno of your caller.

  A stack_index of 2 will provide
      the caller of the function calling this function. Notice that stack_index
      of 2 or more will fail if called from global scope.
  """
  caller = inspect.getframeinfo(inspect.stack()[stack_index][0])

  # Trim the filenames for readability.
  filename = caller.filename
  if root_dir is not None:
    filename = re.sub("^" + root_dir + "/", "", filename)
  return "%s:%d" % (filename, caller.lineno)


def _mlperf_print(key, value=None, benchmark=None, stack_offset=0,
                  tag_set=None, deferred=False, root_dir=None,
                  extra_print=False):
  # pylint: disable=g-doc-args
  # pylint: disable=g-doc-return-or-yield
  """Prints out an MLPerf Log Line.

  key: The MLPerf log key such as 'CLOCK' or 'QUALITY'. See the list of log keys
  in the spec.
  value: The value which contains no newlines.
  benchmark: The short code for the benchmark being run, see the MLPerf log
  spec.
  stack_offset: Increase the value to go deeper into the stack to find the
  callsite. For example, if this
                is being called by a wraper/helper you may want to set
                stack_offset=1 to use the callsite
                of the wraper/helper itself.
  tag_set: The set of tags in which key must belong.
  deferred: The value is not presently known. In that case, a unique ID will
            be assigned as the value of this call and will be returned. The
            caller can then include said unique ID when the value is known
            later.
  root_dir: Directory prefix which will be trimmed when reporting calling file
            for compliance logging.
  extra_print: Print a blank line before logging to clear any text in the line.

  Example output:
    :::MLP-1537375353 MINGO[17] (eval.py:42) QUALITY: 43.7
  """

  return_value = None

  if (tag_set is None and not PATTERN.match(key)) or key not in tag_set:
    raise ValueError("Invalid key for MLPerf print: " + str(key))

  if value is not None and deferred:
    raise ValueError("deferred is set to True, but a value was provided")

  if deferred:
    return_value = str(uuid.uuid4())
    value = "DEFERRED: {}".format(return_value)

  if value is None:
    tag = key
  else:
    str_json = json.dumps(value)
    tag = "{key}: {value}".format(key=key, value=str_json)

  callsite = get_caller(2 + stack_offset, root_dir=root_dir)
  now = time.time()

  message = ":::MLPv0.5.0 {benchmark} {secs:.9f} ({callsite}) {tag}".format(
      secs=now, benchmark=benchmark, callsite=callsite, tag=tag)

  if extra_print:
    print()  # There could be prior text on a line

  if tag in STDOUT_TAG_SET:  # pylint: disable=undefined-variable
    LOGGER.info(message)
  else:
    LOGGER.debug(message)

  return return_value


TRANSFORMER_TAG_SET = set(TRANSFORMER_TAGS)  # pylint: disable=undefined-variable


def transformer_print(key, value=None, stack_offset=2, deferred=False,
                      hparams=None):
  if not hparams or not get_mode(hparams):
    return
  return _mlperf_print(
      key=key,
      value=value,
      benchmark=TRANSFORMER,  # pylint: disable=undefined-variable
      stack_offset=stack_offset,
      tag_set=TRANSFORMER_TAG_SET,
      deferred=deferred,
      root_dir=ROOT_DIR_TRANSFORMER)
