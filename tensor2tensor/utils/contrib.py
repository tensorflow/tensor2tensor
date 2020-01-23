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

"""Wrappers around tf.contrib to dynamically import contrib packages.

This makes sure that libraries depending on T2T and TF2, do not crash at import.
"""

from __future__ import absolute_import
from __future__ import division  # Not necessary in a Python 3-only module
from __future__ import print_function  # Not necessary in a Python 3-only module

from absl import logging
from tensorflow.python import tf2  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
is_tf2 = tf2.enabled()


def err_if_tf2(msg='err'):
  if is_tf2:
    msg = 'contrib is unavailable in tf2.'
    if msg == 'err':
      raise ImportError(msg)
    else:
      logging.info(msg)


def slim():
  err_if_tf2()
  from tensorflow.contrib import slim as contrib_slim  # pylint: disable=g-import-not-at-top
  return contrib_slim


def util():
  err_if_tf2()
  from tensorflow.contrib import util as contrib_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_util


def tfe():
  err_if_tf2(msg='warn')
  from tensorflow.contrib.eager.python import tfe as contrib_eager  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_eager


def framework(msg='err'):
  err_if_tf2(msg=msg)
  from tensorflow.contrib import framework as contrib_framework  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_framework


def nn():
  err_if_tf2(msg='err')
  from tensorflow.contrib import nn as contrib_nn  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_nn


def layers():
  err_if_tf2(msg='err')
  from tensorflow.contrib import layers as contrib_layers  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_layers


def rnn():
  err_if_tf2(msg='err')
  from tensorflow.contrib import rnn as contrib_rnn  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_rnn


def seq2seq():
  err_if_tf2(msg='err')
  from tensorflow.contrib import seq2seq as contrib_seq2seq  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_seq2seq


def tpu():
  err_if_tf2(msg='err')
  from tensorflow.contrib import tpu as contrib_tpu  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_tpu


def training():
  err_if_tf2(msg='err')
  from tensorflow.contrib import training as contrib_training  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_training


def summary():
  err_if_tf2(msg='err')
  from tensorflow.contrib import summary as contrib_summary  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_summary


def metrics():
  err_if_tf2(msg='err')
  from tensorflow.contrib import metrics as contrib_metrics  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_metrics


def opt():
  err_if_tf2(msg='err')
  from tensorflow.contrib import opt as contrib_opt  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_opt


def mixed_precision():
  err_if_tf2(msg='err')
  from tensorflow.contrib import mixed_precision as contrib_mixed_precision  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_mixed_precision


def cluster_resolver():
  err_if_tf2(msg='err')
  from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_cluster_resolver


def distribute():
  err_if_tf2(msg='err')
  from tensorflow.contrib import distribute as contrib_distribute  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_distribute


def learn():
  err_if_tf2(msg='err')
  from tensorflow.contrib import learn as contrib_learn  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_learn


def tf_prof():
  err_if_tf2(msg='err')
  from tensorflow.contrib import tfprof as contrib_tfprof  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_tfprof


def eager():
  err_if_tf2(msg='err')
  from tensorflow.contrib.eager.python import tfe as contrib_eager  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_eager


def image():
  err_if_tf2(msg='err')
  from tensorflow.contrib import image as contrib_image  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return contrib_image
