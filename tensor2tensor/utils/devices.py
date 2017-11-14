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

"""Device placement and data parallelism."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

# pylint: disable=redefined-builtin
from six.moves import xrange
# pylint: enable=redefined-builtin

from tensor2tensor.utils import expert_utils as eu
import tensorflow as tf

# TODO(rsepassi): Rm dep on FLAGS here
FLAGS = tf.flags.FLAGS


def _ps_replicas(all_workers=False):
  if all_workers:
    return list(range(FLAGS.ps_replicas))
  # Worker K will be using replicas {0,...n-1} + K*n if we have n replicas.
  num_replicas = FLAGS.ps_replicas // FLAGS.worker_replicas
  return [d + FLAGS.worker_id * num_replicas for d in xrange(num_replicas)]


def _gpu_order(num_gpus):
  if FLAGS.gpu_order:
    ret = [int(s) for s in FLAGS.gpu_order.split(" ")]
    if len(ret) == num_gpus:
      return ret
  return list(range(num_gpus))


def _ps_gpus(all_workers=False):
  ps_gpus = []
  for d in _ps_replicas(all_workers=all_workers):
    ps_gpus.extend([(d, gpu) for gpu in _gpu_order(FLAGS.ps_gpu)])
  return ps_gpus


def ps_devices(all_workers=False):
  """List of ps devices (where to put the experts).

  Args:
    all_workers: whether the list is for all async workers or just this one.

  Returns:
    a list of device names
  """
  if FLAGS.ps_replicas > 0:
    if FLAGS.ps_gpu > 0:
      return [
          FLAGS.ps_job + "/task:%d/GPU:%d" % (d, gpu)
          for (d, gpu) in _ps_gpus(all_workers=all_workers)
      ]
    else:
      return [
          FLAGS.ps_job + "/task:%d" % d
          for d in _ps_replicas(all_workers=all_workers)
      ]
  else:
    if FLAGS.worker_gpu > 0:
      return ["gpu:%d" % d for d in _gpu_order(FLAGS.worker_gpu)]
    else:
      return [""]


def data_parallelism(all_workers=False):
  """Over which devices do we split each training batch.

  In old-fashioned async mode, we split the batch over all GPUs on the
  current worker.

  In sync mode, we split the batch over all the parameter server GPUs.

  This function returns an expert_utils.Parallelism object, which can be used
  to build the model.  It is configured in a way that any variables created
  by `tf.get_variable` will be assigned to the parameter servers and shared
  between datashards.

  Args:
    all_workers: whether the devices are all async workers or just this one.

  Returns:
    a expert_utils.Parallelism.
  """

  def _replica_device_setter(worker_device):
    if FLAGS.ps_replicas == 0:
      return worker_device
    return tf.train.replica_device_setter(
        worker_device=worker_device,
        ps_tasks=FLAGS.ps_replicas,
        ps_device=FLAGS.ps_job + "/GPU:0" if FLAGS.ps_gpu > 0 else FLAGS.ps_job)

  if FLAGS.schedule in ["train_and_evaluate", "continuous_train_and_eval"]:
    assert not FLAGS.sync
    tf.logging.warn(
        "Schedule=%s. Assuming that training is running on a single machine.",
        FLAGS.schedule)
    datashard_devices = ["gpu:%d" % d for d in _gpu_order(FLAGS.worker_gpu)]
    if FLAGS.locally_shard_to_cpu or FLAGS.worker_gpu < 1:
      datashard_devices += ["cpu:0"]
    caching_devices = None
  elif FLAGS.sync and FLAGS.ps_replicas > 0:
    # compute on ps
    datashard_devices = [
        _replica_device_setter(d) for d in ps_devices(all_workers=all_workers)
    ]
    if FLAGS.ps_gpu > 0 and FLAGS.ps_replicas > 1:
      caching_devices = [
          FLAGS.ps_job + "/task:%d/cpu:0" % d
          for (d, _) in _ps_gpus(all_workers=all_workers)
      ]
    else:
      caching_devices = None
  else:
    # compute on worker - this is either a single-worker setup or asynchronous
    # with parameter servers.
    if FLAGS.worker_gpu > 1:
      datashard_devices = [
          _replica_device_setter(FLAGS.worker_job + "/GPU:%d" % d)
          for d in _gpu_order(FLAGS.worker_gpu)
      ]
      caching_devices = [FLAGS.worker_job + "/GPU:0"] * FLAGS.worker_gpu
    else:
      datashard_devices = [_replica_device_setter(FLAGS.worker_job)]
      caching_devices = None
  tf.logging.info("datashard_devices: %s", datashard_devices)
  tf.logging.info("caching_devices: %s", caching_devices)
  return eu.Parallelism(
      datashard_devices,
      reuse=True,
      caching_devices=caching_devices,
      daisy_chain_variables=FLAGS.daisy_chain_variables)
