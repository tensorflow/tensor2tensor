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

import inspect

# Dependency imports

from tensor2tensor.utils import expert_utils as eu
import tensorflow as tf


def data_parallelism_from_flags(daisy_chain_variables=True, all_workers=False):
  """Over which devices do we split each training batch.

  In old-fashioned async mode, we split the batch over all GPUs on the
  current worker.

  In sync mode, we split the batch over all the parameter server GPUs.

  This function returns an expert_utils.Parallelism object, which can be used
  to build the model.  It is configured in a way that any variables created
  by `tf.get_variable` will be assigned to the parameter servers and shared
  between datashards.

  Args:
    daisy_chain_variables: whether to copy variables in a daisy chain on GPUs.
    all_workers: whether the devices are all async workers or just this one.

  Returns:
    a expert_utils.Parallelism.
  """
  dp_arg_names = inspect.getargspec(data_parallelism).args

  blacklist = ["daisy_chain_variables", "all_workers"]

  kwargs = {}
  for arg in dp_arg_names:
    if arg in blacklist:
      continue
    kwargs[arg] = getattr(tf.flags.FLAGS, arg)

  return data_parallelism(
      daisy_chain_variables=daisy_chain_variables,
      all_workers=all_workers,
      **kwargs)


def data_parallelism(daisy_chain_variables=True,
                     all_workers=False,
                     ps_replicas=0,
                     ps_job="/job:ps",
                     ps_gpu=0,
                     schedule="continuous_train_and_eval",
                     sync=False,
                     worker_gpu=1,
                     worker_replicas=1,
                     worker_id=0,
                     gpu_order="",
                     locally_shard_to_cpu=False,
                     worker_job="/job:localhost",
                     no_data_parallelism=False):
  """See data_parallelism_from_flags."""
  tf.logging.info("schuedule=%s" % schedule)
  tf.logging.info("worker_gpu=%s" % worker_gpu)
  tf.logging.info("sync=%s" % sync)
  def _ps_replicas(all_workers=False):
    if all_workers:
      return list(range(ps_replicas))
    # Worker K will be using replicas {0,...n-1} + K*n if we have n replicas.
    num_replicas = ps_replicas // worker_replicas
    return [d + worker_id * num_replicas for d in range(num_replicas)]

  def _gpu_order(num_gpus):
    if gpu_order:
      ret = [int(s) for s in gpu_order.split(" ")]
      if len(ret) == num_gpus:
        return ret
    return list(range(num_gpus))

  def _ps_gpus(all_workers=False):
    ps_gpus = []
    for d in _ps_replicas(all_workers=all_workers):
      ps_gpus.extend([(d, gpu) for gpu in _gpu_order(ps_gpu)])
    return ps_gpus

  def ps_devices(all_workers=False):
    """List of ps devices (where to put the experts).

    Args:
      all_workers: whether the list is for all async workers or just this one.

    Returns:
      a list of device names
    """
    if ps_replicas > 0:
      if ps_gpu > 0:
        return [
            ps_job + "/task:%d/GPU:%d" % (d, gpu)
            for (d, gpu) in _ps_gpus(all_workers=all_workers)
        ]
      else:
        return [
            ps_job + "/task:%d" % d
            for d in _ps_replicas(all_workers=all_workers)
        ]
    else:
      if worker_gpu > 0:
        return ["gpu:%d" % d for d in _gpu_order(worker_gpu)]
      else:
        return [""]

  def _replica_device_setter(worker_device):
    if ps_replicas == 0:
      return worker_device
    return tf.train.replica_device_setter(
        worker_device=worker_device,
        ps_tasks=ps_replicas,
        ps_device=ps_job + "/GPU:0" if ps_gpu > 0 else ps_job)

  if no_data_parallelism:
    datashard_devices = [""]
    caching_devices = None
  elif schedule in ["train_and_evaluate", "continuous_train_and_eval"]:
    assert not sync
    tf.logging.warn(
        "Schedule=%s. Assuming that training is running on a single machine.",
        schedule)
    datashard_devices = ["gpu:%d" % d for d in _gpu_order(worker_gpu)]
    if locally_shard_to_cpu or worker_gpu < 1:
      datashard_devices += ["cpu:0"]
    caching_devices = None
  elif sync and ps_replicas > 0:
    # compute on ps
    datashard_devices = [
        _replica_device_setter(d) for d in ps_devices(all_workers=all_workers)
    ]
    if ps_gpu > 0 and ps_replicas > 1:
      caching_devices = [
          ps_job + "/task:%d/cpu:0" % d
          for (d, _) in _ps_gpus(all_workers=all_workers)
      ]
    else:
      caching_devices = None
  else:
    # compute on worker - this is either a single-worker setup or asynchronous
    # with parameter servers.
    if worker_gpu > 1:
      datashard_devices = [
          _replica_device_setter(worker_job + "/GPU:%d" % d)
          for d in _gpu_order(worker_gpu)
      ]
      caching_devices = [worker_job + "/GPU:0"] * worker_gpu
    else:
      datashard_devices = [_replica_device_setter(worker_job)]
      caching_devices = None
  tf.logging.info("datashard_devices: %s", datashard_devices)
  tf.logging.info("caching_devices: %s", caching_devices)
  tf.logging.info("ps_devices: %s", ps_devices(all_workers=all_workers))
  return eu.Parallelism(
      datashard_devices,
      caching_devices=caching_devices,
      daisy_chain_variables=daisy_chain_variables,
      ps_devices=ps_devices(all_workers=all_workers))
