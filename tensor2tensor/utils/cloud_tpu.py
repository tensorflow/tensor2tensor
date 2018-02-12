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

"""Launch on TPU on GCP."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import json
import multiprocessing.pool as mp
import os
import random
import signal
import socket
import subprocess as sp
import time

from six.moves import input  # pylint: disable=redefined-builtin
import tensorflow as tf

TPU_IP = "10.240.%d.2"
TPU_PORT = 8470
TPU_PROFILE_PORT = 8466
TB_PORT = 6006

# TODO(rsepassi):
# --cloud_zone
# --cloud_project


class CloudState(object):
  """Manage state across multiple trainer runs."""

  def __init__(self):
    self._tmp_dir = os.path.expanduser("~/.t2t/cloud_state")
    tf.gfile.MakeDirs(self._tmp_dir)

  def cleanup(self, current_vm_name=None, current_tpu_name=None):
    process_pids = os.listdir(self._tmp_dir)
    for pid in process_pids:
      try:
        # Check if trainer pid is still running
        os.kill(int(pid), 0)
      except OSError:
        # Trainer died ungracefully
        pid_file = os.path.join(self._tmp_dir, pid)
        with tf.gfile.Open(pid_file) as f:
          info = json.loads(f.read())

        # Kill possibly zombie tunnel process
        try:
          os.kill(info["tunnel_pid"], signal.SIGTERM)
        except OSError:
          pass

        # Delete VM and TPU if requested
        del_vm = False
        del_tpu = False
        if info["delete_on_done"]:
          if (info["vm_name"] != current_vm_name and
              info["vm_name"] in zip(*list_vm_names_and_ips())[0]):
            print("Old VM %s found. Delete?" % info["vm_name"])
            if confirm():
              del_vm = True
          if (info["tpu_name"] != current_tpu_name and
              info["tpu_name"] in zip(*list_tpu_names_and_ips())[0]):
            print("Old TPU %s found. Delete?" % info["tpu_name"])
            if confirm():
              del_tpu = True

        results = []
        pool = mp.Pool(2)
        if del_vm:
          results.append(pool.apply_async(delete_vm, (info["vm_name"],)))
        if del_tpu:
          results.append(pool.apply_async(delete_tpu, (info["tpu_name"],)))
        _ = [res.get() for res in results]

        # Remove the now cleaned up state file
        tf.gfile.Remove(pid_file)

  def delete_current(self):
    pid_file = os.path.join(self._tmp_dir, str(os.getpid()))
    if tf.gfile.Exists(pid_file):
      tf.gfile.Remove(pid_file)

  def add_current(self, tunnel_pid, vm_name, tpu_name, delete_on_done):
    state = {
        "tunnel_pid": tunnel_pid,
        "vm_name": vm_name,
        "tpu_name": tpu_name,
        "delete_on_done": delete_on_done,
    }

    with tf.gfile.Open(os.path.join(self._tmp_dir, str(os.getpid())), "w") as f:
      f.write(json.dumps(state))


@contextlib.contextmanager
def cloud_tpu(vm_name, tpu_name, delete_on_done=False):
  """Gets or creates a VM and TPU instance, and forwards ports.

  Args:
    vm_name: str, name of VM.
    tpu_name: str, name of TPU instance.
    delete_on_done: bool, whether to delete the instances when done.

  Yields:
    master: str, grpc master pointing to the TPU instance.
  """
  state = CloudState()
  # Read state from previous processes and possibly cleanup
  state.cleanup(current_vm_name=vm_name, current_tpu_name=tpu_name)

  done_str = "" if delete_on_done else "NOT "
  print("Will %sdelete VM and TPU instance on done." % done_str)
  assert confirm()
  _, tpu_ip = create_vm_tpu_pair(vm_name, tpu_name)
  with tpu_tunnel(vm_name, tpu_ip) as (local_ports, tunnel_pid):
    master = "grpc://localhost:%d" % local_ports["tpu"]

    state.add_current(tunnel_pid, vm_name, tpu_name, delete_on_done)

    yield master

  if delete_on_done:
    pool = mp.Pool(2)
    vm_res = pool.apply_async(delete_vm, (vm_name,))
    tpu_res = pool.apply_async(delete_tpu, (tpu_name,))
    vm_res.get()
    tpu_res.get()

  # Cleanup state from this process
  state.delete_current()


class Gcloud(object):
  """gcloud command strings."""
  # Note these can be modified by set_versions
  VM_VERSION = "tf-1-5"
  TPU_VERSION = "1.5"

  @classmethod
  def set_versions(cls, vm, tpu):
    cls.VM_VERSION = vm
    cls.TPU_VERSION = tpu

  @classmethod
  def create_vm(cls):
    create_vm_str = """
    gcloud compute instances create {name} \
      --machine-type=n1-standard-8 \
      --image-family=%s \
      --image-project=ml-images \
      --scopes=https://www.googleapis.com/auth/cloud-platform
    """ % cls.VM_VERSION
    return create_vm_str

  DELETE_VM = "gcloud compute instances delete {name} --quiet"

  @classmethod
  def create_tpu(cls):
    create_tpu_str = """
    gcloud alpha compute tpus create \
      {name} \
      --range={tpu_ip}/29 \
      --version=%s
    """ % cls.TPU_VERSION
    return create_tpu_str

  DELETE_TPU = "gcloud alpha compute tpus delete {name} --quiet"

  LIST_TPU = "gcloud alpha compute tpus list"
  LIST_VM = "gcloud compute instances list"

  SSH_LOCAL_PORT_FORWARD = "-L {local_port}:{host}:{remote_port}"
  SSH_TUNNEL = """
  gcloud compute ssh {name} -- -N
  """

  DEFAULT_PROJECT = "gcloud config get-value project"
  DEFAULT_REGION = "gcloud config get-value compute/region"


@contextlib.contextmanager
def shell_background(cmd_, **kwargs):
  """Run process in background, join on exit."""
  args = format_cmd(cmd_, **kwargs)
  process = sp.Popen(args)
  try:
    yield process
  finally:
    if process.poll() is None:
      process.terminate()
      time.sleep(1)
    if process.poll() is None:
      process.kill()
      time.sleep(1)
    if process.poll() is None:
      raise ValueError(
          "Cannot kill process %d - please kill manually" % process.pid)
    time.sleep(1)


def shell_output(cmd_, **kwargs):
  return sp.check_output(format_cmd(cmd_, **kwargs))


def shell_run(cmd_, **kwargs):
  return sp.check_call(format_cmd(cmd_, **kwargs))


def format_cmd(cmd_, **kwargs):
  return cmd_.format(**kwargs).strip().split()


def default_region():
  return shell_output(Gcloud.DEFAULT_REGION).strip()


def default_project():
  return shell_output(Gcloud.DEFAULT_PROJECT).strip()


def create_vm(vm_name):
  out = shell_output(Gcloud.create_vm(), name=vm_name)
  return out.split("\n")[1:-1][0].split()[4]


def list_tpu_names_and_ips():
  list_out = shell_output(Gcloud.LIST_TPU)
  lines = [l.split() for l in list_out.split("\n")[1:-1]]
  names_and_ips = [(l[0].strip(), l[3].strip().split(":")[0]) for l in lines]
  return names_and_ips


def list_vm_names_and_ips():
  list_out = shell_output(Gcloud.LIST_VM)
  lines = [l.split() for l in list_out.split("\n")[1:-1]]
  names_and_ips = [(l[0].strip(), l[4].strip()) for l in lines]
  return names_and_ips


def unique_tpu_ip(tpu_names_and_ips):
  inuse = [el[1].split(".")[2] for el in tpu_names_and_ips]
  selection = random.choice(list(set(range(256)) - set(inuse)))
  return TPU_IP % selection


def delete_tpu(tpu_name):
  shell_run(Gcloud.DELETE_TPU, name=tpu_name)


def delete_vm(vm_name):
  shell_run(Gcloud.DELETE_VM, name=vm_name)


def create_tpu(tpu_name, tpu_names_and_ips=None):
  tpu_names_and_ips = tpu_names_and_ips or list_tpu_names_and_ips()
  tpu_ip = unique_tpu_ip(tpu_names_and_ips)

  rounded_tpu_ip = tpu_ip
  if rounded_tpu_ip.endswith("2"):
    rounded_tpu_ip = rounded_tpu_ip[:-1] + "0"

  shell_run(Gcloud.create_tpu(), name=tpu_name, tpu_ip=rounded_tpu_ip)
  return tpu_ip


@contextlib.contextmanager
def tpu_tunnel(vm_name, tpu_ip):
  """Forward TPU and TPU profiling ports."""
  local_ports = {
      "tpu": get_open_port(),
      "tpu_profile": get_open_port(),
  }

  tpu = format_cmd(
      Gcloud.SSH_LOCAL_PORT_FORWARD,
      local_port=local_ports["tpu"],
      host=tpu_ip,
      remote_port=TPU_PORT)
  tpu_profile = format_cmd(
      Gcloud.SSH_LOCAL_PORT_FORWARD,
      local_port=local_ports["tpu_profile"],
      host=tpu_ip,
      remote_port=TPU_PROFILE_PORT)

  args = format_cmd(Gcloud.SSH_TUNNEL, name=vm_name) + tpu + tpu_profile
  # Launch process running in background
  with shell_background(" ".join(args)) as tunnel_process:
    time.sleep(1)
    if tunnel_process.poll() is not None:
      raise ValueError("SSH failed")
    tf.logging.info("Set up port fowarding. Local ports: %s", local_ports)
    yield local_ports, tunnel_process.pid


def create_vm_tpu_pair(vm_name, tpu_name, reuse_if_exists=True):
  """Create a VM and paired TPU instance.

  Args:
    vm_name: str, name for VM.
    tpu_name: str, name for TPU instance.
    reuse_if_exists: bool, if True, this will act as a get or create. If False
      and vm_name or tpu_name already exists, will error.

  Returns:
    tuple: (vm_ip, tpu_ip)

  Raises:
    ValueError: if instance exists but reuse_if_exists=False.
  """
  vm_info = list_vm_names_and_ips()
  tpu_info = list_tpu_names_and_ips()

  vm_names = zip(*vm_info)[0]
  tpu_names = zip(*tpu_info)[0]

  make_vm = False
  vm_ip = None
  if vm_name in vm_names:
    if not reuse_if_exists:
      raise ValueError(
          "VM %s already exists and reuse_if_exists=False" % vm_name)
    tf.logging.info("VM %s already exists, reusing.", vm_name)
    vm_ip = vm_info[vm_names.index(vm_name)][1]
  else:
    print("Creating VM %s" % vm_name)
    assert confirm()
    make_vm = True

  make_tpu = False
  tpu_ip = None
  if tpu_name in tpu_names:
    if not reuse_if_exists:
      raise ValueError(
          "TPU instance %s already exists and reuse_if_exists=False" % tpu_name)
    tf.logging.info("TPU %s already exists, reusing.", tpu_name)
    tpu_ip = tpu_info[tpu_names.index(tpu_name)][1]
  else:
    print("Creating TPU instance %s" % tpu_name)
    assert confirm()
    make_tpu = True

  # Create VM and TPU in parallel
  pool = mp.Pool(2)
  vm_res = None
  tpu_res = None
  if make_vm:
    vm_res = pool.apply_async(create_vm, (vm_name,))
  if make_tpu:
    tpu_res = pool.apply_async(create_tpu, (tpu_name, tpu_info))
  if vm_res is not None:
    vm_ip = vm_res.get()
  if tpu_res is not None:
    tpu_ip = tpu_res.get()

  tf.logging.info("VM (Name, IP): %s, %s", vm_name, vm_ip)
  tf.logging.info("TPU (Name, IP): %s, %s", tpu_name, tpu_ip)
  tf.logging.info(
      "To delete the VM, run: %s", Gcloud.DELETE_VM.format(name=vm_name))
  tf.logging.info(
      "To delete the TPU instance, run: %s",
      Gcloud.DELETE_TPU.format(name=tpu_name))
  return vm_ip, tpu_ip


def get_open_port():
  s = socket.socket()
  s.bind(("", 0))
  s.listen(1)
  port = s.getsockname()[1]
  s.close()
  return port


def confirm():
  out = input("Confirm (Y/n)? > ")
  return out == "Y"
