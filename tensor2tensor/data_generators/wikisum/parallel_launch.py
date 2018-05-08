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
# pylint: disable=line-too-long
r"""Launch a script in parallel on GCP.

For each instance (`--num_instances`), the script will copy the code in
`--code_dir` to the instance, run `--setup_command` and then run
`--command_prefix` joined with the task's id or a line in
`--per_instance_suffix_file`.

Note that the machines will attempt to down themselves on completion or failure.
If they do not, you can delete them manually or use delete_instances.sh to
delete many at once.

Example usage:

```
BUCKET=gs://my-bucket
python parallel_launch.py \
  --num_instances=1000 \
  --cpu=4 --mem=4 \
  --name=wikisum-refs-web \
  --code_dir=./ \
  --log_dir=$BUCKET/refs_logs \
  --setup_command="pip3 install aiohttp cchardet aiodns bs4 -q --user" \
  --command_prefix="python3 wikisum/get_references_web.py --out_dir=$BUCKET/wiki_references --shard_id"
```
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import multiprocessing as mp
import os
import socket
import subprocess as sp
import time

from tensor2tensor.utils import cloud_tpu as cloud
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_integer("num_instances", None, "Number of instances to launch.")
flags.DEFINE_string("name", None, "Instance name prefix.")
flags.DEFINE_string("log_dir", None, "GCS bucket to copy logs out to.")
flags.DEFINE_string("code_dir", None, "Directory to copy.")
flags.DEFINE_string("setup_command", None, "Setup command to run.")
flags.DEFINE_string("command_prefix", None, "Command to run, prefix.")
flags.DEFINE_string("per_instance_suffix_file", None,
                    "Command to run, suffix per instance. If None, suffix will "
                    "be instance id.")
flags.DEFINE_integer("cpu", 1, "Number of CPUs per instance.")
flags.DEFINE_integer("mem", 4, "Memory in GB per instance.")
flags.DEFINE_integer("num_threads", 48,
                     "Number of threads to use to spin up jobs.")
flags.DEFINE_bool("debug_keep_up", False,
                  "If True, will keep the machine up. num_instances must be 1.")
flags.DEFINE_string("instance_ids", None,
                    "Comma-separated list of integer instance ids to launch. "
                    "Useful if some failed on a previous run and you only want "
                    "to rerun specific tasks.")


DELETE = "gcloud compute instances delete {name}"
DELETE_SELF = ("gcloud compute instances delete $(hostname) --quiet "
               "--zone={zone}")
CREATE_INSTANCE = ("gcloud compute instances create {instance_name} "
                   "--custom-cpu {cpu} --custom-memory {mem} "
                   "--custom-extensions "
                   "--image-project=ml-images --image-family=tf-1-7 "
                   "--scopes=cloud-platform")
COPY_CODE = "gcloud compute scp --recurse {local_dir} {instance_name}:~/"
SSH = "gcloud compute ssh {instance_name} --command"
SCREEN = "screen -dmS test bash -c \"{command}\""
DEFAULT_ZONE = "gcloud config get-value compute/zone"
LOGS = "> ~/logs-{task_id}.txt 2>&1; gsutil cp ~/logs-{task_id}.txt {bucket}"


def remote_run(cmd, instance_name, detach=False, retries=1):
  """Run command on GCS instance, optionally detached."""
  if detach:
    cmd = SCREEN.format(command=cmd)
  args = SSH.format(instance_name=instance_name).split()
  args.append(cmd)
  for i in range(retries + 1):
    try:
      if i > 0:
        tf.logging.info("Retry %d for %s", i, args)
      return sp.check_call(args)
    except sp.CalledProcessError as e:
      if i == retries:
        raise e


def default_zone():
  return cloud.shell_output(DEFAULT_ZONE).strip()


@contextlib.contextmanager
def safe_socket(timeout=2):
  s = socket.socket()
  s.settimeout(timeout)
  try:
    yield s
  finally:
    s.close()


def wait_for_ssh(ip):
  """Wait for SSH to be available at given IP address."""
  for _ in range(12):
    with safe_socket() as s:
      try:
        s.connect((ip, 22))
        return True
      except socket.timeout:
        pass
    time.sleep(10)
  return False


def create_instance(instance_name, cpu=1, mem=4):
  tf.logging.info("Creating instance %s", instance_name)
  out = cloud.shell_output(CREATE_INSTANCE, instance_name=instance_name,
                           cpu=cpu, mem=mem)
  return out.split("\n")[1:-1][0].split()[8]


def list_vm_names_and_ips():
  list_out = cloud.shell_output(cloud.Gcloud.LIST_VM)
  lines = [l.split() for l in list_out.split("\n")[1:-1]]
  names_and_ips = [(l[0].strip(), l[-2].strip()) for l in lines]
  return names_and_ips


def shell_run_with_retry(cmd, retries=1, **kwargs):
  for i in range(retries + 1):
    try:
      if i > 0:
        tf.logging.info("Retry %d for %s", i, cmd)
      cloud.shell_run(cmd, **kwargs)
      return
    except sp.CalledProcessError as e:
      if i == retries:
        raise e


def delete_instance(instance_name):
  cloud.shell_run(DELETE, name=instance_name)


def launch_instance(instance_name,
                    command,
                    existing_ip=None,
                    cpu=1,
                    mem=4,
                    code_dir=None,
                    setup_command=None):
  """Launch a GCE instance."""
  # Create instance
  ip = existing_ip or create_instance(instance_name, cpu=cpu, mem=mem)
  tf.logging.info("Waiting for SSH %s", instance_name)
  ready = wait_for_ssh(ip)
  if not ready:
    raise ValueError("Instance %s never ready for SSH" % instance_name)

  # Copy code
  if code_dir:
    shell_run_with_retry(COPY_CODE, retries=2,
                         local_dir=code_dir, instance_name=instance_name)

  # Run setup
  if setup_command:
    tf.logging.info("Running setup on %s", instance_name)
    remote_run(setup_command, instance_name)

  # Run command
  tf.logging.info("Running command on %s", instance_name)
  remote_run(command, instance_name, detach=True)


def main(_):
  assert FLAGS.num_instances
  assert FLAGS.name
  zone = default_zone()
  assert zone

  code_dir = None
  if FLAGS.code_dir:
    code_dir = os.path.abspath(os.path.expanduser(FLAGS.code_dir))

  # Suffixes per instance
  if FLAGS.per_instance_suffix_file:
    with tf.gfile.Open(FLAGS.per_instance_suffix_file) as f:
      suffixes = [l.strip() for l in f.readlines()]
  else:
    suffixes = list(range(FLAGS.num_instances))
  assert len(suffixes) == FLAGS.num_instances

  vm_info = list_vm_names_and_ips()
  vm_names = list(zip(*vm_info))[0] if vm_info else []

  pool = mp.Pool(FLAGS.num_threads)
  async_results = []

  assert FLAGS.log_dir
  log_dir = os.path.join(FLAGS.log_dir, FLAGS.name)
  tf.gfile.MakeDirs(log_dir)
  assert log_dir.startswith("gs://")
  if not log_dir.endswith("/"):
    log_dir += "/"
  # Write a test file to make sure gcloud GCS APIs are enabled
  test_filename = os.path.join(log_dir, "check_write")
  with tf.gfile.Open(test_filename, "w") as f:
    f.write("testing GCS write")
  tf.gfile.Remove(test_filename)

  instance_ids = list(range(FLAGS.num_instances))
  if FLAGS.instance_ids:
    instance_ids = [int(i) for i in FLAGS.instance_ids.split(",")]
  tf.logging.info("Launching %d instances", len(instance_ids))

  for i in instance_ids:
    instance_name = "%s-%d" % (FLAGS.name, i)
    existing_ip = (vm_info[vm_names.index(instance_name)][1]
                   if instance_name in vm_names else None)
    logging = LOGS.format(task_id=i, bucket=log_dir) if log_dir else ""
    delete = DELETE_SELF.format(zone=zone)
    if FLAGS.debug_keep_up:
      assert len(instance_ids) == 1
      delete = ""
    command = "{prefix} {suffix} {logging}; {delete}".format(
        prefix=FLAGS.command_prefix,
        suffix=suffixes[i],
        delete=delete,
        logging=logging)
    args = (instance_name, command, existing_ip,
            FLAGS.cpu, FLAGS.mem, code_dir,
            FLAGS.setup_command)
    res = pool.apply_async(launch_instance, args)
    async_results.append((res, instance_name, i))

  failed = []
  for res, instance_name, i in async_results:
    try:
      res.get()
    except Exception as e:  # pylint: disable=broad-except
      failed.append((instance_name, i))
      tf.logging.error("Failed to launch task %s due to exception %s",
                       instance_name, str(e))

  results = []
  if failed:
    ids_for_flag = ",".join([str(i) for i in list(zip(*failed))[1]])
    tf.logging.error("Failed to launch %d jobs. Tasks: %s. "
                     "Attempting delete in case they are still up. Rerun with "
                     "--instance_ids='%s' to attempt relaunch.",
                     len(failed), str(failed), ids_for_flag)
    for instance_name, _ in failed:
      res = pool.apply_async(delete_instance, (instance_name,))
      results.append(res)

  for res in results:
    try:
      res.get()
    except:  # pylint: disable=bare-except
      pass

  tf.logging.info("Launching complete.")


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
