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

"""Launch on GCP's ML Engine."""

import datetime
import os
import shutil
import sys
import tempfile

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
from tensor2tensor.layers import common_hparams
from tensor2tensor.utils import cloud_tpu as cloud
from tensor2tensor.utils import registry
from tensor2tensor.utils import usr_dir as usr_dir_lib
import tensorflow as tf

FLAGS = tf.flags.FLAGS

CONSOLE_URL = 'https://console.cloud.google.com/mlengine/jobs/'

# TODO(rsepassi):
# * Enable multi-machine sync/async training

SETUP_PY = """
from setuptools import find_packages
from setuptools import setup
setup(
    name='DummyUsrDirPackage',
    version='0.1',
    packages=find_packages(),
)
"""


def flags_as_args():
  """Convert FLAGS to list of args suitable for passing on cmd line."""
  args_dict = dict(FLAGS.__dict__['__flags'])
  del args_dict['cloud_mlengine']
  # Configured later
  del args_dict['t2t_usr_dir']
  args = []
  for name, val in args_dict.items():
    if val is None:
      continue
    if name.startswith('autotune'):
      continue
    args.extend(['--%s' % name, str(val)])
  return args


def machine_config(num_gpus=1, use_tpu=False, master_type=None):
  """Return dict specifying machine config for trainingInput."""
  scale_tier = 'BASIC_GPU'
  if use_tpu:
    scale_tier = 'BASIC_TPU'
  elif num_gpus <= 0:
    scale_tier = 'BASIC'
  elif num_gpus > 1:
    scale_tier = 'CUSTOM'

  config = {'scaleTier': scale_tier}

  if scale_tier == 'CUSTOM':
    assert num_gpus > 1
    if num_gpus not in [4, 8]:
      raise ValueError('Must use exactly 1, 4, or 8 GPUs.')
    config['masterType'] = ('complex_model_m_gpu'
                            if num_gpus == 4 else 'complex_model_l_gpu')

  if master_type:
    config['masterType'] = master_type

  return config


def configure_job():
  """Construct jobSpec for ML Engine job."""
  train_dir = FLAGS.output_dir
  assert train_dir.startswith('gs://')

  # See documentation:
  # https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#traininginput
  training_input = {
      'pythonModule': 'tensor2tensor.bin.t2t_trainer',
      'args': flags_as_args(),
      'region': cloud.default_region(),
      'runtimeVersion': '1.4',
      'pythonVersion': '3.5' if sys.version_info.major == 3 else '2.7',
      'jobDir': train_dir,
  }
  training_input.update(
      machine_config(
          num_gpus=FLAGS.worker_gpu,
          use_tpu=FLAGS.use_tpu,
          master_type=FLAGS.cloud_mlengine_master_type))
  if FLAGS.hparams_range:
    assert FLAGS.autotune_objective
    tf.logging.info('Configuring hyperparameter tuning.')
    training_input['hyperparameters'] = configure_autotune(
        FLAGS.hparams_range,
        FLAGS.autotune_objective,
        FLAGS.autotune_maximize,
        FLAGS.autotune_max_trials,
        FLAGS.autotune_parallel_trials,
    )

  if training_input['scaleTier'] == 'CUSTOM':
    assert 'masterType' in training_input

  timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  job_name = '%s_%s_t2t_%s' % (FLAGS.model, FLAGS.problems, timestamp)
  job_spec = {'jobId': job_name, 'trainingInput': training_input}
  return job_spec


def launch_job(job_spec):
  """Launch job on ML Engine."""
  project_id = 'projects/{}'.format(cloud.default_project())
  credentials = GoogleCredentials.get_application_default()
  cloudml = discovery.build('ml', 'v1', credentials=credentials)
  request = cloudml.projects().jobs().create(body=job_spec, parent=project_id)
  request.execute()


def _tar_and_copy(src_dir, target_dir):
  """Tar and gzip src_dir and copy to GCS target_dir."""
  src_dir = src_dir.rstrip('/')
  target_dir = target_dir.rstrip('/')
  tmp_dir = tempfile.gettempdir().rstrip('/')
  src_base = os.path.basename(src_dir)
  cloud.shell_run(
      'tar -zcf {tmp_dir}/{src_base}.tar.gz -C {src_dir} .',
      src_dir=src_dir,
      src_base=src_base,
      tmp_dir=tmp_dir)
  final_destination = '%s/%s.tar.gz' % (target_dir, src_base)
  cloud.shell_run(
      ('gsutil cp {tmp_dir}/{src_base}.tar.gz '
       '{final_destination}'),
      tmp_dir=tmp_dir,
      src_base=src_base,
      final_destination=final_destination)
  return final_destination


def tar_and_copy_t2t(train_dir):
  """Tar Tensor2Tensor and cp to train_dir."""
  tf.logging.info('Tarring and pushing local Tensor2Tensor package.')
  t2t_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
  t2t_tar = _tar_and_copy(t2t_dir, train_dir)
  return t2t_tar


def tar_and_copy_usr_dir(usr_dir, train_dir):
  """Package, tar, and copy usr_dir to GCS train_dir."""
  tf.logging.info('Tarring and pushing t2t_usr_dir.')
  usr_dir = os.path.abspath(os.path.expanduser(usr_dir))
  # Copy usr dir to a temp location
  top_dir = os.path.join(tempfile.gettempdir(), 't2t_usr_container')
  tmp_usr_dir = os.path.join(top_dir, usr_dir_lib.INTERNAL_USR_DIR_PACKAGE)
  shutil.rmtree(top_dir, ignore_errors=True)
  shutil.copytree(usr_dir, tmp_usr_dir)
  # Insert setup.py if one does not exist
  top_setup_fname = os.path.join(top_dir, 'setup.py')
  usr_setup_fname = os.path.join(tmp_usr_dir, 'setup.py')
  if tf.gfile.Exists(usr_setup_fname):
    tf.gfile.Copy(usr_setup_fname, top_setup_fname)
    tf.gfile.Remove(usr_setup_fname)
  else:
    with tf.gfile.Open(top_setup_fname, 'w') as f:
      f.write(SETUP_PY)
  usr_tar = _tar_and_copy(top_dir, train_dir)
  return usr_tar


def autotune_paramspecs(hparams_range):
  rhp = common_hparams.RangedHParams()
  registry.ranged_hparams(hparams_range)(rhp)
  return rhp.to_parameter_specs(name_prefix='hp_')


def configure_autotune(hparams_range,
                       objective,
                       maximize=True,
                       max_trials=10,
                       parallel_trials=1):
  return {
      'goal': 'MAXIMIZE' if maximize else 'MINIMIZE',
      'params': autotune_paramspecs(hparams_range),
      'maxTrials': max_trials,
      'maxParallelTrials': parallel_trials,
      'hyperparameterMetricTag': objective,
  }


def configure_trainer_package(job_spec, t2t_tar):
  assert t2t_tar.startswith('gs://')
  job_spec['trainingInput']['packageUris'] = [t2t_tar]


def configure_usr_dir(job_spec, usr_tar):
  assert usr_tar.startswith('gs://')
  job_spec['trainingInput']['packageUris'].append(usr_tar)
  usr_args = ['--t2t_usr_dir', usr_dir_lib.INTERNAL_USR_DIR_PACKAGE]
  job_spec['trainingInput']['args'].extend(usr_args)


def launch():
  """Launch t2t_trainer on Cloud ML Engine."""
  assert not FLAGS.cloud_tpu
  assert not FLAGS.job_dir
  assert FLAGS.output_dir.startswith('gs://')
  assert FLAGS.data_dir.startswith('gs://')
  assert FLAGS.worker_replicas <= 1
  assert FLAGS.ps_replicas <= 0

  job_spec = configure_job()
  job_name = job_spec['jobId']
  tf.logging.info('Launching job %s with ML Engine spec:\n%s', job_name,
                  job_spec)
  assert cloud.confirm()
  train_dir = FLAGS.output_dir
  t2t_tar = tar_and_copy_t2t(train_dir)
  configure_trainer_package(job_spec, t2t_tar)
  if FLAGS.t2t_usr_dir:
    usr_tar = tar_and_copy_usr_dir(FLAGS.t2t_usr_dir, train_dir)
    configure_usr_dir(job_spec, usr_tar)
  launch_job(job_spec)
  tf.logging.info('Launched %s. See console to track: %s.', job_name,
                  CONSOLE_URL)
