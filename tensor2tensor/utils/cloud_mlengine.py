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

"""Launch on GCP's ML Engine."""

import os
import sys
import tempfile

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
from tensor2tensor.utils import cloud_tpu as cloud
import tensorflow as tf

CONSOLE_URL = 'https://console.cloud.google.com/mlengine/jobs/'

# TODO(rsepassi):
# * Support t2t_usr_dir
# * Support --autotune
# * Add documentation clould_mlengine.md
# * Enable multi-machine sync/async training


def args_dict_as_args(args_dict):
  del args_dict['cloud_mlengine']
  args = []
  for name, val in args_dict.items():
    if val is None:
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


def configure_job(flags_dict):
  """Construct jobSpec for ML Engine job."""
  train_dir = flags_dict['output_dir']
  assert train_dir.startswith('gs://')
  job_name = os.path.basename(train_dir)

  # See documentation:
  # https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#traininginput
  training_input = {
      'packageUris': [os.path.join(train_dir, 'tensor2tensor.tar.gz')],
      'pythonModule': 'tensor2tensor.bin.t2t_trainer',
      'args': args_dict_as_args(flags_dict),
      'region': cloud.default_region(),
      'runtimeVersion': '1.4',
      'pythonVersion': '3.5' if sys.version_info.major == 3 else '2.7',
  }
  training_input.update(
      machine_config(
          num_gpus=flags_dict['worker_gpu'],
          use_tpu=flags_dict['use_tpu'],
          master_type=flags_dict['cloud_mlengine_master_type']))

  if training_input['scaleTier'] == 'CUSTOM':
    assert 'masterType' in training_input

  job_spec = {'jobId': job_name, 'trainingInput': training_input}
  return job_spec


def launch_job(job_spec):
  """Launch job on ML Engine."""
  project_id = 'projects/{}'.format(cloud.default_project())
  credentials = GoogleCredentials.get_application_default()
  cloudml = discovery.build('ml', 'v1', credentials=credentials)
  request = cloudml.projects().jobs().create(body=job_spec, parent=project_id)
  request.execute()


def tar_and_copy_t2t(train_dir, usr_dir):
  """Tar Tensor2Tensor and cp to train_dir."""
  tf.logging.info('Tarring and pushing local Tensor2Tensor package.')
  location = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
  tmp_dir = tempfile.gettempdir()
  cloud.shell_run(
      'tar -zcf {tmp_dir}/tensor2tensor.tar.gz -C {location} .',
      location=location,
      tmp_dir=tmp_dir)
  cloud.shell_run(
      ('gsutil cp {tmp_dir}/tensor2tensor.tar.gz '
       '{train_dir}/tensor2tensor.tar.gz'),
      tmp_dir=tmp_dir,
      train_dir=train_dir.strip('/'))
  if usr_dir:
    raise ValueError('--t2t_usr_dir is not currently supported in conjunction '
                     'with auto-launching on Cloud ML Engine.')


def launch(flags_dict):
  job_spec = configure_job(flags_dict)
  job_name = job_spec['jobId']
  tf.logging.info('Launching job %s with ML Engine spec:\n%s', job_name,
                  job_spec)
  assert cloud.confirm()
  tar_and_copy_t2t(flags_dict['output_dir'], flags_dict['t2t_usr_dir'])
  launch_job(job_spec)
  tf.logging.info('Launched %s. See console to track: %s.', job_name,
                  CONSOLE_URL)
