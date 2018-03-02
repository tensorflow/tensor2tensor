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


def get_setup_file(name, packages=None):
  if not packages:
    packages = []
  return """
from setuptools import find_packages
from setuptools import setup
setup(
    name='{name}',
    version='0.1',
    packages=find_packages(),
    install_requires={pypi_packages}
)
""".format(name=name, pypi_packages=str(list(packages)))


def job_dir():
  # The flag --job-dir is parsed differently before and after switching to absl
  return getattr(FLAGS, 'job-dir', '') or getattr(FLAGS, 'job_dir', '')


def get_requirements(usr_dir):
  requirements_file = os.path.join(usr_dir, 'requirements.txt')
  if not tf.gfile.Exists(requirements_file):
    return []
  with tf.gfile.Open(requirements_file) as f:
    pkg_list = f.readlines()
    return [pkg.strip() for pkg in pkg_list if 'tensor2tensor' not in pkg]


def flags_as_args():
  """Convert FLAGS to list of args suitable for passing on cmd line."""
  if hasattr(FLAGS, 'flag_values_dict'):
    args_dict = FLAGS.flag_values_dict()
  else:
    args_dict = dict(FLAGS.__dict__['__flags'])
  del args_dict['cloud_mlengine']
  # Configured later
  del args_dict['t2t_usr_dir']
  args_dict.pop('h', None)
  args_dict.pop('helpfull', None)
  args_dict.pop('helpshort', None)
  args_dict.pop('help', None)
  args = []
  for name, val in args_dict.items():
    if val is None:
      continue
    if name.startswith('autotune'):
      continue
    args.extend(['--%s' % name, str(val)])
  return args


def get_default_master_type(num_gpus=1, use_tpu=False):
  """Returns master_type for trainingInput."""
  if use_tpu:
    return 'standard_tpu'
  elif num_gpus <= 0:
    return 'standard'
  elif num_gpus == 1:
    return 'standard_p100'
  elif num_gpus == 4:
    return 'complex_model_m_p100'
  elif num_gpus == 8:
    return 'complex_model_l_gpu'
  assert False


def configure_job():
  """Construct jobSpec for ML Engine job."""
  # See documentation:
  # https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#traininginput
  training_input = {
      'pythonModule': 'tensor2tensor.bin.t2t_trainer',
      'args': flags_as_args(),
      'region': cloud.default_region(),
      'runtimeVersion': '1.4',
      'pythonVersion': '3.5' if sys.version_info.major == 3 else '2.7',
      'jobDir': FLAGS.output_dir,
      'scaleTier': 'CUSTOM',
      'masterType': FLAGS.cloud_mlengine_master_type or get_default_master_type(
          num_gpus=FLAGS.worker_gpu,
          use_tpu=FLAGS.use_tpu)
  }
  if FLAGS.hparams_range:
    tf.logging.info('Configuring hyperparameter tuning.')
    training_input['hyperparameters'] = configure_autotune(
        FLAGS.hparams_range,
        FLAGS.autotune_objective,
        FLAGS.autotune_maximize,
        FLAGS.autotune_max_trials,
        FLAGS.autotune_parallel_trials,
    )

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

  output = cloud.shell_output('pip show tensor2tensor').split('\n')
  assert output[1].startswith('Version')
  assert output[7].startswith('Location')
  t2t_version = output[1].split(':')[1].strip()
  t2t_dir = output[7].split(':')[1].strip()

  # A local installation cloned from GitHub will have a setup.py file and a docs
  # folder
  is_local_t2t = all([
      tf.gfile.Exists(os.path.join(t2t_dir, fname))
      for fname in ['setup.py', 'docs/cloud_mlengine.md']
  ])

  if is_local_t2t:
    tf.logging.info('Found local T2T installation. Tarring directory %s',
                    t2t_dir)
  else:
    # PyPI installation
    # Create a folder with just a setup.py file pointing to the right version
    tf.logging.info('Found PyPI T2T installation. Launching tensor2tensor==%s',
                    t2t_version)
    t2t_dir = os.path.join(tempfile.gettempdir(), 'tensor2tensor_tmp')
    shutil.rmtree(t2t_dir, ignore_errors=True)
    os.mkdir(t2t_dir)
    setup_fname = os.path.join(t2t_dir, 'setup.py')
    setup_file_str = get_setup_file(
        name='DummyT2TPackage',
        packages=['tensor2tensor==%s' % t2t_version]
    )
    with tf.gfile.Open(setup_fname, 'w') as f:
      f.write(setup_file_str)
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
  setup_file_str = get_setup_file(
      name='DummyUsrDirPackage',
      packages=get_requirements(usr_dir)
  )
  with tf.gfile.Open(top_setup_fname, 'w') as f:
    f.write(setup_file_str)
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


def validate_flags():
  """Validates flags are set to acceptable values for CloudML Engine runs."""
  assert not FLAGS.cloud_tpu
  assert not job_dir()
  assert FLAGS.output_dir.startswith('gs://')
  assert FLAGS.data_dir.startswith('gs://')
  assert FLAGS.worker_replicas <= 1
  assert FLAGS.ps_replicas <= 0
  if FLAGS.hparams_range:
    assert FLAGS.autotune_objective
  if FLAGS.worker_gpu:
    assert FLAGS.worker_gpu in [1, 4, 8]
  if FLAGS.cloud_mlengine_master_type:
    if FLAGS.use_tpu:
      assert FLAGS.cloud_mlengine_master_type == 'standard_tpu'
    elif FLAGS.worker_gpu:
      if FLAGS.worker_gpu == 1:
        assert FLAGS.cloud_ml_engine_master_type in ['standard_gpu',
                                                     'standard_p100']
      elif FLAGS.worker_gpu == 4:
        assert FLAGS.cloud_ml_engine_master_type in ['complex_model_m_gpu',
                                                     'complex_model_m_p100']
      else:
        assert FLAGS.cloud_ml_engine_master_type == 'complex_model_l_gpu'
    else:
      assert FLAGS.cloud_mlengine_master_type in ['standard', 'large_model',
                                                  'complex_model_s',
                                                  'complex_model_m',
                                                  'complex_model_l']


def launch():
  """Launch t2t_trainer on Cloud ML Engine."""
  validate_flags()
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
