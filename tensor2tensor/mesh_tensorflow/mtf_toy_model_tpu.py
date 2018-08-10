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
"""A toy model using mesh-tensrflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from tensor2tensor.mesh_tensorflow import mesh_tensorflow as mtf
from tensor2tensor.mesh_tensorflow import mtf_layers
from tensor2tensor.mesh_tensorflow import mtf_optimize
from tensor2tensor.mesh_tensorflow import mtf_utils
from tensor2tensor.mesh_tensorflow.simd_mesh_impl import SimdMeshImpl
import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging


FLAGS = flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 64, 'Training batch size.')
tf.flags.DEFINE_integer('io_size', 2, 'Number of channels per feature.')
tf.flags.DEFINE_integer('hidden_size', 2, 'Size of each hidden layer.')
tf.flags.DEFINE_string('mesh_shape', 'all:8', 'mesh shape')
tf.flags.DEFINE_string('layout', 'hidden:all', 'layout rules')
tf.flags.DEFINE_integer('iterations', 100,
                        'Number of iterations per training loop.')
tf.flags.DEFINE_integer('train_steps', 10000, 'max steps')
tf.flags.DEFINE_integer('steps_per_checkpoint', 200, 'steps_per_checkpoint')
tf.flags.DEFINE_string('master', 'local',
                       'BNS name of the TensorFlow master to use.')
tf.flags.DEFINE_string(
    'model_dir',
    default='',
    help='The directory where the model will be stored.')

# Cloud TPU Cluster Resolvers
tf.flags.DEFINE_string(
    'tpu',
    default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

tf.flags.DEFINE_string(
    'gcp_project',
    default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

tf.flags.DEFINE_string(
    'tpu_zone',
    default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')


class ToyModelInput(object):
  """Wrapper class that acts as the input_fn to TPUEstimator."""

  def __init__(self):
    self._num_examples = 10000  # 10k
    self._images = numpy.random.uniform(
        0, 1.0, [self._num_examples, FLAGS.io_size]).astype(numpy.float32)
    self._labels = self._images
    logging.info('init ToyModelInput()')

  def __call__(self, params):
    """Input function which provides a single batch for train or eval."""
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # `tf.contrib.tpu.RunConfig` for details.
    batch_size = params['batch_size']
    logging.info('call ToyModelInput() with batch size {}'.format(batch_size))

    ds = Dataset.from_tensor_slices((self._images, self._labels)).repeat()

    dataset = ds.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size)).prefetch(2)

    return dataset


def toy_model(features, mesh):
  """A toy model implemented by mesh tensorlfow."""
  batch_dim = mtf.Dimension('batch', FLAGS.batch_size)
  hidden_dim = mtf.Dimension('hidden', FLAGS.hidden_size)
  io_dim = mtf.Dimension('io', FLAGS.io_size)

  x = mtf.import_tf_tensor(mesh, features, mtf.Shape([batch_dim, io_dim]))
  h = mtf_layers.dense(x, hidden_dim, name='layer1', use_bias=False)
  y = mtf_layers.dense(h, io_dim, name='layer2', use_bias=False)

  loss = mtf.reduce_sum(mtf.square(y - x))
  return y, loss


def model_fn(features, labels, mode, params):
  """A model is called by TpuEstimator."""
  del labels
  global_step = tf.train.get_global_step()
  graph = mtf.Graph()
  mesh = mtf.Mesh(graph, 'my_mesh')
  mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
  mesh_devices = [''] * mesh_shape.size
  mesh_impl = SimdMeshImpl(
      mesh_shape, mtf.convert_to_layout_rules(FLAGS.layout),
      mesh_devices, params['context'].device_assignment)
  with mtf_utils.outside_all_rewrites():
    logits, loss = toy_model(features, mesh)

  # TRAIN mode
  if mode == tf.estimator.ModeKeys.TRAIN:
    var_grads = mtf.gradients([loss],
                              [v.outputs[0] for v in graph.trainable_variables])
    optimizer = mtf_optimize.AdafactorOptimizer()
    update_ops = []
    for grad, var in zip(var_grads, graph.trainable_variables):
      update_ops.extend(optimizer.apply_grad(grad, var))
  else:
    # for now, we can only export fully-replicated tensors.
    fully_replicated_logits = mtf.anonymize(logits)

  lowering = mtf.Lowering(graph, {mesh: mesh_impl})

  tf_loss = lowering.export_to_tf_tensor(loss)

  if mode == tf.estimator.ModeKeys.TRAIN:
    tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
    tf_update_ops.append(tf.assign_add(global_step, 1))
    tf.logging.info('tf_update_ops: {}'.format(tf_update_ops))
    train_op = tf.group(tf_update_ops)
  else:
    tf_logits = lowering.export_to_tf_tensor(fully_replicated_logits)

  with mtf_utils.outside_all_rewrites():
    # Copy master variables to slices. Must be called first.
    restore_hook = mtf.MtfRestoreHook(lowering)
    if mode == tf.estimator.ModeKeys.TRAIN:
      saver = tf.train.Saver(
          tf.global_variables(),
          sharded=True,
          max_to_keep=10,
          keep_checkpoint_every_n_hours=2,
          defer_build=False,
          save_relative_paths=True)
      tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
      saver_listener = mtf.MtfCheckpointSaverListener(lowering)
      saver_hook = tf.train.CheckpointSaverHook(
          FLAGS.model_dir,
          save_steps=1000,
          saver=saver,
          listeners=[saver_listener])

      return tpu_estimator.TPUEstimatorSpec(
          tf.estimator.ModeKeys.TRAIN,
          loss=tf_loss,
          train_op=train_op,
          training_hooks=[restore_hook, saver_hook])
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(tf_logits):
        mean_logitss = tf.metrics.mean(tf_logits)
        return {'mean_logitss': mean_logitss}

      eval_metrics = (metric_fn, [tf_logits])

      return tpu_estimator.TPUEstimatorSpec(
          tf.estimator.ModeKeys.EVAL,
          evaluation_hooks=[restore_hook],
          loss=tf_loss,
          eval_metrics=eval_metrics)


def run_toy_model_tpu():
  """Run a toy model on TPU."""
  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  iterations_per_loop = FLAGS.iterations
  mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
  config = tpu_config.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=None,  # Disable the default saver
      save_checkpoints_secs=None,  # Disable the default saver
      log_step_count_steps=iterations_per_loop,
      tpu_config=tpu_config.TPUConfig(
          num_shards=mesh_shape.size,
          iterations_per_loop=iterations_per_loop,
          num_cores_per_replica=1,
          per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST))
  classifier = tpu_estimator.TPUEstimator(
      use_tpu=True,
      model_fn=model_fn,
      config=config,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size)
  current_step = estimator_lib._load_global_step_from_checkpoint_dir(FLAGS.model_dir)  # pylint: disable=protected-access,line-too-long
  logging.info('Current step %d', current_step)
  while current_step < FLAGS.train_steps:
    next_checkpoint = min(current_step + FLAGS.steps_per_checkpoint,
                          FLAGS.train_steps)
    classifier.train(input_fn=ToyModelInput(), max_steps=next_checkpoint)
    current_step = next_checkpoint

    logging.info('Starting to evaluate.')
    eval_results = classifier.evaluate(
        input_fn=ToyModelInput(),
        steps=156)  # since we have 10000 examples and batch_size = 64 per host
    logging.info('Eval results: %s', eval_results)
  # classifier.train(input_fn=ToyModelInput(), max_steps=FLAGS.train_steps)


def main(_):
  run_toy_model_tpu()


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
