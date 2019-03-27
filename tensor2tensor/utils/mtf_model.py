# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

"""Mesh-Tensorflow Model in tensor2tensor."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf

import six
from tensor2tensor.utils import hparams_lib
from tensor2tensor.utils import learning_rate
from tensor2tensor.utils import metrics
from tensor2tensor.utils import t2t_model

import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_estimator


class MtfModel(t2t_model.T2TModel):
  """Toy model to test mesh_tensorflow."""

  @classmethod
  def estimator_model_fn(cls,
                         hparams,
                         features,
                         labels,
                         mode,
                         config=None,
                         params=None,
                         decode_hparams=None,
                         use_tpu=False):
    hparams = hparams_lib.copy_hparams(hparams)
    hparams.use_tpu = use_tpu
    # merge decode_hparams into hparams if present
    if mode == tf.estimator.ModeKeys.PREDICT and decode_hparams is not None:
      for k, v in six.iteritems(decode_hparams.values()):
        if hasattr(hparams, k) and getattr(hparams, k) != v:
          tf.logging.warning("Overriding hparams.%s with %s from decode_hparams"
                             % (k, v))
        setattr(hparams, k, v)

    # Instantiate model
    data_parallelism = None
    if not use_tpu and config:
      data_parallelism = config.data_parallelism
    model = cls(
        hparams,
        mode,
        data_parallelism=data_parallelism,
        decode_hparams=decode_hparams)

    global_step = tf.train.get_global_step()

    mesh_shape = mtf.convert_to_shape(hparams.mesh_shape)
    layout_rules = mtf.convert_to_layout_rules(hparams.layout)
    if use_tpu:
      ctx = params["context"]
      num_hosts = ctx.num_hosts
      host_placement_fn = ctx.tpu_host_placement_function
      device_list = [host_placement_fn(host_id=t) for t in range(num_hosts)]
      # TODO(ylc): Better estimation of replica cache size?
      replica_cache_size = 300 * 1000000  # 300M per replica
      # Worker 0 caches all the TPU binaries.
      worker0_mem = replica_cache_size * ctx.num_replicas
      devices_memeory_usage = [worker0_mem] + [0] * (num_hosts - 1)
      var_placer = mtf.utils.BalancedVariablePlacer(device_list,
                                                    devices_memeory_usage)
      mesh_devices = [""] * mesh_shape.size
      mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
          mesh_shape, layout_rules, mesh_devices, ctx.device_assignment)
    else:
      var_placer = None
      if data_parallelism is None or len(data_parallelism.ps_devices) == 1:
        mesh_devices = [""] * mesh_shape.size
      else:
        assert len(data_parallelism.ps_devices) == mesh_shape.size
        mesh_devices = data_parallelism.ps_devices
      mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
          mesh_shape, layout_rules, mesh_devices)

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh", var_placer)
    # PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
      return model.estimator_spec_predict(features, mesh, mesh_impl, use_tpu)

    logits, loss = model.mtf_model_fn(features, mesh)
    if use_tpu and logits is not None:
      logits = mtf.anonymize(logits)

    # TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
      var_grads = mtf.gradients(
          [loss], [v.outputs[0] for v in graph.trainable_variables])
      lr = learning_rate.learning_rate_schedule(hparams)
      tf.summary.scalar("learning_rate", lr)
      mtf_lr = mtf.import_tf_tensor(
          mesh, tf.convert_to_tensor(lr, dtype=tf.float32), mtf.Shape([]))
      optimizer = mtf.optimize.make_optimizer(hparams, mtf_lr)
      update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)

    lowering = mtf.Lowering(graph, {mesh: mesh_impl})

    tf_loss = lowering.export_to_tf_tensor(loss)
    tf_loss = tf.to_float(tf_loss)
    if logits and mode != tf.estimator.ModeKeys.TRAIN:
      tf_logits = lowering.export_to_tf_tensor(logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
      tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
      tf_update_ops.append(tf.assign_add(global_step, 1))
      # tf.logging.info("tf_update_ops: {}".format(tf_update_ops))
      train_op = tf.group(tf_update_ops)

    with mtf.utils.outside_all_rewrites():
      # Copy master variables to slices. Must be called first.
      restore_hook = mtf.MtfRestoreHook(lowering)
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
          hparams.model_dir,
          save_steps=1000,
          saver=saver,
          listeners=[saver_listener])

    # EVAL mode
    if mode == tf.estimator.ModeKeys.EVAL:
      tf_logits = lowering.export_to_tf_tensor(logits)
      return model.estimator_spec_eval(features, tf_logits, labels, tf_loss,
                                       restore_hook, use_tpu)

    if use_tpu:
      # TPU host call. Important: need to be called before remove_summaries()
      if hparams.tpu_enable_host_call:
        host_call = t2t_model.create_host_call(hparams.model_dir)
      else:
        host_call = None

      if hparams.warm_start_from:

        def scaffold_fn():
          t2t_model.initialize_from_ckpt(
              ckpt_dir=hparams.warm_start_from, hparams=hparams)
          return tf.train.Scaffold()
      else:
        scaffold_fn = None

      t2t_model.remove_summaries()
      return tpu_estimator.TPUEstimatorSpec(
          mode=tf.estimator.ModeKeys.TRAIN,
          loss=tf_loss,
          train_op=train_op,
          host_call=host_call,
          training_hooks=[restore_hook, saver_hook],
          scaffold_fn=scaffold_fn)
    else:
      if hparams.warm_start_from:
        t2t_model.initialize_from_ckpt(
            ckpt_dir=hparams.warm_start_from, hparams=hparams)
      return tf.estimator.EstimatorSpec(
          tf.estimator.ModeKeys.TRAIN, loss=tf_loss, train_op=train_op,
          training_chief_hooks=[restore_hook, saver_hook])

  def estimator_spec_eval(
      self, features, logits, labels, loss, restore_hook, use_tpu):
    """Construct EstimatorSpec for EVAL mode."""
    hparams = self.hparams
    problem = hparams.problem
    if logits.get_shape().ndims == 3:
      logits = tf.expand_dims(tf.expand_dims(logits, 2), 3)

    # Support for multiproblem
    task_list = [problem]
    if hasattr(problem, "task_list"):
      task_list = problem.task_list

    eval_metrics_fns = metrics.create_evaluation_metrics(task_list, hparams)

    if use_tpu:
      def metric_fn(tf_logits, labels):
        with tf.device("cpu:0"), mtf.utils.outside_all_rewrites():
          eval_metrics = {}
          for metric_name, metric_fn in six.iteritems(eval_metrics_fns):
            if metric_name.split("/")[-1] not in t2t_model.TPU_METRIC_BLACKLIST:
              eval_metrics[metric_name] = metric_fn(
                  tf_logits, None, tf.identity(labels))
          return eval_metrics
      return tpu_estimator.TPUEstimatorSpec(
          tf.estimator.ModeKeys.EVAL,
          evaluation_hooks=[restore_hook],
          loss=loss,
          eval_metrics=(metric_fn, [logits, labels]))
    else:
      eval_metrics = {}
      predictions = {"predictions": logits}
      for metric_name, metric_fn in six.iteritems(eval_metrics_fns):
        eval_metrics[metric_name] = metric_fn(logits, features,
                                              features["targets"])

      return tf.estimator.EstimatorSpec(
          tf.estimator.ModeKeys.EVAL,
          predictions=predictions,
          eval_metric_ops=eval_metrics,
          evaluation_hooks=[restore_hook],
          loss=loss)

  def estimator_spec_predict(self, features, mesh, mesh_impl, use_tpu):
    mtf_samples = mtf.anonymize(self.sample(features, mesh))
    lowering = mtf.Lowering(mesh.graph, {mesh: mesh_impl})
    outputs = lowering.export_to_tf_tensor(mtf_samples)
    if self.has_input:
      ndims = len(outputs.shape.as_list())
      actual_batch_size = tf.shape(features["inputs"])[0]
      outputs = tf.slice(
          outputs, [0] * ndims, [actual_batch_size] + [-1] * (ndims - 1))
    predictions = {
        "outputs": outputs
    }
    if features.get("infer_targets") is not None:
      predictions["infer_targets"] = features["infer_targets"]

    if features.get("inputs") is not None:
      predictions["inputs"] = features["inputs"]

    if use_tpu:
      t2t_model.remove_summaries()
      return tpu_estimator.TPUEstimatorSpec(
          mode=tf.estimator.ModeKeys.PREDICT,
          predictions=predictions,
          prediction_hooks=[mtf.MtfRestoreHook(lowering)])
    else:
      return tf.estimator.EstimatorSpec(
          tf.estimator.ModeKeys.PREDICT,
          predictions=predictions,
          prediction_hooks=[mtf.MtfRestoreHook(lowering)])

  def sample(self, features, mesh):
    """Sample from the model."""
    raise NotImplementedError("TODO(noam): write generic slow mtf sample.")

  def mtf_model_fn(self, features, mesh):
    raise NotImplementedError("Not implemented")
