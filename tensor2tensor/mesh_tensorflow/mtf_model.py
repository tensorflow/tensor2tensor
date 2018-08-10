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
"""Mesh-Tensorflow Model in tensor2tensor."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import six


from tensor2tensor.mesh_tensorflow import mesh_tensorflow as mtf
from tensor2tensor.mesh_tensorflow import mtf_optimize
from tensor2tensor.mesh_tensorflow import mtf_utils
from tensor2tensor.mesh_tensorflow import placement_mesh_impl
from tensor2tensor.mesh_tensorflow import simd_mesh_impl
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
                         decode_hparams=None):
    hparams = copy.deepcopy(hparams)
    use_tpu = params and params.get("use_tpu", False)
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
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")

    mesh_shape = mtf.convert_to_shape(hparams.mesh_shape)
    layout_rules = mtf.convert_to_layout_rules(hparams.layout)
    if use_tpu:
      mesh_devices = [""] * mesh_shape.size
      mesh_impl = simd_mesh_impl.SimdMeshImpl(
          mesh_shape, layout_rules, mesh_devices,
          params["context"].device_assignment)
    else:
      if len(data_parallelism.ps_devices) == 1:
        mesh_devices = [""] * mesh_shape.size
      else:
        assert len(data_parallelism.ps_devices) == mesh_shape.size
        mesh_devices = data_parallelism.ps_devices
      mesh_impl = placement_mesh_impl.PlacementMeshImpl(
          mesh_shape, layout_rules, mesh_devices)

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
      mtf_lr = mtf.import_tf_tensor(
          mesh, tf.convert_to_tensor(lr, dtype=tf.float32), mtf.Shape([]))
      optimizer = mtf_optimize.make_optimizer(hparams, mtf_lr)
      update_ops = []
      for grad, var in zip(var_grads, graph.trainable_variables):
        update_ops.extend(optimizer.apply_grad(grad, var))

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

    with mtf_utils.outside_all_rewrites():
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
      _remove_summaries()
      return tpu_estimator.TPUEstimatorSpec(
          mode=tf.estimator.ModeKeys.TRAIN,
          loss=tf_loss,
          train_op=train_op,
          training_hooks=[restore_hook, saver_hook])
    else:
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
    eval_metrics_fns = metrics.create_evaluation_metrics([problem], hparams)

    if use_tpu:
      def metric_fn(tf_logits, labels):
        with tf.device("cpu:0"), mtf_utils.outside_all_rewrites():
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
    mtf_samples = self.sample(features, mesh)
    lowering = mtf.Lowering(mesh.graph, {mesh: mesh_impl})
    outputs = lowering.export_to_tf_tensor(mtf_samples)
    if self.has_input:
      ndims = len(outputs.shape.as_list())
      actual_batch_size = tf.shape(features["inputs"])[0]
      outputs = tf.slice(
          outputs, [0] * ndims, [actual_batch_size] + [-1] * (ndims - 1))
    predictions = {
        "outputs": outputs,
        "targets": features.get("infer_targets", features.get("inputs")),
        "inputs": features.get("inputs"),
    }
    if use_tpu:
      _remove_summaries()
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


def _remove_summaries():
  g = tf.get_default_graph()
  key = tf.GraphKeys.SUMMARIES
  del g.get_collection_ref(key)[:]
  assert not g.get_collection(key)


def _create_host_call(model_dir):
  """Construct a host_call writing scalar summaries.

  Args:
    model_dir: String containing path to train

  Returns:
    (fn, args) Pair to be called by TPUEstimator as the host_call.
  """
  graph = tf.get_default_graph()
  summaries = graph.get_collection(tf.GraphKeys.SUMMARIES)

  gs_t = tf.reshape(tf.to_int32(tf.train.get_global_step()), [1])
  summary_kwargs = collections.OrderedDict()
  for t in summaries:
    if t.op.type != "ScalarSummary":
      continue

    name = t.op.name
    tensor = t.op.inputs[1]
    assert tensor.shape.is_compatible_with([])
    if tensor.dtype == tf.int64:
      tensor = tf.to_int32(tensor)
    summary_kwargs[name] = tf.reshape(tensor, [1])
  summary_kwargs["global_step"] = gs_t

  def host_call_fn(**kwargs):
    """Training host call. Creates scalar summaries for training metrics.

    Args:
      **kwargs: Dict of {str: Tensor} , with `Tensor` of shape `[batch]`. Must
        contain key "global_step" with value of current global_step Tensor.

    Returns:
      List of summary ops to run on the CPU host.
    """
    gs = tf.to_int64(kwargs.pop("global_step")[0])
    with tf.contrib.summary.create_file_writer(model_dir).as_default():
      with tf.contrib.summary.always_record_summaries():
        for name, value in sorted(six.iteritems(kwargs)):
          tf.contrib.summary.scalar(
              name, tf.reduce_mean(tf.to_float(value)), step=gs)

        return tf.contrib.summary.all_summary_ops()

  return (host_call_fn, summary_kwargs)
