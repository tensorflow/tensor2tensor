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

"""Model building."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

# Dependency imports

import numpy as np
import six
# pylint: disable=redefined-builtin
from six.moves import xrange
# pylint: enable=redefined-builtin

from tensor2tensor.models import models  # pylint: disable=unused-import
from tensor2tensor.utils import devices
from tensor2tensor.utils import input_fn_builder
from tensor2tensor.utils import registry
from tensor2tensor.utils import yellowfin

import tensorflow as tf
from tensorflow.python.ops import init_ops

# TODO(rsepassi): Rm dep on FLAGS here
FLAGS = tf.flags.FLAGS

# Number of samples to draw for an image input (in such cases as captioning)
IMAGE_DECODE_LENGTH = 100


def log_variable_sizes(var_list, tag):
  """Log the sizes and shapes of variables, and the total size.

  Args:
    var_list: a list of varaibles
    tag: a string
  """
  name_to_var = {v.name: v for v in var_list}
  total_size = 0
  for v_name in sorted(list(name_to_var)):
    v = name_to_var[v_name]
    v_size = int(np.prod(np.array(v.shape.as_list())))
    tf.logging.info("Weight    %s\tshape    %s\tsize    %d",
                    v.name[:-2].ljust(80), str(v.shape).ljust(20), v_size)
    total_size += v_size
  tf.logging.info("%s Total size: %d", tag, total_size)


def build_model_fn(model, hparams):
  """Returns a function to build the model.

  Args:
    model: The name of the model to use.
    hparams: The hyperparameters.

  Returns:
    A function to build the model's graph. This function is called by
    the Estimator object to construct the graph.
  """

  def initializer():
    if hparams.initializer == "orthogonal":
      return tf.orthogonal_initializer(gain=hparams.initializer_gain)
    elif hparams.initializer == "uniform":
      max_val = 0.1 * hparams.initializer_gain
      return tf.random_uniform_initializer(-max_val, max_val)
    elif hparams.initializer == "normal_unit_scaling":
      return init_ops.variance_scaling_initializer(
          hparams.initializer_gain, mode="fan_avg", distribution="normal")
    elif hparams.initializer == "uniform_unit_scaling":
      return init_ops.variance_scaling_initializer(
          hparams.initializer_gain, mode="fan_avg", distribution="uniform")
    else:
      raise ValueError("Unrecognized initializer: %s" % hparams.initializer)

  def learning_rate_decay():
    """Inverse-decay learning rate until warmup_steps, then decay."""
    warmup_steps = tf.to_float(
        hparams.learning_rate_warmup_steps * FLAGS.worker_replicas)
    step = tf.to_float(tf.contrib.framework.get_global_step())
    if hparams.learning_rate_decay_scheme == "noam":
      return 5000.0 * hparams.hidden_size**-0.5 * tf.minimum(
          (step + 1) * warmup_steps**-1.5, (step + 1)**-0.5)
    elif hparams.learning_rate_decay_scheme == "exp100k":
      return 0.94**(step // 100000)
    elif hparams.learning_rate_decay_scheme == "cosine":
      cycle_steps = hparams.learning_rate_cosine_cycle_steps
      return 0.5 * (1 + tf.cos(np.pi * (step % cycle_steps) / cycle_steps))
    elif hparams.learning_rate_decay_scheme == "cyclelinear10x":
      # Cycle the rate linearly by 10x every warmup_steps, up and down.
      cycle_steps = hparams.learning_rate_warmup_steps
      cycle_position = step % (2 * cycle_steps)
      cycle_position = tf.to_float(  # Normalize to the interval [-1, 1].
          cycle_position - cycle_steps) / float(cycle_steps)
      cycle_position = 1.0 - tf.abs(cycle_position)  # 0 to 1 and back to 0.
      return (cycle_position + 0.1) * 3.0  # 10x difference each cycle (0.3-3).

    inv_base = tf.exp(tf.log(0.01) / warmup_steps)
    inv_decay = inv_base**(warmup_steps - step)
    if hparams.learning_rate_decay_scheme == "sqrt":
      decay = _sqrt_decay(step - warmup_steps)
    elif hparams.learning_rate_decay_scheme == "exp10k":
      decay = _exp_decay_after(step - warmup_steps, 0.9995,
                               FLAGS.train_steps - warmup_steps - 10000)
    elif hparams.learning_rate_decay_scheme == "exp50k":
      decay = _exp_decay_after(step - warmup_steps, 0.99995,
                               FLAGS.train_steps - warmup_steps - 50000)
    elif hparams.learning_rate_decay_scheme == "exp500k":
      decay = _exp_decay_after(step - warmup_steps, 0.9999955,
                               FLAGS.train_steps - warmup_steps - 500000)
    elif hparams.learning_rate_decay_scheme == "none":
      decay = tf.constant(1.0)
    else:
      raise ValueError("Unrecognized learning rate decay scheme: %s" %
                       hparams.learning_rate_decay_scheme)
    return tf.cond(
        step < warmup_steps,
        lambda: inv_decay,
        lambda: decay,
        name="learning_rate_decay_warump_cond")

  def model_fn(features, targets, mode):
    """Creates the prediction, loss, and train ops.

    Args:
      features: A dictionary of tensors keyed by the feature name.
      targets: A tensor representing the labels (targets).
      mode: The execution mode, as defined in tf.contrib.learn.ModeKeys.

    Returns:
      A tuple consisting of the prediction, loss, and train_op.
    """
    # Deep-copy the model hparams between modes to eliminate
    # side-effects caused by abuse of the linked problem_hparams
    # objects which are used to share modality objects between
    # problems.  We do not want to share the modality objects between
    # modes, since the modality objects may decide to do something
    # mode-specific.  A better fix would be to stop abusing the
    # hparams in this way and instead use a separate dictionary to
    # share the modality objects between problems.  This dictionary
    # could be created once per mode and passed to the constructor of
    # t2t_model.
    my_hp = copy.deepcopy(hparams)
    if mode == tf.contrib.learn.ModeKeys.INFER:
      if FLAGS.decode_interactive:
        features = _interactive_input_tensor_to_features_dict(features, my_hp)
      elif FLAGS.decode_from_file:
        features = _decode_input_tensor_to_features_dict(features, my_hp)

    if targets is not None:
      features["targets"] = targets

    dp = devices.data_parallelism()

    tf.get_variable_scope().set_initializer(initializer())
    is_training = mode == tf.contrib.learn.ModeKeys.TRAIN

    # Add input statistics for incoming features.
    with tf.name_scope("input_stats"):
      for (k, v) in six.iteritems(features):
        if isinstance(v, tf.Tensor) and v.get_shape().ndims > 1:
          tf.summary.scalar("%s_batch" % k, tf.shape(v)[0] // dp.n)
          tf.summary.scalar("%s_length" % k, tf.shape(v)[1])
          nonpadding = tf.to_float(tf.not_equal(v, 0))
          nonpadding_tokens = tf.reduce_sum(nonpadding)
          if k == "targets":
            targets_nonpadding_tokens = nonpadding_tokens
          tf.summary.scalar("%s_nonpadding_tokens" % k, nonpadding_tokens)
          tf.summary.scalar("%s_nonpadding_fraction" % k,
                            tf.reduce_mean(nonpadding))

      if is_training:
        # The new data reader occasionally emits very small batches, which
        # cause the examples in those batches to be grossly overweighted.
        # We decrease the loss proportionally to the ratio of the size of this
        # batch to the size of the largest training batch ever.
        # TODO(noam): to be more sophisticated, we could keep separate
        # maxima based on problem choice.
        max_nonpadding_var = tf.get_variable(
            "max_nonpadding",
            shape=[],
            initializer=tf.ones_initializer(),
            trainable=False)
        max_nonpadding = tf.maximum(max_nonpadding_var,
                                    targets_nonpadding_tokens)
        with tf.control_dependencies(
            [tf.assign(max_nonpadding_var, max_nonpadding)]):
          small_batch_multiplier = targets_nonpadding_tokens / max_nonpadding
        tf.summary.scalar("small_batch_multiplier", small_batch_multiplier)

    # Get multi-problem logits and loss based on features["problem_choice"].
    loss_variable_names = []

    def nth_model(n):
      """Build the model for the n-th problem, plus some added variables."""
      model_class = registry.model(model)(
          my_hp,
          mode,
          my_hp.problems[n],
          n,
          dp,
          devices.ps_devices(all_workers=True))
      if mode == tf.contrib.learn.ModeKeys.INFER:
        return model_class.infer(
            features,
            beam_size=FLAGS.decode_beam_size,
            top_beams=(FLAGS.decode_beam_size
                       if FLAGS.decode_return_beams else 1),
            last_position_only=FLAGS.decode_use_last_position_only,
            alpha=FLAGS.decode_alpha,
            decode_length=FLAGS.decode_extra_length)
      # In distributed mode, we build graph for problem=0 and problem=worker_id.
      skipping_is_on = my_hp.problem_choice == "distributed" and is_training
      problem_worker_id = FLAGS.worker_id % len(my_hp.problems)
      skip_this_one = n != 0 and n % FLAGS.worker_replicas != problem_worker_id
      # On worker 0 also build graph for problems <= 1.
      # TODO(lukaszkaiser): why is this hack needed for variables init? Repair.
      skip_this_one = skip_this_one and (FLAGS.worker_id != 0 or n > 1)
      if (FLAGS.eval_run_autoregressive and
          mode == tf.contrib.learn.ModeKeys.EVAL):
        sharded_logits, losses_dict = model_class.eval_autoregressive(features)
      else:
        sharded_logits, losses_dict = model_class.model_fn(
            features, skip=(skipping_is_on and skip_this_one))
      with tf.variable_scope("losses_avg"):
        total_loss, ops = 0.0, []
        for loss_key, loss_value in six.iteritems(losses_dict):
          loss_name = "problem_%d/%s_loss" % (n, loss_key)
          loss_moving_avg = tf.get_variable(
              loss_name, initializer=100.0, trainable=False)
          loss_variable_names.append(loss_name)
          ops.append(
              loss_moving_avg.assign(loss_moving_avg * 0.9 + loss_value * 0.1))
          total_loss += loss_value
        try:  # Total loss avg might be reused or not, we try both.
          with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            # Total loss was already constructed on input.
            loss_moving_avg = tf.get_variable("problem_%d/total_loss" % n)
        except ValueError:
          loss_moving_avg = tf.get_variable(
              "problem_%d/total_loss" % n, initializer=100.0, trainable=False)
        ops.append(
            loss_moving_avg.assign(loss_moving_avg * 0.9 + total_loss * 0.1))
      with tf.variable_scope("train_stats"):  # Count steps for this problem.
        problem_steps = tf.get_variable(
            "problem_%d_steps" % n, initializer=0, trainable=False)
        ops.append(problem_steps.assign_add(1))
      with tf.control_dependencies(ops):  # Make sure the ops run.
        # Ensure the loss is a scalar here.
        total_loss = tf.reshape(total_loss, [], name="total_loss_control_id")
      return [total_loss] + sharded_logits  # Need to flatten for cond later.

    result_list = input_fn_builder.cond_on_index(nth_model,
                                                 features["problem_choice"], 0,
                                                 len(my_hp.problems) - 1)

    if mode == tf.contrib.learn.ModeKeys.INFER:
      # Beam search in sequence model returns both decodes withe key "outputs"
      # and scores with they key "scores". If return list is a dict, we expect
      # that it will have keys "outputs", a tensor of int32 and scores, a
      # tensor of floats. This is useful if we want to return scores from
      # estimator.predict
      if not isinstance(result_list, dict):
        ret = {"outputs": result_list}, None, None
      else:
        ret = {
            "outputs": result_list["outputs"],
            "scores": result_list["scores"]
        }, None, None
      if "inputs" in features:
        ret[0]["inputs"] = features["inputs"]
      if "infer_targets" in features:
        ret[0]["targets"] = features["infer_targets"]
      return ret

    sharded_logits, total_loss = result_list[1:], result_list[0]
    if mode == tf.contrib.learn.ModeKeys.EVAL:
      # For evaluation, return the logits layer as our predictions.
      logits = tf.concat(sharded_logits, 0)
      ret = {
          "predictions": logits,
          "problem_choice": features["problem_choice"],
      }
      return ret, total_loss, None

    assert mode == tf.contrib.learn.ModeKeys.TRAIN

    # Some training statistics.
    with tf.name_scope("training_stats"):
      learning_rate = my_hp.learning_rate * learning_rate_decay()
      learning_rate /= math.sqrt(float(FLAGS.worker_replicas))
      tf.summary.scalar("learning_rate", learning_rate)
      global_step = tf.to_float(tf.contrib.framework.get_global_step())
      for n in xrange(len(my_hp.problems)):
        names_and_vars = []
        with tf.variable_scope("losses_avg", reuse=True):
          total_loss_var = tf.get_variable("problem_%d/total_loss" % n)
          names_and_vars.append(("total_loss", total_loss_var))
        with tf.variable_scope("losses_avg", reuse=True):
          for loss_name in loss_variable_names:
            if loss_name.startswith("problem_%d/" % n):
              loss_var = tf.get_variable(loss_name)
              loss_suffix = loss_name[loss_name.index("/") + 1:]
              names_and_vars.append((loss_suffix, loss_var))
        for (loss_name, loss_var) in names_and_vars:
          tf.summary.scalar("loss_avg_%d/%s" % (n, loss_name), loss_var)
        with tf.variable_scope("train_stats", reuse=True):
          nth_steps = tf.get_variable("problem_%d_steps" % n, dtype=tf.int32)
        tf.summary.scalar("problem_%d_frequency" % n,
                          tf.to_float(nth_steps) / (global_step + 1.0))

    # Log trainable weights and add decay.
    total_size, weight_decay_loss = 0, 0.0
    all_weights = {v.name: v for v in tf.trainable_variables()}
    for v_name in sorted(list(all_weights)):
      v = all_weights[v_name]
      v_size = int(np.prod(np.array(v.shape.as_list())))
      total_size += v_size
      if my_hp.weight_decay > 0.0 and len(v.shape.as_list()) > 1:
        # Add weight regularization if set and the weight is not a bias (dim>1).
        with tf.device(v._ref().device):  # pylint: disable=protected-access
          v_loss = tf.nn.l2_loss(v) / v_size
        weight_decay_loss += v_loss
      is_body = len(v_name) > 5 and v_name[:5] == "body/"
      if my_hp.weight_noise > 0.0 and is_body:
        # Add weight noise if set in my_hp.
        with tf.device(v._ref().device):  # pylint: disable=protected-access
          scale = learning_rate * 0.001
          noise = tf.truncated_normal(v.shape) * my_hp.weight_noise * scale
          noise_op = v.assign_add(noise)
        with tf.control_dependencies([noise_op]):
          total_loss = tf.identity(total_loss)
    if my_hp.weight_decay > 0.0:
      total_loss += weight_decay_loss * my_hp.weight_decay
    if is_training:
      total_loss *= small_batch_multiplier
    total_loss = tf.identity(total_loss, name="total_loss")
    log_variable_sizes(tf.trainable_variables(), "Trainable Variables")
    diet_vars = [v for v in tf.global_variables() if hasattr(v, "optimizer")]
    log_variable_sizes(diet_vars, "Diet Varaibles")
    # Define the train_op for the TRAIN mode.
    opt = _ConditionalOptimizer(my_hp.optimizer, learning_rate, my_hp)
    tf.logging.info("Computing gradients for global model_fn.")
    opt_summaries = ["learning_rate", "loss"]
    if hparams.summarize_grads:
      opt_summaries.extend(["gradients", "gradient_norm"])
    train_op = tf.contrib.layers.optimize_loss(
        name="training",
        loss=total_loss,
        global_step=tf.train.get_global_step(),
        learning_rate=learning_rate,
        clip_gradients=my_hp.clip_grad_norm or None,
        gradient_noise_scale=hparams.grad_noise_scale or None,
        optimizer=opt,
        summaries=opt_summaries,
        colocate_gradients_with_ops=True)

    # Remove summaries that will fail to run because they are in conditionals.
    # TODO(cwhipkey): Test with this code removed, later in 2017.
    summaries = tf.get_collection_ref(tf.GraphKeys.SUMMARIES)
    for i in range(len(summaries) - 1, -1, -1):
      if summaries[i].name.startswith("cond_"):
        del summaries[i]

    tf.logging.info("Global model_fn finished.")
    return {"problem_choice": features["problem_choice"]}, total_loss, train_op

  return model_fn


class _ConditionalOptimizer(tf.train.Optimizer):
  """Conditional optimizer."""

  def __init__(self, optimizer_name, lr, hparams):
    if optimizer_name == "Adam":
      # We change the default epsilon for Adam and re-scale lr.
      # Using LazyAdam as it's much faster for large vocabulary embeddings.
      self._opt = tf.contrib.opt.LazyAdamOptimizer(
          lr / 500.0,
          beta1=hparams.optimizer_adam_beta1,
          beta2=hparams.optimizer_adam_beta2,
          epsilon=hparams.optimizer_adam_epsilon)
    elif optimizer_name == "Momentum":
      self._opt = tf.train.MomentumOptimizer(
          lr, momentum=hparams.optimizer_momentum_momentum)
    elif optimizer_name == "YellowFin":
      tf.logging.info("Init YellowFin Optimizer.")
      self._opt = yellowfin.YellowFinOptimizer(
          learning_rate=lr, momentum=hparams.optimizer_momentum_momentum)
    else:
      self._opt = tf.contrib.layers.OPTIMIZER_CLS_NAMES[optimizer_name](lr)

  def compute_gradients(self, loss, var_list, colocate_gradients_with_ops):
    return self._opt.compute_gradients(
        loss, var_list, colocate_gradients_with_ops=colocate_gradients_with_ops)

  def apply_gradients(self, gradients, global_step=None, name=None):
    return self._opt.apply_gradients(
        gradients, global_step=global_step, name=name)


def _sqrt_decay(step):
  """Decay like 1 / sqrt(step), multiplied by 500 to normalize."""
  return 500.0 / tf.sqrt(tf.maximum(step, 1.0))


def _exp_decay_after(step, rate, from_which_step):
  """Decay exponentially by rate (per step) starting at from_which_step."""
  return tf.cond(
      step < from_which_step,
      lambda: tf.constant(1.0),
      lambda: rate**(step - from_which_step),
      name="exponential_decay_step_cond")


def _interactive_input_tensor_to_features_dict(feature_map, hparams):
  """Convert the interactive input format (see above) to a dictionary.

  Args:
    feature_map: a dictionary with keys `problem_choice` and `input` containing
      Tensors.
    hparams: model hyperparameters

  Returns:
    a features dictionary, as expected by the decoder.
  """
  inputs = tf.constant(feature_map["inputs"])
  input_is_image = False if len(inputs.shape) < 3 else True

  def input_fn(problem_choice, x=inputs):  # pylint: disable=missing-docstring
    p_hparams = hparams.problems[problem_choice]
    if not input_is_image:
      # Remove the batch dimension.
      num_samples = x[0]
      length = x[2]
      x = tf.slice(x, [3], tf.to_int32([length]))
      x = tf.reshape(x, [1, -1, 1, 1])
      # Transform into a batch of size num_samples to get that many random
      # decodes.
      x = tf.tile(x, tf.to_int32([num_samples, 1, 1, 1]))
    else:
      x = tf.image.resize_images(x, [299, 299])
      x = tf.reshape(x, [1, 299, 299, -1])
      x = tf.to_int32(x)
    return (tf.constant(p_hparams.input_space_id),
            tf.constant(p_hparams.target_space_id), x)

  input_space_id, target_space_id, x = input_fn_builder.cond_on_index(
      input_fn, feature_map["problem_choice"], 0, len(hparams.problems) - 1)

  features = {}
  features["problem_choice"] = tf.constant(feature_map["problem_choice"])
  features["input_space_id"] = input_space_id
  features["target_space_id"] = target_space_id
  features["decode_length"] = (IMAGE_DECODE_LENGTH
                               if input_is_image else inputs[1])
  features["inputs"] = x
  return features


def _decode_input_tensor_to_features_dict(feature_map, hparams):
  """Convert the interactive input format (see above) to a dictionary.

  Args:
    feature_map: a dictionary with keys `problem_choice` and `input` containing
      Tensors.
    hparams: model hyperparameters

  Returns:
    a features dictionary, as expected by the decoder.
  """
  inputs = tf.constant(feature_map["inputs"])
  input_is_image = False

  def input_fn(problem_choice, x=inputs):  # pylint: disable=missing-docstring
    p_hparams = hparams.problems[problem_choice]
    # Add a third empty dimension dimension
    x = tf.expand_dims(x, axis=[2])
    x = tf.to_int32(x)
    return (tf.constant(p_hparams.input_space_id),
            tf.constant(p_hparams.target_space_id), x)

  input_space_id, target_space_id, x = input_fn_builder.cond_on_index(
      input_fn, feature_map["problem_choice"], 0, len(hparams.problems) - 1)

  features = {}
  features["problem_choice"] = feature_map["problem_choice"]
  features["input_space_id"] = input_space_id
  features["target_space_id"] = target_space_id
  features["decode_length"] = (IMAGE_DECODE_LENGTH
                               if input_is_image else tf.shape(x)[1] + 50)
  features["inputs"] = x
  return features
