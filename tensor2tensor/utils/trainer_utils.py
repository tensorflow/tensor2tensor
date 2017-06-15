# Copyright 2017 Google Inc.
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

"""Utilities for trainer binary."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import operator
import os
import sys

# Dependency imports

import numpy as np
import six
# pylint: disable=redefined-builtin
from six.moves import input
from six.moves import xrange
from six.moves import zip
# pylint: enable=redefined-builtin

from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.models import models  # pylint: disable=unused-import
from tensor2tensor.utils import data_reader
from tensor2tensor.utils import expert_utils as eu
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.ops import init_ops


# Number of samples to draw for an image input (in such cases as captioning)
IMAGE_DECODE_LENGTH = 100

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool("registry_help", False,
                  "If True, logs the contents of the registry and exits.")
flags.DEFINE_string("master", "", "Address of TensorFlow master.")
flags.DEFINE_string("schedule", "local_run",
                    "Method of tf.contrib.learn.Experiment to run.")
flags.DEFINE_string("output_dir", "", "Base output directory for run.")
flags.DEFINE_string("model", "", "Which model to use.")
flags.DEFINE_string("hparams_set", "", "Which parameters to use.")
flags.DEFINE_string("hparams_range", "", "Parameters range.")
flags.DEFINE_string(
    "hparams", "",
    """A comma-separated list of `name=value` hyperparameter values. This flag
    is used to override hyperparameter settings either when manually selecting
    hyperparameters or when using Vizier. If a hyperparameter setting is
    specified by this flag then it must be a valid hyperparameter name for the
    model.""")
flags.DEFINE_string("problems", "", "Dash separated list of problems to "
                    "solve.")
flags.DEFINE_string("data_dir", "/tmp/data", "Directory with training data.")
flags.DEFINE_string("worker_job", "/job:worker", "name of worker job")
flags.DEFINE_integer("worker_gpu", 1, "How many GPUs to use.")
flags.DEFINE_integer("worker_replicas", 1, "How many workers to use.")
flags.DEFINE_integer("worker_id", 0, "Which worker task are we.")
flags.DEFINE_integer("ps_gpu", 0, "How many GPUs to use per ps.")
flags.DEFINE_string("gpu_order", "", "Optional order for daisy-chaining gpus."
                    " e.g. \"1 3 2 4\"")
flags.DEFINE_string("ps_job", "/job:ps", "name of ps job")
flags.DEFINE_integer("ps_replicas", 0, "How many ps replicas.")
flags.DEFINE_bool("experimental_optimize_placement", False,
                  "Optimize ops placement with experimental session options.")
flags.DEFINE_bool("sync", False, "Sync compute on PS.")
flags.DEFINE_bool("infer_use_last_position_only", False,
                  "In inference, use last position only for speedup.")
flags.DEFINE_integer("train_steps", 250000,
                     "The number of steps to run training for.")
flags.DEFINE_integer("eval_steps", 10, "Number of steps in evaluation.")
flags.DEFINE_integer("keep_checkpoint_max", 20,
                     "How many recent checkpoints to keep.")
flags.DEFINE_bool("interactive", False, "Interactive local inference mode.")
flags.DEFINE_bool("endless_dec", False, "Run decoding endlessly. Temporary.")
flags.DEFINE_bool("save_images", False, "Save inference input images.")
flags.DEFINE_string("decode_from_file", None, "Path to decode file")
flags.DEFINE_string("decode_to_file", None, "Path to inference output file")
flags.DEFINE_integer("decode_shards", 1, "How many shards to decode.")
flags.DEFINE_integer("decode_problem_id", 0, "Which problem to decode.")
flags.DEFINE_integer("decode_extra_length", 50, "Added decode length.")
flags.DEFINE_integer("decode_batch_size", 32, "Batch size for decoding. "
                     "The decodes will be written to <filename>.decodes in"
                     "format result\tinput")
flags.DEFINE_integer("beam_size", 4, "The beam size for beam decoding")
flags.DEFINE_float("alpha", 0.6, "Alpha for length penalty")
flags.DEFINE_bool("return_beams", False,
                  "Whether to return 1 (False) or all (True) beams. The \n "
                  "output file will have the format "
                  "<beam1>\t<beam2>..\t<input>")
flags.DEFINE_bool("daisy_chain_variables", True,
                  "copy variables around in a daisy chain")


def make_experiment_fn(data_dir, model_name, train_steps, eval_steps):
  """Returns experiment_fn for learn_runner. Wraps create_experiment."""

  def experiment_fn(output_dir):
    return create_experiment(
        output_dir=output_dir,
        data_dir=data_dir,
        model_name=model_name,
        train_steps=train_steps,
        eval_steps=eval_steps)

  return experiment_fn


def create_experiment(output_dir, data_dir, model_name, train_steps,
                      eval_steps):
  hparams = create_hparams(FLAGS.hparams_set, FLAGS.data_dir)
  estimator, input_fns = create_experiment_components(
      hparams=hparams,
      output_dir=output_dir,
      data_dir=data_dir,
      model_name=model_name)
  return tf.contrib.learn.Experiment(
      estimator=estimator,
      train_input_fn=input_fns["train"],
      eval_input_fn=input_fns["eval"],
      eval_metrics=metrics.create_evaluation_metrics(FLAGS.problems.split("-")),
      train_steps=train_steps,
      eval_steps=eval_steps,
      train_monitors=[])


def create_experiment_components(hparams, output_dir, data_dir, model_name):
  """Constructs and returns Estimator and train/eval input functions."""
  hparams.problems = [
      problem_hparams.problem_hparams(problem, hparams)
      for problem in FLAGS.problems.split("-")
  ]

  num_datashards = data_parallelism().n

  tf.logging.info("Creating experiment, storing model files in %s", output_dir)

  train_problems_data = get_datasets_for_mode(data_dir,
                                              tf.contrib.learn.ModeKeys.TRAIN)
  train_input_fn = get_input_fn(
      mode=tf.contrib.learn.ModeKeys.TRAIN,
      hparams=hparams,
      data_file_patterns=train_problems_data,
      num_datashards=num_datashards)

  eval_problems_data = get_datasets_for_mode(data_dir,
                                             tf.contrib.learn.ModeKeys.EVAL)
  eval_input_fn = get_input_fn(
      mode=tf.contrib.learn.ModeKeys.EVAL,
      hparams=hparams,
      data_file_patterns=eval_problems_data,
      num_datashards=num_datashards)
  estimator = tf.contrib.learn.Estimator(
      model_fn=model_builder(model_name, hparams=hparams),
      model_dir=output_dir,
      config=tf.contrib.learn.RunConfig(
          master=FLAGS.master,
          model_dir=output_dir,
          session_config=session_config(),
          keep_checkpoint_max=20))
  return estimator, {"train": train_input_fn, "eval": eval_input_fn}


def log_registry():
  tf.logging.info(registry.help_string())
  if FLAGS.registry_help:
    sys.exit(0)


def create_hparams(params_id, data_dir):
  """Returns hyperparameters, including any flag value overrides.

  If the hparams FLAG is set, then it will use any values specified in
  hparams to override any individually-set hyperparameter. This logic
  allows tuners to override hyperparameter settings to find optimal values.

  Args:
    params_id: which set of parameters to choose (must be in _PARAMS above).
    data_dir: the directory containing the training data.

  Returns:
    The hyperparameters as a tf.contrib.training.HParams object.
  """
  hparams = registry.hparams(params_id)()
  hparams.add_hparam("data_dir", data_dir)
  # Command line flags override any of the preceding hyperparameter values.
  if FLAGS.hparams:
    hparams = hparams.parse(FLAGS.hparams)
  return hparams


def run(data_dir, model, output_dir, train_steps, eval_steps, schedule):
  """Runs an Estimator locally or distributed.

  This function chooses one of two paths to execute:

  1. Running locally if schedule=="local_run".
  3. Distributed training/evaluation otherwise.

  Args:
    data_dir: The directory the data can be found in.
    model: The name of the model to use.
    output_dir: The directory to store outputs in.
    train_steps: The number of steps to run training for.
    eval_steps: The number of steps to run evaluation for.
    schedule: (str) The schedule to run. The value here must
      be the name of one of Experiment's methods.
  """
  if schedule == "local_run":
    # Run the local demo.
    run_locally(
        data_dir=data_dir,
        model=model,
        output_dir=output_dir,
        train_steps=train_steps,
        eval_steps=eval_steps)
  else:
    # Perform distributed training/evaluation.
    learn_runner.run(
        experiment_fn=make_experiment_fn(
            data_dir=data_dir,
            model_name=model,
            train_steps=train_steps,
            eval_steps=eval_steps),
        schedule=schedule,
        output_dir=FLAGS.output_dir)


def validate_flags():
  if not FLAGS.model:
    raise ValueError("Must specify a model with --model.")
  if not FLAGS.problems:
    raise ValueError("Must specify a set of problems with --problems.")
  if not (FLAGS.hparams_set or FLAGS.hparams_range):
    raise ValueError("Must specify either --hparams_set or --hparams_range.")
  if not FLAGS.schedule:
    raise ValueError("Must specify --schedule.")
  if not FLAGS.output_dir:
    FLAGS.output_dir = "/tmp/tensor2tensor"
    tf.logging.warning("It is strongly recommended to specify --output_dir. "
                       "Using default output_dir=%s.", FLAGS.output_dir)


def session_config():
  """The TensorFlow Session config to use."""
  graph_options = tf.GraphOptions(optimizer_options=tf.OptimizerOptions(
      opt_level=tf.OptimizerOptions.L1, do_function_inlining=False))
  if FLAGS.experimental_optimize_placement:
    rewrite_options = tf.RewriterConfig(optimize_tensor_layout=True)
    rewrite_options.optimizers.append("pruning")
    rewrite_options.optimizers.append("constfold")
    rewrite_options.optimizers.append("layout")
    graph_options = tf.GraphOptions(
        rewrite_options=rewrite_options, infer_shapes=True)
  config = tf.ConfigProto(
      allow_soft_placement=True, graph_options=graph_options)

  return config


def model_builder(model, hparams):
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
    if mode == tf.contrib.learn.ModeKeys.INFER and FLAGS.interactive:
      features = _interactive_input_tensor_to_features_dict(features, hparams)
    if mode == tf.contrib.learn.ModeKeys.INFER and FLAGS.decode_from_file:
      features = _decode_input_tensor_to_features_dict(features, hparams)
    # A dictionary containing:
    #  - problem_choice: A Tensor containing an integer indicating which problem
    #                    was selected for this run.
    #  - predictions: A Tensor containing the model's output predictions.
    run_info = dict()
    run_info["problem_choice"] = features["problem_choice"]

    if targets is not None:
      features["targets"] = targets

    dp = data_parallelism()

    # Add input statistics for incoming features.
    with tf.name_scope("input_stats"):
      for (k, v) in six.iteritems(features):
        if isinstance(v, tf.Tensor) and v.get_shape().ndims > 1:
          tf.summary.scalar("%s_batch" % k, tf.shape(v)[0] // dp.n)
          tf.summary.scalar("%s_length" % k, tf.shape(v)[1])
          nonpadding = tf.to_float(tf.not_equal(v, 0))
          tf.summary.scalar("%s_nonpadding_tokens" % k,
                            tf.reduce_sum(nonpadding))
          tf.summary.scalar("%s_nonpadding_fraction" % k,
                            tf.reduce_mean(nonpadding))

    tf.get_variable_scope().set_initializer(initializer())
    train = mode == tf.contrib.learn.ModeKeys.TRAIN

    # Get multi-problem logits and loss based on features["problem_choice"].
    def nth_model(n):
      """Build the model for the n-th problem, plus some added variables."""
      model_class = registry.model(model)(
          hparams, hparams.problems[n], n, dp, _ps_devices(all_workers=True))
      if mode == tf.contrib.learn.ModeKeys.INFER:
        return model_class.infer(
            features,
            beam_size=FLAGS.beam_size,
            top_beams=FLAGS.beam_size if FLAGS.return_beams else 1,
            last_position_only=FLAGS.infer_use_last_position_only,
            alpha=FLAGS.alpha,
            decode_length=FLAGS.decode_extra_length)
      # In distributed mode, we build graph for problem=0 and problem=worker_id.
      skipping_is_on = hparams.problem_choice == "distributed" and train
      problem_worker_id = FLAGS.worker_id % len(hparams.problems)
      skip_this_one = n != 0 and n % FLAGS.worker_replicas != problem_worker_id
      # On worker 0 also build graph for problems <= 1.
      # TODO(lukaszkaiser): why is this hack needed for variables init? Repair.
      skip_this_one = skip_this_one and (FLAGS.worker_id != 0 or n > 1)
      sharded_logits, training_loss, extra_loss = model_class.model_fn(
          features, train, skip=(skipping_is_on and skip_this_one))
      with tf.variable_scope("losses_avg", reuse=True):
        loss_moving_avg = tf.get_variable("problem_%d/training_loss" % n)
        o1 = loss_moving_avg.assign(loss_moving_avg * 0.9 + training_loss * 0.1)
        loss_moving_avg = tf.get_variable("problem_%d/extra_loss" % n)
        o2 = loss_moving_avg.assign(loss_moving_avg * 0.9 + extra_loss * 0.1)
        loss_moving_avg = tf.get_variable("problem_%d/total_loss" % n)
        total_loss = training_loss + extra_loss
        o3 = loss_moving_avg.assign(loss_moving_avg * 0.9 + total_loss * 0.1)
      with tf.variable_scope("train_stats"):  # Count steps for this problem.
        problem_steps = tf.get_variable(
            "problem_%d_steps" % n, initializer=0, trainable=False)
        o4 = problem_steps.assign_add(1)
      with tf.control_dependencies([o1, o2, o3, o4]):  # Make sure the ops run.
        total_loss = tf.identity(total_loss)
      return [total_loss] + sharded_logits  # Need to flatten for cond later.

    result_list = _cond_on_index(nth_model, features["problem_choice"], 0,
                                 len(hparams.problems) - 1)

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
      logits = tf.concat(sharded_logits, 0)
      # For evaluation, return the logits layer as our predictions.
      run_info["predictions"] = logits
      train_op = None
      return run_info, total_loss, None

    assert mode == tf.contrib.learn.ModeKeys.TRAIN

    # Some training statistics.
    with tf.name_scope("training_stats"):
      learning_rate = hparams.learning_rate * learning_rate_decay()
      learning_rate /= math.sqrt(float(FLAGS.worker_replicas))
      tf.summary.scalar("learning_rate", learning_rate)
      global_step = tf.to_float(tf.contrib.framework.get_global_step())
      for n in xrange(len(hparams.problems)):
        with tf.variable_scope("losses_avg", reuse=True):
          total_loss_var = tf.get_variable("problem_%d/total_loss" % n)
          training_loss_var = tf.get_variable("problem_%d/training_loss" % n)
          extra_loss_var = tf.get_variable("problem_%d/extra_loss" % n)
        tf.summary.scalar("loss_avg_%d/total_loss" % n, total_loss_var)
        tf.summary.scalar("loss_avg_%d/training_loss" % n, training_loss_var)
        tf.summary.scalar("loss_avg_%d/extra_loss" % n, extra_loss_var)
        with tf.variable_scope("train_stats", reuse=True):
          nth_steps = tf.get_variable("problem_%d_steps" % n, dtype=tf.int32)
        tf.summary.scalar("problem_%d_frequency" % n,
                          tf.to_float(nth_steps) / (global_step + 1.0))

    # Log trainable weights and add decay.
    total_size, total_embedding, weight_decay_loss = 0, 0, 0.0
    all_weights = {v.name: v for v in tf.trainable_variables()}
    for v_name in sorted(list(all_weights)):
      v = all_weights[v_name]
      v_size = int(np.prod(np.array(v.shape.as_list())))
      tf.logging.info("Weight    %s\tshape    %s\tsize    %d",
                      v.name[:-2].ljust(80), str(v.shape).ljust(20), v_size)
      if "embedding" in v_name:
        total_embedding += v_size
      total_size += v_size
      if hparams.weight_decay > 0.0 and len(v.shape.as_list()) > 1:
        # Add weight regularization if set and the weight is not a bias (dim>1).
        with tf.device(v._ref().device):  # pylint: disable=protected-access
          v_loss = tf.nn.l2_loss(v) / v_size
        weight_decay_loss += v_loss
      is_body = len(v_name) > 5 and v_name[:5] == "body/"
      if hparams.weight_noise > 0.0 and is_body:
        # Add weight noise if set in hparams.
        with tf.device(v._ref().device):  # pylint: disable=protected-access
          scale = learning_rate * 0.001
          noise = tf.truncated_normal(v.shape) * hparams.weight_noise * scale
          noise_op = v.assign_add(noise)
        with tf.control_dependencies([noise_op]):
          total_loss = tf.identity(total_loss)
    tf.logging.info("Total trainable variables size: %d", total_size)
    tf.logging.info("Total embedding variables size: %d", total_embedding)
    tf.logging.info("Total non-embedding variables size: %d",
                    total_size - total_embedding)
    total_loss += weight_decay_loss * hparams.weight_decay

    # Define the train_op for the TRAIN mode.
    opt = _ConditionalOptimizer(hparams.optimizer, learning_rate, hparams)
    tf.logging.info("Computing gradients for global model_fn.")
    train_op = tf.contrib.layers.optimize_loss(
        name="training",
        loss=total_loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=learning_rate,
        clip_gradients=hparams.clip_grad_norm or None,
        optimizer=opt,
        colocate_gradients_with_ops=True)

    tf.logging.info("Global model_fn finished.")
    return run_info, total_loss, train_op

  return model_fn


def run_locally(data_dir, model, output_dir, train_steps, eval_steps):
  """Runs an Estimator locally.

  This function demonstrates model training, evaluation, inference locally.

  Args:
    data_dir: The directory the data can be found in.
    model: The name of the model to use.
    output_dir: The directory to store outputs in.
    train_steps: The number of steps to run training for.
    eval_steps: The number of steps to run evaluation for.
  """
  train_problems_data = get_datasets_for_mode(data_dir,
                                              tf.contrib.learn.ModeKeys.TRAIN)

  # For a local run, we can train, evaluate, predict.
  hparams = create_hparams(FLAGS.hparams_set, FLAGS.data_dir)
  hparams.problems = [
      problem_hparams.problem_hparams(problem, hparams)
      for problem in FLAGS.problems.split("-")
  ]

  estimator = tf.contrib.learn.Estimator(
      model_fn=model_builder(model, hparams=hparams),
      model_dir=output_dir,
      config=tf.contrib.learn.RunConfig(
          session_config=session_config(),
          keep_checkpoint_max=FLAGS.keep_checkpoint_max))

  num_datashards = data_parallelism().n

  if train_steps > 0:
    # Train.
    tf.logging.info("Performing local training.")
    estimator.fit(
        input_fn=get_input_fn(
            mode=tf.contrib.learn.ModeKeys.TRAIN,
            hparams=hparams,
            data_file_patterns=train_problems_data,
            num_datashards=num_datashards),
        steps=train_steps,
        monitors=[])

  if eval_steps > 0:
    # Evaluate.
    tf.logging.info("Performing local evaluation.")
    eval_problems_data = get_datasets_for_mode(data_dir,
                                               tf.contrib.learn.ModeKeys.EVAL)
    eval_input_fn = get_input_fn(
        mode=tf.contrib.learn.ModeKeys.EVAL,
        hparams=hparams,
        data_file_patterns=eval_problems_data,
        num_datashards=num_datashards)
    unused_metrics = estimator.evaluate(
        input_fn=eval_input_fn,
        steps=eval_steps,
        metrics=metrics.create_evaluation_metrics(FLAGS.problems.split("-")))

  # Predict.
  if FLAGS.interactive:
    infer_input_fn = _interactive_input_fn(hparams)
    for problem_idx, example in infer_input_fn:
      targets_vocab = hparams.problems[problem_idx].vocabulary["targets"]
      result_iter = estimator.predict(input_fn=lambda e=example: e)
      for result in result_iter:
        if FLAGS.return_beams:
          beams = np.split(result["outputs"], FLAGS.beam_size, axis=0)
          scores = None
          if "scores" in result:
            scores = np.split(result["scores"], FLAGS.beam_size, axis=0)
          for k, beam in enumerate(beams):
            tf.logging.info("BEAM %d:" % k)
            if scores is not None:
              tf.logging.info("%s\tScore:%f" %
                              (targets_vocab.decode(beam.flatten()), scores[k]))
            else:
              tf.logging.info(targets_vocab.decode(beam.flatten()))
        else:
          tf.logging.info(targets_vocab.decode(result["outputs"].flatten()))
  # Predict from file
  elif FLAGS.decode_from_file is not None:
    problem_id = FLAGS.decode_problem_id
    inputs_vocab = hparams.problems[problem_id].vocabulary["inputs"]
    targets_vocab = hparams.problems[problem_id].vocabulary["targets"]
    tf.logging.info("Performing Decoding from a file.")
    sorted_inputs, sorted_keys = _get_sorted_inputs()
    num_decode_batches = (len(sorted_inputs) - 1) // FLAGS.decode_batch_size + 1
    input_fn = _decode_batch_input_fn(problem_id, num_decode_batches,
                                      sorted_inputs, inputs_vocab)

    # strips everything after the first <EOS> id, which is assumed to be 1
    def _save_until_eos(hyp):  #  pylint: disable=missing-docstring
      ret = []
      index = 0
      # until you reach <EOS> id
      while index < len(hyp) and hyp[index] != 1:
        ret.append(hyp[index])
        index += 1
      return np.array(ret)

    decodes = []
    for _ in range(num_decode_batches):
      result_iter = estimator.predict(input_fn=input_fn.next, as_iterable=True)
      for result in result_iter:

        def log_fn(inputs, outputs):
          decoded_inputs = inputs_vocab.decode(
              _save_until_eos(inputs.flatten()))
          tf.logging.info("Inference results INPUT: %s" % decoded_inputs)

          decoded_outputs = targets_vocab.decode(
              _save_until_eos(outputs.flatten()))
          tf.logging.info("Inference results OUTPUT: %s" % decoded_outputs)
          return decoded_outputs

        if FLAGS.return_beams:
          beam_decodes = []
          output_beams = np.split(result["outputs"], FLAGS.beam_size, axis=0)
          for k, beam in enumerate(output_beams):
            tf.logging.info("BEAM %d:" % k)
            beam_decodes.append(log_fn(result["inputs"], beam))
          decodes.append(str.join("\t", beam_decodes))

        else:
          decodes.append(log_fn(result["inputs"], result["outputs"]))

    # Reversing the decoded inputs and outputs because they were reversed in
    # _decode_batch_input_fn
    sorted_inputs.reverse()
    decodes.reverse()
    # Dumping inputs and outputs to file FLAGS.decode_from_file.decodes in
    # format result\tinput in the same order as original inputs
    if FLAGS.decode_shards > 1:
      base_filename = FLAGS.decode_from_file + ("%.2d" % FLAGS.worker_id)
    else:
      base_filename = FLAGS.decode_from_file
    decode_filename = (
        base_filename + "." + FLAGS.model + "." + FLAGS.hparams_set + ".beam" +
        str(FLAGS.beam_size) + ".a" + str(FLAGS.alpha) + ".alpha" +
        str(FLAGS.alpha) + ".decodes")
    tf.logging.info("Writing decodes into %s" % decode_filename)
    outfile = tf.gfile.Open(decode_filename, "w")
    for index in range(len(sorted_inputs)):
      outfile.write("%s\t%s\n" % (decodes[sorted_keys[index]],
                                  sorted_inputs[sorted_keys[index]]))
  else:
    for i, problem in enumerate(FLAGS.problems.split("-")):
      inputs_vocab = hparams.problems[i].vocabulary.get("inputs", None)
      targets_vocab = hparams.problems[i].vocabulary["targets"]
      tf.logging.info("Performing local inference.")
      infer_problems_data = get_datasets_for_mode(
          data_dir, tf.contrib.learn.ModeKeys.INFER)
      infer_input_fn = get_input_fn(
          mode=tf.contrib.learn.ModeKeys.INFER,
          hparams=hparams,
          data_file_patterns=infer_problems_data,
          num_datashards=num_datashards,
          fixed_problem=i)
      result_iter = estimator.predict(
          input_fn=infer_input_fn, as_iterable=FLAGS.endless_dec)

      def log_fn(inputs, targets, outputs, problem, j):
        """Log inference results."""
        if "image" in problem and FLAGS.save_images:
          save_path = os.path.join(FLAGS.output_dir,
                                   "%s_prediction_%d.jpg" % (problem, j))
          show_and_save_image(inputs / 255., save_path)
        elif inputs_vocab:
          decoded_inputs = inputs_vocab.decode(inputs.flatten())
          tf.logging.info("Inference results INPUT: %s" % decoded_inputs)

        decoded_outputs = targets_vocab.decode(outputs.flatten())
        decoded_targets = targets_vocab.decode(targets.flatten())
        tf.logging.info("Inference results OUTPUT: %s" % decoded_outputs)
        if FLAGS.decode_to_file:
          output_filepath = FLAGS.decode_to_file + ".outputs." + problem
          output_file = tf.gfile.Open(output_filepath, "a")
          output_file.write(decoded_outputs + "\n")
          target_filepath = FLAGS.decode_to_file + ".targets." + problem
          target_file = tf.gfile.Open(target_filepath, "a")
          target_file.write(decoded_targets + "\n")

      # The function predict() returns an iterable over the network's
      # predictions from the test input. if FLAGS.endless_dec is set, it will
      # decode over the dev set endlessly, looping over it. We use the returned
      # iterator to log inputs and decodes.
      if FLAGS.endless_dec:
        tf.logging.info("Warning: Decoding endlessly")
        for j, result in enumerate(result_iter):
          inputs, targets, outputs = (result["inputs"], result["targets"],
                                      result["outputs"])
          if FLAGS.return_beams:
            output_beams = np.split(outputs, FLAGS.beam_size, axis=0)
            for k, beam in enumerate(output_beams):
              tf.logging.info("BEAM %d:" % k)
              log_fn(inputs, targets, beam, problem, j)
          else:
            log_fn(inputs, targets, outputs, problem, j)
      else:
        for j, (inputs, targets, outputs) in enumerate(
            zip(result_iter["inputs"], result_iter["targets"], result_iter[
                "outputs"])):
          if FLAGS.return_beams:
            output_beams = np.split(outputs, FLAGS.beam_size, axis=0)
            for k, beam in enumerate(output_beams):
              tf.logging.info("BEAM %d:" % k)
              log_fn(inputs, targets, beam, problem, j)
          else:
            log_fn(inputs, targets, outputs, problem, j)


def _decode_batch_input_fn(problem_id, num_decode_batches, sorted_inputs,
                           vocabulary):
  tf.logging.info(" batch %d" % num_decode_batches)
  # First reverse all the input sentences so that if you're going to get OOMs,
  # you'll see it in the first batch
  sorted_inputs.reverse()
  for b in range(num_decode_batches):
    tf.logging.info("Deocding batch %d" % b)
    batch_length = 0
    batch_inputs = []
    for inputs in sorted_inputs[b * FLAGS.decode_batch_size:(
        b + 1) * FLAGS.decode_batch_size]:
      input_ids = vocabulary.encode(inputs)
      input_ids.append(1)  # Assuming EOS=1.
      batch_inputs.append(input_ids)
      if len(input_ids) > batch_length:
        batch_length = len(input_ids)
    final_batch_inputs = []
    for input_ids in batch_inputs:
      assert len(input_ids) <= batch_length
      x = input_ids + [0] * (batch_length - len(input_ids))
      final_batch_inputs.append(x)
    yield {
        "inputs": np.array(final_batch_inputs),
        "problem_choice": np.array(problem_id)
    }


def get_datasets_for_mode(data_dir, mode):
  return data_reader.get_datasets(FLAGS.problems, data_dir, mode)


def _cond_on_index(fn, index_tensor, cur_idx, max_idx):
  """Call fn(index_tensor) using tf.cond in [cur_id, max_idx]."""
  if cur_idx == max_idx:
    return fn(cur_idx)
  return tf.cond(
      tf.equal(index_tensor, cur_idx), lambda: fn(cur_idx),
      lambda: _cond_on_index(fn, index_tensor, cur_idx + 1, max_idx))


def _interactive_input_fn(hparams):
  """Generator that reads from the terminal and yields "interactive inputs".

  Due to temporary limitations in tf.learn, if we don't want to reload the
  whole graph, then we are stuck encoding all of the input as one fixed-size
  numpy array.

  We yield int64 arrays with shape [const_array_size].  The format is:
  [num_samples, decode_length, len(input ids), <input ids>, <padding>]

  Args:
    hparams: model hparams
  Yields:
    numpy arrays

  Raises:
    Exception: when `input_type` is invalid.
  """
  num_samples = 3
  decode_length = 100
  input_type = "text"
  problem_id = 0
  p_hparams = hparams.problems[problem_id]
  has_input = "inputs" in p_hparams.input_modality
  vocabulary = p_hparams.vocabulary["inputs" if has_input else "targets"]
  # This should be longer than the longest input.
  const_array_size = 10000
  while True:
    prompt = ("INTERACTIVE MODE  num_samples=%d  decode_length=%d  \n"
              "  it=<input_type>     ('text' or 'image')\n"
              "  pr=<problem_num>    (set the problem number)\n"
              "  in=<input_problem>  (set the input problem number)\n"
              "  ou=<output_problem> (set the output problem number)\n"
              "  ns=<num_samples>    (changes number of samples)\n"
              "  dl=<decode_length>  (changes decode legnth)\n"
              "  <%s>                (decode)\n"
              "  q                   (quit)\n"
              ">" % (num_samples, decode_length, "source_string"
                     if has_input else "target_prefix"))
    input_string = input(prompt)
    if input_string == "q":
      return
    elif input_string[:3] == "pr=":
      problem_id = int(input_string[3:])
      p_hparams = hparams.problems[problem_id]
      has_input = "inputs" in p_hparams.input_modality
      vocabulary = p_hparams.vocabulary["inputs" if has_input else "targets"]
    elif input_string[:3] == "in=":
      problem = int(input_string[3:])
      p_hparams.input_modality = hparams.problems[problem].input_modality
      p_hparams.input_space_id = hparams.problems[problem].input_space_id
    elif input_string[:3] == "ou=":
      problem = int(input_string[3:])
      p_hparams.target_modality = hparams.problems[problem].target_modality
      p_hparams.target_space_id = hparams.problems[problem].target_space_id
    elif input_string[:3] == "ns=":
      num_samples = int(input_string[3:])
    elif input_string[:3] == "dl=":
      decode_length = int(input_string[3:])
    elif input_string[:3] == "it=":
      input_type = input_string[3:]
    else:
      if input_type == "text":
        input_ids = vocabulary.encode(input_string)
        if has_input:
          input_ids.append(1)  # assume 1 means end-of-source
        x = [num_samples, decode_length, len(input_ids)] + input_ids
        assert len(x) < const_array_size
        x += [0] * (const_array_size - len(x))
        yield problem_id, {
            "inputs": np.array(x),
            "problem_choice": np.array(problem_id)
        }
      elif input_type == "image":
        input_path = input_string
        img = read_image(input_path)
        yield problem_id, {
            "inputs": img,
            "problem_choice": np.array(problem_id)
        }
      else:
        raise Exception("Unsupported input type.")


def read_image(path):
  try:
    import matplotlib.image as im  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    tf.logging.warning(
        "Reading an image requires matplotlib to be installed: %s", e)
    raise NotImplementedError("Image reading not implemented.")
  return im.imread(path)


def show_and_save_image(img, save_path):
  try:
    import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    tf.logging.warning("Showing and saving an image requires matplotlib to be "
                       "installed: %s", e)
    raise NotImplementedError("Image display and save not implemented.")
  plt.imshow(img)
  plt.savefig(save_path)


def _get_sorted_inputs():
  """Returning inputs sorted according to length.

  Returns:
    a sorted list of inputs

  """
  tf.logging.info("Getting sorted inputs")
  # read file and sort inputs according them according to input length.
  if FLAGS.decode_shards > 1:
    decode_filename = FLAGS.decode_from_file + ("%.2d" % FLAGS.worker_id)
  else:
    decode_filename = FLAGS.decode_from_file
  inputs = [line.strip() for line in tf.gfile.Open(decode_filename)]
  input_lens = [(i, len(line.strip().split())) for i, line in enumerate(inputs)]
  sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1))
  # We'll need the keys to rearrange the inputs back into their original order
  sorted_keys = {}
  sorted_inputs = []
  for i, (index, _) in enumerate(sorted_input_lens):
    sorted_inputs.append(inputs[index])
    sorted_keys[index] = i
  return sorted_inputs, sorted_keys


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

  input_space_id, target_space_id, x = _cond_on_index(
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

  input_space_id, target_space_id, x = _cond_on_index(
      input_fn, feature_map["problem_choice"], 0, len(hparams.problems) - 1)

  features = {}
  features["problem_choice"] = feature_map["problem_choice"]
  features["input_space_id"] = input_space_id
  features["target_space_id"] = target_space_id
  features["decode_length"] = (IMAGE_DECODE_LENGTH
                               if input_is_image else tf.shape(x)[1] + 50)
  features["inputs"] = x
  return features


def get_input_fn(mode,
                 hparams,
                 data_file_patterns=None,
                 num_datashards=None,
                 fixed_problem=None):
  """Provides input to the graph, either from disk or via a placeholder.

  This function produces an input function that will feed data into
  the network. There are two modes of operation:

  1. If data_file_pattern and all subsequent arguments are None, then
     it creates a placeholder for a serialized tf.Example proto.
  2. If data_file_pattern is defined, it will read the data from the
     files at the given location. Use this mode for training,
     evaluation, and testing prediction.

  Args:
    mode: The execution mode, as defined in tf.contrib.learn.ModeKeys.
    hparams: HParams object.
    data_file_patterns: The list of file patterns to use to read in data. Set to
      `None` if you want to create a placeholder for the input data. The
      `problems` flag is a list of problem names joined by the `-` character.
      The flag's string is then split along the `-` and each problem gets its
      own example queue.
    num_datashards: An integer.
    fixed_problem: An integer indicating the problem to fetch data for, or None
      if the input is to be randomly selected.

  Returns:
    A function that returns a dictionary of features and the target labels.
  """

  def input_fn():
    """Supplies input to our model.

    This function supplies input to our model, where this input is a
    function of the mode. For example, we supply different data if
    we're performing training versus evaluation.

    Returns:
      A tuple consisting of 1) a dictionary of tensors whose keys are
      the feature names, and 2) a tensor of target labels if the mode
      is not INFER (and None, otherwise).

    Raises:
      ValueError: if one of the parameters has an unsupported value.
    """
    problem_count, batches = len(data_file_patterns), []
    with tf.name_scope("input_queues"):
      for n in xrange(problem_count):
        if fixed_problem is not None and n != fixed_problem:
          continue
        with tf.name_scope("problem_%d" % n):
          with tf.device("/cpu:0"):  # Input queues are on CPU.
            capacity = hparams.problems[n].max_expected_batch_size_per_shard
            capacity *= num_datashards
            examples = data_reader.input_pipeline(data_file_patterns[n],
                                                  capacity, mode)
            drop_long_sequences = mode == tf.contrib.learn.ModeKeys.TRAIN
            batch_size_multiplier = hparams.problems[n].batch_size_multiplier
            feature_map = data_reader.batch_examples(
                examples,
                data_reader.hparams_to_batching_scheme(
                    hparams,
                    shard_multiplier=num_datashards,
                    drop_long_sequences=drop_long_sequences,
                    length_multiplier=batch_size_multiplier))

        # Reverse inputs and targets features if the problem was reversed.
        if hparams.problems[n].was_reversed:
          inputs = feature_map["inputs"]
          targets = feature_map["targets"]
          feature_map["inputs"] = targets
          feature_map["targets"] = inputs

        # Use the inputs as the targets if the problem is a copy problem.
        if hparams.problems[n].was_copy:
          feature_map["targets"] = feature_map["inputs"]

        # Ensure inputs and targets are proper rank.
        while len(feature_map["inputs"].get_shape()) != 4:
          feature_map["inputs"] = tf.expand_dims(feature_map["inputs"], axis=-1)
        while len(feature_map["targets"].get_shape()) != 4:
          feature_map["targets"] = tf.expand_dims(
              feature_map["targets"], axis=-1)

        batches.append(
            (feature_map["inputs"], feature_map["targets"], tf.constant(n),
             tf.constant(hparams.problems[n].input_space_id),
             tf.constant(hparams.problems[n].target_space_id)))

    # We choose which problem to process.
    loss_moving_avgs = []  # Need loss moving averages for that.
    for n in xrange(problem_count):
      with tf.variable_scope("losses_avg"):
        loss_moving_avgs.append(
            tf.get_variable(
                "problem_%d/total_loss" % n, initializer=100.0,
                trainable=False))
        tf.get_variable(
            "problem_%d/training_loss" % n, initializer=100.0, trainable=False)
        tf.get_variable(
            "problem_%d/extra_loss" % n, initializer=100.0, trainable=False)
    if fixed_problem is None:
      if (hparams.problem_choice == "uniform" or
          mode != tf.contrib.learn.ModeKeys.TRAIN):
        problem_choice = tf.random_uniform(
            [], maxval=problem_count, dtype=tf.int32)
      elif hparams.problem_choice == "adaptive":
        loss_moving_avgs = tf.stack(loss_moving_avgs)
        problem_choice = tf.multinomial(
            tf.reshape(loss_moving_avgs, [1, -1]), 1)
        problem_choice = tf.to_int32(tf.squeeze(problem_choice))
      elif hparams.problem_choice == "distributed":
        assert FLAGS.worker_replicas >= problem_count
        assert FLAGS.worker_replicas % problem_count == 0
        problem_choice = tf.to_int32(FLAGS.worker_id % problem_count)
      else:
        raise ValueError("Value of hparams.problem_choice is %s and must be "
                         "one of [uniform, adaptive, distributed]",
                         hparams.problem_choice)

      # Inputs and targets conditional on problem_choice.
      rand_inputs, rand_target, choice, inp_id, tgt_id = _cond_on_index(
          lambda n: batches[n], problem_choice, 0, problem_count - 1)
    else:
      problem_choice = tf.constant(fixed_problem)
      # Take the only constructed batch, which is the fixed_problem.
      rand_inputs, rand_target, choice, inp_id, tgt_id = batches[0]

    # Set shapes so the ranks are clear.
    rand_inputs.set_shape([None, None, None, None])
    rand_target.set_shape([None, None, None, None])
    choice.set_shape([])
    inp_id.set_shape([])
    tgt_id.set_shape([])
    #  Forced shape obfuscation is necessary for inference.
    if mode == tf.contrib.learn.ModeKeys.INFER:
      rand_inputs._shape = tf.TensorShape([None, None, None, None])  # pylint: disable=protected-access
      rand_target._shape = tf.TensorShape([None, None, None, None])  # pylint: disable=protected-access

    # Final feature map.
    rand_feature_map = {
        "inputs": rand_inputs,
        "problem_choice": choice,
        "input_space_id": inp_id,
        "target_space_id": tgt_id
    }
    if mode == tf.contrib.learn.ModeKeys.INFER:
      rand_feature_map["infer_targets"] = rand_target
      rand_target = None
    return rand_feature_map, rand_target

  return input_fn


class _ConditionalOptimizer(tf.train.Optimizer):
  """Conditional optimizer."""

  def __init__(self, optimizer_name, lr, hparams, skip_condition_tensor=False):
    self._skip_condition = skip_condition_tensor
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
    else:
      self._opt = tf.contrib.layers.OPTIMIZER_CLS_NAMES[optimizer_name](lr)

  def compute_gradients(self, loss, var_list, colocate_gradients_with_ops):
    return self._opt.compute_gradients(
        loss, var_list, colocate_gradients_with_ops=colocate_gradients_with_ops)

  def apply_gradients(self, gradients, global_step=None, name=None):

    def opt_gradients():
      return self._opt.apply_gradients(
          gradients, global_step=global_step, name=name)

    if self._skip_condition is False:
      return opt_gradients()
    return tf.cond(
        self._skip_condition,
        tf.no_op,
        opt_gradients,
        name="conditional_optimizer_gradients_skip_cond")


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


def _ps_replicas(all_workers=False):
  if all_workers:
    return list(range(FLAGS.ps_replicas))
  # Worker K will be using replicas {0,...n-1} + K*n if we have n replicas.
  num_replicas = FLAGS.ps_replicas // FLAGS.worker_replicas
  return [d + FLAGS.worker_id * num_replicas for d in xrange(num_replicas)]


def _gpu_order(num_gpus):
  if FLAGS.gpu_order:
    ret = [int(s) for s in FLAGS.gpu_order.split(" ")]
    if len(ret) == num_gpus:
      return ret
  return list(range(num_gpus))


def _ps_gpus(all_workers=False):
  ps_gpus = []
  for d in _ps_replicas(all_workers=all_workers):
    ps_gpus.extend([(d, gpu) for gpu in _gpu_order(FLAGS.ps_gpu)])
  return ps_gpus


def _ps_devices(all_workers=False):
  """List of ps devices (where to put the experts).

  Args:
    all_workers: whether the list is for all async workers or just this one.

  Returns:
    a list of device names
  """
  if FLAGS.ps_replicas > 0:
    if FLAGS.ps_gpu > 0:
      return [
          FLAGS.ps_job + "/task:%d/GPU:%d" % (d, gpu)
          for (d, gpu) in _ps_gpus(all_workers=all_workers)
      ]
    else:
      return [
          FLAGS.ps_job + "/task:%d" % d
          for d in _ps_replicas(all_workers=all_workers)
      ]
  else:
    if FLAGS.worker_gpu > 0:
      return ["gpu:%d" % d for d in _gpu_order(FLAGS.worker_gpu)]
    else:
      return [""]


def data_parallelism(all_workers=False):
  """Over which devices do we split each training batch.

  In old-fashioned async mode, we split the batch over all GPUs on the
  current worker.

  In sync mode, we split the batch over all the parameter server GPUs.

  This function returns an expert_utils.Parallelism object, which can be used
  to build the model.  It is configured in a way that any variables created
  by `tf.get_variable` will be assigned to the parameter servers and shared
  between datashards.

  Args:
    all_workers: whether the devices are all async workers or just this one.

  Returns:
    a expert_utils.Parallelism.
  """

  def _replica_device_setter(worker_device):
    if FLAGS.ps_replicas == 0:
      return worker_device
    return tf.train.replica_device_setter(
        worker_device=worker_device,
        ps_tasks=FLAGS.ps_replicas,
        ps_device=FLAGS.ps_job + "/GPU:0" if FLAGS.ps_gpu > 0 else FLAGS.ps_job)

  if FLAGS.schedule == "local_run":
    assert not FLAGS.sync
    datashard_devices = ["gpu:%d" % d for d in _gpu_order(FLAGS.worker_gpu)]
    caching_devices = None
  elif FLAGS.sync:
    assert FLAGS.ps_replicas > 0
    datashard_devices = [
        _replica_device_setter(d) for d in _ps_devices(all_workers=all_workers)
    ]
    if FLAGS.ps_gpu > 0 and FLAGS.ps_replicas > 1:
      caching_devices = [
          FLAGS.ps_job + "/task:%d/cpu:0" % d
          for (d, _) in _ps_gpus(all_workers=all_workers)
      ]
    else:
      caching_devices = None
  else:
    # old fashioned async - compute on worker
    if FLAGS.worker_gpu > 1:
      datashard_devices = [
          _replica_device_setter(FLAGS.worker_job + "/GPU:%d" % d)
          for d in _gpu_order(FLAGS.worker_gpu)
      ]
      caching_devices = [FLAGS.worker_job + "/GPU:0"] * FLAGS.worker_gpu
    else:
      datashard_devices = [_replica_device_setter(FLAGS.worker_job)]
      caching_devices = None
  tf.logging.info("datashard_devices: %s", datashard_devices)
  tf.logging.info("caching_devices: %s", caching_devices)
  return eu.Parallelism(
      datashard_devices,
      reuse=True,
      caching_devices=caching_devices,
      daisy_chain_variables=FLAGS.daisy_chain_variables)
