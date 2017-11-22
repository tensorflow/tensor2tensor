# Copyright 2017 Google Inc. All Rights Reserved.
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
# ==============================================================================
"""For training NMT models."""
from __future__ import print_function

import math
import os
import random
import time

import tensorflow as tf

from . import attention_model
from . import gnmt_model
from . import inference
from . import model as nmt_model
from . import model_helper
from .utils import misc_utils as utils
from .utils import nmt_utils

utils.check_tensorflow_version()

__all__ = [
    "run_sample_decode", "run_internal_eval", "run_external_eval",
    "run_full_eval", "init_stats", "update_stats", "check_stats", "train"
]


def run_sample_decode(infer_model, infer_sess, model_dir, hparams,
                      summary_writer, src_data, tgt_data):
  """Sample decode a random sentence from src_data."""
  with infer_model.graph.as_default():
    loaded_infer_model, global_step = model_helper.create_or_load_model(
        infer_model.model, model_dir, infer_sess, "infer")

  _sample_decode(loaded_infer_model, global_step, infer_sess, hparams,
                 infer_model.iterator, src_data, tgt_data,
                 infer_model.src_placeholder,
                 infer_model.batch_size_placeholder, summary_writer)


def run_internal_eval(
    eval_model, eval_sess, model_dir, hparams, summary_writer,
    use_test_set=True):
  """Compute internal evaluation (perplexity) for both dev / test."""
  with eval_model.graph.as_default():
    loaded_eval_model, global_step = model_helper.create_or_load_model(
        eval_model.model, model_dir, eval_sess, "eval")

  dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
  dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
  dev_eval_iterator_feed_dict = {
      eval_model.src_file_placeholder: dev_src_file,
      eval_model.tgt_file_placeholder: dev_tgt_file
  }

  dev_ppl = _internal_eval(loaded_eval_model, global_step, eval_sess,
                           eval_model.iterator, dev_eval_iterator_feed_dict,
                           summary_writer, "dev")
  test_ppl = None
  if use_test_set and hparams.test_prefix:
    test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
    test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
    test_eval_iterator_feed_dict = {
        eval_model.src_file_placeholder: test_src_file,
        eval_model.tgt_file_placeholder: test_tgt_file
    }
    test_ppl = _internal_eval(loaded_eval_model, global_step, eval_sess,
                              eval_model.iterator, test_eval_iterator_feed_dict,
                              summary_writer, "test")
  return dev_ppl, test_ppl


def run_external_eval(infer_model, infer_sess, model_dir, hparams,
                      summary_writer, save_best_dev=True, use_test_set=True):

  """Compute external evaluation (bleu, rouge, etc.) for both dev / test."""
  with infer_model.graph.as_default():
    loaded_infer_model, global_step = model_helper.create_or_load_model(
        infer_model.model, model_dir, infer_sess, "infer")

  dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
  dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
  dev_infer_iterator_feed_dict = {
      infer_model.src_placeholder: inference.load_data(dev_src_file),
      infer_model.batch_size_placeholder: hparams.infer_batch_size,
  }
  dev_scores = _external_eval(
      loaded_infer_model,
      global_step,
      infer_sess,
      hparams,
      infer_model.iterator,
      dev_infer_iterator_feed_dict,
      dev_tgt_file,
      "dev",
      summary_writer,
      save_on_best=save_best_dev)

  test_scores = None
  if use_test_set and hparams.test_prefix:
    test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
    test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
    test_infer_iterator_feed_dict = {
        infer_model.src_placeholder: inference.load_data(test_src_file),
        infer_model.batch_size_placeholder: hparams.infer_batch_size,
    }
    test_scores = _external_eval(
        loaded_infer_model,
        global_step,
        infer_sess,
        hparams,
        infer_model.iterator,
        test_infer_iterator_feed_dict,
        test_tgt_file,
        "test",
        summary_writer,
        save_on_best=False)
  return dev_scores, test_scores, global_step


def run_full_eval(model_dir, infer_model, infer_sess, eval_model, eval_sess,
                  hparams, summary_writer, sample_src_data, sample_tgt_data):
  """Wrapper for running sample_decode, internal_eval and external_eval."""
  run_sample_decode(infer_model, infer_sess, model_dir, hparams, summary_writer,
                    sample_src_data, sample_tgt_data)
  dev_ppl, test_ppl = run_internal_eval(
      eval_model, eval_sess, model_dir, hparams, summary_writer)
  dev_scores, test_scores, global_step = run_external_eval(
      infer_model, infer_sess, model_dir, hparams, summary_writer)

  result_summary = _format_results("dev", dev_ppl, dev_scores, hparams.metrics)
  if hparams.test_prefix:
    result_summary += ", " + _format_results("test", test_ppl, test_scores,
                                             hparams.metrics)

  return result_summary, global_step, dev_scores, test_scores, dev_ppl, test_ppl


def init_stats():
  """Initialize statistics that we want to keep."""
  return {"step_time": 0.0, "loss": 0.0, "predict_count": 0.0,
          "total_count": 0.0, "grad_norm": 0.0}


def update_stats(stats, summary_writer, start_time, step_result):
  """Update stats: write summary and accumulate statistics."""
  (_, step_loss, step_predict_count, step_summary, global_step,
   step_word_count, batch_size, grad_norm, learning_rate) = step_result

  # Write step summary.
  summary_writer.add_summary(step_summary, global_step)

  # update statistics
  stats["step_time"] += (time.time() - start_time)
  stats["loss"] += (step_loss * batch_size)
  stats["predict_count"] += step_predict_count
  stats["total_count"] += float(step_word_count)
  stats["grad_norm"] += grad_norm
  stats["learning_rate"] = learning_rate

  return global_step


def check_stats(stats, global_step, steps_per_stats, hparams, log_f):
  """Print statistics and also check for overflow."""
  # Print statistics for the previous epoch.
  avg_step_time = stats["step_time"] / steps_per_stats
  avg_grad_norm = stats["grad_norm"] / steps_per_stats
  train_ppl = utils.safe_exp(
      stats["loss"] / stats["predict_count"])
  speed = stats["total_count"] / (1000 * stats["step_time"])
  utils.print_out(
      "  global step %d lr %g "
      "step-time %.2fs wps %.2fK ppl %.2f gN %.2f %s" %
      (global_step, stats["learning_rate"],
       avg_step_time, speed, train_ppl, avg_grad_norm,
       _get_best_results(hparams)),
      log_f)

  # Check for overflow
  is_overflow = False
  if math.isnan(train_ppl) or math.isinf(train_ppl) or train_ppl > 1e20:
    utils.print_out("  step %d overflow, stop early" % global_step, log_f)
    is_overflow = True

  return is_overflow


def train(hparams, scope=None, target_session=""):
  """Train a translation model."""
  log_device_placement = hparams.log_device_placement
  out_dir = hparams.out_dir
  num_train_steps = hparams.num_train_steps
  steps_per_stats = hparams.steps_per_stats
  steps_per_external_eval = hparams.steps_per_external_eval
  steps_per_eval = 10 * steps_per_stats
  if not steps_per_external_eval:
    steps_per_external_eval = 5 * steps_per_eval

  if not hparams.attention:
    model_creator = nmt_model.Model
  elif hparams.attention_architecture == "standard":
    model_creator = attention_model.AttentionModel
  elif hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
    model_creator = gnmt_model.GNMTModel
  else:
    raise ValueError("Unknown model architecture")

  train_model = model_helper.create_train_model(model_creator, hparams, scope)
  eval_model = model_helper.create_eval_model(model_creator, hparams, scope)
  infer_model = model_helper.create_infer_model(model_creator, hparams, scope)

  # Preload data for sample decoding.
  dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
  dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
  sample_src_data = inference.load_data(dev_src_file)
  sample_tgt_data = inference.load_data(dev_tgt_file)

  summary_name = "train_log"
  model_dir = hparams.out_dir

  # Log and output files
  log_file = os.path.join(out_dir, "log_%d" % time.time())
  log_f = tf.gfile.GFile(log_file, mode="a")
  utils.print_out("# log_file=%s" % log_file, log_f)

  avg_step_time = 0.0

  # TensorFlow model
  config_proto = utils.get_config_proto(
      log_device_placement=log_device_placement)

  train_sess = tf.Session(
      target=target_session, config=config_proto, graph=train_model.graph)
  eval_sess = tf.Session(
      target=target_session, config=config_proto, graph=eval_model.graph)
  infer_sess = tf.Session(
      target=target_session, config=config_proto, graph=infer_model.graph)

  with train_model.graph.as_default():
    loaded_train_model, global_step = model_helper.create_or_load_model(
        train_model.model, model_dir, train_sess, "train")

  # Summary writer
  summary_writer = tf.summary.FileWriter(
      os.path.join(out_dir, summary_name), train_model.graph)

  # First evaluation
  run_full_eval(
      model_dir, infer_model, infer_sess,
      eval_model, eval_sess, hparams,
      summary_writer, sample_src_data,
      sample_tgt_data)

  last_stats_step = global_step
  last_eval_step = global_step
  last_external_eval_step = global_step

  # This is the training loop.
  stats = init_stats()
  speed, train_ppl = 0.0, 0.0
  start_train_time = time.time()

  utils.print_out(
      "# Start step %d, lr %g, %s" %
      (global_step, loaded_train_model.learning_rate.eval(session=train_sess),
       time.ctime()),
      log_f)

  # Initialize all of the iterators
  skip_count = hparams.batch_size * hparams.epoch_step
  utils.print_out("# Init train iterator, skipping %d elements" % skip_count)
  train_sess.run(
      train_model.iterator.initializer,
      feed_dict={train_model.skip_count_placeholder: skip_count})

  while global_step < num_train_steps:
    ### Run a step ###
    start_time = time.time()
    try:
      step_result = loaded_train_model.train(train_sess)
      hparams.epoch_step += 1
    except tf.errors.OutOfRangeError:
      # Finished going through the training dataset.  Go to next epoch.
      hparams.epoch_step = 0
      utils.print_out(
          "# Finished an epoch, step %d. Perform external evaluation" %
          global_step)
      run_sample_decode(infer_model, infer_sess,
                        model_dir, hparams, summary_writer, sample_src_data,
                        sample_tgt_data)
      dev_scores, test_scores, _ = run_external_eval(
          infer_model, infer_sess, model_dir,
          hparams, summary_writer)
      train_sess.run(
          train_model.iterator.initializer,
          feed_dict={train_model.skip_count_placeholder: 0})
      continue

    # Write step summary and accumulate statistics
    global_step = update_stats(stats, summary_writer, start_time, step_result)

    # Once in a while, we print statistics.
    if global_step - last_stats_step >= steps_per_stats:
      last_stats_step = global_step
      is_overflow = check_stats(stats, global_step, steps_per_stats, hparams,
                                log_f)
      if is_overflow:
        break

      # Reset statistics
      stats = init_stats()

    if global_step - last_eval_step >= steps_per_eval:
      last_eval_step = global_step

      utils.print_out("# Save eval, global step %d" % global_step)
      utils.add_summary(summary_writer, global_step, "train_ppl", train_ppl)

      # Save checkpoint
      loaded_train_model.saver.save(
          train_sess,
          os.path.join(out_dir, "translate.ckpt"),
          global_step=global_step)

      # Evaluate on dev/test
      run_sample_decode(infer_model, infer_sess,
                        model_dir, hparams, summary_writer, sample_src_data,
                        sample_tgt_data)
      dev_ppl, test_ppl = run_internal_eval(
          eval_model, eval_sess, model_dir, hparams, summary_writer)

    if global_step - last_external_eval_step >= steps_per_external_eval:
      last_external_eval_step = global_step

      # Save checkpoint
      loaded_train_model.saver.save(
          train_sess,
          os.path.join(out_dir, "translate.ckpt"),
          global_step=global_step)
      run_sample_decode(infer_model, infer_sess,
                        model_dir, hparams, summary_writer, sample_src_data,
                        sample_tgt_data)
      dev_scores, test_scores, _ = run_external_eval(
          infer_model, infer_sess, model_dir,
          hparams, summary_writer)

  # Done training
  loaded_train_model.saver.save(
      train_sess,
      os.path.join(out_dir, "translate.ckpt"),
      global_step=global_step)

  result_summary, _, dev_scores, test_scores, dev_ppl, test_ppl = run_full_eval(
      model_dir, infer_model, infer_sess,
      eval_model, eval_sess, hparams,
      summary_writer, sample_src_data,
      sample_tgt_data)
  utils.print_out(
      "# Final, step %d lr %g "
      "step-time %.2f wps %.2fK ppl %.2f, %s, %s" %
      (global_step, loaded_train_model.learning_rate.eval(session=train_sess),
       avg_step_time, speed, train_ppl, result_summary, time.ctime()),
      log_f)
  utils.print_time("# Done training!", start_train_time)

  summary_writer.close()

  utils.print_out("# Start evaluating saved best models.")
  for metric in hparams.metrics:
    best_model_dir = getattr(hparams, "best_" + metric + "_dir")
    summary_writer = tf.summary.FileWriter(
        os.path.join(best_model_dir, summary_name), infer_model.graph)
    result_summary, best_global_step, _, _, _, _ = run_full_eval(
        best_model_dir, infer_model, infer_sess, eval_model, eval_sess, hparams,
        summary_writer, sample_src_data, sample_tgt_data)
    utils.print_out("# Best %s, step %d "
                    "step-time %.2f wps %.2fK, %s, %s" %
                    (metric, best_global_step, avg_step_time, speed,
                     result_summary, time.ctime()), log_f)
    summary_writer.close()

  return (dev_scores, test_scores, dev_ppl, test_ppl, global_step)


def _format_results(name, ppl, scores, metrics):
  """Format results."""
  result_str = "%s ppl %.2f" % (name, ppl)
  if scores:
    for metric in metrics:
      result_str += ", %s %s %.1f" % (name, metric, scores[metric])
  return result_str


def _get_best_results(hparams):
  """Summary of the current best results."""
  tokens = []
  for metric in hparams.metrics:
    tokens.append("%s %.2f" % (metric, getattr(hparams, "best_" + metric)))
  return ", ".join(tokens)


def _internal_eval(model, global_step, sess, iterator, iterator_feed_dict,
                   summary_writer, label):
  """Computing perplexity."""
  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
  ppl = model_helper.compute_perplexity(model, sess, label)
  utils.add_summary(summary_writer, global_step, "%s_ppl" % label, ppl)
  return ppl


def _sample_decode(model, global_step, sess, hparams, iterator, src_data,
                   tgt_data, iterator_src_placeholder,
                   iterator_batch_size_placeholder, summary_writer):
  """Pick a sentence and decode."""
  decode_id = random.randint(0, len(src_data) - 1)
  utils.print_out("  # %d" % decode_id)

  iterator_feed_dict = {
      iterator_src_placeholder: [src_data[decode_id]],
      iterator_batch_size_placeholder: 1,
  }
  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

  nmt_outputs, attention_summary = model.decode(sess)

  if hparams.beam_width > 0:
    # get the top translation.
    nmt_outputs = nmt_outputs[0]

  translation = nmt_utils.get_translation(
      nmt_outputs,
      sent_id=0,
      tgt_eos=hparams.eos,
      subword_option=hparams.subword_option)
  utils.print_out(b"    src: " + src_data[decode_id])
  utils.print_out(b"    ref: " + tgt_data[decode_id])
  utils.print_out(b"    nmt: " + translation)

  # Summary
  if attention_summary is not None:
    summary_writer.add_summary(attention_summary, global_step)


def _external_eval(model, global_step, sess, hparams, iterator,
                   iterator_feed_dict, tgt_file, label, summary_writer,
                   save_on_best):
  """External evaluation such as BLEU and ROUGE scores."""
  out_dir = hparams.out_dir
  decode = global_step > 0
  if decode:
    utils.print_out("# External evaluation, global step %d" % global_step)

  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

  output = os.path.join(out_dir, "output_%s" % label)
  scores = nmt_utils.decode_and_evaluate(
      label,
      model,
      sess,
      output,
      ref_file=tgt_file,
      metrics=hparams.metrics,
      subword_option=hparams.subword_option,
      beam_width=hparams.beam_width,
      tgt_eos=hparams.eos,
      decode=decode)
  # Save on best metrics
  if decode:
    for metric in hparams.metrics:
      utils.add_summary(summary_writer, global_step, "%s_%s" % (label, metric),
                        scores[metric])
      # metric: larger is better
      if save_on_best and scores[metric] > getattr(hparams, "best_" + metric):
        setattr(hparams, "best_" + metric, scores[metric])
        model.saver.save(
            sess,
            os.path.join(
                getattr(hparams, "best_" + metric + "_dir"), "translate.ckpt"),
            global_step=model.global_step)
    utils.save_hparams(out_dir, hparams)
  return scores
