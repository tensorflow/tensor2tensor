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

import os
import random
import time

import tensorflow as tf

from tensorflow.python.ops import lookup_ops

from . import attention_model
from . import model as nmt_model
from . import model_helper
from . import inference
from .utils import iterator_utils
from .utils import misc_utils as utils
from .utils import nmt_utils
from .utils import vocab_utils

__all__ = ["train"]


def create_train_model(model_creator,
                       hparams,
                       src_vocab_file,
                       tgt_vocab_file,
                       scope=None):
  train_src_file = "%s.%s" % (hparams.train_prefix, hparams.src)
  train_tgt_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)

  train_graph = tf.Graph()

  with train_graph.as_default():
    src_vocab_table = lookup_ops.index_table_from_file(
        src_vocab_file, default_value=vocab_utils.UNK_ID)
    tgt_vocab_table = lookup_ops.index_table_from_file(
        tgt_vocab_file, default_value=vocab_utils.UNK_ID)
    train_src_dataset = tf.contrib.data.TextLineDataset(train_src_file)
    train_tgt_dataset = tf.contrib.data.TextLineDataset(train_tgt_file)
    train_iterator = iterator_utils.get_iterator(
        train_src_dataset, train_tgt_dataset, hparams,
        src_vocab_table, tgt_vocab_table, hparams.batch_size,
        src_max_len=hparams.src_max_len, tgt_max_len=hparams.tgt_max_len)
    train_model = model_creator(
        hparams,
        iterator=train_iterator,
        mode=tf.contrib.learn.ModeKeys.TRAIN,
        source_vocab_table=src_vocab_table,
        target_vocab_table=tgt_vocab_table,
        scope=scope)

  return train_graph, train_model, train_iterator

def create_eval_model(model_creator,
                      hparams,
                      src_vocab_file,
                      tgt_vocab_file,
                      scope=None):
  eval_graph = tf.Graph()

  with eval_graph.as_default():
    src_vocab_table = lookup_ops.index_table_from_file(
        src_vocab_file, default_value=vocab_utils.UNK_ID)
    tgt_vocab_table = lookup_ops.index_table_from_file(
        tgt_vocab_file, default_value=vocab_utils.UNK_ID)
    eval_src_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
    eval_tgt_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
    eval_src_dataset = tf.contrib.data.TextLineDataset(
        eval_src_file_placeholder)
    eval_tgt_dataset = tf.contrib.data.TextLineDataset(
        eval_tgt_file_placeholder)
    eval_iterator = iterator_utils.get_iterator(
        eval_src_dataset,
        eval_tgt_dataset,
        hparams,
        src_vocab_table,
        tgt_vocab_table,
        hparams.batch_size,
        src_max_len=hparams.src_max_len,
        tgt_max_len=hparams.tgt_max_len)
    eval_model = model_creator(
        hparams,
        iterator=eval_iterator,
        mode=tf.contrib.learn.ModeKeys.EVAL,
        source_vocab_table=src_vocab_table,
        target_vocab_table=tgt_vocab_table,
        scope=scope)
  return (eval_graph, eval_model, eval_src_file_placeholder,
          eval_tgt_file_placeholder, eval_iterator)


def train(hparams, scope=None):
  """Train a translation model."""
  log_device_placement = hparams.log_device_placement
  out_dir = hparams.out_dir
  num_epochs = hparams.num_epochs
  batch_size = hparams.batch_size
  bpe_delimiter = hparams.bpe_delimiter
  steps_per_checkpoint = hparams.steps_per_checkpoint
  steps_per_external_eval = hparams.steps_per_external_eval
  steps_per_eval = 10 * steps_per_checkpoint

  # Load data
  #(train_set, train_total_size, train_bucket_sizes,
  # dev_batches, test_batches, infer_dev_batches, infer_test_batches) = (
  #     load_all_data(hparams))

  test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
  test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
  dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
  dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)

  if not hparams.attention:
    model_creator = nmt_model.Model
  elif hparams.attention_architecture == "top":
    model_creator = attention_model.AttentionModel
  else:
    raise ValueError("Unknown model architecture")

  assert hparams.vocab_prefix
  src_vocab_file = "%s.%s" % (hparams.vocab_prefix, hparams.src)
  tgt_vocab_file = "%s.%s" % (hparams.vocab_prefix, hparams.tgt)

  train_graph, train_model, train_iterator = create_train_model(
      model_creator, hparams, src_vocab_file, tgt_vocab_file, scope)
  (eval_graph, eval_model, eval_src_file_placeholder,
   eval_tgt_file_placeholder, eval_iterator) = create_eval_model(
       model_creator, hparams, src_vocab_file, tgt_vocab_file, scope)
  infer_graph, infer_model, infer_src_placeholder, infer_iterator = (
      inference.create_infer_model(model_creator, hparams, src_vocab_file,
                                   tgt_vocab_file, scope))

  # Log and output files
  log_file = os.path.join(out_dir, "log")
  log_f = tf.gfile.GFile(log_file, mode="w")
  utils.print_out("# log_file=%s" % log_file, log_f)

  avg_step_time = 0.0

  # TensorFlow model
  config_proto = utils.get_config_proto(
      log_device_placement=log_device_placement)

  server = tf.train.Server.create_local_server()
  train_sess = tf.Session(server.target, config=config_proto, graph=train_graph)
  eval_sess = tf.Session(server.target, config=config_proto, graph=eval_graph)
  infer_sess = tf.Session(server.target, config=config_proto, graph=infer_graph)

  with train_graph.as_default():
    model_helper.create_or_load_model(train_model, out_dir, train_sess, hparams)

  # Create/Load model.
  # TODO(thangluong): move this to create_or_load_model
  global_step = train_model.global_step.eval(session=train_sess)
  epoch = 0

  # Summary writer
  summary_writer = tf.summary.FileWriter(
      os.path.join(out_dir, "train_log"), train_graph)

  utils.print_out("Starting epoch 0")

  # First evaluation
  # dev_ppl = _internal_eval(
  #     model, sess, dev_iterator, "dev", global_step, summary_writer)
  # test_ppl = _internal_eval(
  #     model, sess, test_iterator, "test", global_step, summary_writer)
  # dev_scores = _external_eval(
  #     model,
  #     sess,
  #     hparams,
  #     infer_dev_iterator,
  #     "dev",
  #     global_step,
  #     summary_writer,
  #     save_on_best=True)
  # test_scores = _external_eval(
  #     model,
  #     sess,
  #     hparams,
  #     infer_test_iterator,
  #     "test",
  #     global_step,
  #     summary_writer,
  #     save_on_best=False)
  # all_dev_perplexities = [dev_ppl]
  # all_test_perplexities = [test_ppl]
  # all_steps = [global_step]

  # This is the training loop.
  step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0
  checkpoint_total_count = 0.0
  speed, train_ppl = 0.0, 0.0
  start_train_time = time.time()
  while epoch < num_epochs:
    # Get a batch and make a step.
    start_time = time.time()
    # utils.print_out("# Start epoch %d, step %d, lr %g, %s" %
    #                 (epoch, global_step, model.learning_rate.eval(),
    #                  time.ctime()), log_f)
    # utils.print_out("  sample train data:")
    # nmt_utils.print_translation(
    #     batch["encoder_inputs"][:, -1],
    #     batch["decoder_outputs"][:, -1],
    #     None,
    #     src_vocab, tgt_vocab, source_reverse,
    #     bpe_delimiter=bpe_delimiter
    # )

    # Initialize all of the iterators
    train_sess.run(train_iterator.initializer)
    model_step = 0
    while True:
      ### Run a step ###
      try:
        utils.print_out("Train step: %d.  Global step: %d"
                        % (model_step, global_step))
        step_result = train_model.train(train_sess)
        model_step += 1
      except tf.errors.OutOfRangeError:
        # Finished going through the training dataset.  Go to next epoch.
        utils.print_out("Finished epoch %d.  Step %d." % (epoch, model_step))
        epoch += 1
        break

      _, step_loss, _, step_summary, global_step = step_result

      # Write step summary.
      summary_writer.add_summary(step_summary, global_step)

      # update statistics
      step_time += (time.time() - start_time)

  #   checkpoint_loss += (step_loss * batch["size"])
  #   checkpoint_predict_count += step_predict_count
  #   checkpoint_total_count += np.sum(batch["encoder_lengths"])
  #   if "decoder_lengths" in batch:
  #     checkpoint_total_count += np.sum(batch["decoder_lengths"])
  #   global_step += 1

  # Once in a while, we save checkpoint, print statistics, and run evals.
    if global_step % steps_per_checkpoint == 0:
      # Print statistics for the previous epoch.
      avg_step_time = step_time / steps_per_checkpoint
      train_ppl = utils.safe_exp(checkpoint_loss / checkpoint_predict_count)
      speed = checkpoint_total_count / (1000 * step_time)
      utils.print_out("  epoch %d step %d lr %g "
                      "step-time %.2fs wps %.2fK ppl %.2f %s" %
                      (epoch,
                       global_step,
                       train_model.learning_rate.eval(session=train_sess),
                       avg_step_time, speed, train_ppl,
                       _get_best_results(hparams)), log_f)

      # Reset timer and loss.
      step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0
      checkpoint_total_count = 0.0

    # Evaluate on dev / test
    if global_step % steps_per_eval == 0:
      eval_sess.run(eval_iterator.initializer, feed_dict={
          eval_src_file_placeholder: test_src_file,
          eval_tgt_file_placeholder: test_tgt_file
      })
      # TODO(ebrevdo): run eval_graph.eval() until done
      eval_sess.run(eval_iterator.initializer, feed_dict={
          eval_src_file_placeholder: dev_src_file,
          eval_tgt_file_placeholder: dev_tgt_file
      })
      # TODO(ebrevdo): run eval_graph.eval() until done
      infer_sess.run(infer_iterator.initializer, feed_dict={
          infer_src_placeholder: [""] * 100,  # 100 empty string examples
      })
      # TODO(ebrevdo): fill in placeholder values above and run infer
      # graph until done.

    #   dev_ppl, test_ppl = _save_eval_decode(
    #       model, sess, hparams, train_ppl,
    #       dev_batches, test_batches, infer_dev_batches,
    #       global_step, summary_writer)
    #   all_dev_perplexities.append(dev_ppl)
    #   all_test_perplexities.append(test_ppl)
    #   all_steps.append(global_step)
    #   if math.isnan(dev_ppl): break

    # if global_step % steps_per_external_eval == 0:
    #   dev_scores, test_scores = _external_eval(
    #       model, sess, hparams, infer_dev_batches, infer_test_batches,
    #       global_step, summary_writer)

    #   # End of epoch
    #   if global_step % steps_per_epoch == 0:
    #     epoch += 1

    # # Done training
    # utils.print_out("# Best %s" % _get_best_results(hparams))
    # eval_results = _format_results("dev", dev_ppl, dev_scores, hparams.metrics)
    # if test_batches:
    #   eval_results += ", " + _format_results(
    #       "test", test_ppl, test_scores, hparams.metrics)
    # utils.print_out(
    #     "# Final, epoch %d step %d lr %g "
    #     "step-time %.2f wps %.2fK ppl %.2f, %s, %s" %
    #     (epoch, global_step, model.learning_rate.eval(),
    #      avg_step_time, speed, train_ppl, eval_results, time.ctime()), log_f)
    # summary_writer.close()
    # utils.print_time("# Done training!", start_train_time)

    # return all_dev_perplexities, all_test_perplexities, all_steps


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


def _save_eval_decode(model, sess, hparams, train_ppl,
                      dev_batches, test_batches, infer_dev_batches,
                      global_step, summary_writer):
  """Save checkpoint, compute perplexity, and sample decode."""
  utils.print_out("# Save eval decode, global step %d" % global_step)
  utils.add_summary(summary_writer, global_step, "train_ppl", train_ppl)

  # Save checkpoint
  model.saver.save(sess, os.path.join(hparams.out_dir, "translate.ckpt"),
                   global_step=model.global_step)

  # Internal evaluation
  dev_ppl, test_ppl = _internal_eval(
      model, sess, dev_batches, test_batches, global_step, summary_writer)

  # Sample decoding
  _sample_decode(model, sess, hparams, infer_dev_batches,
                 global_step, summary_writer)

  return dev_ppl, test_ppl


def _internal_eval(model, sess, iterator, label,
                   global_step, summary_writer):
  """Computing perplexity."""
  # Compute perplexity
  ppl = model_helper.compute_perplexity(model, sess, iterator, label)
  # Summary
  utils.add_summary(summary_writer, global_step, "%s_ppl" % label, ppl)

  return ppl


def _sample_decode(model, sess, hparams, infer_dev_batches,
                   global_step, summary_writer):
  """Pick a sentence and decode."""
  src_vocab = hparams.src_vocab
  tgt_vocab = hparams.tgt_vocab
  bpe_delimiter = hparams.bpe_delimiter
  source_reverse = hparams.source_reverse

  # Pick a sent to decode
  decode_id = random.randint(0, len(infer_dev_batches) - 1)
  utils.print_out("  # %d" % decode_id)
  sample_ids, attention_summary = model.decode(sess)

  # Print translation
  nmt_utils.print_translation(
      infer_dev_batches[decode_id]["encoder_inputs"][:, 0],
      infer_dev_batches[decode_id]["decoder_outputs"][:, 0],
      sample_ids,
      src_vocab, tgt_vocab,
      source_reverse,
      bpe_delimiter=bpe_delimiter
  )

  # Summary
  if attention_summary is not None:
    summary_writer.add_summary(attention_summary, global_step)


def _external_eval(model, sess, hparams, prefix,
                   iterator, label, global_step, summary_writer, save_on_best):
  """External evaluation such as BLEU and ROUGE scores."""
  decode = global_step > 0
  if decode:
    utils.print_out("# External evaluation, global step %d" % global_step)
  out_dir = hparams.out_dir
  tgt_eos_id = hparams.tgt_eos_id
  tgt_vocab = hparams.tgt_vocab
  bpe_delimiter = hparams.bpe_delimiter
  ignore_map = hparams.ignore_map

  # External evaluation on dev
  tgt = prefix + "." + hparams.tgt
  output = os.path.join(out_dir, "output_%s" % label)
  scores = nmt_utils.decode_and_evaluate(
      label,
      model, sess, output,
      iterator,
      tgt_vocab, tgt_eos_id,
      hparams.metrics,
      bpe_delimiter=bpe_delimiter,
      ignore_map=ignore_map,
      decode=decode,
  )
  # Save on best metrics
  if decode:
    for metric in hparams.metrics:
      utils.add_summary(summary_writer, global_step, "%s_%s" % (label, metric),
                        scores[metric])
      # TODO(thangluong): make clear the semantic, larger is better
      if save_on_best and scores[metric] > getattr(hparams, "best_" + metric):
        setattr(hparams, "best_" + metric, scores[metric])
        model.saver.save(
            sess, os.path.join(getattr(hparams, "best_" + metric + "_dir"),
                               "translate.ckpt"),
            global_step=model.global_step)
    utils.save_hparams(out_dir, hparams)
  return scores
