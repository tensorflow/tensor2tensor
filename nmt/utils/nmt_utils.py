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

"""Utility functions specifically for NMT."""
from __future__ import print_function

import time

import tensorflow as tf

import utils.evaluation_utils as evaluation_utils
import utils.misc_utils as utils

__all__ = ["decode_and_evaluate", "get_translation", "print_translation"]


def decode_and_evaluate(name,
                        model,
                        sess,
                        trans_file,
                        ref_file,
                        hparams,
                        decode=True):
  """Decode a test set and compute a score according to the evaluation task."""
  metrics = hparams.metrics
  bpe_delimiter = hparams.bpe_delimiter
  ignore_map = hparams.ignore_map

  # Decode
  if decode:
    utils.print_out("  decoding to output %s." % trans_file)

    start_time = time.time()
    num_sentences = 0
    with tf.gfile.GFile(trans_file, mode="w") as trans_f:
      trans_f.write("")  # Write empty string to ensure file is created.

      while True:
        try:
          nmt_outputs, _ = model.decode(sess)
          num_sentences += len(nmt_outputs)
          print("num output", len(nmt_outputs))
          for sent_id in range(len(nmt_outputs)):
            translation = get_translation(nmt_outputs, sent_id, hparams)
            trans_f.write("%s\n" % translation)
        except tf.errors.OutOfRangeError:
          utils.print_time("  done, num sentences %d" % num_sentences,
                           start_time)
          break

  # Evaluation
  #  Fix this for inference during training.
  evaluation_scores = {}
  if ref_file and tf.gfile.Exists(trans_file):
    for metric in metrics:
      score = evaluation_utils.evaluate(
          ref_file,
          trans_file,
          metric,
          ignore_map=ignore_map,
          bpe_delimiter=bpe_delimiter)
      evaluation_scores[metric] = score
      utils.print_out("  %s %s: %.1f" % (metric, name, score))

  return evaluation_scores


def get_translation(nmt_outputs, sent_id, hparams):
  """Given batch decoding outputs, select a sentence and turn to text."""
  tgt_vocab = hparams.tgt_vocab
  tgt_eos_id = hparams.tgt_eos_id
  bpe_delimiter = hparams.bpe_delimiter
  ignore_map = hparams.ignore_map

  # Select a sentence
  if hparams.task == "seq2label":
    if hasattr(nmt_outputs, "__len__"):  # for numpy array
      output = nmt_outputs[sent_id]
    else:  # single-sent batch
      output = nmt_outputs
  elif hparams.task == "seq2seq":
    output = nmt_outputs[sent_id, :].tolist()

    # If there is an eos symbol in outputs, cut them at that point.
    if tgt_eos_id and tgt_eos_id in output:
      output = output[:output.index(tgt_eos_id)]
  else:
    raise ValueError("Unknown task %s" % hparams.task)

  # Turn integers into text
  if not bpe_delimiter:
    translation = utils.int2text(
        output, tgt_vocab, ignore_map=ignore_map)
  else:  # BPE
    translation = utils.bpe_int2text(
        output, tgt_vocab,
        ignore_map=ignore_map,
        delimiter=bpe_delimiter)

  return translation


def print_translation(batch, sent_id, nmt_outputs, src_vocab, tgt_vocab,
                      hparams, bpe_delimiter=None):
  """Print translation in text format (sent_id=-1 means last)."""
  # src
  src = batch["encoder_inputs"][:, sent_id]
  if hparams.source_reverse:
    utils.print_out(
        "    src_reverse: %s" % utils.int2text(reversed(src), src_vocab))
  else:
    utils.print_out("    src: %s" % utils.int2text(src, src_vocab))

  # ref
  if "decoder_outputs" in batch:
    if hparams.task == "seq2label":
      ref = batch["decoder_outputs"][sent_id]
    elif hparams.task == "seq2seq":
      ref = batch["decoder_outputs"][:, sent_id]
    else:
      raise ValueError("Unknown task %s" % hparams.task)

    utils.print_out("    ref: %s" % utils.int2text(ref, tgt_vocab))

  if nmt_outputs is not None:
    # BPE
    if bpe_delimiter:
      assert hparams.task == "seq2seq"
      nmt = nmt_outputs[sent_id, :].tolist()
      utils.print_out("    bpe: %s" % utils.int2text(nmt, tgt_vocab))

    translation = get_translation(nmt_outputs, sent_id, hparams)
    utils.print_out("    nmt: %s" % translation)
