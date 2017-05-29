"""Utility for evaluating various tasks, e.g., translation & summarization."""
import codecs
import os
import re
import subprocess

import tensorflow as tf

import scripts.bleu as bleu
import scripts.rouge as rouge


__all__ = ["evaluate"]


def evaluate(ref_file, trans_file, metric,
             ignore_map=None, bpe_delimiter=None):
  """Pick a metric and evaluate depending on task."""
  # BLEU scores for translation task
  if metric.lower() == "bleu":
    evaluation_score = _bleu(ref_file, trans_file,
                             ignore_map=ignore_map,
                             bpe_delimiter=bpe_delimiter)
  # ROUGE scores for summarization tasks
  elif metric.lower() == "rouge":
    evaluation_score = _rouge(ref_file, trans_file,
                              ignore_map=ignore_map,
                              bpe_delimiter=bpe_delimiter)
  elif metric.lower() == "accuracy":
    evaluation_score = _accuracy(ref_file, trans_file)
  else:
    raise ValueError("Unknown metric %s" % metric)

  return evaluation_score


def _clean(sentence, ignore_map, bpe_delimiter):
  """Clean, handle BPE delimiter, and ignore tokens in ignore map."""
  sentence = sentence.strip()

  # BPE
  if bpe_delimiter:
    sentence = re.sub(bpe_delimiter + " ", "", sentence)

  # Ignore map
  if ignore_map:
    tokens = []
    for token in sentence.strip().split(" "):
      if token not in ignore_map:
        tokens.append(token)
    sentence = " ".join(tokens)

  return sentence


# Follow //transconsole/localization/machine_translation/metrics/bleu_calc.py
def _bleu(ref_file, trans_file, ignore_map=None, bpe_delimiter=None):
  """Compute BLEU scores, ignoring tokens in ignore map and handling BPE."""
  max_order = 4
  smooth = False

  ref_files = [ref_file]
  reference_text = []
  for reference_filename in ref_files:
    with codecs.getreader("utf-8")(
        tf.gfile.GFile(reference_filename, "r")) as fh:
      reference_text.append(fh.readlines())

  per_segment_references = []
  for references in zip(*reference_text):
    reference_list = []
    for reference in references:
      reference = _clean(reference, ignore_map, bpe_delimiter)
      reference_list.append(reference.split(" "))
    per_segment_references.append(reference_list)

  translations = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "r")) as fh:
    for line in fh:
      line = _clean(line, ignore_map, bpe_delimiter)
      translations.append(line.split(" "))

  # bleu_score, precisions, bp, ratio, translation_length, reference_length
  bleu_score, _, _, _, _, _ = bleu.compute_bleu(
      per_segment_references, translations, max_order, smooth)
  return 100 * bleu_score


def _rouge(ref_file, summarization_file, ignore_map=None, bpe_delimiter=None):
  """Compute ROUGE scores, ignoring tokens in ignore map and handling BPE."""

  references = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(ref_file, "r")) as fh:
    for line in fh:
      references.append(_clean(line, ignore_map, bpe_delimiter))

  hypotheses = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(summarization_file, "r")) as fh:
    for line in fh:
      hypotheses.append(_clean(line, ignore_map, bpe_delimiter))

  rouge_score_map = rouge.rouge(hypotheses, references)
  return 100 * rouge_score_map["rouge_l/f_score"]


def _accuracy(label_file, pred_file):
  """Compute accuracy, each line contains a label."""

  with codecs.getreader("utf-8")(tf.gfile.GFile(label_file, "r")) as label_fh:
    with codecs.getreader("utf-8")(tf.gfile.GFile(pred_file, "r")) as pred_fh:
      count = 0.0
      match = 0.0
      for label in label_fh:
        label = label.strip()
        pred = pred_fh.readline().strip()
        if label == pred:
          match += 1
        count += 1
  return 100 * match / count


def _moses_bleu(multi_bleu_script, tgt_test, trans_file, bpe_delimiter=None):
  """Compute BLEU scores using Moses multi-bleu.perl script."""
  # BPE
  if bpe_delimiter:
    debpe_tgt_test = tgt_test + ".debpe"
    if not os.path.exists(debpe_tgt_test):
      #  not use shell=True, can be a security hazard
      subprocess.call("cp %s %s" % (tgt_test, debpe_tgt_test), shell=True)
      subprocess.call("sed s/%s //g %s" % (bpe_delimiter, debpe_tgt_test),
                      shell=True)
    tgt_test = debpe_tgt_test

  cmd = "%s %s < %s" % (multi_bleu_script, tgt_test, trans_file)

  # subprocess
  bleu_output = subprocess.check_output(cmd, shell=True)

  # extract BLEU score
  m = re.search("BLEU = (.+?),", bleu_output)
  bleu_score = float(m.group(1))

  return bleu_score
