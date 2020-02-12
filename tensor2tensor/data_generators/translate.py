# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tarfile
import zipfile
from tensor2tensor.data_generators import cleaner_en_xx
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import bleu_hook
from tensor2tensor.utils import contrib
from tensor2tensor.utils import mlperf_log

import tensorflow.compat.v1 as tf


class TranslateProblem(text_problems.Text2TextProblem):
  """Base class for translation problems."""

  @property
  def is_generate_per_split(self):
    return True

  @property
  def approx_vocab_size(self):
    return 2**15

  @property
  def datatypes_to_clean(self):
    return None

  def source_data_files(self, dataset_split):
    """Files to be passed to compile_data."""
    raise NotImplementedError()

  def vocab_data_files(self):
    """Files to be passed to get_or_generate_vocab."""
    return self.source_data_files(problem.DatasetSplit.TRAIN)

  def generate_samples(
      self,
      data_dir,
      tmp_dir,
      dataset_split,
      custom_iterator=text_problems.text2text_txt_iterator):
    datasets = self.source_data_files(dataset_split)
    tag = "dev"
    datatypes_to_clean = None
    if dataset_split == problem.DatasetSplit.TRAIN:
      tag = "train"
      datatypes_to_clean = self.datatypes_to_clean
    data_path = compile_data(
        tmp_dir, datasets, "%s-compiled-%s" % (self.name, tag),
        datatypes_to_clean=datatypes_to_clean)

    return custom_iterator(data_path + ".lang1", data_path + ".lang2")

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    return generator_utils.generate_lines_for_vocab(tmp_dir,
                                                    self.vocab_data_files())

  @property
  def decode_hooks(self):
    return [compute_bleu_summaries]


def compute_bleu_summaries(hook_args):
  """Compute BLEU core summaries using the decoder output.

  Args:
    hook_args: DecodeHookArgs namedtuple
  Returns:
    A list of tf.Summary values if hook_args.hparams contains the
    reference file and the translated file.
  """
  decode_hparams = hook_args.decode_hparams

  if not (decode_hparams.decode_reference and decode_hparams.decode_to_file):
    return None

  values = []
  bleu = 100 * bleu_hook.bleu_wrapper(
      decode_hparams.decode_reference, decode_hparams.decode_to_file)
  values.append(tf.Summary.Value(tag="BLEU", simple_value=bleu))
  tf.logging.info("%s: BLEU = %6.2f" % (decode_hparams.decode_to_file, bleu))
  if hook_args.hparams.mlperf_mode:
    current_step = decode_hparams.mlperf_decode_step
    mlperf_log.transformer_print(
        key=mlperf_log.EVAL_TARGET, value=decode_hparams.mlperf_threshold)
    mlperf_log.transformer_print(
        key=mlperf_log.EVAL_ACCURACY,
        value={
            "epoch": max(current_step // decode_hparams.iterations_per_loop - 1,
                         0),
            "value": bleu
        })
    mlperf_log.transformer_print(key=mlperf_log.EVAL_STOP)

  if bleu >= decode_hparams.mlperf_threshold:
    decode_hparams.set_hparam("mlperf_success", True)

  return values


def _preprocess_sgm(line, is_sgm):
  """Preprocessing to strip tags in SGM files."""
  if not is_sgm:
    return line
  # In SGM files, remove <srcset ...>, <p>, <doc ...> lines.
  if line.startswith("<srcset") or line.startswith("</srcset"):
    return ""
  if line.startswith("<doc") or line.startswith("</doc"):
    return ""
  if line.startswith("<p>") or line.startswith("</p>"):
    return ""
  # Strip <seg> tags.
  line = line.strip()
  if line.startswith("<seg") and line.endswith("</seg>"):
    i = line.index(">")
    return line[i + 1:-6]  # Strip first <seg ...> and last </seg>.


def _clean_sentences(sentence_pairs):
  res_pairs = []
  for cleaned in cleaner_en_xx.clean_en_xx_pairs(sentence_pairs):
    res_pairs.append(cleaned)
  return res_pairs


def _tmx_to_source_target(tmx_file, source_resfile, target_resfile,
                          do_cleaning=False):
  source_target_pairs = cleaner_en_xx.paracrawl_v3_pairs(tmx_file)
  if do_cleaning:
    source_target_pairs = cleaner_en_xx.clean_en_xx_pairs(source_target_pairs)
  for source, target in source_target_pairs:
    source_resfile.write(source)
    source_resfile.write("\n")
    target_resfile.write(target)
    target_resfile.write("\n")


def compile_data(tmp_dir, datasets, filename, datatypes_to_clean=None):
  """Concatenates all `datasets` and saves to `filename`."""
  datatypes_to_clean = datatypes_to_clean or []
  filename = os.path.join(tmp_dir, filename)
  lang1_fname = filename + ".lang1"
  lang2_fname = filename + ".lang2"
  if tf.gfile.Exists(lang1_fname) and tf.gfile.Exists(lang2_fname):
    tf.logging.info("Skipping compile data, found files:\n%s\n%s", lang1_fname,
                    lang2_fname)
    return filename
  with tf.gfile.GFile(lang1_fname, mode="w") as lang1_resfile:
    with tf.gfile.GFile(lang2_fname, mode="w") as lang2_resfile:
      for dataset in datasets:
        url = dataset[0]
        compressed_filename = os.path.basename(url)
        compressed_filepath = os.path.join(tmp_dir, compressed_filename)
        if url.startswith("http"):
          generator_utils.maybe_download(tmp_dir, compressed_filename, url)
        if compressed_filename.endswith(".zip"):
          zipfile.ZipFile(os.path.join(compressed_filepath),
                          "r").extractall(tmp_dir)

        if dataset[1][0] == "tmx":
          cleaning_requested = "tmx" in datatypes_to_clean
          tmx_filename = os.path.join(tmp_dir, dataset[1][1])
          if tmx_filename.endswith(".gz"):
            with gzip.open(tmx_filename, "rb") as tmx_file:
              _tmx_to_source_target(tmx_file, lang1_resfile, lang2_resfile,
                                    do_cleaning=cleaning_requested)
          else:
            with tf.gfile.Open(tmx_filename) as tmx_file:
              _tmx_to_source_target(tmx_file, lang1_resfile, lang2_resfile,
                                    do_cleaning=cleaning_requested)

        elif dataset[1][0] == "tsv":
          _, src_column, trg_column, glob_pattern = dataset[1]
          filenames = tf.gfile.Glob(os.path.join(tmp_dir, glob_pattern))
          if not filenames:
            # Capture *.tgz and *.tar.gz too.
            mode = "r:gz" if compressed_filepath.endswith("gz") else "r"
            with tarfile.open(compressed_filepath, mode) as corpus_tar:
              corpus_tar.extractall(tmp_dir)
            filenames = tf.gfile.Glob(os.path.join(tmp_dir, glob_pattern))
          for tsv_filename in filenames:
            if tsv_filename.endswith(".gz"):
              new_filename = tsv_filename.strip(".gz")
              generator_utils.gunzip_file(tsv_filename, new_filename)
              tsv_filename = new_filename
            with tf.gfile.Open(tsv_filename) as tsv_file:
              for line in tsv_file:
                if line and "\t" in line:
                  parts = line.split("\t")
                  source, target = parts[src_column], parts[trg_column]
                  source, target = source.strip(), target.strip()
                  clean_pairs = [(source, target)]
                  if "tsv" in datatypes_to_clean:
                    clean_pairs = cleaner_en_xx.clean_en_xx_pairs(clean_pairs)
                  for source, target in clean_pairs:
                    if source and target:
                      lang1_resfile.write(source)
                      lang1_resfile.write("\n")
                      lang2_resfile.write(target)
                      lang2_resfile.write("\n")

        else:
          lang1_filename, lang2_filename = dataset[1]
          lang1_filepath = os.path.join(tmp_dir, lang1_filename)
          lang2_filepath = os.path.join(tmp_dir, lang2_filename)
          is_sgm = (
              lang1_filename.endswith("sgm") and lang2_filename.endswith("sgm"))

          if not (tf.gfile.Exists(lang1_filepath) and
                  tf.gfile.Exists(lang2_filepath)):
            # For .tar.gz and .tgz files, we read compressed.
            mode = "r:gz" if compressed_filepath.endswith("gz") else "r"
            with tarfile.open(compressed_filepath, mode) as corpus_tar:
              corpus_tar.extractall(tmp_dir)
          if lang1_filepath.endswith(".gz"):
            new_filepath = lang1_filepath.strip(".gz")
            generator_utils.gunzip_file(lang1_filepath, new_filepath)
            lang1_filepath = new_filepath
          if lang2_filepath.endswith(".gz"):
            new_filepath = lang2_filepath.strip(".gz")
            generator_utils.gunzip_file(lang2_filepath, new_filepath)
            lang2_filepath = new_filepath

          for example in text_problems.text2text_txt_iterator(
              lang1_filepath, lang2_filepath):
            line1res = _preprocess_sgm(example["inputs"], is_sgm)
            line2res = _preprocess_sgm(example["targets"], is_sgm)
            clean_pairs = [(line1res, line2res)]
            if "txt" in datatypes_to_clean:
              clean_pairs = cleaner_en_xx.clean_en_xx_pairs(clean_pairs)
            for line1res, line2res in clean_pairs:
              if line1res and line2res:
                lang1_resfile.write(line1res)
                lang1_resfile.write("\n")
                lang2_resfile.write(line2res)
                lang2_resfile.write("\n")

  return filename


class TranslateDistillProblem(TranslateProblem):
  """Base class for translation problems."""

  def is_generate_per_split(self):
    return True

  def example_reading_spec(self):
    data_fields = {"dist_targets": tf.VarLenFeature(tf.int64)}

    if self.has_inputs:
      data_fields["inputs"] = tf.VarLenFeature(tf.int64)

    # hack: ignoring true targets and putting dist_targets in targets
    data_items_to_decoders = {
        "inputs": contrib.slim().tfexample_decoder.Tensor("inputs"),
        "targets": contrib.slim().tfexample_decoder.Tensor("dist_targets"),
    }

    return (data_fields, data_items_to_decoders)

  def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
    """Get vocab for distill problems."""
    # We assume that vocab file is present in data_dir directory where the
    # data generated will be stored.
    vocab_filepath = os.path.join(data_dir, self.vocab_filename)
    encoder = text_encoder.SubwordTextEncoder(vocab_filepath)
    return encoder

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    vocab = self.get_or_create_vocab(data_dir, tmp_dir)
    # For each example, encode the text and append EOS ID.
    for sample in generator:
      if self.has_inputs:
        sample["inputs"] = vocab.encode(sample["inputs"])
        sample["inputs"].append(text_encoder.EOS_ID)
        sample["targets"] = vocab.encode(sample["targets"])
        sample["targets"].append(text_encoder.EOS_ID)
        sample["dist_targets"] = vocab.encode(sample["dist_targets"])
        sample["dist_targets"].append(text_encoder.EOS_ID)
        yield sample

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    data_path = self.source_data_files(dataset_split)
    assert tf.gfile.Exists(data_path)
    return text_problems.text2text_distill_iterator(data_path + "inputs",
                                                    data_path + "gold",
                                                    data_path + "prediction")
