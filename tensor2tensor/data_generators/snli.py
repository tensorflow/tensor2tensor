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

"""Data generators for the SNLI data-set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import zipfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import tokenizer

import tensorflow as tf

_EOS = 1
_SEP = 2

_LABEL_INDEX = 0
_PARSE1_INDEX = 3
_PARSE2_INDEX = 4
_SENT1_INDEX = 5
_SENT2_INDEX = 6

_LABEL_TO_ID = {
    'contradiction': 0,
    'entailment': 1,
    'neutral': 2,
}

_EXAMPLES_FILE = 'examples.txt'
_SNLI_DATA_PATH = 'snli_1.0/snli_1.0_%s.txt'
_SNLI_ZIP = 'snli_1.0.zip'
_SNLI_URL = 'https://nlp.stanford.edu/projects/snli/' + _SNLI_ZIP


def _download_and_parse_dataset(tmp_dir, train):
  """Downloads and prepairs the dataset to be parsed by the data_generator."""
  file_path = generator_utils.maybe_download(tmp_dir, _SNLI_ZIP, _SNLI_URL)
  zip_ref = zipfile.ZipFile(file_path, 'r')
  zip_ref.extractall(tmp_dir)
  zip_ref.close()

  file_name = 'train' if train else 'dev'
  dataset_file_path = os.path.join(tmp_dir, _SNLI_DATA_PATH % file_name)
  _parse_dataset(dataset_file_path, tmp_dir, train)


def _get_tokens_and_tags(parse_str):
  """Parse str to tokens and pos tags."""
  tokens = []
  parse_split = parse_str.split(' ')
  for p in parse_split:
    assert p.startswith('(') or p.endswith(')')
    if p.endswith(')'):
      token = p.replace(')', '')
      tokens.append(token)

  return tokens


def _parse_dataset(file_path, tmp_dir, train):
  """Convert the dataset in to a simpler format.

  This function creates two files. One for being processed to produce a vocab
  and another to generate the data.

  Args:
    file_path: string, path to the file to parse.
    tmp_dir: string, path to the directory to output the files.
    train: bool, indicating if we are parsing the training set.
  """
  input_path = file_path
  file_name = 'train' if train else 'dev'
  gen_output_path = os.path.join(tmp_dir, file_name + '.txt')
  example_output_path = os.path.join(tmp_dir, _EXAMPLES_FILE)

  print('input path: ' + input_path)
  print('gen_output_path: ' + gen_output_path)
  print('example_output_path: ' + example_output_path)

  input_file = tf.gfile.Open(input_path, mode='r')
  examples = []
  for counter, line in enumerate(input_file):
    if counter == 0:  # Ignore first line since its a header.
      continue
    # Get the token and embedding vector.
    line_split = line.split('\t')

    parse1 = line_split[_PARSE1_INDEX]
    parse2 = line_split[_PARSE2_INDEX]
    consensus_label = line_split[_LABEL_INDEX]

    tokens1 = _get_tokens_and_tags(parse1)
    tokens2 = _get_tokens_and_tags(parse2)

    tokens1_str = ' '.join(tokens1)
    tokens2_str = ' '.join(tokens2)

    if consensus_label != '-':
      examples.append([tokens1_str, tokens2_str, consensus_label])

  input_file.close()

  # Output tab delimited file of lines of examples (sentence1, sentence2, label)
  with tf.gfile.GFile(gen_output_path, 'w') as f:
    for tokens1_str, tokens2_str, consensus_label in examples:
      f.write('%s\t%s\t%s\n' % (tokens1_str, tokens2_str, consensus_label))

  if train:
    # Output file containing all the sentences for generating the vocab from.
    with tf.gfile.GFile(example_output_path, 'w') as f:
      for tokens1_str, tokens2_str, consensus_label in examples:
        f.write('%s %s\n' % (tokens1_str, tokens2_str))


def _get_or_generate_vocab(tmp_dir, vocab_filename, vocab_size):
  """Read or create vocabulary."""
  vocab_filepath = os.path.join(tmp_dir, vocab_filename)
  print('Vocab file written to: ' + vocab_filepath)

  if tf.gfile.Exists(vocab_filepath):
    gs = text_encoder.SubwordTextEncoder(vocab_filepath)
    return gs
  example_file = os.path.join(tmp_dir, _EXAMPLES_FILE)
  gs = text_encoder.SubwordTextEncoder()
  token_counts = tokenizer.corpus_token_counts(
      example_file, corpus_max_lines=1000000)
  gs = gs.build_to_target_size(
      vocab_size, token_counts, min_val=1, max_val=1e3)
  gs.store_to_file(vocab_filepath)
  return gs


def snli_token_generator(tmp_dir, train, vocab_size):
  _download_and_parse_dataset(tmp_dir, train)

  symbolizer_vocab = _get_or_generate_vocab(
      tmp_dir, 'vocab.subword_text_encoder', vocab_size)

  file_name = 'train' if train else 'dev'
  data_file = os.path.join(tmp_dir, file_name + '.txt')
  with tf.gfile.GFile(data_file, mode='r') as f:
    for line in f:
      sent1, sent2, label = line.strip().split('\t')
      sent1_enc = symbolizer_vocab.encode(sent1)
      sent2_enc = symbolizer_vocab.encode(sent2)

      inputs = sent1_enc + [_SEP] + sent2_enc + [_EOS]
      yield {
          'inputs': inputs,
          'targets': [_LABEL_TO_ID[label]],
      }
