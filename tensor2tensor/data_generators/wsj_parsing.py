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

"""Data generators for parsing data-sets."""

import os

# Dependency imports

from tensor2tensor.data_generators import generator_utils

import tensorflow as tf


def words_and_tags_from_wsj_tree(tree_string):
  """Generates linearized trees and tokens from the wsj tree format.

  It uses the linearized algorithm described in https://arxiv.org/abs/1412.7449.

  Args:
    tree_string: tree in wsj format

  Returns:
    tuple: (words, linearized tree)
  """
  stack, tags, words = [], [], []
  for tok in tree_string.strip().split():
    if tok[0] == "(":
      symbol = tok[1:]
      tags.append(symbol)
      stack.append(symbol)
    else:
      assert tok[-1] == ")"
      stack.pop()  # Pop the POS-tag.
      while tok[-2] == ")":
        tags.append("/" + stack.pop())
        tok = tok[:-1]
      words.append(tok[:-1])
  return str.join(" ", words), str.join(" ", tags[1:-1])  # Strip "TOP" tag.


def token_generator(tree_path, source_token_vocab, target_token_vocab,
                    eos=None):
  """Generator for parsing as a sequence-to-sequence task that uses tokens.

  This generator assumes the files at source_path and target_path have
  the same number of lines and yields dictionaries of "inputs" and "targets"
  where inputs and targets are token ids from source and taret lines
  converted to integers using the token_map.

  Args:
    tree_path: path to the file with wsj format trees, one per line.
    source_token_vocab: GenericVocabulary object for source vocabulary.
    target_token_vocab: GenericVocabulary object for target vocabulary.
    eos: integer to append at the end of each sequence (default: None).

  Yields:
    A dictionary {"inputs": source-line, "targets": target-line} where
    the lines are integer lists converted from tokens in the file lines.
  """
  eos_list = [] if eos is None else [eos]
  with tf.gfile.GFile(tree_path, mode="r") as tree_file:
    tree_line = tree_file.readline()
    while tree_line:
      source, target = words_and_tags_from_wsj_tree(tree_line)
      source_ints = source_token_vocab.encode(source.strip()) + eos_list
      target_ints = target_token_vocab.encode(target.strip()) + eos_list
      yield {"inputs": source_ints, "targets": target_ints}
      tree_line = tree_file.readline()


def parsing_token_generator(tmp_dir, train, source_vocab_size,
                            target_vocab_size):
  """Generator for parsing as a sequence-to-sequence task that uses tokens.

  This generator assumes the files parsing_{train,dev}.wsj, which contain trees
  in wsj format and wsj_{source,target}.tokens.vocab.<vocab_size> exist in
  tmp_dir.

  Args:
    tmp_dir: path to the file with source sentences.
    train: path to the file with target sentences.
    source_vocab_size: source vocab size.
    target_vocab_size: target vocab size.

  Returns:
    A generator to a dictionary of inputs and outputs.
  """
  source_symbolizer_vocab = generator_utils.get_or_generate_vocab(
      tmp_dir, "wsj_source.tokens.vocab.%d" % source_vocab_size,
      source_vocab_size)
  target_symbolizer_vocab = generator_utils.get_or_generate_vocab(
      tmp_dir, "wsj_target.tokens.vocab.%d" % target_vocab_size,
      target_vocab_size)
  filename = "parsing_%s.trees" % ("train" if train else "dev")
  tree_filepath = os.path.join(tmp_dir, filename)
  return token_generator(tree_filepath, source_symbolizer_vocab,
                         target_symbolizer_vocab, 1)
