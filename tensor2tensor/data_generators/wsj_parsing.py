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

"""Data generators for parsing data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
import tensorflow.compat.v1 as tf

flags.DEFINE_string("parsing_path", "", "Path to parsing files in tmp_dir.")


FLAGS = flags.FLAGS


@registry.register_problem
class WsjParsing(text_problems.Text2textTmpdir):
  """Generate vocabulary and training data for parsing.
  """

  # These files are used for vocab generation
  TRAIN_FILES = ("wsj.train.text.txt", "wsj.train.tags.txt")

  # These files are used for generating encoded samples
  TRAIN_FILES_TREE = "wsjTrain.trees"
  EVAL_FILES_TREE = "wsjEval.trees"

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del data_dir
    is_training = dataset_split == problem.DatasetSplit.TRAIN
    tree_file = self.TRAIN_FILES_TREE if is_training else self.EVAL_FILES_TREE
    tree_file_path = os.path.join(tmp_dir, tree_file)
    with tf.gfile.GFile(tree_file_path, mode="r") as cur_tree_file:
      for line in cur_tree_file:
        (words, tags) = words_and_tags_from_wsj_tree(line)
        yield {"inputs": words, "targets": tags}

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    encoder = self.get_or_create_vocab(data_dir, tmp_dir)
    return text_problems.text2text_generate_encoded(generator, encoder,
                                                    has_inputs=self.has_inputs)

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    files = [os.path.join(tmp_dir, f) for f in self.TRAIN_FILES]
    inputs_file, targets_file = files
    for i, sample in enumerate(text_problems.text2text_txt_iterator(inputs_file,
                                                                    targets_file
                                                                   )):
      yield sample["inputs"]
      yield sample["targets"]
      if self.max_samples_for_vocab and (i + 1) >= self.max_samples_for_vocab:
        break

  @property
  def max_samples_for_vocab(self):
    return 1000


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
  where inputs and targets are token ids from source and target lines
  converted to integers using the token_map.

  Args:
    tree_path: path to the file with WSJ format trees, one per line.
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


def parsing_token_generator(data_dir, tmp_dir, train, source_vocab_size,
                            target_vocab_size):
  """Generator for parsing as a sequence-to-sequence task that uses tokens.

  This generator assumes the files parsing_{train,dev}.trees, which contain
  trees in WSJ format.

  Args:
    data_dir: path to the data directory.
    tmp_dir: path to temporary storage directory.
    train: whether we're training or not.
    source_vocab_size: source vocab size.
    target_vocab_size: target vocab size.

  Returns:
    A generator to a dictionary of inputs and outputs.
  """
  # TODO(lukaszkaiser): Correct these calls to generate vocabularies. No data
  # sources are being passed.
  del (data_dir, tmp_dir, train, source_vocab_size, target_vocab_size)
  assert False, "Vocabulary generation not implemented"
  # source_symbolizer_vocab = generator_utils.get_or_generate_vocab(
  #     data_dir, tmp_dir, "wsj_source.vocab.%d" % source_vocab_size,
  #     source_vocab_size)
  # target_symbolizer_vocab = generator_utils.get_or_generate_vocab(
  #     data_dir, tmp_dir, "wsj_target.vocab.%d" % target_vocab_size,
  #     target_vocab_size)
  # filename = "%s_%s.trees" % (FLAGS.parsing_path, "train" if train else "dev")
  # tree_filepath = os.path.join(tmp_dir, filename)
  # return token_generator(tree_filepath, source_symbolizer_vocab,
  #                        target_symbolizer_vocab, 1)
