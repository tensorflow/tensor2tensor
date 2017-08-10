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

# This module implements the ice_parsing_* problems, which
# parse plain text into flattened parse trees and POS tags.
# The training data is stored in files named `parsing_train.pairs`
# and `parsing_dev.pairs`. These files are UTF-8 text files where
# each line contains an input sentence and a target parse tree,
# separated by a tab character.

import os

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators.wmt import tabbed_generator
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer

import tensorflow as tf


# End-of-sentence marker.
EOS = text_encoder.EOS_ID


def tabbed_parsing_token_generator(data_dir, tmp_dir, train, prefix,
                                   source_vocab_size, target_vocab_size):
  """Generate source and target data from a single file."""
  filename = "parsing_{0}.pairs".format("train" if train else "dev")
  source_vocab = generator_utils.get_or_generate_tabbed_vocab(
      data_dir, tmp_dir, filename, 0,
      prefix + "_source.tokens.vocab.%d" % source_vocab_size, source_vocab_size)
  target_vocab = generator_utils.get_or_generate_tabbed_vocab(
      data_dir, tmp_dir, filename, 1,
      prefix + "_target.tokens.vocab.%d" % target_vocab_size, target_vocab_size)
  pair_filepath = os.path.join(tmp_dir, filename)
  return tabbed_generator(pair_filepath, source_vocab, target_vocab, EOS)


def tabbed_parsing_character_generator(tmp_dir, train):
  """Generate source and target data from a single file."""
  character_vocab = text_encoder.ByteTextEncoder()
  filename = "parsing_{0}.pairs".format("train" if train else "dev")
  pair_filepath = os.path.join(tmp_dir, filename)
  return tabbed_generator(pair_filepath, character_vocab, character_vocab, EOS)


@registry.register_problem("ice_parsing_tokens")
class IceParsingTokens(problem.Problem):
  """Problem spec for parsing tokenized Icelandic text to
    constituency trees, also tokenized but to a smaller vocabulary."""

  @property
  def source_vocab_size(self):
    return 2**14  # 16384

  @property
  def target_vocab_size(self):
    return 2**8  # 256

  def feature_encoders(self, data_dir):
    source_vocab_filename = os.path.join(
        data_dir, "ice_source.tokens.vocab.%d" % self.source_vocab_size)
    target_vocab_filename = os.path.join(
        data_dir, "ice_target.tokens.vocab.%d" % self.target_vocab_size)
    source_subtokenizer = text_encoder.SubwordTextEncoder(source_vocab_filename)
    target_subtokenizer = text_encoder.SubwordTextEncoder(target_vocab_filename)
    return {
        "inputs": source_subtokenizer,
        "targets": target_subtokenizer,
    }

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    generator_utils.generate_dataset_and_shuffle(
        tabbed_parsing_token_generator(data_dir, tmp_dir, True, "ice",
                                       self.source_vocab_size,
                                       self.target_vocab_size),
        self.training_filepaths(data_dir, 1, shuffled=False),
        tabbed_parsing_token_generator(data_dir, tmp_dir, False, "ice",
                                       self.source_vocab_size,
                                       self.target_vocab_size),
        self.dev_filepaths(data_dir, 1, shuffled=False))

  def hparams(self, defaults, model_hparams):
    p = defaults
    source_vocab_size = self._encoders["inputs"].vocab_size
    p.input_modality = {"inputs": (registry.Modalities.SYMBOL, source_vocab_size)}
    p.target_modality = (registry.Modalities.SYMBOL, self.target_vocab_size)
    p.input_space_id = problem.SpaceID.ICE_TOK
    p.target_space_id = problem.SpaceID.ICE_PARSE_TOK
    p.loss_multiplier = 2.5 # Rough estimate of avg number of tokens per word


@registry.register_hparams
def transformer_parsing_ice():
  """Hparams for parsing Icelandic text."""
  hparams = transformer.transformer_base_single_gpu()
  hparams.batch_size = 4096
  hparams.shared_embedding_and_softmax_weights = int(False)
  return hparams


@registry.register_hparams
def transformer_parsing_ice_big():
  """Hparams for parsing Icelandic text, bigger model."""
  hparams = transformer_parsing_ice()
  hparams.batch_size = 2048 # 4096 gives Out-of-memory on 8 GB 1080 GTX GPU
  hparams.attention_dropout = 0.05
  hparams.residual_dropout = 0.05
  hparams.max_length = 512
  hparams.hidden_size = 1024
  return hparams

