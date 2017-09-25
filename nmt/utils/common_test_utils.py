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

"""Common utility functions for tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import lookup_ops

from ..utils import iterator_utils


def create_test_hparams(unit_type="lstm",
                        encoder_type="uni",
                        num_layers=4,
                        attention="",
                        attention_architecture=None,
                        use_residual=False,
                        inference_indices=None,
                        num_translations_per_input=1,
                        beam_width=0,
                        init_op="uniform"):
  """Create training and inference test hparams."""
  num_residual_layers = 0
  if use_residual:
    # TODO(rzhao): Put num_residual_layers computation logic into
    # `model_utils.py`, so we can also test it here.
    num_residual_layers = 2

  return tf.contrib.training.HParams(
      # Networks
      num_units=5,
      num_layers=num_layers,
      dropout=0.5,
      unit_type=unit_type,
      encoder_type=encoder_type,
      num_residual_layers=num_residual_layers,
      time_major=True,
      num_embeddings_partitions=0,

      # Attention mechanisms
      attention=attention,
      attention_architecture=attention_architecture,
      pass_hidden_state=True,

      # Train
      optimizer="sgd",
      init_op=init_op,
      init_weight=0.1,
      max_gradient_norm=5.0,
      max_emb_gradient_norm=None,
      learning_rate=1.0,
      learning_rate_warmup_steps=0,
      learning_rate_warmup_factor=1.0,
      start_decay_step=0,
      decay_factor=0.98,
      decay_steps=100,
      colocate_gradients_with_ops=True,
      batch_size=128,
      num_buckets=5,

      # Infer
      tgt_max_len_infer=100,
      infer_batch_size=32,
      beam_width=beam_width,
      length_penalty_weight=0.0,
      num_translations_per_input=num_translations_per_input,

      # Misc
      forget_bias=0.0,
      num_gpus=1,
      share_vocab=False,
      random_seed=3,

      # Vocab
      src_vocab_size=5,
      tgt_vocab_size=5,
      eos="eos",
      sos="sos",

      # For inference.py test
      source_reverse=False,
      bpe_delimiter="@@",
      subword_option="bpe",
      src="src",
      tgt="tgt",
      src_max_len=400,
      tgt_eos_id=0,
      # TODO(rzhao): Remove this after adding in-graph id to string lookup.
      tgt_vocab=["eos", "test1", "test2", "test3", "test4", "test5"],
      src_max_len_infer=None,
      inference_indices=inference_indices,
      metrics=["bleu"],
  )


def create_test_iterator(hparams, mode):
  """Create test iterator."""
  src_vocab_table = lookup_ops.index_table_from_tensor(
      tf.constant([hparams.eos, "a", "b", "c", "d"]))
  tgt_vocab_mapping = tf.constant([hparams.sos, hparams.eos, "a", "b", "c"])
  tgt_vocab_table = lookup_ops.index_table_from_tensor(tgt_vocab_mapping)
  if mode == tf.contrib.learn.ModeKeys.INFER:
    reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_tensor(
        tgt_vocab_mapping)

  src_dataset = tf.contrib.data.Dataset.from_tensor_slices(
      tf.constant(["a a b b c", "a b b"]))

  if mode != tf.contrib.learn.ModeKeys.INFER:
    tgt_dataset = tf.contrib.data.Dataset.from_tensor_slices(
        tf.constant(["a b c b c", "a b c b"]))
    return (
        iterator_utils.get_iterator(
            src_dataset=src_dataset,
            tgt_dataset=tgt_dataset,
            src_vocab_table=src_vocab_table,
            tgt_vocab_table=tgt_vocab_table,
            batch_size=hparams.batch_size,
            sos=hparams.sos,
            eos=hparams.eos,
            source_reverse=hparams.source_reverse,
            random_seed=hparams.random_seed,
            num_buckets=hparams.num_buckets),
        src_vocab_table,
        tgt_vocab_table)
  else:
    return (
        iterator_utils.get_infer_iterator(
            src_dataset=src_dataset,
            src_vocab_table=src_vocab_table,
            eos=hparams.eos,
            source_reverse=hparams.source_reverse,
            batch_size=hparams.batch_size),
        src_vocab_table,
        tgt_vocab_table,
        reverse_tgt_vocab_table)
