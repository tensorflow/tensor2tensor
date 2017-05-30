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

"""Attention-based sequence-to-sequence model with dynamic RNN support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import model
from . import model_helper

__all__ = ["AttentionModel"]


class AttentionModel(model.Model):
  """Sequence-to-sequence dynamic model with attention.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  (Luong et al., EMNLP'2015) paper: https://arxiv.org/pdf/1508.04025v5.pdf.
  This class also allows to use GRU cells in addition to LSTM cells with
  support for dropout.
  """

  def __init__(self, hparams, mode, iterator,
               source_vocab_table, target_vocab_table, scope=None):
    super(AttentionModel, self).__init__(
        hparams=hparams,
        mode=mode,
        iterator=iterator,
        source_vocab_table=source_vocab_table,
        target_vocab_table=target_vocab_table,
        scope=scope)
    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_summary = (
          _create_attention_images_summary(self.final_context_state,
                                           hparams.alignment_history))

  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Build a RNN cell with attention mechanism that can be used by decoder."""
    attention_option = hparams.attention
    attention_architecture = hparams.attention_architecture

    if attention_architecture != "standard":
      raise ValueError(
          "Unknown attention architecture %s" % attention_architecture)

    num_units = hparams.num_units
    num_layers = hparams.num_layers
    num_residual_layers = hparams.num_residual_layers
    num_gpus = hparams.num_gpus

    dtype = tf.float32

    attention_mechanism = create_attention_mechanism(
        attention_option, num_units, encoder_outputs, self.time_major,
        source_sequence_length)

    cell = model_helper.create_rnn_cell(
        hparams, num_layers, num_residual_layers, self.mode)

    cell = tf.contrib.seq2seq.AttentionWrapper(
        cell,
        attention_mechanism,
        attention_layer_size=num_units,
        alignment_history=hparams.alignment_history,
        name="attention")

    # TODO(thangluong): do we need num_layers, num_gpus?
    cell = tf.contrib.rnn.DeviceWrapper(
        cell, model_helper.get_device_str(num_layers - 1, num_gpus))

    batch_size = self.get_batch_size(encoder_outputs)
    decoder_initial_state = cell.zero_state(batch_size, dtype).clone(
        cell_state=encoder_state)

    return cell, decoder_initial_state


def create_attention_mechanism(attention_option, num_units, encoder_outputs,
                               time_major, source_sequence_length):
  """Create attention mechanism based on the attention_option."""
  if time_major:
    attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
  else:
    attention_states = encoder_outputs

  # Mechanism
  if attention_option == "luong":
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units,
        attention_states,
        memory_sequence_length=source_sequence_length)
  elif attention_option == "scaled_luong":
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units,
        attention_states,
        memory_sequence_length=source_sequence_length,
        scale=True)
  elif attention_option == "bahdanau":
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units,
        attention_states,
        memory_sequence_length=source_sequence_length)
  elif attention_option == "normed_bahdanau":
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units,
        attention_states,
        memory_sequence_length=source_sequence_length,
        normalize=True)
  else:
    raise ValueError("Unknown attention option %s" % attention_option)

  return attention_mechanism


def _create_attention_images_summary(final_context_state, alignment_history):
  """create attention image and attention summary."""
  if alignment_history:
    attention_images = (
        final_context_state.alignment_history.stack())
  else:
    attention_images = tf.zeros([1, 1, 1])

  # Reshape to (batch, src_seq_len, tgt_seq_len,1)
  attention_images = tf.expand_dims(
      tf.transpose(attention_images, [1, 2, 0]), -1)
  # Scale to range [0, 255]
  attention_images *= 255
  attention_summary = tf.summary.image("attention_images", attention_images)
  return attention_summary
