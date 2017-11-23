"""This file extends the TF-NMT architecture by the alternating stacked encoder

https://arxiv.org/abs/1606.04199
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import gnmt_model
from . import model_helper

__all__ = ["AlternatingEncoderModel"]

class AlternatingEncoderModel(gnmt_model.GNMTEncoderModel):
  """This model supports various encoder types. We use the terminology
  from https://arxiv.org/abs/1707.07631
    - uni: Unidirectional encoder
    - bi: Bidirectional "deep transition" encoder
    - gnmt: GNMT-style encoder "biunidirectional stacked encoder"
    - alternating: Alternating stacked encoder
  """

  def _build_encoder(self, hparams):
    """Build an (alternating) encoder."""
    if hparams.encoder_type != "alternating":
      return super(AlternatingEncoderModel, self)._build_encoder(hparams)
    source = self.iterator.source
    if self.time_major:
      source = tf.transpose(source)
    left_outputs = source
    right_outputs = source 
    encoder_state = []
    for i in xrange(hparams.num_layers):
      with tf.variable_scope("alternating_layer_%d" % i):
        residual = i < hparams.num_residual_layers
        with tf.variable_scope("left"):
          left_outputs, left_state = self._encode_single(
            left_outputs, hparams, reverse=i % 2 == 1, residual=residual)
        with tf.variable_scope("right"):
          right_outputs, right_state = self._encode_single(
            right_outputs, hparams, reverse=i % 2 == 0, residual=residual)
        encoder_state.append(left_state)
        #encoder_state.append(left_state if i % 4 < 2 else right_state)
    outputs = tf.concat([left_outputs, right_outputs], axis=2)
    return outputs, tuple(encoder_state)

  def _encode_single(self, inputs, hparams, reverse=False, residual=True):
    sequence_length = self.iterator.source_sequence_length
    cell = self._build_shallow_encoder_cell(hparams, residual)
    if reverse:
      # This is adapted from the implementation of
      # tf.nn.bidirectional_dynamic_rnn
      if not self.time_major:
        time_dim = 1
        batch_dim = 0
      else:
        time_dim = 0
        batch_dim = 1
      def _reverse(input_, seq_lengths, seq_dim, batch_dim):
        return tf.reverse_sequence(
              input=input_, seq_lengths=seq_lengths,
              seq_dim=seq_dim, batch_dim=batch_dim)
      inputs_reverse = _reverse(
          inputs, seq_lengths=sequence_length,
          seq_dim=time_dim, batch_dim=batch_dim)
      tmp, encoder_state =  tf.nn.dynamic_rnn(
          cell, inputs_reverse, sequence_length=sequence_length,
          time_major=self.time_major, dtype=tf.float32)
      encoder_outputs = _reverse(
          tmp, seq_lengths=sequence_length,
          seq_dim=time_dim, batch_dim=batch_dim)
    else:
      # Encoding in forward direction.
      encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
          cell,
          inputs,
          sequence_length=sequence_length,
          time_major=self.time_major, dtype=tf.float32)
    return encoder_outputs, encoder_state

  def _build_shallow_encoder_cell(self, hparams, residual=True):
    return model_helper.create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=hparams.num_units,
        num_layers=1,
        num_residual_layers=1 if residual else 0,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=hparams.num_gpus,
        base_gpu=1,
        mode=self.mode,
        single_cell_fn=self.single_cell_fn)
