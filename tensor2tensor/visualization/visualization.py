# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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

"""Shared code for visualizing transformer attentions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np

# To register the hparams set
from tensor2tensor import models  # pylint: disable=unused-import
from tensor2tensor import problems
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib

import tensorflow as tf

EOS_ID = 1


class AttentionVisualizer(object):
  """Helper object for creating Attention visualizations."""

  def __init__(
      self, hparams_set, model_name, data_dir, problem_name, beam_size=1):
    inputs, targets, samples, att_mats = build_model(
        hparams_set, model_name, data_dir, problem_name, beam_size=beam_size)

    # Fetch the problem
    ende_problem = problems.problem(problem_name)
    encoders = ende_problem.feature_encoders(data_dir)

    self.inputs = inputs
    self.targets = targets
    self.att_mats = att_mats
    self.samples = samples
    self.encoders = encoders

  def encode(self, input_str):
    """Input str to features dict, ready for inference."""
    inputs = self.encoders['inputs'].encode(input_str) + [EOS_ID]
    batch_inputs = np.reshape(inputs, [1, -1, 1, 1])  # Make it 3D.
    return batch_inputs

  def decode(self, integers):
    """List of ints to str."""
    integers = list(np.squeeze(integers))
    return self.encoders['inputs'].decode(integers)

  def decode_list(self, integers):
    """List of ints to list of str."""
    integers = list(np.squeeze(integers))
    return self.encoders['inputs'].decode_list(integers)

  def get_vis_data_from_string(self, sess, input_string):
    """Constructs the data needed for visualizing attentions.

    Args:
      sess: A tf.Session object.
      input_string: The input setence to be translated and visulized.

    Returns:
      Tuple of (
          output_string: The translated sentence.
          input_list: Tokenized input sentence.
          output_list: Tokenized translation.
          att_mats: Tuple of attention matrices; (
              enc_atts: Encoder self attention weights.
                A list of `num_layers` numpy arrays of size
                (batch_size, num_heads, inp_len, inp_len)
              dec_atts: Decoder self attention weights.
                A list of `num_layers` numpy arrays of size
                (batch_size, num_heads, out_len, out_len)
              encdec_atts: Encoder-Decoder attention weights.
                A list of `num_layers` numpy arrays of size
                (batch_size, num_heads, out_len, inp_len)
          )
    """
    encoded_inputs = self.encode(input_string)

    # Run inference graph to get the translation.
    out = sess.run(self.samples, {
        self.inputs: encoded_inputs,
    })

    # Run the decoded translation through the training graph to get the
    # attention tensors.
    att_mats = sess.run(self.att_mats, {
        self.inputs: encoded_inputs,
        self.targets: np.reshape(out, [1, -1, 1, 1]),
    })

    output_string = self.decode(out)
    input_list = self.decode_list(encoded_inputs)
    output_list = self.decode_list(out)

    return output_string, input_list, output_list, att_mats


def build_model(hparams_set, model_name, data_dir, problem_name, beam_size=1):
  """Build the graph required to featch the attention weights.

  Args:
    hparams_set: HParams set to build the model with.
    model_name: Name of model.
    data_dir: Path to directory contatining training data.
    problem_name: Name of problem.
    beam_size: (Optional) Number of beams to use when decoding a traslation.
        If set to 1 (default) then greedy decoding is used.

  Returns:
    Tuple of (
        inputs: Input placeholder to feed in ids to be translated.
        targets: Targets placeholder to feed to translation when fetching
            attention weights.
        samples: Tensor representing the ids of the translation.
        att_mats: Tensors representing the attention weights.
    )
  """
  hparams = trainer_lib.create_hparams(
      hparams_set, data_dir=data_dir, problem_name=problem_name)
  translate_model = registry.model(model_name)(
      hparams, tf.estimator.ModeKeys.EVAL)

  inputs = tf.placeholder(tf.int32, shape=(1, None, 1, 1), name='inputs')
  targets = tf.placeholder(tf.int32, shape=(1, None, 1, 1), name='targets')
  translate_model({
      'inputs': inputs,
      'targets': targets,
  })

  # Must be called after building the training graph, so that the dict will
  # have been filled with the attention tensors. BUT before creating the
  # interence graph otherwise the dict will be filled with tensors from
  # inside a tf.while_loop from decoding and are marked unfetchable.
  att_mats = get_att_mats(translate_model)

  with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    samples = translate_model.infer({
        'inputs': inputs,
    }, beam_size=beam_size)['outputs']

  return inputs, targets, samples, att_mats


def get_att_mats(translate_model):
  """Get's the tensors representing the attentions from a build model.

  The attentions are stored in a dict on the Transformer object while building
  the graph.

  Args:
    translate_model: Transformer object to fetch the attention weights from.

  Returns:
  Tuple of attention matrices; (
      enc_atts: Encoder self attention weights.
        A list of `num_layers` numpy arrays of size
        (batch_size, num_heads, inp_len, inp_len)
      dec_atts: Decoder self attetnion weights.
        A list of `num_layers` numpy arrays of size
        (batch_size, num_heads, out_len, out_len)
      encdec_atts: Encoder-Decoder attention weights.
        A list of `num_layers` numpy arrays of size
        (batch_size, num_heads, out_len, inp_len)
  )
  """
  enc_atts = []
  dec_atts = []
  encdec_atts = []

  prefix = 'transformer/body/'
  postfix = '/multihead_attention/dot_product_attention'

  for i in range(translate_model.hparams.num_hidden_layers):
    enc_att = translate_model.attention_weights[
        '%sencoder/layer_%i/self_attention%s' % (prefix, i, postfix)]
    dec_att = translate_model.attention_weights[
        '%sdecoder/layer_%i/self_attention%s' % (prefix, i, postfix)]
    encdec_att = translate_model.attention_weights[
        '%sdecoder/layer_%i/encdec_attention%s' % (prefix, i, postfix)]
    enc_atts.append(enc_att)
    dec_atts.append(dec_att)
    encdec_atts.append(encdec_att)

  return enc_atts, dec_atts, encdec_atts
