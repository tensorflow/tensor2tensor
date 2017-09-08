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

"""Module for postprocessing and displaying tranformer attentions.

This module is designed to be called from an ipython notebook.
"""

import json
import os

import IPython.display as display

import numpy as np

vis_html = """
  <span style="user-select:none">
    Layer: <select id="layer"></select>
    Attention: <select id="att_type">
      <option value="all">All</option>
      <option value="inp_inp">Input - Input</option>
      <option value="inp_out">Input - Output</option>
      <option value="out_out">Output - Output</option>
    </select>
  </span>
  <div id='vis'></div>
"""


__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
vis_js = open(os.path.join(__location__, 'attention.js')).read()


def show(inp_text, out_text, enc_atts, dec_atts, encdec_atts):
  attention = _get_attention(
      inp_text, out_text, enc_atts, dec_atts, encdec_atts)
  att_json = json.dumps(attention)
  _show_attention(att_json)


def _show_attention(att_json):
  display.display(display.HTML(vis_html))
  display.display(display.Javascript('window.attention = %s' % att_json))
  display.display(display.Javascript(vis_js))


def _get_attention(inp_text, out_text, enc_atts, dec_atts, encdec_atts):
  """Compute representation of the attention ready for the d3 visualization.

  Args:
    inp_text: list of strings, words to be displayed on the left of the vis
    out_text: list of strings, words to be displayed on the right of the vis
    enc_atts: numpy array, encoder self-attentions
        [num_layers, batch_size, num_heads, enc_length, enc_length]
    dec_atts: numpy array, decoder self-attentions
        [num_layers, batch_size, num_heads, dec_length, dec_length]
    encdec_atts: numpy array, encoder-decoder attentions
        [num_layers, batch_size, num_heads, enc_length, dec_length]

  Returns:
    Dictionary of attention representations with the structure:
    {
      'all': Representations for showing all attentions at the same time.
      'inp_inp': Representations for showing encoder self-attentions
      'inp_out': Representations for showing encoder-decoder attentions
      'out_out': Representations for showing decoder self-attentions
    }
    and each sub-dictionary has structure:
    {
      'att': list of inter attentions matrices, one for each attention head
      'top_text': list of strings, words to be displayed on the left of the vis
      'bot_text': list of strings, words to be displayed on the right of the vis
    }
  """
  def get_full_attention(layer):
    """Get the full input+output - input+output attentions."""
    enc_att = enc_atts[layer][0]
    dec_att = dec_atts[layer][0]
    encdec_att = encdec_atts[layer][0]
    enc_att = np.transpose(enc_att, [0, 2, 1])
    dec_att = np.transpose(dec_att, [0, 2, 1])
    encdec_att = np.transpose(encdec_att, [0, 2, 1])
    # [heads, query_length, memory_length]
    enc_length = enc_att.shape[1]
    dec_length = dec_att.shape[1]
    num_heads = enc_att.shape[0]
    first = np.concatenate([enc_att, encdec_att], axis=2)
    second = np.concatenate(
        [np.zeros((num_heads, dec_length, enc_length)), dec_att], axis=2)
    full_att = np.concatenate([first, second], axis=1)
    return [ha.T.tolist() for ha in full_att]

  def get_inp_inp_attention(layer):
    att = np.transpose(enc_atts[layer][0], (0, 2, 1))
    return [ha.T.tolist() for ha in att]

  def get_out_inp_attention(layer):
    att = np.transpose(encdec_atts[layer][0], (0, 2, 1))
    return [ha.T.tolist() for ha in att]

  def get_out_out_attention(layer):
    att = np.transpose(dec_atts[layer][0], (0, 2, 1))
    return [ha.T.tolist() for ha in att]

  def get_attentions(get_attention_fn):
    num_layers = len(enc_atts)
    attentions = []
    for i in range(num_layers):
      attentions.append(get_attention_fn(i))

    return attentions

  attentions = {
      'all': {
          'att': get_attentions(get_full_attention),
          'top_text': inp_text + out_text,
          'bot_text': inp_text + out_text,
      },
      'inp_inp': {
          'att': get_attentions(get_inp_inp_attention),
          'top_text': inp_text,
          'bot_text': inp_text,
      },
      'inp_out': {
          'att': get_attentions(get_out_inp_attention),
          'top_text': inp_text,
          'bot_text': out_text,
      },
      'out_out': {
          'att': get_attentions(get_out_out_attention),
          'top_text': out_text,
          'bot_text': out_text,
      },
  }

  return attentions
