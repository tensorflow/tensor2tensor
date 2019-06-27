# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

"""Tests for NasSeq2Seq."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import parameterized
import numpy as np
from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.layers import common_attention
from tensor2tensor.models import transformer
from tensor2tensor.models.neural_architecture_search import nas_layers as layers
from tensor2tensor.models.neural_architecture_search import nas_model as translation_nas_net
import tensorflow as tf

_BATCH_SIZE = 5
_INPUT_LENGTH = 5
_TARGET_LENGTH = 6
_VOCAB_SIZE = 8
_HIDDEN_SIZE = 512
_EMBEDDING_DEPTH = _HIDDEN_SIZE


def _list_product(num_list):
  """Computes product of all elements in a list."""
  product = 1
  for num in num_list:
    product *= num
  return product


def _get_transformer_branching_encoder_config():
  """Returns config for the Transformer encoder."""
  num_cells = 2
  left_inputs = [0, 1, 2, 3]
  left_layers = [
      layers.STANDARD_ATTENTION_REGISTRY_KEY,
      layers.STANDARD_CONV_1X1_REGISTRY_KEY,
      layers.STANDARD_CONV_1X1_REGISTRY_KEY, layers.IDENTITY_REGISTRY_KEY
  ]
  left_output_dims = [512, 2048, 512, 512]
  right_inputs = [0, 1, 1, 3]
  right_layers = [
      layers.IDENTITY_REGISTRY_KEY, translation_nas_net.DEAD_BRANCH_KEY,
      layers.IDENTITY_REGISTRY_KEY, translation_nas_net.DEAD_BRANCH_KEY
  ]
  right_output_dims = [512, 512, 512, 512]
  combiner_functions = [
      translation_nas_net.ADD_COMBINER_FUNC_KEY,
      translation_nas_net.ADD_COMBINER_FUNC_KEY,
      translation_nas_net.ADD_COMBINER_FUNC_KEY,
      translation_nas_net.ADD_COMBINER_FUNC_KEY
  ]
  dummy_activations = [translation_nas_net.NONE_ACTIVATION_KEY] * 4
  dummy_norms = [translation_nas_net.NO_NORM_KEY] * 4
  layer_registry = layers.ENCODER_LAYERS
  is_decoder = False
  final_combiner_function = translation_nas_net.CONCAT_COMBINER_FUNC_KEY

  return (num_cells, left_inputs, left_layers, left_output_dims, right_inputs,
          right_layers, right_output_dims, combiner_functions,
          final_combiner_function, dummy_activations, dummy_norms,
          layer_registry, is_decoder)


def _get_transformer_branching_decoder_config():
  """Returns config for the Transformer decoder."""
  num_cells = 2
  left_inputs = [0, 1, 2, 3, 4]
  left_layers = [
      layers.STANDARD_ATTENTION_REGISTRY_KEY,
      layers.ATTEND_TO_ENCODER_REGISTRY_KEY,
      layers.STANDARD_CONV_1X1_REGISTRY_KEY,
      layers.STANDARD_CONV_1X1_REGISTRY_KEY, layers.IDENTITY_REGISTRY_KEY
  ]
  left_output_dims = [512, 512, 1024, 256, 512]
  right_inputs = [0, 1, 2, 3, 2]
  right_layers = [
      layers.IDENTITY_REGISTRY_KEY, layers.IDENTITY_REGISTRY_KEY,
      layers.STANDARD_CONV_1X1_REGISTRY_KEY,
      layers.STANDARD_CONV_1X1_REGISTRY_KEY, layers.IDENTITY_REGISTRY_KEY
  ]
  right_output_dims = [512, 512, 1024, 256, 512]
  combiner_functions = [
      translation_nas_net.ADD_COMBINER_FUNC_KEY,
      translation_nas_net.ADD_COMBINER_FUNC_KEY,
      translation_nas_net.CONCAT_COMBINER_FUNC_KEY,
      translation_nas_net.CONCAT_COMBINER_FUNC_KEY,
      translation_nas_net.ADD_COMBINER_FUNC_KEY
  ]
  dummy_activations = [translation_nas_net.NONE_ACTIVATION_KEY] * 5
  dummy_norms = [translation_nas_net.NO_NORM_KEY] * 5
  layer_registry = layers.DECODER_LAYERS
  is_decoder = True
  final_combiner_function = translation_nas_net.CONCAT_COMBINER_FUNC_KEY

  return (num_cells, left_inputs, left_layers, left_output_dims, right_inputs,
          right_layers, right_output_dims, combiner_functions,
          final_combiner_function, dummy_activations, dummy_norms,
          layer_registry, is_decoder)


def _add_transformer_branching_hparams(hparams):
  (encoder_num_cells, encoder_left_inputs, encoder_left_layers,
   encoder_left_output_dims, encoder_right_inputs, encoder_right_layers,
   encoder_right_output_dims, encoder_combiner_functions,
   encoder_final_combiner_function, encoder_dummy_activations,
   encoder_dummy_norms, _, _) = _get_transformer_branching_encoder_config()

  # Transformer encoder.
  hparams.add_hparam("encoder_left_inputs", encoder_left_inputs)
  hparams.add_hparam("encoder_left_layers", encoder_left_layers)
  hparams.add_hparam("encoder_left_activations", encoder_dummy_activations)
  hparams.add_hparam("encoder_left_output_dims", encoder_left_output_dims)
  hparams.add_hparam("encoder_left_norms", encoder_dummy_norms)
  hparams.add_hparam("encoder_right_inputs", encoder_right_inputs)
  hparams.add_hparam("encoder_right_layers", encoder_right_layers)
  hparams.add_hparam("encoder_right_activations", encoder_dummy_activations)
  hparams.add_hparam("encoder_right_output_dims", encoder_right_output_dims)
  hparams.add_hparam("encoder_right_norms", encoder_dummy_norms)
  hparams.add_hparam("encoder_combiner_functions", encoder_combiner_functions)
  hparams.add_hparam("encoder_num_cells", encoder_num_cells)
  hparams.add_hparam("encoder_final_combiner_function",
                     encoder_final_combiner_function)

  (decoder_num_cells, decoder_left_inputs, decoder_left_layers,
   decoder_left_output_dims, decoder_right_inputs, decoder_right_layers,
   decoder_right_output_dims, decoder_combiner_functions,
   decoder_final_combiner_function, decoder_dummy_activations,
   decoder_dummy_norms, _, _) = _get_transformer_branching_decoder_config()

  # Transformer decoder.
  hparams.add_hparam("decoder_left_inputs", decoder_left_inputs)
  hparams.add_hparam("decoder_left_layers", decoder_left_layers)
  hparams.add_hparam("decoder_left_activations", decoder_dummy_activations)
  hparams.add_hparam("decoder_left_output_dims", decoder_left_output_dims)
  hparams.add_hparam("decoder_left_norms", decoder_dummy_norms)
  hparams.add_hparam("decoder_right_inputs", decoder_right_inputs)
  hparams.add_hparam("decoder_right_layers", decoder_right_layers)
  hparams.add_hparam("decoder_right_activations", decoder_dummy_activations)
  hparams.add_hparam("decoder_right_output_dims", decoder_right_output_dims)
  hparams.add_hparam("decoder_right_norms", decoder_dummy_norms)
  hparams.add_hparam("decoder_combiner_functions", decoder_combiner_functions)
  hparams.add_hparam("decoder_num_cells", decoder_num_cells)
  hparams.add_hparam("decoder_final_combiner_function",
                     decoder_final_combiner_function)


class NasSeq2SeqTest(parameterized.TestCase, tf.test.TestCase):

  def _test_model(self, model_cls, hparams):
    """Test a Translation Nas Net model."""
    tf.reset_default_graph()

    hparams.filter_size = 32
    hparams.num_heads = 1
    hparams.layer_prepostprocess_dropout = 0.0
    hparams.hidden_size = _HIDDEN_SIZE

    p_hparams = problem_hparams.test_problem_hparams(_VOCAB_SIZE, _VOCAB_SIZE,
                                                     hparams)
    hparams.problems = [p_hparams]

    inputs = -1 + np.random.random_integers(
        _VOCAB_SIZE, size=(_BATCH_SIZE, _INPUT_LENGTH, 1, 1))
    targets = -1 + np.random.random_integers(
        _VOCAB_SIZE, size=(_BATCH_SIZE, _TARGET_LENGTH, 1, 1))
    features = {
        "inputs": tf.constant(inputs, dtype=tf.int32, name="inputs"),
        "targets": tf.constant(targets, dtype=tf.int32, name="targets"),
        "target_space_id": tf.constant(1, dtype=tf.int32)
    }

    model = model_cls(hparams, tf.estimator.ModeKeys.TRAIN, p_hparams)
    logits, _ = model(features)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape,
                     (_BATCH_SIZE, _TARGET_LENGTH, 1, 1, _VOCAB_SIZE))

  def _get_encoder_hparams(self):
    hparams = transformer.transformer_small()
    hparams.add_hparam("encoder_layer_list",
                       layers.ENCODER_LAYERS.get_layer_names())
    hparams.add_hparam("encoder_output_dim_list", [32] + [64] *
                       (len(hparams.encoder_layer_list) - 2) + [32])
    hparams.add_hparam("encoder_activation_list", ["none"] + ["relu"] *
                       (len(hparams.encoder_layer_list) - 1))
    hparams.add_hparam("encoder_norm_list", ["none"] + ["layer_norm"] *
                       (len(hparams.encoder_layer_list) - 1))
    return hparams

  def test_nas_seq2seq(self):
    hparams = self._get_encoder_hparams()
    _add_transformer_branching_hparams(hparams)
    self._test_model(translation_nas_net.NasSeq2Seq, hparams)

  def _get_wrong_output_dim_decoder_hparams(self):
    tf.reset_default_graph()

    hparams = transformer.transformer_base()
    _add_transformer_branching_hparams(hparams)
    hparams.num_heads = 1
    # Purposely scale up the final embedding depth.
    wrong_output_size = _EMBEDDING_DEPTH + 1
    hparams.decoder_left_output_dims[
        -2] = hparams.decoder_left_output_dims[-2] + 1
    hparams.decoder_left_output_dims[-1] = wrong_output_size

    return hparams, wrong_output_size

  def test_nas_decoder_resizing_output(self):
    hparams, wrong_size = self._get_wrong_output_dim_decoder_hparams()
    hparams.enforce_output_size = False
    input_tensor = tf.zeros([_BATCH_SIZE, _INPUT_LENGTH, _EMBEDDING_DEPTH])
    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(_INPUT_LENGTH))
    with tf.variable_scope("wrong"):
      wrong_size_decoder_output = translation_nas_net.nas_decoder(
          decoder_input=input_tensor,
          encoder_cell_outputs=[input_tensor] * hparams.encoder_num_cells,
          decoder_self_attention_bias=decoder_self_attention_bias,
          encoder_decoder_attention_bias=None,
          hparams=hparams)

    # Now add the correction.
    hparams.enforce_output_size = True
    with tf.variable_scope("correct"):
      correct_size_decoder_output = translation_nas_net.nas_decoder(
          decoder_input=input_tensor,
          encoder_cell_outputs=[input_tensor] * hparams.encoder_num_cells,
          decoder_self_attention_bias=decoder_self_attention_bias,
          encoder_decoder_attention_bias=None,
          hparams=hparams)

    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      wrong_output, correct_output = session.run(
          [wrong_size_decoder_output, correct_size_decoder_output])
    self.assertEqual(wrong_output.shape,
                     (_BATCH_SIZE, _INPUT_LENGTH, wrong_size))
    self.assertEqual(correct_output.shape,
                     (_BATCH_SIZE, _INPUT_LENGTH, _EMBEDDING_DEPTH))

  @parameterized.parameters([(_get_transformer_branching_encoder_config,
                              [512, 512, 2048, 512, 512]),
                             (_get_transformer_branching_decoder_config,
                              [512, 512, 512, 2048, 512, 512])])
  def test_calculate_branching_model_parameters_transformer(
      self, get_config, expected_hidden_depths):
    tf.reset_default_graph()

    (num_cells, left_inputs, left_layers, left_output_dims, right_inputs,
     right_layers, right_output_dims, combiner_functions,
     final_combiner_function, dummy_activations, dummy_norms, layer_registry,
     is_decoder) = get_config()

    # Get predicted number of parameters.
    (predicted_num_params, output_size, hidden_depths,
     _) = translation_nas_net.calculate_branching_model_parameters(
         encoding_depth=_EMBEDDING_DEPTH,
         left_inputs=left_inputs,
         left_layers=left_layers,
         left_output_dims=left_output_dims,
         right_inputs=right_inputs,
         right_layers=right_layers,
         right_output_dims=right_output_dims,
         combiner_functions=combiner_functions,
         final_combiner_function=final_combiner_function,
         layer_registry=layer_registry,
         num_cells=num_cells,
         encoder_depth=_EMBEDDING_DEPTH)

    # Create model graph.
    input_tensor = tf.zeros([32, _INPUT_LENGTH, _EMBEDDING_DEPTH])
    hparams = transformer.transformer_small()

    if is_decoder:
      nonpadding = None
      mask_future = True
      decoder_self_attention_bias = (
          common_attention.attention_bias_lower_triangle(_INPUT_LENGTH))
      encoder_cell_outputs = [input_tensor] * 6
    else:
      nonpadding = tf.ones([32, _INPUT_LENGTH])
      mask_future = False
      decoder_self_attention_bias = None
      encoder_cell_outputs = None

    translation_nas_net.apply_nas_layers(
        input_tensor=input_tensor,
        left_inputs=left_inputs,
        left_layers=left_layers,
        left_activations=dummy_activations,
        left_output_dims=left_output_dims,
        left_norms=dummy_norms,
        right_inputs=right_inputs,
        right_layers=right_layers,
        right_activations=dummy_activations,
        right_output_dims=right_output_dims,
        right_norms=dummy_norms,
        combiner_functions=combiner_functions,
        final_combiner_function=final_combiner_function,
        num_cells=num_cells,
        nonpadding=nonpadding,
        layer_registry=layer_registry,
        mask_future=mask_future,
        hparams=hparams,
        var_scope="test",
        encoder_decoder_attention_bias=None,
        encoder_cell_outputs=encoder_cell_outputs,
        decoder_self_attention_bias=decoder_self_attention_bias,
        final_layer_norm=False)

    # Count graph variables.
    trainable_variables_list = tf.trainable_variables()
    empirical_num_params = 0
    for variable_tensor in trainable_variables_list:
      empirical_num_params += _list_product(variable_tensor.shape.as_list())

    # Compare.
    self.assertEqual(empirical_num_params, predicted_num_params)
    self.assertEqual(output_size, _EMBEDDING_DEPTH)
    self.assertEqual(hidden_depths, expected_hidden_depths)

  @parameterized.parameters([True, False])
  def test_calculate_branching_model_parameters_decoder_resize(
      self, enforce_output_size):
    tf.reset_default_graph()

    hparams, _ = self._get_wrong_output_dim_decoder_hparams()
    hparams.enforce_output_size = enforce_output_size
    hparams.decoder_left_norms = [translation_nas_net.NO_NORM_KEY] * 5
    hparams.decoder_right_norms = [translation_nas_net.NO_NORM_KEY] * 5

    # Get predicted number of parameters.
    (predicted_num_params, _, _,
     _) = translation_nas_net.calculate_branching_model_parameters(
         encoding_depth=_EMBEDDING_DEPTH,
         left_inputs=hparams.decoder_left_inputs,
         left_layers=hparams.decoder_left_layers,
         left_output_dims=hparams.decoder_left_output_dims,
         right_inputs=hparams.decoder_right_inputs,
         right_layers=hparams.decoder_right_layers,
         right_output_dims=hparams.decoder_right_output_dims,
         combiner_functions=hparams.decoder_combiner_functions,
         final_combiner_function=hparams.decoder_final_combiner_function,
         layer_registry=layers.DECODER_LAYERS,
         num_cells=hparams.decoder_num_cells,
         encoder_depth=_EMBEDDING_DEPTH,
         enforce_output_size=enforce_output_size)

    # Count graph variables.
    input_tensor = tf.zeros([_BATCH_SIZE, _INPUT_LENGTH, _EMBEDDING_DEPTH])
    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(_INPUT_LENGTH))
    _ = translation_nas_net.nas_decoder(
        decoder_input=input_tensor,
        encoder_cell_outputs=[input_tensor] * hparams.encoder_num_cells,
        decoder_self_attention_bias=decoder_self_attention_bias,
        encoder_decoder_attention_bias=None,
        hparams=hparams,
        final_layer_norm=False)
    trainable_variables_list = tf.trainable_variables()
    empirical_num_params = 0
    for variable_tensor in trainable_variables_list:
      empirical_num_params += _list_product(variable_tensor.shape.as_list())

    self.assertEqual(empirical_num_params, predicted_num_params)

  def test_calculate_branching_model_parameters_output_size_only_final(self):
    left_inputs = [0, 1, 2, 3]
    right_inputs = [0, 1, 2, 3]
    left_output_dims = [1, 10, 100, 1000]
    right_output_dims = [10000, 100000, 1000000, 10000000]
    right_layers = [
        layers.IDENTITY_REGISTRY_KEY, layers.STANDARD_CONV_1X1_REGISTRY_KEY,
        layers.STANDARD_CONV_1X1_REGISTRY_KEY, layers.IDENTITY_REGISTRY_KEY
    ]
    combiner_functions = [
        translation_nas_net.ADD_COMBINER_FUNC_KEY,
        translation_nas_net.ADD_COMBINER_FUNC_KEY,
        translation_nas_net.MULTIPLY_COMBINER_FUNC_KEY,
        translation_nas_net.CONCAT_COMBINER_FUNC_KEY
    ]

    (num_cells, _, left_layers, _, _, _, _, _, final_combiner_function,
     dummy_activations, dummy_norms, layer_registry,
     _) = _get_transformer_branching_encoder_config()

    # Get predicted number of parameters.
    (_, output_size, _,
     _) = translation_nas_net.calculate_branching_model_parameters(
         encoding_depth=_EMBEDDING_DEPTH,
         left_inputs=left_inputs,
         left_layers=left_layers,
         left_output_dims=left_output_dims,
         right_inputs=right_inputs,
         right_layers=right_layers,
         right_output_dims=right_output_dims,
         combiner_functions=combiner_functions,
         final_combiner_function=final_combiner_function,
         layer_registry=layer_registry,
         num_cells=num_cells,
         encoder_depth=_EMBEDDING_DEPTH,
         enforce_output_size=False,
         enforce_fixed_output_sizes=False)

    self.assertEqual(output_size, 10001000)

  def test_calculate_branching_model_parameters_output_size_last_two(self):
    left_inputs = [0, 1, 2, 2]
    right_inputs = [0, 1, 2, 2]
    left_output_dims = [1, 10, 100, 1000]
    right_output_dims = [10000, 100000, 1000000, 10000000]
    right_layers = [
        layers.IDENTITY_REGISTRY_KEY, layers.STANDARD_CONV_1X1_REGISTRY_KEY,
        layers.STANDARD_CONV_1X1_REGISTRY_KEY, layers.IDENTITY_REGISTRY_KEY
    ]
    combiner_functions = [
        translation_nas_net.ADD_COMBINER_FUNC_KEY,
        translation_nas_net.ADD_COMBINER_FUNC_KEY,
        translation_nas_net.MULTIPLY_COMBINER_FUNC_KEY,
        translation_nas_net.CONCAT_COMBINER_FUNC_KEY
    ]

    (num_cells, _, left_layers, _, _, _, _, _, final_combiner_function,
     dummy_activations, dummy_norms, layer_registry,
     _) = _get_transformer_branching_encoder_config()

    # Get predicted number of parameters.
    (_, output_size, _,
     _) = translation_nas_net.calculate_branching_model_parameters(
         encoding_depth=_EMBEDDING_DEPTH,
         left_inputs=left_inputs,
         left_layers=left_layers,
         left_output_dims=left_output_dims,
         right_inputs=right_inputs,
         right_layers=right_layers,
         right_output_dims=right_output_dims,
         combiner_functions=combiner_functions,
         final_combiner_function=final_combiner_function,
         layer_registry=layer_registry,
         num_cells=num_cells,
         encoder_depth=_EMBEDDING_DEPTH,
         enforce_output_size=False,
         enforce_fixed_output_sizes=False)

    self.assertEqual(output_size, 11001000)


if __name__ == "__main__":
  tf.test.main()
