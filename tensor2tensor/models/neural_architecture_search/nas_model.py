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

"""NasSeq2Seq class which can be configured to produce a variety of models.

This was the class used in the Evolved Transformer paper
(https://arxiv.org/abs/1901.11117) to create configurable models. It can be used
to train models in the search space as was done in the paper.

To use NasSeq2Seq:
  - set model=nas_seq2_seq.
  - set hparams_set=nas_seq2seq_base.
  - use hparams to specify the configuration you want to run. See
    nas_seq2seq_base() for an example.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import six
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.models.neural_architecture_search import nas_layers as layers
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
import tensorflow as tf


# Keys for the activation map.
LEAKY_RELU_ACTIVATION_KEY = "leaky_relu"
NONE_ACTIVATION_KEY = "none"
RELU_ACTIVATION_KEY = "relu"
SIGMOID_ACTIVATION_KEY = "sigmoid"
SWISH_ACTIVATION_KEY = "swish"
SOFTMAX_ACTIVATION_KEY = "softmax"

# Mapping from string names to activation function.
ACTIVATION_MAP = {
    SWISH_ACTIVATION_KEY: tf.nn.swish,
    LEAKY_RELU_ACTIVATION_KEY: tf.nn.leaky_relu,
    RELU_ACTIVATION_KEY: tf.nn.relu,
    NONE_ACTIVATION_KEY: None,
    SIGMOID_ACTIVATION_KEY: tf.nn.sigmoid,
    SOFTMAX_ACTIVATION_KEY: tf.nn.softmax
}

# Norm strings.
LAYER_NORM_KEY = "layer_norm"
NO_NORM_KEY = "none"

# Combiner function strings.
ADD_COMBINER_FUNC_KEY = "add"
MULTIPLY_COMBINER_FUNC_KEY = "multiply"
CONCAT_COMBINER_FUNC_KEY = "concat"

# Layers that force the output_dim to be equal to the input_dim if
# enforce_fixed_output_sizes is True.
LAYERS_TO_FIX_OUTPUT_SIZE = [
    layers.IDENTITY_REGISTRY_KEY,
]

# Depthwise layers that the output dimension will need to be changed for
# if channel multiplier cannot be changed to match output dimension.
DEPTHWISE_LAYERS = [
    layers.DEPTHWISE_CONV_3X1_REGISTRY_KEY,
    layers.DEPTHWISE_CONV_5X1_REGISTRY_KEY,
    layers.DEPTHWISE_CONV_7X1_REGISTRY_KEY
]

DEAD_BRANCH_KEY = "dead_branch"


def should_alter_output_dim(layer_name, enforce_fixed_output_sizes, input_depth,
                            output_depth):
  """Check if the output_depth for the specified layer should be changed."""
  # Check to see if output_depth should be changed if we are using
  # a depthwise operation and the channel multiplier is returned as 1,
  # which means that the depthwise multiplier could not be set to match
  # output_depth.
  change_dim_for_depthwise = ((layer_name in DEPTHWISE_LAYERS) and
                              (layers.calculate_depthwise_channel_multiplier(
                                  input_depth, output_depth) == 1))
  # See if layer is in LAYERS_TO_FIX_OUTPUT_SIZE and if it is then we
  # know that the output_dim must be input_dim.
  change_dim_for_other = layer_name in LAYERS_TO_FIX_OUTPUT_SIZE
  # Must be sure enforce_fixed_output_sizes is true.
  return ((change_dim_for_depthwise or change_dim_for_other) and
          enforce_fixed_output_sizes)


def get_activation_names():
  return ACTIVATION_MAP.keys()


def _pad_shallow_tensors(tensors, pad_value):
  """Pads the shorter tensors to be as long as the longest."""
  max_dim = 0
  for tensor in tensors:
    dim = tensor.shape.as_list()[-1]
    if dim > max_dim:
      max_dim = dim

  output_tensors = []
  for tensor in tensors:
    dim = tensor.shape.as_list()[-1]
    if tensor.shape.as_list()[-1] < max_dim:
      output_tensors.append(
          tf.pad(
              tensor, [[0, 0], [0, 0], [0, max_dim - dim]],
              constant_values=pad_value))
    else:
      output_tensors.append(tensor)
  print(output_tensors)

  return output_tensors


class CombinerFunction(object):
  """Interface for combiner functions."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def combine_tensors(self, tensors):
    """Combines `tensors`.

    Args:
      tensors: List of tensors to combine.

    Returns:
      Combined tensor.
    """

  @abc.abstractmethod
  def combined_output_dim(self, output_dims):
    """Determines the output dimension of the combined tensor.

    Args:
      output_dims: List of output dimensions of combined tensors.

    Returns:
      Output dimension of the combined tensor.
    """


class AddCombiner(CombinerFunction):
  """Addition CombinerFunction."""

  def combine_tensors(self, tensors):
    assert tensors

    if len(tensors) == 1:
      return tensors[0]

    tensors_to_combine = _pad_shallow_tensors(tensors, 0)

    output_tensor = tensors_to_combine[0] + tensors_to_combine[1]
    for tensor in tensors_to_combine[2:]:
      output_tensor += tensor

    return output_tensor

  def combined_output_dim(self, output_dims):
    return max(output_dims)


class MultiplyCombiner(CombinerFunction):
  """Multiply CombinerFunction."""

  def combine_tensors(self, tensors):
    assert tensors

    if len(tensors) == 1:
      return tensors[0]

    tensors_to_combine = _pad_shallow_tensors(tensors, 1)

    output_tensor = tensors_to_combine[0] * tensors_to_combine[1]
    for tensor in tensors_to_combine[2:]:
      output_tensor *= tensor

    return output_tensor

  def combined_output_dim(self, output_dims):
    return max(output_dims)


class ConcatCombiner(CombinerFunction):
  """Concat CombinerFunction."""

  def combine_tensors(self, tensors):
    assert tensors

    if len(tensors) == 1:
      return tensors[0]

    return tf.concat(tensors, 2)

  def combined_output_dim(self, output_dims):
    concat_tensor_dim = 0
    for output_dim in output_dims:
      concat_tensor_dim += output_dim

    return concat_tensor_dim


# Dict of combiner functions where each key is the function key string and each
# value is a function that takes a list of tensors and outputs the tensors'
# combination.
COMBINER_FUNCTIONS = {
    ADD_COMBINER_FUNC_KEY: AddCombiner,
    MULTIPLY_COMBINER_FUNC_KEY: MultiplyCombiner,
    CONCAT_COMBINER_FUNC_KEY: ConcatCombiner,
}


@registry.register_model
class NasSeq2Seq(transformer.Transformer):
  """Configurable seq2seq model used for Neural Architecture Search.

  Models are defined by 26 hparam fields. They are:
    - <encoder/decoder>_num_cells: The number of cells in the <encoder/decoder>.
    - <encoder/decoder>_<left/right>_layers: List of layers used the
                                             <encoder/decoder> <left/right>
                                             branch. For available layers, see
                                             the nas_layers.py file.
    - <encoder/decoder>_<left/right_inputs>: List of inputs to the
                                             <encoder/decoder> <left/right>
                                             layers. Each index i specifies the
                                             i_th layer's output with 0
                                             representing the cell input
                                             tensor.
    - <encoder/decoder>_<left/right>_output_dims: List of absolute output
                                                  dimensions for each layer.
    - <encoder/decoder>_<left/right>_activation: List of activations applied
                                                 after each layer.
                                                 ACTIVATION_MAP holds the valid
                                                 activations.
    - <encoder/decoder>_<left/right>_norms: List of norms applied before each
                                            layer. Must be either "layer_norm"
                                            or "none".
    - <encoder/decoder>_combiner_functions: List of functions used to combine
                                            each left/right branch pair.
                                            Options are listed in
                                            COMBINER_FUNCTIONS.
    - <encoder/decoder>_final_combiner_function: Function applied to combine
                                                 all the block outputs that are
                                                 not used as inputs to other
                                                 blocks. Options are listed in
                                                 COMBINER_FUNCTIONS.

  For an example of how to set these hparams, please see nas_seq2seq_base().
  """
  __metaclass__ = abc.ABCMeta

  def encode(self, inputs, target_space, hparams, features=None, losses=None):
    """Encode inputs using _encoder().

    This performs the same way as transformer.Transformer.encode with the
    encoder portion replaced with _encoder().

    Args:
      inputs: Input [batch_size, input_length, input_height, hidden_dim] tensor
        which will be flattened along the two spatial dimensions.
      target_space: scalar, target space ID.
      hparams: Hyperparmeters for model.
      features: Optionally pass the entire features dictionary as well. This is
        needed now for "packed" datasets.
      losses: Unused list of losses.

    Returns:
      Tuple of:
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_decoder_attention_bias: Bias and mask weights for
              encodre-decoder attention. [batch_size, input_length]

    Raises:
      ValueError: If encoder type not found.
    """
    inputs = common_layers.flatten4d3d(inputs)

    encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
        transformer.transformer_prepare_encoder(
            inputs, target_space, hparams, features=features))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    encoder_output = self._encoder(
        encoder_input,
        self_attention_bias,
        hparams,
        nonpadding=transformer.features_to_nonpadding(features, "inputs"),
        save_weights_to=self.attention_weights)

    return encoder_output, encoder_decoder_attention_bias

  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_self_attention_bias,
             hparams,
             cache=None,
             nonpadding=None,
             losses=None):
    """Decode inputs using _decoder().

    This performs the same way as transformer.Transformer.decode with the
    decoder portion replaced with _decoder().

    Args:
      decoder_input: Inputs to bottom of the model. [batch_size, decoder_length,
        hidden_dim]
      encoder_output: Encoder representation. [batch_size, input_length,
        hidden_dim]
      encoder_decoder_attention_bias: Bias and mask weights for encoder-decoder
        attention. [batch_size, input_length]
      decoder_self_attention_bias: Bias and mask weights for decoder
        self-attention. [batch_size, decoder_length]
      hparams: Hyperparmeters for model.
      cache: Dict, containing tensors which are the results of previous
        attentions, used for fast decoding.
      nonpadding: Optional Tensor with shape [batch_size, decoder_length]
      losses: Unused losses.

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    decoder_output = self._decoder(
        decoder_input,
        encoder_output,
        decoder_self_attention_bias,
        encoder_decoder_attention_bias,
        hparams,
        cache=cache,
        nonpadding=nonpadding,
        save_weights_to=self.attention_weights)

    if (common_layers.is_xla_compiled() and
        hparams.mode == tf.estimator.ModeKeys.TRAIN):
      # TPU does not react kindly to extra dimensions.
      return decoder_output

    # Expand since t2t expects 4d tensors.
    return tf.expand_dims(decoder_output, axis=2)

  def _encoder(self,
               encoder_input,
               encoder_self_attention_bias,
               hparams,
               nonpadding=None,
               save_weights_to=None):
    encoder_output, encoder_cell_outputs = nas_encoder(
        encoder_input, encoder_self_attention_bias, hparams, nonpadding)
    self._encoder_cell_outputs = encoder_cell_outputs
    return encoder_output

  def _decoder(self,
               decoder_input,
               encoder_output,
               decoder_self_attention_bias,
               encoder_decoder_attention_bias,
               hparams,
               cache=None,
               nonpadding=None,
               save_weights_to=None):
    assert self._encoder_cell_outputs
    return nas_decoder(decoder_input, self._encoder_cell_outputs,
                       decoder_self_attention_bias,
                       encoder_decoder_attention_bias, hparams)

  def estimator_spec_eval(self, features, logits, labels, loss, losses_dict):
    """Construct EstimatorSpec for EVAL mode."""
    if self.hparams.use_tpu:
      return self._tpu_estimator_spec_eval(features, logits, labels, loss,
                                           losses_dict)
    return self._gpu_estimator_spec_eval(features, logits, labels, loss,
                                         losses_dict)

  # This function is overridden because py_func is not supported on distributed
  # training, which is necessary for NAS. This function works
  # the exact same way as the original Transformer.estimator_spec_eval(),
  # except only neg log perplexity is accepted as a metric.
  def _gpu_estimator_spec_eval(self, features, logits, labels, loss,
                               losses_dict):
    """Construct EstimatorSpec for GPU EVAL mode."""
    hparams = self.hparams

    if not hasattr(hparams, "problem"):
      raise NotImplementedError(
          "hparams is missing attribute `problem`. NasSeq2Seq must "
          "be used with a problem.")

    # TPU is not supported.
    eval_metrics_fns = metrics.create_evaluation_metrics([hparams.problem],
                                                         hparams)
    eval_metrics = {}
    for metric_name, metric_fn in six.iteritems(eval_metrics_fns):
      if "rouge" not in metric_name and "bleu" not in metric_name:
        eval_metrics[metric_name] = metric_fn(logits, features,
                                              features["targets"])

    return tf.estimator.EstimatorSpec(
        tf.estimator.ModeKeys.EVAL,
        predictions={"predictions": logits},
        eval_metric_ops=eval_metrics,
        loss=loss)

  def _tpu_estimator_spec_eval(self, features, logits, labels, loss,
                               losses_dict):
    """Construct EstimatorSpec for TPU EVAL mode."""
    del losses_dict
    hparams = self.hparams

    if not hasattr(hparams, "problem"):
      raise NotImplementedError(
          "hparams is missing attribute `problem`. NasSeq2Seq must "
          "be used with a problem.")

    problem = hparams.problem
    t2t_model.remove_summaries()
    eval_metrics_fn = t2t_model.create_tpu_eval_metrics_fn(problem, hparams)
    if isinstance(logits, dict):
      # For TPU, logits dict will be passed as keyword arguments to
      # eval_metrics_fn. Here we add the labels to those arguments.
      logits.update({"labels": labels})
      return tf.contrib.tpu.TPUEstimatorSpec(
          tf.estimator.ModeKeys.EVAL,
          eval_metrics=(eval_metrics_fn, logits),
          loss=loss)
    else:
      return tf.contrib.tpu.TPUEstimatorSpec(
          tf.estimator.ModeKeys.EVAL,
          eval_metrics=(eval_metrics_fn, [logits, labels]),
          loss=loss)

  def _beam_decode(self, features, decode_length, beam_size, top_beams, alpha,
                   use_tpu):
    """Forced slow beam decode.

    Args:
      features: an map of string to `Tensor`.
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
      use_tpu: Whether or not TPU is being used.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length].
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1).
      }
    """
    return self._beam_decode_slow(features, decode_length, beam_size, top_beams,
                                  alpha, use_tpu)


def _apply_layer_norm(input_tensor, nonpadding, hparams):
  """Applies Tensor2Tensor layer_norm to |input_tensor|."""
  input_depth = input_tensor.shape.as_list()[-1]
  if nonpadding is not None:
    nonpadding_input_tiled = tf.tile(
        tf.expand_dims(nonpadding, 2), [1, 1, input_depth])
    output_tensor = input_tensor * nonpadding_input_tiled

  output_tensor = common_layers.layer_preprocess(input_tensor, hparams)
  if nonpadding is not None:
    output_tensor *= nonpadding_input_tiled

  return output_tensor


def _apply_nas_branch(norm, layer_norm_dict, hidden_states, nonpadding, hparams,
                      input_index, layer_name, activation_name, layer_registry,
                      output_dim, branch_scope_name, mask_future,
                      dropout_broadcast_dims, encoder_decoder_attention_bias,
                      encoder_cell_outputs, decoder_self_attention_bias,
                      cell_number):
  """Applies a single NAS branch."""
  with tf.variable_scope(branch_scope_name):
    # Apply layer norm to an individual layer at most one time.
    if norm == LAYER_NORM_KEY:
      try:
        output_tensor = layer_norm_dict[input_index]
      except KeyError:
        output_tensor = _apply_layer_norm(hidden_states[input_index],
                                          nonpadding, hparams)
        layer_norm_dict[input_index] = output_tensor
    elif norm == NO_NORM_KEY:
      output_tensor = hidden_states[input_index]
    else:
      raise ValueError("norm must be either '%s' or '%s'. Got %s" %
                       (LAYER_NORM_KEY, NO_NORM_KEY, norm))

    layer_class = layer_registry.get(layer_name)
    activation = ACTIVATION_MAP[activation_name]

    postprocess_dropout = layer_name != layers.IDENTITY_REGISTRY_KEY
    output_tensor = layer_class.apply_layer(
        output_tensor,
        None,
        int(output_dim),
        activation,
        hparams,
        branch_scope_name,
        mask_future=mask_future,
        layer_preprocess_fn=None,
        postprocess_dropout=postprocess_dropout,
        nonpadding=nonpadding,
        attention_dropout_broadcast_dims=dropout_broadcast_dims,
        encoder_decoder_attention_bias=encoder_decoder_attention_bias,
        encoder_cell_outputs=encoder_cell_outputs,
        cell_number=cell_number,
        decoder_self_attention_bias=decoder_self_attention_bias)

    return output_tensor


def apply_nas_layers(input_tensor,
                     left_inputs,
                     left_layers,
                     left_activations,
                     left_output_dims,
                     left_norms,
                     right_inputs,
                     right_layers,
                     right_activations,
                     right_output_dims,
                     right_norms,
                     combiner_functions,
                     final_combiner_function,
                     num_cells,
                     nonpadding,
                     layer_registry,
                     mask_future,
                     hparams,
                     var_scope,
                     encoder_decoder_attention_bias=None,
                     encoder_cell_outputs=None,
                     decoder_self_attention_bias=None,
                     final_layer_norm=True,
                     enforce_fixed_output_sizes=True):
  """Applies layers with NasNet search space style branching.

  Args:
    input_tensor: Input [batch_size, input_length, hidden_dim] sequence tensor.
    left_inputs: Int list of left branch hidden layer input indexes.
    left_layers: String list of left branch layers.
    left_activations: String list of left branch activations.
    left_output_dims: String list of left branch output dimensions.
    left_norms: String list of left branch norms.
    right_inputs: Int list of right branch hidden layer input indexes.
    right_layers: String list of right branch layers.
    right_activations: String list of right branch activations.
    right_output_dims: String list of right branch output dimensions.
    right_norms: String list of right branch norms.
    combiner_functions: String list of branch combining functions.
    final_combiner_function: String. The final combiner function that combines
      all the unused hidden layers in a cell.
    num_cells: The number of cells. This is the number of times the given
      layers will be repeated.
    nonpadding: Tensor with 1s at all nonpadding time step positions and 0s
      everywhere else.
    layer_registry: The LayerRegistry that holds all valid layers.
    mask_future: Whether or not to mask future sequence values.
    hparams: Hyperparameters for the model.
    var_scope: The variable scope name.
    encoder_decoder_attention_bias: The attention bias for decoder attending to
      `encoder_output`.
    encoder_cell_outputs: List of tensors. The encoder cell outputs, listed in
      order.
    decoder_self_attention_bias: The self attention bias for decoders. This
      needs to be set for decoders.
    final_layer_norm: Whether or not to apply a final layer_norm to the output
      of the model.
    enforce_fixed_output_sizes: Whether or not to automatically resize output
      dimensions to match the input dimension if `should_alter_output_dim()`
      returns True.

  Raises:
    ValueError: When branching inputs are not of the same length.
    ValueError: If item in left_norms is not LAYER_NORM_KEY or NO_NORM_KEY.
    ValueError: If item in right_norms is not LAYER_NORM_KEY or NO_NORM_KEY.

  Returns:
    Output of applied layers and list of each cell's outputs in order.
  """

  if not (len(left_inputs) == len(left_layers) == len(left_activations) ==
          len(left_output_dims) == len(left_norms) == len(right_inputs) ==
          len(right_layers) == len(right_activations) == len(right_output_dims)
          == len(right_norms) == len(combiner_functions)):
    raise ValueError("All branching inputs must be of the same length.")

  cell_output = None
  modified_left_inputs = [
      left_inputs[i]
      for i in range(len(left_inputs))
      if left_layers[i] != DEAD_BRANCH_KEY
  ]
  modified_right_inputs = [
      right_inputs[i]
      for i in range(len(right_inputs))
      if right_layers[i] != DEAD_BRANCH_KEY
  ]
  unused_cell_hidden_states = [
      i for i in range(len(left_inputs) + 1)
      if i not in modified_left_inputs and i not in modified_right_inputs
  ]
  assert unused_cell_hidden_states

  cell_outputs = []

  with tf.variable_scope(var_scope):
    dropout_broadcast_dims = (
        common_layers.comma_separated_string_to_integer_list(
            getattr(hparams, "attention_dropout_broadcast_dims", "")))

    for cell_num in range(num_cells):
      # h_0 is the input tensor.
      # Keep a dict for layer norm states.
      if cell_output is not None:
        cell_hidden_states = [cell_output]
      else:
        cell_hidden_states = [input_tensor]
      layer_norm_dict = {}

      with tf.variable_scope("cell_%d" % cell_num):

        for i, (left_input, left_layer_name, left_activation_name,
                left_output_dim, left_norm, right_input, right_layer_name,
                right_activation_name, right_output_dim, right_norm,
                combiner) in enumerate(
                    zip(left_inputs, left_layers, left_activations,
                        left_output_dims, left_norms, right_inputs,
                        right_layers, right_activations, right_output_dims,
                        right_norms, combiner_functions)):
          left_input = int(left_input)
          right_input = int(right_input)

          with tf.variable_scope("layer_%d" % i):

            assert not (left_layer_name == DEAD_BRANCH_KEY and
                        right_layer_name == DEAD_BRANCH_KEY)

            if left_layer_name != DEAD_BRANCH_KEY:

              left_raw_input_tensor = cell_hidden_states[left_input]
              left_input_dim = left_raw_input_tensor.shape.as_list()[-1]
              if should_alter_output_dim(left_layer_name,
                                         enforce_fixed_output_sizes,
                                         left_input_dim, left_output_dim):
                left_output_dim = left_input_dim

              # First process the left branch.
              left_tensor = _apply_nas_branch(
                  norm=left_norm,
                  layer_norm_dict=layer_norm_dict,
                  hidden_states=cell_hidden_states,
                  nonpadding=nonpadding,
                  hparams=hparams,
                  input_index=left_input,
                  layer_name=left_layer_name,
                  activation_name=left_activation_name,
                  layer_registry=layer_registry,
                  output_dim=left_output_dim,
                  branch_scope_name="left_%s" % str(i),
                  mask_future=mask_future,
                  dropout_broadcast_dims=dropout_broadcast_dims,
                  encoder_decoder_attention_bias=encoder_decoder_attention_bias,
                  encoder_cell_outputs=encoder_cell_outputs,
                  decoder_self_attention_bias=decoder_self_attention_bias,
                  cell_number=cell_num)

            if right_layer_name != DEAD_BRANCH_KEY:
              right_raw_input_tensor = cell_hidden_states[right_input]
              right_input_dim = right_raw_input_tensor.shape.as_list()[-1]
              if should_alter_output_dim(right_layer_name,
                                         enforce_fixed_output_sizes,
                                         right_input_dim, right_output_dim):
                right_output_dim = right_input_dim
              # Next process the right branch.
              right_tensor = _apply_nas_branch(
                  norm=right_norm,
                  layer_norm_dict=layer_norm_dict,
                  hidden_states=cell_hidden_states,
                  nonpadding=nonpadding,
                  hparams=hparams,
                  input_index=right_input,
                  layer_name=right_layer_name,
                  activation_name=right_activation_name,
                  layer_registry=layer_registry,
                  output_dim=right_output_dim,
                  branch_scope_name="right_%s" % str(i),
                  mask_future=mask_future,
                  dropout_broadcast_dims=dropout_broadcast_dims,
                  encoder_decoder_attention_bias=encoder_decoder_attention_bias,
                  encoder_cell_outputs=encoder_cell_outputs,
                  decoder_self_attention_bias=decoder_self_attention_bias,
                  cell_number=cell_num)

            # Combine the branches.
            if left_layer_name == DEAD_BRANCH_KEY:
              hidden_tensor = right_tensor
            elif right_layer_name == DEAD_BRANCH_KEY:
              hidden_tensor = left_tensor
            else:
              hidden_tensor = COMBINER_FUNCTIONS[combiner]().combine_tensors(
                  [left_tensor, right_tensor])
            cell_hidden_states.append(hidden_tensor)

      states_to_combine = [
          cell_hidden_states[j] for j in unused_cell_hidden_states
      ]
      cell_output = COMBINER_FUNCTIONS[final_combiner_function](
      ).combine_tensors(states_to_combine)
      cell_outputs.append(cell_output)

  if final_layer_norm:
    final_output = common_layers.layer_preprocess(cell_output, hparams)
    cell_outputs = [
        common_layers.layer_preprocess(cell_output, hparams)
        for cell_output in cell_outputs
    ]
    return final_output, cell_outputs
  else:
    return cell_output, cell_outputs


def nas_encoder(encoder_input,
                encoder_self_attention_bias,
                hparams,
                nonpadding=None,
                final_layer_norm=True):
  """Encoder for configurable NAS model.

  Args:
    encoder_input: Input tensor.
    encoder_self_attention_bias: Attention bias tensor with 0s for all valid
      postions and large negative numbers for the padding positions.
    hparams: transformer.Transformer hparams that must also contain:
      + encoder_<left|right>_inputs: List of ints specifying the hidden layer
        input indexes for the <left|right> branches.
      + encoder_<left|right>_layers: String list of layers. Each string must be
        the name of a TranslationLayer registered in layers.py's ENCODER_LAYERS.
      + encoder_<left|right>_activations: String list of activations. Each
        string in this list must have a corresponding activation in
        ACTIVATION_MAP.
      + encoder_<left|right>_output_dims: Int list of output dimensions for
        <left|right> branch layers.
      + encoder_<left|right>_norms: String list of norms to apply to the
        <left|right> layer branches. Each item must be either LAYER_NORM_KEY or
        NO_NORM_KEY.
      + encoder_num_cells: The number of cells in the encoder. This determines
        how many times the given layers will be repeated.
      + encoder_combiner_functions: String list of functions used to combine
        left and right branches. Must be a COMBINER_FUNCTION key.
    nonpadding: Tensor with 1s at all nonpadding positions and 0s everywhere
      else. If None (default), then nonpadding will be determined from
      encoder_self_attention_bias.
    final_layer_norm: Whether or not to apply a final layer_norm to the output
      of the encoder.

  Returns:
    Encoder output and list of each encoder cell's output in order.
  """
  if nonpadding is None:
    padding = common_attention.attention_bias_to_padding(
        encoder_self_attention_bias)
    nonpadding = 1.0 - padding
  return apply_nas_layers(
      input_tensor=encoder_input,
      left_inputs=hparams.encoder_left_inputs,
      left_layers=hparams.encoder_left_layers,
      left_activations=hparams.encoder_left_activations,
      left_output_dims=hparams.encoder_left_output_dims,
      left_norms=hparams.encoder_left_norms,
      right_inputs=hparams.encoder_right_inputs,
      right_layers=hparams.encoder_right_layers,
      right_activations=hparams.encoder_right_activations,
      right_output_dims=hparams.encoder_right_output_dims,
      right_norms=hparams.encoder_right_norms,
      num_cells=hparams.encoder_num_cells,
      combiner_functions=hparams.encoder_combiner_functions,
      final_combiner_function=hparams.encoder_final_combiner_function,
      nonpadding=nonpadding,
      layer_registry=layers.ENCODER_LAYERS,
      mask_future=False,
      hparams=hparams,
      var_scope="encoder",
      final_layer_norm=final_layer_norm)


def nas_decoder(decoder_input,
                encoder_cell_outputs,
                decoder_self_attention_bias,
                encoder_decoder_attention_bias,
                hparams,
                final_layer_norm=True):
  """Decoder for configurable model.

  Args:
    decoder_input: Input tensor.
    encoder_cell_outputs: List of tensors. The encoder cell outputs, listed in
      order.
    decoder_self_attention_bias: Attention bias that the decoder uses when
      attending to itself. This should have 0s for all valid positions and large
      negative numbers for all hidden future positions.
    encoder_decoder_attention_bias: Attention bias that the decoder uses when
      attending to the encoder. This should be 0s at all valid positions and
      large negative numbers for all padded positions.
    hparams: transformer.Transformer hparams that must also contain:
      + decoder_<left|right>_inputs: List of ints specifying the hidden layer
        input indexes for the <left|right> branches.
      + decoder_<left|right>_layers: String list of layers. Each string must be
        the name of a TranslationLayer registered in layers.py's DECODER_LAYERS.
      + decoder_<left|right>_activations: String list of activations. Each
        string in this list must have a corresponding activation in
        ACTIVATION_MAP.
      + decoder_<left|right>_output_dims: Int list of output dimensions for
        <left|right> branch layers.
      + decoder_<left|right>_norms: String list of norms to apply to the
        <left|right> layer branches. Each item must be either LAYER_NORM_KEY or
        NO_NORM_KEY.
      + decoder_num_cells: The number of cells in the decoder. This determines
        how many times the given layers will be repeated.
      + decoder_combiner_functions: String list of functions used to combine
        left and right branches. Must be a COMBINER_FUNCTION key.
      hparams may also optionally contain:
      + enforce_output_size: Boolean that determines whether or not the decoder
        output must be resized to hparams.hidden_size. If True, the output will
        be resized if it not equal to hparams.hidden_size. If False, the output
        will not be resized. If this field is not set, behavior defaults to
        True.
    final_layer_norm: Whether or not to apply a final layer norm to the output
      of the decoder.

  Returns:
    Decoder output tensor.
  """
  # Enforce that the output tensor depth is equal to the depth of the encoding.
  (_, output_depth, _, _) = calculate_branching_model_parameters(
      encoding_depth=hparams.hidden_size,
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
      encoder_depth=hparams.hidden_size)
  improper_output_size = output_depth != hparams.hidden_size

  try:
    enforce_output_size = hparams.enforce_output_size
  except AttributeError:
    enforce_output_size = True
  resize_output = enforce_output_size and improper_output_size

  decoder_cells_output, _ = apply_nas_layers(
      input_tensor=decoder_input,
      left_inputs=hparams.decoder_left_inputs,
      left_layers=hparams.decoder_left_layers,
      left_activations=hparams.decoder_left_activations,
      left_output_dims=hparams.decoder_left_output_dims,
      left_norms=hparams.decoder_left_norms,
      right_inputs=hparams.decoder_right_inputs,
      right_layers=hparams.decoder_right_layers,
      right_activations=hparams.decoder_right_activations,
      right_output_dims=hparams.decoder_right_output_dims,
      right_norms=hparams.decoder_right_norms,
      num_cells=hparams.decoder_num_cells,
      combiner_functions=hparams.decoder_combiner_functions,
      final_combiner_function=hparams.decoder_final_combiner_function,
      nonpadding=None,
      layer_registry=layers.DECODER_LAYERS,
      mask_future=True,
      hparams=hparams,
      var_scope="decoder",
      decoder_self_attention_bias=decoder_self_attention_bias,
      encoder_decoder_attention_bias=encoder_decoder_attention_bias,
      encoder_cell_outputs=encoder_cell_outputs,
      final_layer_norm=final_layer_norm)

  if not resize_output:
    return decoder_cells_output

  # Resize output if necessary.
  dense_layer = layers.DECODER_LAYERS.get(layers.STANDARD_CONV_1X1_REGISTRY_KEY)
  output = dense_layer.apply_layer(
      decoder_cells_output,
      None,
      hparams.hidden_size,
      None,
      hparams,
      "decoder_resize_dense",
      mask_future=True,
      layer_preprocess_fn=None,
      postprocess_dropout=True,
      nonpadding=None,
      attention_dropout_broadcast_dims=None,
      encoder_decoder_attention_bias=None,
      encoder_cell_outputs=None,
      decoder_self_attention_bias=None,
  )
  if final_layer_norm:
    output = common_layers.layer_preprocess(output, hparams)

  return output


def calculate_branching_model_parameters(encoding_depth,
                                         left_inputs,
                                         left_layers,
                                         left_output_dims,
                                         right_inputs,
                                         right_layers,
                                         right_output_dims,
                                         combiner_functions,
                                         layer_registry,
                                         num_cells,
                                         final_combiner_function,
                                         encoder_depth=None,
                                         enforce_output_size=False,
                                         enforce_fixed_output_sizes=True):
  """Calculates the number of parameters in the given model portion.

  Args:
    encoding_depth: Integer. The depth of the initial input tensor.
    left_inputs: Integer list. The indexes of the hidden layer inputs for the
      left branch.
    left_layers: String list. The names of the left branch layers.
    left_output_dims: Integer list. The output dimensions for each of the left
      branch layers.
    right_inputs: Integer list. The indexes of the hidden layer inputs for the
      right branch.
    right_layers: String list. The names of the right branch layers.
    right_output_dims: Integer list. The output dimensions of each of the right
      branch layers.
    combiner_functions: String list. The functions used to combine the left and
      right branch tensors.
    layer_registry: layers.LayerRegistry. The LayerRegistry that contains the
      layers.TranslationLayers needed to construct the model.
    num_cells: Integer. The number of times the given layers are repeated to
      produce the model.
    final_combiner_function: String. The COMBINER_FUNCTIONS key for the combiner
      used to combine the unused hidden dimensions.
    encoder_depth: Integer. The depth of the final encoder layer.
    enforce_output_size: Boolean. If True, include parameters for the addition
      of a dense layer that projects the final output to the appropriate
      `encoding_depth` if it is not already that size. If False, do not add any
      additional parameters.
    enforce_fixed_output_sizes: Whether or not to automatically resize output
      dimensions to match the input dimension if `should_alter_output_dim()`
      returns True.

  Raises:
    ValueError: When the layer config lists are not of equal length.

  Returns:
    total_parameters: The total number of parameters in the model, accounting
      for repeated cells.
    output_depth: The depth of the cell output tensor.
    hidden_depths: The depths of the hidden layers.
    unused_outputs: List of integer indexes of the hidden layers that are not
      used as input, and therefore are concatenated to produce the cell
      output.
  """
  if not (len(left_inputs) == len(left_layers) == len(left_output_dims) ==
          len(right_inputs) == len(right_layers) == len(right_output_dims) ==
          len(combiner_functions)):
    raise ValueError("Layer configs must be of equal length.")

  total_parameters = 0
  output_depth = encoding_depth
  for _ in range(num_cells):
    hidden_depths = [output_depth]
    unused_outputs = set(range(len(left_inputs) + 1))

    for (left_input, left_layer, left_output_dim, right_input,
         right_layer, right_output_dim, combiner_function) in zip(
             left_inputs, left_layers, left_output_dims, right_inputs,
             right_layers, right_output_dims, combiner_functions):

      assert not (left_layer == DEAD_BRANCH_KEY and
                  right_layer == DEAD_BRANCH_KEY)

      if left_layer == DEAD_BRANCH_KEY:
        left_parameters = 0

      else:
        left_input_dim = hidden_depths[left_input]
        if should_alter_output_dim(left_layer, enforce_fixed_output_sizes,
                                   left_input_dim, left_output_dim):
          left_output_dim = left_input_dim

        left_parameters = layer_registry.get(left_layer).num_params(
            left_input_dim, left_output_dim, encoder_depth=encoder_depth)

      if right_layer == DEAD_BRANCH_KEY:
        right_parameters = 0

      else:
        right_input_dim = hidden_depths[right_input]
        if should_alter_output_dim(right_layer, enforce_fixed_output_sizes,
                                   right_input_dim, right_output_dim):
          right_output_dim = right_input_dim

        right_parameters = layer_registry.get(right_layer).num_params(
            right_input_dim, right_output_dim, encoder_depth=encoder_depth)

      total_parameters += left_parameters + right_parameters

      if left_layer == DEAD_BRANCH_KEY:
        hidden_dim = right_output_dim
      elif right_layer == DEAD_BRANCH_KEY:
        hidden_dim = left_output_dim
      else:
        hidden_dim = COMBINER_FUNCTIONS[combiner_function](
        ).combined_output_dim([left_output_dim, right_output_dim])
      hidden_depths.append(hidden_dim)

      try:
        if left_layer != DEAD_BRANCH_KEY:
          unused_outputs.remove(left_input)
      except KeyError:
        pass
      try:
        if right_layer != DEAD_BRANCH_KEY:
          unused_outputs.remove(right_input)
      except KeyError:
        pass

    # All unused outputs combined_together.
    unused_hidden_depths = [hidden_depths[index] for index in unused_outputs]
    output_depth = COMBINER_FUNCTIONS[final_combiner_function](
    ).combined_output_dim(unused_hidden_depths)

  # Add the resizing layer if needed.
  if output_depth != encoding_depth and enforce_output_size:
    total_parameters += layer_registry.get(
        layers.STANDARD_CONV_1X1_REGISTRY_KEY).num_params(
            output_depth, encoding_depth, encoder_depth=encoder_depth)

  return (total_parameters, output_depth, hidden_depths, unused_outputs)


@registry.register_hparams
def nas_seq2seq_base():
  """Base parameters for Nas Seq2Seq model.

  The default parameters are set to create the Transformer.

  Returns:
    Hyperparameters for Nas Seq2Seq model.
  """
  hparams = transformer.transformer_base()

  hparams.add_hparam("encoder_num_cells", 6)
  hparams.add_hparam("encoder_left_inputs", [0, 1, 2, 3])
  hparams.add_hparam("encoder_left_layers", [
      "standard_attention", "standard_conv_1x1", "standard_conv_1x1", "identity"
  ])
  hparams.add_hparam("encoder_left_output_dims", [512, 2048, 512, 512])
  hparams.add_hparam("encoder_left_activations",
                     ["none", "relu", "none", "none"])
  hparams.add_hparam("encoder_left_norms",
                     ["layer_norm", "layer_norm", "none", "none"])
  hparams.add_hparam("encoder_right_inputs", [0, 1, 1, 1])
  hparams.add_hparam("encoder_right_layers",
                     ["identity", "dead_branch", "identity", "dead_branch"])
  hparams.add_hparam("encoder_right_activations",
                     ["none", "none", "none", "none"])
  hparams.add_hparam("encoder_right_output_dims", [512, 512, 512, 512])
  hparams.add_hparam("encoder_right_norms", ["none", "none", "none", "none"])
  hparams.add_hparam("encoder_combiner_functions", ["add", "add", "add", "add"])
  hparams.add_hparam("encoder_final_combiner_function", "add")

  hparams.add_hparam("decoder_num_cells", 6)
  hparams.add_hparam("decoder_left_inputs", [0, 1, 2, 3, 4])
  hparams.add_hparam("decoder_left_layers", [
      "standard_attention", "attend_to_encoder", "standard_conv_1x1",
      "standard_conv_1x1", "identity"
  ])
  hparams.add_hparam("decoder_left_activations",
                     ["none", "none", "relu", "none", "none"])
  hparams.add_hparam("decoder_left_output_dims", [512, 512, 2048, 512, 512])
  hparams.add_hparam("decoder_left_norms",
                     ["layer_norm", "layer_norm", "layer_norm", "none", "none"])
  hparams.add_hparam("decoder_right_inputs", [0, 1, 2, 2, 4])
  hparams.add_hparam(
      "decoder_right_layers",
      ["identity", "identity", "dead_branch", "identity", "dead_branch"])
  hparams.add_hparam("decoder_right_activations",
                     ["none", "none", "none", "none", "none"])
  hparams.add_hparam("decoder_right_output_dims", [512, 512, 512, 512, 512])
  hparams.add_hparam("decoder_right_norms",
                     ["none", "none", "none", "none", "none"])
  hparams.add_hparam("decoder_combiner_functions",
                     ["add", "add", "add", "add", "add"])
  hparams.add_hparam("decoder_final_combiner_function", "add")

  return hparams
