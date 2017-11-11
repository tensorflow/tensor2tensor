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

"""Utilities for attention."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import math

# Dependency imports
import numpy as np

from six.moves import range  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_layers
from tensor2tensor.utils import expert_utils

import tensorflow as tf

from tensorflow.python.framework import function


# Struct conatining the sequences ids and order on a batch (are send to the
# expert to allow them to compute the bias mask)
BatchInfo = collections.namedtuple(
    "BatchInfo", "coordinates, order")

_expert_count = 0


def get_standardized_layers(hparams, dp=None, ps_devices=None):
  """Get the common attention and feed-forward layers.

  The returned layer functions will have the following signature:

    y, extra_loss = fct(x)

  extra_loss is set to 0.0 if the layer doesn't have extra loss.
  If dp is provided, the layers will be distributed within the devices.
  If moe wants to be used, both dp and model need to be set.

  Args:
    hparams (tf.HParams): the model hparameters
    dp (expert_utils.Parallelism): A data paralelism object. If not given,
      the dp calls are simply ignored.
    ps_devices: a reference to model._ps_devices (only used by the moe layer)

  Returns:
    dict[str:fct]: A dictionary containing the standardized functions
  """

  def partial(fct, *args, **kwargs):
    """Same as functools.partial but with functools.wraps."""
    return functools.wraps(fct)(functools.partial(fct, *args, **kwargs))

  def register_layer(
      fct,
      default_args=None,
      default_kwargs=None,
      use_dp=True,
  ):
    """Turn a function into its standardized version.

    Args:
      fct (fct): The function to register
      default_args (list): The default parameters to add to the function.
      default_kwargs (dict): The default parameters to add to the function.
        Those arguments can be overwriten when calling the function.
      use_dp (bool): Wrap the function call within a dataparalellism object if
        dp is available. Some layers (like moe) must be called without dp.

    Returns:
      fct: the standardized layer function.
    """
    # The kwargs given when calling the function overwrite the default ones
    fct = partial(fct, *(default_args or []), **(default_kwargs or {}))

    @functools.wraps(fct)
    def decorator(x, *args, **kwargs):
      """Call the layer function."""
      # Eventually use dp (if given and not MoE)
      if use_dp and dp is not None:
        y = dp(fct, x, *args, **kwargs)
      else:
        y = fct(x, *args, **kwargs)

      # Eventually capture the extra loss
      extra_loss = 0.0
      if isinstance(y, tuple):
        y, extra_loss = y

      return y, extra_loss
    return decorator

  total_key_depth = hparams.attention_key_channels or hparams.hidden_size
  total_value_depth = hparams.attention_value_channels or hparams.hidden_size
  is_train = hparams.mode == tf.estimator.ModeKeys.TRAIN

  moe_hidden_sizes = [int(s) for s in hparams.moe_hidden_sizes.split(",")]
  # Use filter size if moe_hidden_sizes was not given
  if not moe_hidden_sizes:
    moe_hidden_sizes = [hparams.filter_size]
  expert_fn = expert_utils.ffn_expert_fn(
      hparams.hidden_size, moe_hidden_sizes, hparams.hidden_size)

  # Attention layers:

  # === Multi-head full attention layer ===
  multihead_attention_fn = register_layer(
      multihead_attention,
      default_kwargs=dict(
          memory_antecedent=None,  # Self-attention by default
          bias=None,
          total_key_depth=total_key_depth,
          total_value_depth=total_value_depth,
          output_depth=hparams.hidden_size,
          num_heads=hparams.num_heads,
          dropout_rate=hparams.attention_dropout,
      )
  )

  # === Local attention layer ===
  # Reuse same parameters as multihead_attention
  # Only works for self attention. Always mask the future.
  local_attention_fn = partial(
      multihead_attention_fn,
      block_length=hparams.attention_loc_block_length,
      attention_type="local_mask_right",
  )

  # === Memory-compressed multihead self attention layer ===
  # Only works for self attention. Always mask the future.
  compressed_attention_fn = register_layer(
      multihead_self_attention_reduced,
      default_kwargs=dict(
          factor=hparams.attention_red_factor,
          nonlinearity=hparams.attention_red_nonlinearity,
          reduction_type=hparams.attention_red_type,
          multihead_params=dict(
              total_key_depth=total_key_depth,
              total_value_depth=total_value_depth,
              num_heads=hparams.num_heads,
              dropout_rate=hparams.attention_dropout,
          ),
      ),
  )

  # Feed-forwards layers:

  # === Mixture of expert layer ===
  distributed_moe = register_layer(
      expert_utils.distributed_moe,
      default_args=[
          dp,
          ps_devices,
      ],
      default_kwargs=dict(
          train=is_train,
          input_size=hparams.hidden_size,
          expert_fn=expert_fn,
          num_experts=hparams.moe_num_experts,
          k=hparams.moe_k,
          loss_coef=hparams.moe_loss_coef,
      ),
      use_dp=False,
  )

  # === FC layer ===
  conv_hidden_relu = register_layer(
      common_layers.conv_hidden_relu,
      default_kwargs=dict(
          hidden_size=hparams.filter_size,
          output_size=hparams.hidden_size,
          dropout=hparams.relu_dropout,
      ),
  )

  # === Separable convolution layer ===
  # No mask applied
  sep_conv_relu = partial(
      conv_hidden_relu,
      padding="SAME",
      # Parameters copied from the transformer model, could add hparams
      kernel_size=(3, 1),
      second_kernel_size=(31, 1),
  )

  # === Separable convolution layer (masked version) ===
  # Mask the future
  sep_conv_relu_masked = partial(
      sep_conv_relu,
      padding="LEFT",  # Mask future for decoder
  )

  # Define all available layers

  layers = dict(
      a=multihead_attention_fn,  # Multihead full attention
      loc=local_attention_fn,  # Local attention
      red=compressed_attention_fn,  # Memory-compressed attention
      mem=None,  # Memory efficient
      fc=conv_hidden_relu,
      sep=sep_conv_relu,  # Fully connected
      sepm=sep_conv_relu_masked,  # masked separable convolution
      moe=distributed_moe,  # Mixture of expert layer
  )
  return layers


def add_standard_attention_hparams(hparams):
  """Adds the hparams used by get_standadized_layers."""
  # All hyperparameters ending in "dropout" are automatically set to 0.0
  # when not in training mode.

  # hparams used and which should have been defined outside (in
  # common_hparams):
  # Global flags
  # hparams.mode
  # hparams.hidden_size
  # Pre-post processing flags
  # hparams.layer_preprocess_sequence
  # hparams.layer_postprocess_sequence
  # hparams.layer_prepostprocess_dropout
  # hparams.norm_type
  # hparams.norm_epsilon
  # Mixture-of-Expert flags
  # hparams.moe_hidden_sizes
  # hparams.moe_num_experts
  # hparams.moe_k
  # hparams.moe_loss_coef

  # Attention layers flags
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  hparams.add_hparam("attention_dropout", 0.0)
  # Attention: Local
  hparams.add_hparam("attention_loc_block_length", 256)
  # Attention: Memory-compressed
  hparams.add_hparam("attention_red_factor", 3)
  hparams.add_hparam("attention_red_type", "conv")
  hparams.add_hparam("attention_red_nonlinearity", "none")

  # Fully connected layers flags
  # To be more concistent, should use filter_size to also controle the moe
  # size if moe_hidden_sizes not set
  hparams.add_hparam("filter_size", 2048)
  hparams.add_hparam("relu_dropout", 0.0)

  return hparams


@expert_utils.add_name_scope()
def get_timing_signal_1d(
    length, channels, min_timescale=1.0, max_timescale=1.0e4):
  """Gets a bunch of sinusoids of different frequencies.

  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.

  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.

  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  experessed in terms of y, sin(x) and cos(x).

  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.

  Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float

  Returns:
    a Tensor of timing signals [1, length, channels]
  """
  position = tf.to_float(tf.range(length))
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
  signal = tf.reshape(signal, [1, length, channels])
  return signal


@expert_utils.add_name_scope()
def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.

  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.

  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.

  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  experessed in terms of y, sin(x) and cos(x).

  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.

  Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float

  Returns:
    a Tensor the same shape as x.
  """
  length = tf.shape(x)[1]
  channels = tf.shape(x)[2]
  signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
  return x + signal


@expert_utils.add_name_scope()
def add_timing_signal_1d_given_position(x, position, min_timescale=1.0,
                                        max_timescale=1.0e4):
  """Adds sinusoids of diff frequencies to a Tensor, with timing position given.

  Args:
    x: a Tensor with shape [batch, length, channels]
    position: a Tensor with shape [batch, length]
    min_timescale: a float
    max_timescale: a float

  Returns:
    a Tensor the same shape as x.
  """
  channels = tf.shape(x)[2]
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = (tf.expand_dims(tf.to_float(position), 2) *
                 tf.expand_dims(tf.expand_dims(inv_timescales, 0), 0))
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=2)
  signal = tf.pad(signal, [[0, 0], [0, 0], [0, tf.mod(channels, 2)]])
  return x + signal


@expert_utils.add_name_scope()
def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.

  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase in one of the positional dimensions.

  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.

  The use of relative position is possible because sin(a+b) and cos(a+b) can be
  experessed in terms of b, sin(a) and cos(a).

  x is a Tensor with n "positional" dimensions, e.g. one dimension for a
  sequence or two dimensions for an image

  We use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels // (n * 2). For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.

  Args:
    x: a Tensor with shape [batch, d1 ... dn, channels]
    min_timescale: a float
    max_timescale: a float

  Returns:
    a Tensor the same shape as x.
  """
  static_shape = x.get_shape().as_list()
  num_dims = len(static_shape) - 2
  channels = tf.shape(x)[-1]
  num_timescales = channels // (num_dims * 2)
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  for dim in xrange(num_dims):
    length = tf.shape(x)[dim + 1]
    position = tf.to_float(tf.range(length))
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
        inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    prepad = dim * 2 * num_timescales
    postpad = channels - (dim + 1) * 2 * num_timescales
    signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
    for _ in xrange(1 + dim):
      signal = tf.expand_dims(signal, 0)
    for _ in xrange(num_dims - 1 - dim):
      signal = tf.expand_dims(signal, -2)
    x += signal
  return x


@expert_utils.add_name_scope()
def add_positional_embedding_nd(x, max_length, name):
  """Add n-dimensional positional embedding.

  Adds embeddings to represent the positional dimensions of the tensor.
  The input tensor has n positional dimensions - i.e. 1 for text, 2 for images,
  3 for video, etc.

  Args:
    x: a Tensor with shape [batch, p1 ... pn, depth]
    max_length: an integer.  static maximum size of any dimension.
    name: a name for this layer.

  Returns:
    a Tensor the same shape as x.
  """
  static_shape = x.get_shape().as_list()
  dynamic_shape = tf.shape(x)
  num_dims = len(static_shape) - 2
  depth = static_shape[-1]
  base_shape = [1] * (num_dims + 1) + [depth]
  base_start = [0] * (num_dims + 2)
  base_size = [-1] + [1] * num_dims + [depth]
  for i in xrange(num_dims):
    shape = base_shape[:]
    start = base_start[:]
    size = base_size[:]
    shape[i + 1] = max_length
    size[i + 1] = dynamic_shape[i + 1]
    var = (tf.get_variable(
        name + "_%d" % i,
        shape,
        initializer=tf.random_normal_initializer(0, depth**-0.5)) *
           (depth**0.5))
    x += tf.slice(var, start, size)
  return x


class LshGating(object):
  """Class to split key/queries into separate buckets."""

  def __init__(self, depth, nb_hyperplanes, nb_replicat=1, trainable=False):
    """Construct the gating function parameters.

    Compute the gates for a single head.

    Args:
      depth (int): Dimension of the key/queries to dispatch
      nb_hyperplanes (int): Nb of vectors use to split the space. Will determine
        the number of buckets (2^nb_hyperplanes - 1).
      nb_replicat (int): Redundancy to avoid the edge cases (to be in one bucket
        the input should be in a majority)
      trainable (bool): If True, a balance loss is added to force the hyperplane
        to divide the key/query space evenly
    """
    self.depth = depth
    self.nb_hyperplanes = nb_hyperplanes
    self.nb_buckets = 2**nb_hyperplanes
    self.nb_replicat = nb_replicat  # Unused for now
    self.trainable = trainable  # Unused for now

    self.dispatchers = {}

    assert self.nb_replicat == 1  # For now

    with tf.variable_scope("lsh_gating"):
      # Vectors defining the hyperplanes
      self.t_vectors = tf.get_variable(
          "vector",
          shape=(self.depth, self.nb_hyperplanes * self.nb_replicat),
          dtype=tf.float32,
          trainable=self.trainable,
      )
      # Projection vector from the bit space to similarity score space
      self.t_group = tf.constant([
          self._idx_to_bits(i)
          for i in range(self.nb_buckets)
      ], dtype=tf.float32, name="group")

  def _idx_to_bits(self, i):
    """Convert an group index to its bit representation."""
    bits = bin(i)[2:].zfill(self.nb_hyperplanes)  # Pad the bits str with 0
    return [-1.0 if b == "0" else 1.0 for b in bits]

  @expert_utils.add_name_scope("lsh_gating")
  def get_gates(self, x):
    """Return the bucket id of the given tensor.

    Args:
      x (tf.Tensor): float32 of shape [length, depth]

    Returns:
      tf.Tensor: One-hot vector int64 of shape [heads, length, nb_buckets]
        containing the id of the bucket
    """

    # The balance loss don't propagate to the rest of the network
    x = tf.stop_gradient(x)
    # [length, depth] * [depth, nb_vectors * replicat]
    x = tf.matmul(x, self.t_vectors)
    # [length, nb_vector * replicat]
    x = tf.sign(x)  # Get on which side of the hyperplane the keys are.

    # x = tf.reshape(x, [-1, nb_replicat, nb_vector])
    # [length, replicat, nb_vector] * [nb_vector, 2^nb_vector - 1]

    x = tf.matmul(x, self.t_group, transpose_b=True) / self.nb_hyperplanes
    # We get a similarity score for each of the group between [-1, 1]
    # [length, (replicat,) 2^nb_vector - 1]
    # Do an argmax to get the most likely group for each replicat
    x = tf.argmax(x, axis=-1)
    # [length(, replicat)]
    # One-hot for compatibility with the sparse dispatcher
    x = tf.one_hot(x, self.nb_buckets)
    # TODO(epot): Use a loss to force an even distribution
    return x


@expert_utils.add_name_scope()
def embedding_to_padding(emb):
  """Calculates the padding mask based on which embeddings are all zero.

  We have hacked symbol_modality to return all-zero embeddings for padding.

  Args:
    emb: a Tensor with shape [..., depth].
  Returns:
    a float Tensor with shape [...].
  """
  emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
  return tf.to_float(tf.equal(emb_sum, 0.0))


@expert_utils.add_name_scope()
def attention_bias_local(length, max_backward, max_forward):
  """Create an bias tensor to be added to attention logits.

  A position may attend to positions at most max_distance from it,
  forward and backwards.

  This does not actually save any computation.

  Args:
    length: an integer Scalar.
    max_backward: an int64 Scalar - maximum distance backward to attend.
      negative values indicate unlimited.
    max_forward: an int64 Scalar - maximum distance forward to attend.
      negative values indicate unlimited.

  Returns:
    a `Tensor` with shape [1, 1, length, length].
  """
  band = tf.matrix_band_part(
      tf.ones([length, length]), max_backward, max_forward)
  ret = -1e9 * (1.0 - band)
  return tf.reshape(ret, [1, 1, length, length])


@expert_utils.add_name_scope()
def attention_bias_lower_triangle(length):
  """Create an bias tensor to be added to attention logits.

  Allows a query to attend to all positions up to and including its own.

  Args:
   length: a Scalar.

  Returns:
    a `Tensor` with shape [1, 1, length, length].
  """
  return attention_bias_local(length, -1, 0)


@expert_utils.add_name_scope()
def attention_bias_ignore_padding(memory_padding):
  """Create an bias tensor to be added to attention logits.

  Args:
    memory_padding: a float `Tensor` with shape [batch, memory_length].

  Returns:
    a `Tensor` with shape [batch, 1, 1, memory_length].
  """
  ret = memory_padding * -1e9
  return tf.expand_dims(tf.expand_dims(ret, axis=1), axis=1)


@expert_utils.add_name_scope()
def attention_bias_to_padding(attention_bias):
  """Inverse of attention_bias_ignore_padding().

  Args:
    attention_bias: a `Tensor` with shape [batch, 1, 1, memory_length], as
      returned by attention_bias_ignore_padding().

  Returns:
    a Tensor with shape [batch, memory_length] with 1.0 in padding positions
    and 0.0 in non-padding positions.
  """
  # `attention_bias` is a large negative number in padding positions and 0.0
  # elsewhere.
  return tf.squeeze(tf.to_float(tf.less(attention_bias, -1)), axis=[1, 2])


@expert_utils.add_name_scope()
def attention_bias_prepend_inputs_full_attention(padding):
  """Create a bias tensor for prepend_mode="prepend_inputs_full_attention".

  See prepend_inputs in common_hparams.py.

  Produces a bias tensor to be used in self-attention.

  This bias tensor allows for full connectivity in the "inputs" part of
  the sequence and masked connectivity in the targets part.

  Args:
    padding: a float `Tensor` with shape [batch, length] with
      ones in positions corresponding to padding.  In each row, a single
      padding position separates the input part from the target part.

  Returns:
    a `Tensor` with shape [batch, 1, length, length].
  """
  # Everything past the first padding position is part of the target.
  # This Tensor has zeros for the source portion and separator,
  # and ones for the target portion.
  in_target = tf.cumsum(padding, axis=1, exclusive=True)
  # The position within the target, or 0 if part of the source.
  target_pos = tf.cumsum(in_target, axis=1)
  # A position with a lesser target_pos cannot see a position with greater
  # target_pos.
  illegal_connections = tf.greater(tf.expand_dims(target_pos, 1),
                                   tf.expand_dims(target_pos, 2))
  bias = tf.to_float(illegal_connections) * -1e9
  bias = tf.expand_dims(bias, 1)
  return bias


@expert_utils.add_name_scope()
def attention_bias_proximal(length):
  """Bias for self-attention to encourage attention to close positions.

  Args:
    length: an integer scalar.

  Returns:
    a Tensor with shape [1, 1, length, length]
  """
  r = tf.to_float(tf.range(length))
  diff = tf.expand_dims(r, 0) - tf.expand_dims(r, 1)
  return tf.expand_dims(tf.expand_dims(-tf.log(1 + tf.abs(diff)), 0), 0)


@expert_utils.add_name_scope()
def attention_bias_batch(
    batch_coordinates_q,
    batch_coordinates_k=None,
    condition_fn=None,
):
  """Generate a mask to prevent the batch to attend to each others.

  Args:
    batch_coordinates_q (tf.Tensor): int32 of shape [length_q, 1] containing the
      coordinates of the batches
    batch_coordinates_k (tf.Tensor): int32 of shape [length_k, 1] containing the
      coordinates of the batches. If None, do self attention (q and k identical)
    condition_fn (fct): A function defining which type of mask build

  Returns:
    tf.Tensor: float32 mask of shape [length_q, length_k] containing either 0 or
      -infinity (-1e9)
  """
  if batch_coordinates_k is None:
    batch_coordinates_k = batch_coordinates_q

  # Convert to float first because of b/25387198
  def to_float(bc):
    bc = tf.squeeze(bc, 1)
    bc = tf.to_float(bc)
    return bc

  bc_v = tf.expand_dims(to_float(batch_coordinates_q), 1)
  bc_h = tf.expand_dims(to_float(batch_coordinates_k), 0)
  bias_batch = bc_h - bc_v  # Broadcast to create [length_q, length_k] mask
  # Theshold non zeros to 1.0
  bias_batch = condition_fn(bias_batch)
  bias_batch *= -1e9  # Set non zeros to -infinity
  return bias_batch


# Mask to prevent individual sequences of the same batch to attend to each other
attention_bias_coordinates = functools.partial(
    attention_bias_batch,
    condition_fn=lambda bias: tf.minimum(1.0, tf.abs(bias)),
)


# Mask similar to upper triangular mask, but allow dispatching
attention_bias_future = functools.partial(
    attention_bias_batch,
    # Elems can attend to themself (otherwise would use bias_batch + 1.0)
    # No tf.abs to consider the order
    # tf.maximum and tf.minimum to threshold the values
    condition_fn=lambda bias: tf.maximum(0.0, tf.minimum(1.0, bias)),
)


@expert_utils.add_name_scope()
def split_last_dimension(x, n):
  """Reshape x so that the last dimension becomes two dimensions.

  The first of these two dimensions is n.

  Args:
    x: a Tensor with shape [..., m]
    n: an integer.

  Returns:
    a Tensor with shape [..., n, m/n]
  """
  old_shape = x.get_shape().dims
  last = old_shape[-1]
  new_shape = old_shape[:-1] + [n] + [last // n if last else None]
  ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
  ret.set_shape(new_shape)
  return ret


@expert_utils.add_name_scope()
def combine_last_two_dimensions(x):
  """Reshape x so that the last two dimension become one.

  Args:
    x: a Tensor with shape [..., a, b]

  Returns:
    a Tensor with shape [..., ab]
  """
  old_shape = x.get_shape().dims
  a, b = old_shape[-2:]
  new_shape = old_shape[:-2] + [a * b if a and b else None]
  ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
  ret.set_shape(new_shape)
  return ret


@expert_utils.add_name_scope()
def combine_first_two_dimensions(x):
  """Reshape x so that the first two dimension become one.

  Args:
    x: a Tensor with shape [a, b, ...]

  Returns:
    a Tensor with shape [ab, ...]
  """
  ret = tf.reshape(x, tf.concat([[-1], tf.shape(x)[2:]], 0))
  old_shape = x.get_shape().dims
  a, b = old_shape[:2]
  new_shape = [a * b if a and b else None] + old_shape[2:]
  ret.set_shape(new_shape)
  return ret


@expert_utils.add_name_scope()
def split_heads(x, num_heads):
  """Split channels (dimension 3) into multiple heads (becomes dimension 1).

  Args:
    x: a Tensor with shape [batch, length, channels]
    num_heads: an integer

  Returns:
    a Tensor with shape [batch, num_heads, length, channels / num_heads]
  """
  return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


@expert_utils.add_name_scope()
def split_heads_2d(x, num_heads):
  """Split channels (dimension 4) into multiple heads (becomes dimension 1).

  Args:
    x: a Tensor with shape [batch, height, width, channels]
    num_heads: an integer

  Returns:
    a Tensor with shape [batch, num_heads, height, width, channels / num_heads]
  """
  return tf.transpose(split_last_dimension(x, num_heads), [0, 3, 1, 2, 4])


@expert_utils.add_name_scope()
def combine_heads(x):
  """Inverse of split_heads.

  Args:
    x: a Tensor with shape [batch, num_heads, length, channels / num_heads]

  Returns:
    a Tensor with shape [batch, length, channels]
  """
  return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


@expert_utils.add_name_scope()
def combine_heads_2d(x):
  """Inverse of split_heads_2d.

  Args:
    x: a Tensor with shape
      [batch, num_heads, height, width, channels / num_heads]

  Returns:
    a Tensor with shape [batch, height, width, channels]
  """
  return combine_last_two_dimensions(tf.transpose(x, [0, 2, 3, 1, 4]))


def attention_image_summary(attn, image_shapes=None):
  """Compute color image summary.

  Args:
    attn: a Tensor with shape [batch, num_heads, query_length, memory_length]
    image_shapes: optional tuple of integer scalars.
      If the query positions and memory positions represent the
      pixels of flattened images, then pass in their dimensions:
        (query_rows, query_cols, memory_rows, memory_cols).
      If the query positions and memory positions represent the
      pixels x channels of flattened images, then pass in their dimensions:
        (query_rows, query_cols, query_channels,
         memory_rows, memory_cols, memory_channels).
  """
  num_heads = tf.shape(attn)[1]
  # [batch, query_length, memory_length, num_heads]
  image = tf.transpose(attn, [0, 2, 3, 1])
  image = tf.pow(image, 0.2)  # for high-dynamic-range
  # Each head will correspond to one of RGB.
  # pad the heads to be a multiple of 3
  image = tf.pad(image, [[0, 0], [0, 0], [0, 0], [0, tf.mod(-num_heads, 3)]])
  image = split_last_dimension(image, 3)
  image = tf.reduce_max(image, 4)
  if image_shapes is not None:
    if len(image_shapes) == 4:
      q_rows, q_cols, m_rows, m_cols = list(image_shapes)
      image = tf.reshape(image, [-1, q_rows, q_cols, m_rows, m_cols, 3])
      image = tf.transpose(image, [0, 1, 3, 2, 4, 5])
      image = tf.reshape(image, [-1, q_rows * m_rows, q_cols * m_cols, 3])
    else:
      assert len(image_shapes) == 6
      q_rows, q_cols, q_channnels, m_rows, m_cols, m_channels = list(
          image_shapes)
      image = tf.reshape(image, [
          -1, q_rows, q_cols, q_channnels, m_rows, m_cols, m_channels, 3
      ])
      image = tf.transpose(image, [0, 1, 4, 3, 2, 5, 6, 7])
      image = tf.reshape(image, [
          -1, q_rows * m_rows * q_channnels, q_cols * m_cols * m_channels, 3
      ])
  tf.summary.image("attention", image, max_outputs=1)


def grouped_attention_multihead(query_antecedent,
                                memory_antecedent,
                                total_key_depth,
                                total_value_depth,
                                output_depth,
                                num_heads,
                                num_groups,
                                memory_target_density=2.0,
                                multiplicative_overhead=1.25,
                                additive_overhead=8.0,
                                mask_right=False,
                                make_image_summary=True,
                                name=None):
  """Multi-head dot-product attention with sparsity.

  For each attention head, the queries are partitioned into groups.
  For each group, only a subset of the key-value pairs are considered.

  The choices of groups are selected based on trained predictors of
  the total attention given the group inclusion.

  memory_target_density indicates the average how many groups in which
  a key-value pair should participate.

  We use auxialiary losses to ensure that each group contains roughly
  the same number of queries and the same number of key-value pairs.
  If for a given sequence, the actual number of queries/pairs sent to
  an expert exceeds this target by a factor of more than
  multiplicative_overhead, then the last ones are dropped.  We use
  this drop-last policy to avoid bleeding information backwards, which
  is necessary when using this function with autoregressive
  prediction.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    num_groups: an integer
    memory_target_density: a floating point scalar
    multiplicative_overhead: a floating point scalar
    additive_overhead: a floating point scalar
    mask_right: a boolean
    make_image_summary: a boolean
    name: an optional string

  Returns:
    A Tensor with shape [batch, length_q, output_depth]

  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  batch = tf.shape(query_antecedent)[0]
  length_q = tf.shape(query_antecedent)[1]
  length_kv = tf.shape(memory_antecedent)[1]

  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  depth_qk = total_key_depth // num_heads
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  depth_v = total_value_depth // num_heads
  with tf.variable_scope(
      name,
      default_name="multihead_attention_sparse",
      values=[query_antecedent, memory_antecedent]):
    q = common_layers.conv1d(
        query_antecedent, total_key_depth, 1, name="q_transform")
    kv = common_layers.conv1d(
        memory_antecedent, total_key_depth + total_value_depth,
        1, name="kv_transform")
    q = split_heads(q, num_heads)
    kv = split_heads(kv, num_heads)
    # Make predictions about q_total and m_total.
    # These are used to determine group inclusion.
    # We will train these by auxiliary losses.  We use stop_gradient here
    # to keep these losses from back-propagating to the rest of the model.
    # We add biases that help balance the usage of the experts.
    q_pred = common_layers.conv1d(
        tf.stop_gradient(query_antecedent), num_heads * num_groups, 1,
        name="q_pred")
    q_pred = split_heads(q_pred, num_heads)
    q_bias = tf.get_variable("q_bias", [1, num_heads, 1, num_groups])
    q_pred_biased = q_pred + q_bias
    m_pred = common_layers.conv1d(tf.stop_gradient(
        memory_antecedent), num_heads * num_groups, 1, name="m_pred")
    m_pred = split_heads(m_pred, num_heads)
    m_bias = tf.get_variable("m_bias", [1, num_heads, 1, num_groups])
    m_pred_biased = m_pred + m_bias
    q *= depth_qk**-0.5
    # q, kv, q_pred, m_pred are all [batch, heads, length_[q/m], ?]
    # now reshape them all to [batch * heads, length, ?]
    q = combine_first_two_dimensions(q)
    kv = combine_first_two_dimensions(kv)
    q_pred = combine_first_two_dimensions(q_pred)
    m_pred = combine_first_two_dimensions(m_pred)
    q_pred_biased = combine_first_two_dimensions(q_pred_biased)
    m_pred_biased = combine_first_two_dimensions(m_pred_biased)
    q_group = tf.argmax(q_pred_biased, axis=2)
    q_requests = tf.one_hot(q_group, num_groups, axis=-1)
    m_requests = tf.to_float(tf.greater(m_pred_biased, 0.0))
    # include first memory position in all groups, to avoid division by zero.
    m_requests = tf.maximum(
        m_requests, tf.reshape(tf.one_hot([0], length_kv), [1, length_kv, 1]))
    q_group_size = tf.reduce_sum(q_requests, 1)
    m_group_size = tf.reduce_sum(m_requests, 1)
    q_group_target_size = tf.to_float(length_q) / tf.to_float(num_groups)
    m_group_target_size = (
        tf.to_float(length_kv) * memory_target_density
        / tf.to_float(num_groups))
    capacity_q = tf.minimum(length_q, tf.to_int32(
        q_group_target_size * multiplicative_overhead + additive_overhead))
    capacity_m = tf.minimum(length_kv, tf.to_int32(
        m_group_target_size * multiplicative_overhead + additive_overhead))
    q_dispatcher = expert_utils.TruncatingDispatcher(q_requests, capacity_q)
    m_dispatcher = expert_utils.TruncatingDispatcher(m_requests, capacity_m)
    q_gates = q_dispatcher.gates()
    m_gates = m_dispatcher.gates()
    dispatched_q = q_dispatcher.dispatch(q)
    dispatched_kv = m_dispatcher.dispatch(kv)
    # dispatched_q: [batch * num_heads, num_groups, capacity_q, depth_qk]
    # dispatched_kv:
    #   [batch * num_heads, num_groups, capacity_m, depth_qk + depth_v]
    k, v = tf.split(dispatched_kv, [depth_qk, depth_v], axis=3)
    logits = tf.matmul(dispatched_q, k, transpose_b=True)
    bias = tf.expand_dims((m_dispatcher.nonpadding() - 1.0) * 1e9, 2)
    if mask_right:
      q_coordinate = tf.to_float(
          tf.expand_dims(q_dispatcher.length_coordinate(), 3))
      m_coordinate = tf.to_float(
          tf.expand_dims(m_dispatcher.length_coordinate(), 2))
      bias += tf.to_float(tf.greater(m_coordinate, q_coordinate)) * -1e9
    logits += bias
    log_weights = tf.nn.log_softmax(logits)
    weights = tf.exp(log_weights)
    # For each query, this is the log of the sum of the unnormalized weights.
    q_total = tf.stop_gradient(logits[:, :, :, :1] - log_weights[:, :, :, :1])
    # For each key, this is the sum of the normalized weights.
    m_total = tf.expand_dims(
        tf.reduce_sum(tf.stop_gradient(weights), axis=2), -1)
    o = tf.matmul(weights, v)
    o = q_dispatcher.combine(o)

    o = tf.reshape(o, [batch, num_heads, length_q, depth_v])
    o = combine_heads(o)
    o = common_layers.conv1d(o, output_depth, 1, name="output_transform")

    m_total = m_dispatcher.combine(m_total)
    q_total = q_dispatcher.combine(q_total)
    q_total = tf.squeeze(q_total, -1)
    m_total = tf.squeeze(m_total, -1)
    # Compute summed m predictions for all groups
    m_pred_used = tf.reduce_sum(tf.exp(m_pred) * m_dispatcher.gates(), axis=2)
    q_pred_used = tf.reduce_sum(q_pred * q_dispatcher.gates(), axis=2)
    epsilon = 1e-3
    m_pred_used = tf.log(m_pred_used + epsilon)
    m_total = tf.log(m_total + epsilon)
    m_loss = tf.nn.l2_loss(m_total - m_pred_used)
    q_loss = tf.nn.l2_loss(
        (q_total - q_pred_used) * tf.reduce_sum(q_gates, axis=2))

    q_loss /= tf.to_float(batch * length_q)
    m_loss /= tf.to_float(batch * length_kv)

    # We would like the query groups to be equal sized.  The group
    # size is discrete, so we need some trick here.  We add a loss
    # proportional to the product of the group size and the
    # predictions for that group.  This encourages the predictions to
    # decrease for groups that are too big.
    q_group_deviation = (q_group_size / q_group_target_size) - 1.0
    q_balance_loss = tf.reduce_sum(
        tf.reduce_mean(q_pred_biased, axis=1) * q_group_deviation
    ) / tf.to_float(batch)
    m_group_deviation = (m_group_size / m_group_target_size) - 1.0
    m_balance_loss = tf.reduce_sum(
        tf.reduce_mean(m_pred_biased, axis=1) * m_group_deviation
    ) / tf.to_float(batch)

    # The losses in this function only propagate back to variables
    # defined in this function, and the losses outside of this
    # function only propagate back to variables outside of this
    # function.  Assuming some kind of adaptive learning algorithm,
    # it should not matter how much we scale the losses in this function.
    # Still we scale them down a lot so that they should not show up
    # much in the overall loss for the model.
    extra_loss_multiplier = 1e-3
    extra_loss = q_loss + m_loss + q_balance_loss + m_balance_loss
    extra_loss *= extra_loss_multiplier

    # Show a bunch of summaries.
    if (not tf.get_variable_scope().reuse and
        # Summaries don't work well within tf.while_loop()
        "/while/" not in tf.contrib.framework.get_name_scope() and
        make_image_summary):
      tf.summary.histogram("q_group_size", q_group_size)
      tf.summary.histogram("m_group_size", m_group_size)
      tf.summary.scalar("q_loss", q_loss)
      tf.summary.scalar("m_loss", m_loss)
      tf.summary.scalar("q_balance_loss", q_balance_loss)
      tf.summary.scalar("m_balance_loss", m_balance_loss)
      tf.summary.histogram("m_pred_used", m_pred_used)
      tf.summary.histogram("m_total", m_total)
      tf.summary.histogram("q_pred_used", q_pred_used)
      tf.summary.histogram("q_total", q_total)
      if make_image_summary:
        # image summaries are expensive.
        # So we restrict them to head_num<4, query_position<512, batch_index=0.
        trunc_heads = min(4, num_heads)
        trunc_length_q = tf.minimum(length_q, 512)
        # We recompute the attention for the first example, in an inefficient
        # way - masking.  This lets us show pretty pictures.
        # [trunc_heads, length_q, group]
        q_gates_trunc = q_gates[:trunc_heads, :trunc_length_q, :]
        # [trunc_heads, length_kv, group]
        m_gates_trunc = m_gates[:trunc_heads, :, :]
        grouping_mask = tf.matmul(
            q_gates_trunc, m_gates_trunc, transpose_b=True)
        q_trunc = q[:trunc_heads, :trunc_length_q, :]
        k_trunc = kv[:trunc_heads, :, :depth_qk]
        logits_trunc = tf.matmul(q_trunc, k_trunc, transpose_b=True)
        if mask_right:
          band = tf.matrix_band_part(
              tf.ones([trunc_length_q, length_kv]), -1, 0)
          trunc_bias = tf.expand_dims((1.0 - band) * -1e9, 0)
          logits_trunc += trunc_bias
        att_trunc = tf.nn.softmax(logits_trunc)
        mask_coverage = tf.reduce_sum(grouping_mask * att_trunc) / (
            tf.to_float(trunc_length_q) * trunc_heads)
        tf.summary.scalar("coverage", mask_coverage)
        att_trunc_hdr = tf.pow(att_trunc, 0.2)  # for high-dynamic-range
        mask_channel = grouping_mask * tf.maximum(att_trunc_hdr, 0.3)
        image = tf.stack([att_trunc_hdr, mask_channel, mask_channel], axis=3)
        tf.summary.image("att", image, max_outputs=trunc_heads)
        # show one group for each head.
        att_per_group = tf.expand_dims(weights[:trunc_heads, 0, :, :], -1)
        tf.summary.image(
            "att_per_group_%d", tf.pow(att_per_group, 0.2),
            max_outputs=trunc_heads)
    return o, extra_loss


def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          image_shapes=None,
                          name=None,
                          make_image_summary=True):
  """dot-product attention.

  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    image_shapes: optional tuple of integer scalars.
      see comments for attention_image_summary()
    name: an optional string
    make_image_summary: True if you want an image summary.

  Returns:
    A Tensor.
  """
  with tf.variable_scope(
      name, default_name="dot_product_attention", values=[q, k, v]):
    # [batch, num_heads, query_length, memory_length]
    logits = tf.matmul(q, k, transpose_b=True)
    if bias is not None:
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    # dropping out the attention links for each of the heads
    weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
    if (not tf.get_variable_scope().reuse and
        # Summaries don't work well within tf.while_loop()
        "/while/" not in tf.contrib.framework.get_name_scope() and
        make_image_summary):
      attention_image_summary(weights, image_shapes)
    return tf.matmul(weights, v)


def _generate_relative_positions_matrix(length, max_relative_position):
  """Generates matrix of relative positions between inputs."""
  range_vec = tf.range(length)
  range_mat = tf.reshape(tf.tile(range_vec, [length]), [length, length])
  distance_mat = range_mat - tf.transpose(range_mat)
  distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position,
                                          max_relative_position)
  # Shift values to be >= 0. Each integer still uniquely identifies a relative
  # position difference.
  final_mat = distance_mat_clipped + max_relative_position
  return final_mat


def _generate_relative_positions_embeddings(heads, length, depth,
                                            max_relative_position, name):
  """Generates tensor of size [heads, length, length, depth]."""
  with tf.variable_scope(name):
    relative_positions_matrix = _generate_relative_positions_matrix(
        length, max_relative_position)
    vocab_size = max_relative_position * 2 + 1
    # Generates embedding for each relative position of dimension heads * depth.
    embeddings_table = tf.get_variable("embeddings",
                                       [vocab_size, heads * depth])
    embeddings = tf.gather(embeddings_table, relative_positions_matrix)
    # Split embeddings per head.
    embeddings = tf.reshape(embeddings, [length, length, heads, depth])
    # Transpose to shape [heads, length, length, depth].
    embeddings = tf.transpose(embeddings, [2, 0, 1, 3])
    return embeddings


def _relative_attention_inner(x, y, z, transpose):
  """Relative position-aware dot-product attention inner calculation.

  This batches matrix multiply calculations to avoid unnecessary broadcasting.

  Args:
    x: Tensor with shape [batch_size, heads, length, length or depth].
    y: Tensor with shape [batch_size, heads, length, depth].
    z: Tensor with shape [heads, length, length, depth].
    transpose: Whether to tranpose inner matrices of y and z. Should be true if
        last dimension of x is depth, not length.

  Returns:
    A Tensor with shape [batch_size, heads, length, a].
  """
  xy_matmul = tf.matmul(x, y, transpose_b=transpose)
  x_t = tf.transpose(x, [1, 2, 0, 3])
  x_tz_matmul = tf.matmul(x_t, z, transpose_b=transpose)
  x_tz_matmul_t = tf.transpose(x_tz_matmul, [2, 0, 1, 3])
  return xy_matmul + x_tz_matmul_t


def dot_product_attention_relative(q,
                                   k,
                                   v,
                                   bias,
                                   max_relative_position,
                                   dropout_rate=0.0,
                                   image_shapes=None,
                                   name=None):
  """Calculate relative position-aware dot-product self-attention.

  The attention calculation is augmented with learned representations for the
  relative position between each element in q and each element in k and v.

  Args:
    q: a Tensor with shape [batch, heads, length, depth].
    k: a Tensor with shape [batch, heads, length, depth].
    v: a Tensor with shape [batch, heads, length, depth].
    bias: bias Tensor.
    max_relative_position: an integer specifying the maxmimum distance between
        inputs that unique position embeddings should be learned for.
    dropout_rate: a floating point number.
    image_shapes: optional tuple of integer scalars.
    name: an optional string.

  Returns:
    A Tensor.

  Raises:
    ValueError: if max_relative_position is not > 0.
  """
  if not max_relative_position:
    raise ValueError("Max relative position (%s) should be > 0 when using "
                     "relative self attention." % (max_relative_position))
  with tf.variable_scope(
      name, default_name="dot_product_attention_relative", values=[q, k, v]):

    # This calculation only works for self attention.
    # q, k and v must therefore have the same shape.
    q.get_shape().assert_is_compatible_with(k.get_shape())
    q.get_shape().assert_is_compatible_with(v.get_shape())

    # Use separate embeddings suitable for keys and values.
    heads = q.get_shape().as_list()[1]
    depth = q.get_shape().as_list()[3]
    length = tf.shape(q)[2]
    relations_keys = _generate_relative_positions_embeddings(
        heads, length, depth, max_relative_position, "relative_positions_keys")
    relations_values = _generate_relative_positions_embeddings(
        heads, length, depth, max_relative_position,
        "relative_positions_values")

    # Compute self attention considering the relative position embeddings.
    logits = _relative_attention_inner(q, k, relations_keys, True)
    if bias is not None:
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
    if not tf.get_variable_scope().reuse:
      attention_image_summary(weights, image_shapes)
    return _relative_attention_inner(weights, v, relations_values, False)


def masked_local_attention_1d(
    q, k, v, block_length=128, name=None):
  """Attention to the source position and a neighborhood to the left of it.

  The sequence is divided into blocks of length block_size.
  Attention for a given query position can only see memory positions
  less than or equal to the query position, in the corresponding block
  and the previous block.

  If mask_right is True, then a target position cannot see greater source
  positions.

  Args:
    q: a Tensor with shape [batch, heads, length, depth_k]
    k: a Tensor with shape [batch, heads, length, depth_k]
    v: a Tensor with shape [batch, heads, length, depth_v]
    block_length: an integer
    name: an optional string

  Returns:
    a Tensor of shape [batch, heads, length, depth_v]
  """
  with tf.variable_scope(name, default_name="local_attention_1d",
                         values=[q, k, v]):
    v_shape = v.get_shape()
    batch = common_layers.shape_dim(q, 0)
    heads = common_layers.shape_dim(q, 1)
    length = common_layers.shape_dim(q, 2)
    if isinstance(block_length, tf.Tensor):
      const = tf.contrib.util.constant_value(block_length)
      if const is not None:
        block_length = int(const)

    # If (length < 2 * block_length), then we use only one block.
    if isinstance(length, int) and isinstance(block_length, int):
      block_length = length if length < block_length * 2 else block_length
    else:
      block_length = tf.where(tf.less(length, block_length * 2),
                              length, block_length)
    depth_k = tf.shape(k)[3]
    depth_v = tf.shape(v)[3]
    original_length = length
    padding_size = tf.mod(-length, block_length)
    length += padding_size
    padding = [[0, 0], [0, 0], [0, padding_size], [0, 0]]
    q = tf.pad(q, padding)
    k = tf.pad(k, padding)
    v = tf.pad(v, padding)
    num_blocks = tf.div(length, block_length)

    # compute attention for the first query block.
    first_q = tf.slice(q, [0, 0, 0, 0], [-1, -1, block_length, -1])
    first_k = tf.slice(k, [0, 0, 0, 0], [-1, -1, block_length, -1])
    first_v = tf.slice(v, [0, 0, 0, 0], [-1, -1, block_length, -1])
    first_output = dot_product_attention(
        first_q, first_k, first_v, attention_bias_lower_triangle(block_length),
        name="fist_block")

    # compute attention for all subsequent query blocks.
    q = tf.reshape(q, [batch, heads, num_blocks, block_length, depth_k])
    k = tf.reshape(k, [batch, heads, num_blocks, block_length, depth_k])
    v = tf.reshape(v, [batch, heads, num_blocks, block_length, depth_v])

    def local(x):
      """Create a local version of the keys or values."""
      prev_block = tf.slice(
          x, [0, 0, 0, 0, 0], [-1, -1, num_blocks - 1, -1, -1])
      cur_block = tf.slice(
          x, [0, 0, 1, 0, 0], [-1, -1, -1, -1, -1])
      return tf.concat([prev_block, cur_block], 3)
    local_k = local(k)
    local_v = local(v)
    tail_q = tf.slice(q, [0, 0, 1, 0, 0], [-1, -1, -1, -1, -1])

    local_length = tf.shape(local_k)[3]

    # [batch, heads, num_blocks - 1, block_length, local_length]
    attention = tf.matmul(tail_q, local_k, transpose_b=True)

    # make sure source_pos <= target_pos
    good_part = tf.matrix_band_part(
        tf.ones([block_length, local_length]), -1, tf.to_int64(block_length))
    mask = (1.0 - good_part) * -1e9
    attention += tf.reshape(mask, [1, 1, 1, block_length, local_length])
    attention = tf.nn.softmax(attention)
    # TODO(noam): figure out how to show a summary for the remaining blocks.
    # The naive way currently causes errors due to empty tensors.
    # output: [batch, heads, num_blocks-1, block_length, depth_v]
    output = tf.matmul(attention, local_v)
    output = tf.reshape(output, [batch, heads, -1, depth_v])
    output = tf.concat([first_output, output], axis=2)
    output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
    output.set_shape(v_shape)
    return output


def local_attention_1d(q,
                       k,
                       v,
                       block_length=128,
                       filter_width=100,
                       name=None):
  """strided block local self-attention.

  Args:
    q: a Tensor with shape [batch, heads, length, depth_k]
    k: a Tensor with shape [batch, heads, length, depth_k]
    v: a Tensor with shape [batch, heads, length, depth_v]
    block_length: an integer
    filter_width: an integer indicating how much to look left.
    name: an optional string

  Returns:
    a Tensor of shape [batch, heads, length, depth_v]
  """
  with tf.variable_scope(
      name, default_name="local_self_attention_1d", values=[q, k, v]):
    v_shape = v.get_shape()
    depth_v = tf.shape(v)[3]
    batch_size = tf.shape(q)[0]
    num_heads = tf.shape(q)[1]
    original_length = tf.shape(q)[2]

    # making sure q is a multiple of d
    def pad_to_multiple(x, pad_length):
      x_length = tf.shape(x)[2]
      return tf.pad(x, [[0, 0], [0, 0], [0, -x_length % pad_length], [0, 0]])

    def pad_l_and_r(x, pad_length):
      return tf.pad(x, [[0, 0], [0, 0], [pad_length, pad_length], [0, 0]])

    q = pad_to_multiple(q, block_length)
    k = pad_to_multiple(k, block_length)
    v = pad_to_multiple(v, block_length)

    # Setting up q blocks
    new_q_shape = tf.shape(q)
    # Setting up q blocks
    q = tf.reshape(q, [
        new_q_shape[0], new_q_shape[1], new_q_shape[2] // block_length,
        block_length, new_q_shape[3]
    ])

    # Setting up k and v values
    k = pad_l_and_r(k, filter_width)
    v = pad_l_and_r(v, filter_width)

    length = tf.shape(k)[2]
    full_filter_width = block_length + 2 * filter_width
    # getting gather indices
    indices = tf.range(0, length, delta=1, name="index_range")
    # making indices [1, length, 1] to appy convs
    indices = tf.reshape(indices, [1, -1, 1])
    kernel = tf.expand_dims(tf.eye(full_filter_width), axis=1)
    gather_indices = tf.nn.conv1d(
        tf.cast(indices, tf.float32),
        kernel,
        block_length,
        padding="VALID",
        name="gather_conv")

    gather_indices = tf.squeeze(tf.cast(gather_indices, tf.int32), axis=0)

    # [length, batch, heads, dim]
    k_t = tf.transpose(k, [2, 0, 1, 3])
    k_new = tf.gather(k_t, gather_indices)

    # [batch, heads, blocks, block_length, dim]
    k_new = tf.transpose(k_new, [2, 3, 0, 1, 4])

    attention_bias = tf.expand_dims(embedding_to_padding(k_new) * -1e9, axis=-2)

    v_t = tf.transpose(v, [2, 0, 1, 3])
    v_new = tf.gather(v_t, gather_indices)
    v_new = tf.transpose(v_new, [2, 3, 0, 1, 4])

    output = dot_product_attention(
        q, k_new, v_new, attention_bias, dropout_rate=0., name="local_1d",
        make_image_summary=False)
    output = tf.reshape(output, [batch_size, num_heads, -1, depth_v])
    # Remove the padding if introduced
    output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
    output.set_shape(v_shape)
    return output


def reshape_by_blocks(x, x_shape, memory_block_size):
  x = tf.reshape(x, [
      x_shape[0], x_shape[1], x_shape[2] // memory_block_size,
      memory_block_size, x_shape[3]
  ])
  return x


def dilated_self_attention_1d(q,
                              k,
                              v,
                              query_block_size=128,
                              memory_block_size=128,
                              gap_size=2,
                              num_memory_blocks=2,
                              name=None):
  """dilated self-attention.

  Args:
    q: a Tensor with shape [batch, heads, length, depth_k]
    k: a Tensor with shape [batch, heads, length, depth_k]
    v: a Tensor with shape [batch, heads, length, depth_v]
    query_block_size: an integer indicating size of query block
    memory_block_size: an integer indicating the size of a memory block.
    gap_size: an integer indicating the gap size
    num_memory_blocks: how many memory blocks to look at to the left and right.
      Each will be separated by gap_size.
    name: an optional string

  Returns:
    a Tensor of shape [batch, heads, length, depth_v]
  """
  with tf.variable_scope(
      name, default_name="dilated_self_attention_1d", values=[q, k, v]):
    v_list_shape = v.get_shape().as_list()
    v_shape = tf.shape(v)
    depth_v = v_shape[3]
    batch_size = v_shape[0]
    num_heads = v_shape[1]
    original_length = tf.shape(q)[2]
    # making sure q is a multiple of query block size
    def pad_to_multiple(x, pad_length):
      x_length = tf.shape(x)[2]
      return tf.pad(x, [[0, 0], [0, 0], [0, -x_length % pad_length], [0, 0]])

    def pad_l_and_r(x, pad_length):
      return tf.pad(x, [[0, 0], [0, 0], [pad_length, pad_length], [0, 0]])

    q = pad_to_multiple(q, query_block_size)
    v = pad_to_multiple(v, query_block_size)
    k = pad_to_multiple(k, query_block_size)

    q.set_shape(v_list_shape)
    v.set_shape(v_list_shape)
    k.set_shape(v_list_shape)
    # Setting up q blocks
    new_q_shape = tf.shape(q)
    # Setting up q blocks
    q = reshape_by_blocks(q, new_q_shape, query_block_size)
    self_k_part = reshape_by_blocks(k, new_q_shape, query_block_size)
    self_v_part = reshape_by_blocks(v, new_q_shape, query_block_size)

    # Setting up k and v windows
    k_v_padding = (gap_size + memory_block_size) * num_memory_blocks
    k = pad_l_and_r(k, k_v_padding)
    v = pad_l_and_r(v, k_v_padding)
    # getting gather indices
    index_length = (new_q_shape[2] - query_block_size + memory_block_size)
    indices = tf.range(0, index_length, delta=1, name="index_range")
    # making indices [1, length, 1] to appy convs
    indices = tf.reshape(indices, [1, -1, 1])
    kernel = tf.expand_dims(tf.eye(memory_block_size), axis=1)
    gather_indices = tf.nn.conv1d(
        tf.cast(indices, tf.float32),
        kernel,
        query_block_size,
        padding="VALID",
        name="gather_conv")

    gather_indices = tf.squeeze(tf.cast(gather_indices, tf.int32), axis=0)

    # get left and right memory blocks for each query
    # [length, batch, heads, dim]
    k_t = tf.transpose(k, [2, 0, 1, 3])
    v_t = tf.transpose(v, [2, 0, 1, 3])
    left_k = gather_dilated_memory_blocks(k_t[:-k_v_padding, :, :, :],
                                          num_memory_blocks, gap_size,
                                          query_block_size, memory_block_size,
                                          gather_indices)
    left_v = gather_dilated_memory_blocks(v_t[:-k_v_padding, :, :, :],
                                          num_memory_blocks, gap_size,
                                          query_block_size, memory_block_size,
                                          gather_indices)

    right_k = gather_dilated_memory_blocks(k_t[k_v_padding:, :, :, :],
                                           num_memory_blocks, gap_size,
                                           query_block_size, memory_block_size,
                                           gather_indices, direction="right")
    right_v = gather_dilated_memory_blocks(v_t[k_v_padding:, :, :, :],
                                           num_memory_blocks, gap_size,
                                           query_block_size, memory_block_size,
                                           gather_indices, direction="right")

    k_windows = tf.concat([left_k, self_k_part, right_k], axis=3)
    v_windows = tf.concat([left_v, self_v_part, right_v], axis=3)
    attention_bias = tf.expand_dims(
        embedding_to_padding(k_windows) * -1e9, axis=-2)

    output = dot_product_attention(
        q, k_windows, v_windows, attention_bias, dropout_rate=0.,
        name="dilated_1d", make_image_summary=False)
    output = tf.reshape(output, [batch_size, num_heads, -1, depth_v])
    # Remove the padding if introduced
    output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
    output.set_shape(v_list_shape)
    return output


def gather_dilated_memory_blocks(x, num_memory_blocks, gap_size,
                                 query_block_size, memory_block_size,
                                 gather_indices, direction="left"):
  """Gathers blocks with gaps in between.

  Args:
    x: A tensor of shape [length, batch, heads, depth]
    num_memory_blocks:     num_memory_blocks: how many memory blocks to look
      in "direction". Each will be separated by gap_size.
    gap_size: an integer indicating the gap size
    query_block_size: an integer indicating size of query block
    memory_block_size: an integer indicating the size of a memory block.
    gather_indices: The indices to gather from.
    direction: left or right
  Returns:
    a tensor of shape [batch, heads, blocks, block_length, depth]
  """

  gathered_blocks = []
  # gathering memory blocks
  for block_id in range(num_memory_blocks):
    block_end_index = -(query_block_size +
                        gap_size * (block_id+1) + memory_block_size *
                        block_id) - 1
    block_start_index = (
        (memory_block_size + gap_size) *
        (num_memory_blocks - (block_id + 1))
    )
    if direction != "left":
      [block_end_index, block_start_index] = [
          -block_start_index - 1, -block_end_index + 1
      ]
    def gather_dilated_1d_blocks(x, gather_indices):
      x_new = tf.gather(x, gather_indices)
      # [batch, heads, blocks, block_length, dim]
      return tf.transpose(x_new, [2, 3, 0, 1, 4])

    gathered_blocks.append(
        gather_dilated_1d_blocks(x[block_start_index:block_end_index],
                                 gather_indices))
  return tf.concat(gathered_blocks, 3)


def masked_dilated_self_attention_1d(q,
                                     k,
                                     v,
                                     query_block_size=64,
                                     memory_block_size=64,
                                     gap_size=2,
                                     num_memory_blocks=2,
                                     name=None):
  """dilated self-attention.

  Args:
    q: a Tensor with shape [batch, heads, length, depth_k]
    k: a Tensor with shape [batch, heads, length, depth_k]
    v: a Tensor with shape [batch, heads, length, depth_v]
    query_block_size: an integer
    memory_block_size: an integer indicating how much to look left.
    gap_size: an integer indicating the gap size
    num_memory_blocks: how many memory blocks to look at to the left. Each will
      be separated by gap_size.
    name: an optional string

  Returns:
    a Tensor of shape [batch, heads, length, depth_v]
  """
  with tf.variable_scope(
      name, default_name="masked_dilated_self_attention_1d", values=[q, k, v]):
    v_list_shape = v.get_shape().as_list()
    v_shape = tf.shape(v)
    depth_v = v_shape[3]
    batch_size = v_shape[0]
    num_heads = v_shape[1]
    original_length = tf.shape(q)[2]
    # making sure q is a multiple of query block size
    def pad_to_multiple(x, pad_length):
      x_length = tf.shape(x)[2]
      return tf.pad(x, [[0, 0], [0, 0], [0, -x_length % pad_length], [0, 0]])

    def pad_l(x, left_pad_length):
      return tf.pad(x, [[0, 0], [0, 0], [left_pad_length, 0], [0, 0]])

    q = pad_to_multiple(q, query_block_size)
    v = pad_to_multiple(v, query_block_size)
    k = pad_to_multiple(k, query_block_size)
    q.set_shape(v_list_shape)
    v.set_shape(v_list_shape)
    k.set_shape(v_list_shape)
    # Setting up q blocks
    new_q_shape = tf.shape(q)

    # Setting up q blocks
    q = reshape_by_blocks(q, new_q_shape, query_block_size)
    self_k_part = reshape_by_blocks(k, new_q_shape, query_block_size)
    self_v_part = reshape_by_blocks(v, new_q_shape, query_block_size)
    # Setting up k and v windows
    k_v_padding = (gap_size + memory_block_size) * num_memory_blocks
    k = pad_l(k, k_v_padding)
    v = pad_l(v, k_v_padding)
    # getting gather indices
    index_length = (new_q_shape[2] - query_block_size + memory_block_size)

    indices = tf.range(0, index_length, delta=1, name="index_range")
    # making indices [1, length, 1] to appy convs
    indices = tf.reshape(indices, [1, -1, 1])
    kernel = tf.expand_dims(tf.eye(memory_block_size), axis=1)
    gather_indices = tf.nn.conv1d(
        tf.cast(indices, tf.float32),
        kernel,
        query_block_size,
        padding="VALID",
        name="gather_conv")
    gather_indices = tf.squeeze(tf.cast(gather_indices, tf.int32), axis=0)

    # get left and right memory blocks for each query
    # [length, batch, heads, dim]
    k_t = tf.transpose(k, [2, 0, 1, 3])
    v_t = tf.transpose(v, [2, 0, 1, 3])

    k_unmasked_windows = gather_dilated_memory_blocks(k_t, num_memory_blocks,
                                                      gap_size,
                                                      query_block_size,
                                                      memory_block_size,
                                                      gather_indices)
    v_unmasked_windows = gather_dilated_memory_blocks(v_t, num_memory_blocks,
                                                      gap_size,
                                                      query_block_size,
                                                      memory_block_size,
                                                      gather_indices)

    # combine memory windows
    block_q_shape = tf.shape(q)
    masked_attention_bias = tf.tile(tf.expand_dims(
        attention_bias_lower_triangle(query_block_size), axis=0),
                                    [block_q_shape[0], block_q_shape[1],
                                     block_q_shape[2], 1, 1])
    padding_attention_bias = tf.expand_dims(
        embedding_to_padding(k_unmasked_windows) * -1e9, axis=-2)
    padding_attention_bias = tf.tile(padding_attention_bias,
                                     [1, 1, 1, query_block_size, 1])
    attention_bias = tf.concat([masked_attention_bias, padding_attention_bias],
                               axis=-1)
    # combine memory windows
    k_windows = tf.concat([self_k_part, k_unmasked_windows], 3)
    v_windows = tf.concat([self_v_part, v_unmasked_windows], 3)
    output = dot_product_attention(
        q, k_windows, v_windows, attention_bias, dropout_rate=0.,
        name="dilated_1d", make_image_summary=False)
    output = tf.reshape(output, [batch_size, num_heads, -1, depth_v])
    # Remove the padding if introduced
    output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
    output.set_shape(v_list_shape)
    return output


def local_attention_2d(q,
                       k,
                       v,
                       query_shape=(8, 16),
                       memory_flange=(8, 16),
                       name=None):
  """strided block local self-attention.

  Args:
    q: a Tensor with shape [batch, heads, h, w, depth_k]
    k: a Tensor with shape [batch, heads, h, w, depth_k]
    v: a Tensor with shape [batch, heads, h, w, depth_v]
    query_shape: an tuple indicating the height and width of each query block.
    memory_flange: an integer indicating how much to look in height and width
      from each query block.
    name: an optional string

  Returns:
    a Tensor of shape [batch, heads, h, w, depth_v]
  """
  with tf.variable_scope(
      name, default_name="local_self_attention_2d", values=[q, k, v]):
    q_shape = q.get_shape().as_list()
    v_shape = tf.shape(v)

    q = pad_to_multiple_2d(q, query_shape)
    k = pad_to_multiple_2d(k, query_shape)
    v = pad_to_multiple_2d(v, query_shape)
    padded_q_shape = tf.shape(q)
    # Setting up k and v values
    paddings = [[0, 0], [0, 0], [memory_flange[0], memory_flange[1]],
                [memory_flange[0], memory_flange[1]], [0, 0]]
    k = tf.pad(k, paddings)
    v = tf.pad(v, paddings)

    # Setting up q blocks
    q_indices = gather_indices_2d(q, query_shape, query_shape)
    q_new = gather_blocks_2d(q, q_indices)

    # Setting up k and v blocks
    memory_shape = (query_shape[0]+2*memory_flange[0],
                    query_shape[1]+2*memory_flange[1])
    k_and_v_indices = gather_indices_2d(k, memory_shape, query_shape)
    k_new = gather_blocks_2d(k, k_and_v_indices)
    v_new = gather_blocks_2d(v, k_and_v_indices)

    attention_bias = tf.expand_dims(
        tf.to_float(embedding_to_padding(k_new)) * -1e9, axis=-2)

    output = dot_product_attention(q_new, k_new, v_new, attention_bias,
                                   dropout_rate=0., name="local_2d",
                                   make_image_summary=False)
    # putting the representations back in the right place
    output = scatter_blocks_2d(output, q_indices, padded_q_shape)
    # Remove the padding if introduced
    output = tf.slice(output, [0, 0, 0, 0, 0],
                      [-1, -1, v_shape[2], v_shape[3], -1])
    output.set_shape(q_shape)
    return output


def pad_to_multiple_2d(x, block_shape):
  """Making sure x is a multiple of shape. x is [batch, heads, h, w, depth]."""
  old_shape = x.get_shape().dims
  last = old_shape[-1]
  height_padding = -tf.shape(x)[2] % block_shape[0]
  width_padding = -tf.shape(x)[3] % block_shape[1]
  paddings = [[0, 0], [0, 0], [0, height_padding],
              [0, width_padding], [0, 0]]
  padded_x = tf.pad(x, paddings)
  padded_shape = padded_x.get_shape().as_list()
  padded_shape = padded_shape[:-1]+[last]
  padded_x.set_shape(padded_shape)
  return padded_x


def reshape_range(tensor, i, j, shape):
  """Reshapes a tensor between dimensions i and j."""
  target_shape = tf.concat(
      [tf.shape(tensor)[:i], shape, tf.shape(tensor)[j:]],
      axis=0)
  return tf.reshape(tensor, target_shape)


def gather_blocks_2d(x, indices):
  """Gathers flattened blocks from x."""
  x_shape = tf.shape(x)
  x = reshape_range(x, 2, 4, [tf.reduce_prod(x_shape[2:4])])
  # [length, batch, heads, dim]
  x_t = tf.transpose(x, [2, 0, 1, 3])
  x_new = tf.gather(x_t, indices)
  # returns [batch, heads, num_blocks, block_length ** 2, dim]
  return tf.transpose(x_new, [2, 3, 0, 1, 4])


def scatter_blocks_2d(x, indices, shape):
  """scatters blocks from x into shape with indices."""
  x_shape = tf.shape(x)
  # [length, batch, heads, dim]
  x_t = tf.transpose(tf.reshape(x, [x_shape[0], x_shape[1], -1, x_shape[-1]]),
                     [2, 0, 1, 3])
  x_t_shape = tf.shape(x_t)
  indices = tf.reshape(indices, [-1, 1])
  scattered_x = tf.scatter_nd(indices, x_t, x_t_shape)
  scattered_x = tf.transpose(scattered_x, [1, 2, 0, 3])
  return tf.reshape(scattered_x, shape)


def gather_indices_2d(x, block_shape, block_stride):
  """Getting gather indices."""
  # making an identity matrix kernel
  kernel = tf.eye(block_shape[0]*block_shape[1])
  kernel = reshape_range(kernel, 0, 1, [block_shape[0], block_shape[1], 1])
  # making indices [1, h, w, 1] to appy convs
  indices = tf.range(0, tf.shape(x)[2] * tf.shape(x)[3], delta=1)
  indices = tf.reshape(indices, [1, tf.shape(x)[2], tf.shape(x)[3], 1])
  indices = tf.nn.conv2d(
      tf.cast(indices, tf.float32),
      kernel,
      strides=[1, block_stride[0], block_stride[1], 1],
      padding="VALID")
  # making indices [num_blocks, dim] to gather
  num_blocks = tf.reduce_prod(tf.shape(indices)[:3])
  indices = tf.reshape(indices, [num_blocks, -1])
  return tf.cast(indices, tf.int32)


def make_2d_block_raster_mask(query_shape, memory_flange):
  """creates a mask for 2d block raster scany.

  The query mask can look to the left, top left, top, and top right, but
  not to the right. Inside the query, we have the standard raster scan
  masking.
  Args:
    query_shape: A tuple of ints (query_height, query_width)
    memory_flange: A tuple of ints
      (memory_flange_height, memory_flange_width)

  Returns:
    A tensor of shape query_size, memory_size
  """
  # mask inside the query block
  query_triangle = tf.matrix_band_part(
      tf.ones([np.prod(query_shape), np.prod(query_shape)]), -1, 0)
  split_query_masks = tf.split(query_triangle, query_shape[0], axis=1)
  # adding mask for left and right
  mask_pieces = [
      tf.concat(
          [tf.ones([np.prod(query_shape), memory_flange[1]]),
           split_query_masks[i],
           tf.zeros([np.prod(query_shape), memory_flange[1]])
          ], axis=1) for i in range(query_shape[0])]
  # adding mask for top
  final_mask = tf.concat(
      [tf.ones(
          [np.prod(query_shape),
           (query_shape[1]+2*memory_flange[1])*memory_flange[0]]),
       tf.concat(mask_pieces, axis=1)
      ], axis=1)
  # 0. is visible location, 1.0 is masked.
  return 1. - final_mask


def get_memory_region(x,
                      query_block_shape,
                      memory_flange,
                      q_indices):
  """Get the memory regions that surround a 2d query.

    The memory regions will be the left and top right.

  Args:
    x: A tensor with shape [batch, heads, height, width, depth]
    query_block_shape: a 2-d tuple of integers
    memory_flange: a 2-d tuple of integers
    q_indices: a tensor of indices for each of the center blocks.
      [num_blocks, block_length]
  Returns:
    x_flange: A tensor of shape [batch, heads, #blocks, block_length, depth]
  """
  # Padding x to be multiple of query_shape and then
  # extracting the memory blocks from the same regions as the query blocks
  x_query_padded = pad_to_multiple_2d(x, query_block_shape)
  x_center = gather_blocks_2d(x_query_padded, q_indices)
  # Then padding the flange region
  paddings = [[0, 0], [0, 0], [memory_flange[0], 0],
              [memory_flange[1], memory_flange[1]], [0, 0]]
  x_memory_padded = tf.pad(x_query_padded, paddings)
  left_x = None
  top_x = None
  # Extracting the memory regions around the query block. left_x_region extends
  # to the left and the top_x_region is the combination of top left, top, and
  # top right of the query block
  # if no left region
  if memory_flange[1] > 0:
    left_x_region = x_memory_padded[:, :, memory_flange[0]:,
                                    :-(query_block_shape[1]+memory_flange[1]),
                                    :]
    left_memory_shape = (query_block_shape[0], memory_flange[1])
    left_indices = gather_indices_2d(left_x_region, left_memory_shape,
                                     query_block_shape)
    left_x = gather_blocks_2d(left_x_region, left_indices)
  # if no top region
  if memory_flange[0] > 0:
    top_x_region = x_memory_padded[:, :, :-query_block_shape[0], :, :]

    top_memory_shape = (memory_flange[0],
                        query_block_shape[1]+2*memory_flange[1])

    top_indices = gather_indices_2d(top_x_region, top_memory_shape,
                                    query_block_shape)

    top_x = gather_blocks_2d(top_x_region, top_indices)
  x_flange = None
  if top_x is not None and left_x is not None:
    x_flange = tf.concat([top_x, left_x], axis=3)
  else:
    x_flange = top_x if top_x is not None else left_x
  return x_flange, x_center


def get_shifted_center_blocks(x, indices):
  """Get right shifted blocks for masked local attention 2d.

  Args:
    x: A tensor with shape [batch, heads, height, width, depth]
    indices: The indices to gather blocks

  Returns:
    x_shifted: a tensor of extracted blocks, each block right shifted along
      length.
  """
  center_x = gather_blocks_2d(x, indices)
  # Shift right along the length dimension
  def shift_right_2d_blocks(x):
    """Shift the second to last dimension of x right by one."""
    shifted_targets = (
        tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0], [0, 0]])[:, :, :, :-1, :]
    )
    return shifted_targets
  x_shifted = shift_right_2d_blocks(center_x)
  return x_shifted


def right_shift_blockwise(x, query_shape, name=None):
  """Right shifts once in every block.

  Args:
    x: a tensor of shape [batch, height, width, depth]
    query_shape: A 2d tuple of ints
    name: a string

  Returns:
    output: a tensor of the same shape as x
  """
  with tf.variable_scope(
      name, default_name="right_shift_blockwise", values=[x]):
    x_list_shape = x.get_shape().as_list()
    x_shape = tf.shape(x)
    # Add a dummy dimension for heads
    x = tf.expand_dims(x, axis=1)
    x = pad_to_multiple_2d(x, query_shape)
    padded_x_shape = tf.shape(x)
    # Setting up q blocks
    x_indices = gather_indices_2d(x, query_shape, query_shape)
    x_new = get_shifted_center_blocks(x, x_indices)

    # putting the representations back in the right place
    output = scatter_blocks_2d(x_new, x_indices, padded_x_shape)
    # Removing the dummy head dimension
    output = tf.squeeze(output, axis=1)
    # Remove the padding if introduced
    output = tf.slice(output, [0, 0, 0, 0],
                      [-1, x_shape[1], x_shape[2], -1])
    output.set_shape(x_list_shape)
    return output


def masked_local_attention_2d(q,
                              k,
                              v,
                              query_shape=(8, 16),
                              memory_flange=(8, 16),
                              name=None):
  """strided block local self-attention.

    Each position in a query block can attend to all the generated queries in
    the query block, which are generated in raster scan, and positions that are
    generated to the left and top. The shapes are specified by query shape and
    memory flange. Note that if you're using this function, you do not need to
    right shift. Right shifting happens inside this function separately for each
    block.

  Args:
    q: a Tensor with shape [batch, heads, h, w, depth_k]
    k: a Tensor with shape [batch, heads, h, w, depth_k]
    v: a Tensor with shape [batch, heads, h, w, depth_v]
    query_shape: an tuple indicating the height and width of each query block.
      query_shape = block_shape
    memory_flange: an integer indicating how much to look in height and width
      from each query block.
      memory shape = query_shape + (block_flange[0], 2*block_flange[1])
    name: an optional string

  Returns:
    a Tensor of shape [batch, heads, h, w, depth_v]
  """
  with tf.variable_scope(
      name, default_name="local_masked_self_attention_2d", values=[q, k, v]):
    q_shape = q.get_shape().as_list()
    v_shape = tf.shape(v)

    q = pad_to_multiple_2d(q, query_shape)
    padded_q_shape = tf.shape(q)
    # Setting up q blocks
    q_indices = gather_indices_2d(q, query_shape, query_shape)
    q_new = gather_blocks_2d(q, q_indices)
    # Setting up k and v blocks
    k_flange, k_center = get_memory_region(k, query_shape, memory_flange,
                                           q_indices)
    v_flange, v_center = get_memory_region(v, query_shape, memory_flange,
                                           q_indices)
    if k_flange is not None:
      k_new = tf.concat([k_flange, k_center], axis=3)
      v_new = tf.concat([v_flange, v_center], axis=3)
    else:
      k_new = k_center
      v_new = v_center
    # Getting the masks ready
    query_elements = np.prod(query_shape)
    padding_mask = None
    if k_flange is not None:
      padding_mask = tf.expand_dims(
          embedding_to_padding(k_flange)*-1e9, axis=-2)
      padding_mask = tf.tile(padding_mask, [1, 1, 1, query_elements, 1])

    center_attention_bias = attention_bias_lower_triangle(
        np.prod(query_elements))
    center_attention_bias = tf.reshape(center_attention_bias,
                                       [1, 1, 1, query_elements, query_elements]
                                      )
    v_center_shape = tf.shape(v_center)
    center_attention_bias = tf.tile(center_attention_bias,
                                    [v_center_shape[0],
                                     v_center_shape[1],
                                     v_center_shape[2],
                                     1, 1])
    if padding_mask is not None:
      # Combining the mask for padding and visible region
      attention_bias = tf.concat([padding_mask, center_attention_bias], axis=4)
    else:
      attention_bias = center_attention_bias

    output = dot_product_attention(q_new, k_new, v_new, attention_bias,
                                   dropout_rate=0., name="masked_local_2d",
                                   make_image_summary=False)
    # putting the representations back in the right place
    output = scatter_blocks_2d(output, q_indices, padded_q_shape)
    # Remove the padding if introduced
    output = tf.slice(output, [0, 0, 0, 0, 0],
                      [-1, -1, v_shape[2], v_shape[3], -1])
    output.set_shape(q_shape)
    return output


def compute_qkv(query_antecedent, memory_antecedent, total_key_depth,
                total_value_depth, q_filter_width=1, kv_filter_width=1,
                q_padding="VALID", kv_padding="VALID"):
  """Computes query, key and value.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    total_key_depth: an integer
    total_value_depth: and integer
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
    to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    kv_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.

  Returns:
    q, k, v : [batch, length, depth] tensors
  """
  if memory_antecedent is None and q_filter_width == kv_filter_width == 1:
    # self attention with single position q, k, and v
    combined = common_layers.conv1d(
        query_antecedent,
        total_key_depth * 2 + total_value_depth,
        1,
        name="qkv_transform")
    q, k, v = tf.split(
        combined, [total_key_depth, total_key_depth, total_value_depth],
        axis=2)
    return q, k, v

  if memory_antecedent is None:
    # self attention
    q = common_layers.conv1d(
        query_antecedent,
        total_key_depth,
        q_filter_width,
        padding=q_padding,
        name="q_transform")
    kv_combined = common_layers.conv1d(
        query_antecedent,
        total_key_depth + total_value_depth,
        kv_filter_width,
        padding=kv_padding,
        name="kv_transform")
    k, v = tf.split(kv_combined, [total_key_depth, total_value_depth],
                    axis=2)
    return q, k, v

  # encoder-decoder attention
  q = common_layers.conv1d(
      query_antecedent, total_key_depth, q_filter_width, padding=q_padding,
      name="q_transform")
  combined = common_layers.conv1d(
      memory_antecedent,
      total_key_depth + total_value_depth,
      1,
      padding=kv_padding,
      name="kv_transform")
  k, v = tf.split(combined, [total_key_depth, total_value_depth], axis=2)

  return q, k, v


def compute_qkv_2d(query_antecedent, memory_antecedent, total_key_depth,
                   total_value_depth):
  """Computes query, key and value.

  Args:
    query_antecedent: a Tensor with shape [batch, h, w, depth_k]
    memory_antecedent: a Tensor with shape [batch, h, w, depth_k]
    total_key_depth: an integer
    total_value_depth: and integer

  Returns:
    q, k, v : [batch, h, w, depth_k] tensors
  """
  # self attention with single position q, k, and v
  if memory_antecedent is None:
    combined = tf.layers.conv2d(
        query_antecedent,
        total_key_depth * 2 + total_value_depth, (1, 1),
        name="qkv_transform")
    q, k, v = tf.split(
        combined, [total_key_depth, total_key_depth, total_value_depth],
        axis=-1)
    return q, k, v

  # Encoder decoder attention
  q = common_layers.conv1d(
      query_antecedent, total_key_depth, 1, name="q_transform")
  combined = common_layers.conv1d(
      memory_antecedent,
      total_key_depth + total_value_depth,
      1,
      name="kv_transform")
  k, v = tf.split(combined, [total_key_depth, total_value_depth], axis=2)

  return q, k, v


def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        max_relative_position=None,
                        image_shapes=None,
                        attention_type="dot_product",
                        block_length=128,
                        block_width=128,
                        q_filter_width=1,
                        kv_filter_width=1,
                        q_padding="VALID",
                        kv_padding="VALID",
                        cache=None,
                        gap_size=0,
                        num_memory_blocks=2,
                        name=None,
                        **kwargs):
  """Multihead scaled-dot-product attention with input/output transformations.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    max_relative_position: Maximum distance between inputs to generate
                           unique relation embeddings for. Only relevant
                           when using "dot_product_relative" attention.
    image_shapes: optional tuple of integer scalars.
                  see comments for attention_image_summary()
    attention_type: a string, either "dot_product", "dot_product_relative",
                    "local_mask_right", "local_unmasked", "masked_dilated_1d",
                    "unmasked_dilated_1d" or any attention function with the
                    signature (query, key, value, **kwargs)
    block_length: an integer - relevant for "local_mask_right"
    block_width: an integer - relevant for "local_unmasked"
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
                     to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
               kv_padding: One of "VALID", "SAME" or "LEFT". Default is "VALID":
               no padding.
    cache: dict containing Tensors which are the results of previous
           attentions, used for fast decoding. Expects the dict to contrain two
           keys ('k' and 'v'), for the initial call the values for these keys
           should be empty Tensors of the appropriate shape.
               'k' [batch_size, 0, key_channels]
               'v' [batch_size, 0, value_channels]
    gap_size: Integer option for dilated attention to indicate spacing between
              memory blocks.
    num_memory_blocks: Integer option to indicate how many memory blocks to look
                       at.
    name: an optional string
    **kwargs (dict): Parameters for the attention function

  Caching:
    WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
    the caching assumes that the bias contains future masking.

    The caching works by saving all the previous key and value values so that
    you are able to send just the last query location to this attention
    function. I.e. if the cache dict is provided it assumes the query is of the
    shape [batch_size, 1, hiddem_dim] rather than the full memory.

  Returns:
    The result of the attention transformation. The output shape is
        [batch_size, length_q, hidden_dim]
    unless the cache dict is provided in which case only the last memory
    position is calculated and the output shape is [batch_size, 1, hidden_dim]
    Optionaly returns an additional loss parameters (ex: load balance loss for
    the experts) returned by the attention_type function.

  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  with tf.variable_scope(
      name,
      default_name="multihead_attention",
      values=[query_antecedent, memory_antecedent]):
    q, k, v = compute_qkv(query_antecedent, memory_antecedent, total_key_depth,
                          total_value_depth, q_filter_width, kv_filter_width,
                          q_padding, kv_padding)

    if cache is not None:
      if attention_type != "dot_product":
        raise NotImplementedError(
            "Caching is not guaranteed to work with attention types other than"
            " dot_product.")
      if bias is None:
        raise ValueError("Bias required for caching. See function docstring "
                         "for details.")
      k = cache["k"] = tf.concat([cache["k"], k], axis=1)
      v = cache["v"] = tf.concat([cache["v"], v], axis=1)

    q = split_heads(q, num_heads)
    k = split_heads(k, num_heads)
    v = split_heads(v, num_heads)
    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head**-0.5

    additional_returned_value = None
    if callable(attention_type):  # Generic way to extend multihead_attention
      x = attention_type(q, k, v, **kwargs)
      if isinstance(x, tuple):
        x, additional_returned_value = x  # Unpack
    elif attention_type == "dot_product":
      x = dot_product_attention(q, k, v, bias, dropout_rate, image_shapes)
    elif attention_type == "dot_product_relative":
      x = dot_product_attention_relative(q, k, v, bias, max_relative_position,
                                         dropout_rate, image_shapes)
    elif attention_type == "local_mask_right":
      x = masked_local_attention_1d(q, k, v, block_length=block_length)
    elif attention_type == "local_unmasked":
      x = local_attention_1d(
          q, k, v, block_length=block_length, filter_width=block_width)
    elif attention_type == "masked_dilated_1d":
      x = masked_dilated_self_attention_1d(q, k, v, block_length,
                                           block_width,
                                           gap_size,
                                           num_memory_blocks)
    else:
      assert attention_type == "unmasked_dilated_1d"
      x = dilated_self_attention_1d(q, k, v, block_length,
                                    block_width,
                                    gap_size,
                                    num_memory_blocks)
    x = combine_heads(x)
    x = common_layers.conv1d(x, output_depth, 1, name="output_transform")
    if additional_returned_value is not None:
      return x, additional_returned_value
    return x


def multihead_attention_2d(query_antecedent,
                           memory_antecedent,
                           total_key_depth,
                           total_value_depth,
                           output_depth,
                           num_heads,
                           attention_type="local_attention_2d",
                           query_shape=(8, 16),
                           memory_flange=(8, 16),
                           name=None):
  """2d Multihead scaled-dot-product attention with inp/output transformations.

  Args:
    query_antecedent: a Tensor with shape [batch, h, w, depth_k]
    memory_antecedent: a Tensor with shape [batch, h, w, depth_k]
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    attention_type: String, type of attention function to use.
    query_shape: an tuple indicating the height and width of each query block.
    memory_flange: an integer indicating how much to look in height and width
    name: an optional string

  Returns:
    A Tensor of shape [batch, h, w, depth_k]

  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  with tf.variable_scope(
      name,
      default_name="multihead_attention_2d",
      values=[query_antecedent, memory_antecedent]):
    q, k, v = compute_qkv_2d(query_antecedent, memory_antecedent,
                             total_key_depth, total_value_depth)
    # after splitting, shape is [batch, heads, h, w, depth]
    q = split_heads_2d(q, num_heads)
    k = split_heads_2d(k, num_heads)
    v = split_heads_2d(v, num_heads)
    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head**-0.5
    if attention_type == "local_attention_2d":
      x = local_attention_2d(
          q, k, v, query_shape=query_shape, memory_flange=memory_flange)
    else:
      assert attention_type == "masked_local_attention_2d"
      x = masked_local_attention_2d(q, k, v, query_shape=query_shape,
                                    memory_flange=memory_flange)
    x = combine_heads_2d(x)
    x = tf.layers.conv2d(
        x,
        output_depth,
        (1, 1),
        name="output_transform")
    return x


def ffn_self_attention_layer(x,
                             filter_depth,
                             output_depth,
                             num_parts,
                             dropout_rate,
                             share_kv=False,
                             name=None):
  """Self-attention feedforward layer.

  We use self-attention to do feedforward computations. We apply this function
  positionwise where for each position, we linearly transform the output to have
  depth filter_depth, and break up the result depth-wise into num_parts
  contiguous parts.  The parts self-attentd, we concatenate the results
  depth-wise, and we linearly transform to a depth of output_depth. The
  goal is to get multiplicative interactions between components of a
  representation.

  Args:
    x: a Tensor with shape [batch, length, channels]
    filter_depth: an integer
    output_depth: an integer
    num_parts: an integer dividing filter depth
    dropout_rate: a floating point number
    share_kv: Share the key value transform
    name: an optional string

  Returns:
    A Tensor.
  """

  with tf.variable_scope(
      name, default_name="feedforward_self_attention", values=[x]):
    x_shape = tf.shape(x)
    part_depth = filter_depth // num_parts
    if not share_kv:
      combined = common_layers.conv1d(
          x, filter_depth * 3, 1, name="qkv_transform")
      combined = tf.expand_dims(combined, axis=2)
      q, k, v = tf.split(combined, 3, axis=3)
    else:
      q = tf.expand_dims(
          common_layers.conv1d(x, filter_depth, 1, name="q_transform"), axis=2)
      kv_combined = tf.expand_dims(
          common_layers.conv1d(
              tf.concat([x, x], axis=1), filter_depth, 1, name="kv_transform"),
          axis=2)
      k, v = tf.split(kv_combined, [x_shape[1], x_shape[1]], axis=1)

    batch_q = tf.reshape(q, [-1, 1, num_parts, part_depth])
    batch_k = tf.reshape(k, [-1, 1, num_parts, part_depth])
    batch_v = tf.reshape(v, [-1, 1, num_parts, part_depth])

    batch_q *= part_depth**-0.5
    # non-masked bias
    bias = None
    x = dot_product_attention(batch_q, batch_k, batch_v, bias, dropout_rate)
    x = tf.reshape(x, [x_shape[0], x_shape[1], filter_depth])
    x = common_layers.conv1d(x, output_depth, 1, name="output_transform")
    return x


def parameter_attention(x,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        memory_rows,
                        num_heads,
                        dropout_rate,
                        name=None):
  """Attention over parameters.

  We use the same multi-headed attention as in the other layers, but the memory
  keys and values are model parameters.  There are no linear transformation
  on the keys or values.

  We are also a bit more careful about memory usage, since the number of
  memory positions may be very large.

  Args:
    x: a Tensor with shape [batch, length_q, channels]
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    memory_rows: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    name: an optional string

  Returns:
    A Tensor.
  """
  with tf.variable_scope(name, default_name="parameter_attention", values=[x]):
    head_size_k = total_key_depth // num_heads
    head_size_v = total_value_depth // num_heads
    var_shape_k = [num_heads, memory_rows, head_size_k]
    var_shape_v = [num_heads, memory_rows, head_size_v]
    k = tf.get_variable(
        "k",
        var_shape_k,
        initializer=tf.random_normal_initializer(0, output_depth**-0.5)) * (
            num_heads**0.5)
    v = tf.get_variable(
        "v",
        var_shape_v,
        initializer=tf.random_normal_initializer(0, output_depth**-0.5)) * (
            output_depth**0.5)
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]
    q = common_layers.conv1d(x, total_key_depth, 1, name="q_transform")
    if dropout_rate:
      # This is a cheaper form of attention dropout where we use to use
      # the same dropout decisions across batch elemets and query positions,
      # but different decisions across heads and memory positions.
      v = tf.nn.dropout(
          v, 1.0 - dropout_rate, noise_shape=[num_heads, memory_rows, 1])
    # query is [batch, length, hidden_size]
    # reshape and transpose it to [heads, batch * length, head_size]
    q = tf.reshape(q, [batch_size, length, num_heads, head_size_k])
    q = tf.transpose(q, [2, 0, 1, 3])
    q = tf.reshape(q, [num_heads, batch_size * length, head_size_k])
    weights = tf.matmul(q, k, transpose_b=True)
    weights = tf.nn.softmax(weights)
    y = tf.matmul(weights, v)
    y = tf.reshape(y, [num_heads, batch_size, length, head_size_v])
    y = tf.transpose(y, [1, 2, 0, 3])
    y = tf.reshape(y, [batch_size, length, total_value_depth])
    y.set_shape([None, None, total_value_depth])
    y = common_layers.conv1d(y, output_depth, 1, name="output_transform")
    return y


@expert_utils.add_name_scope()
def coordinate_tensor(shape, axis):
  """Return a tensor with given shape containing coordinte along given axis.

  Args:
    shape: a Tensor representing the shape of the output Tensor
    axis: an integer

  Returns:
    A tensor with shape shape and type tf.int32, where each elements its
    coordinate along the given axis.
  """
  if axis < 0:
    axis = tf.size(shape) + axis  # Convert to positive for the one_hot indice

  r = tf.range(shape[axis])
  r_shape = tf.one_hot(
      axis, tf.size(shape), on_value=-1, off_value=1, dtype=tf.int32)
  return tf.zeros(shape, dtype=tf.int32) + tf.reshape(r, r_shape)


def self_attention_expert(
    x,
    batch_coordinate,
    mask_right=True,
    split_batch=False,
    attention_num_head=1,
    attention_kq_size=None,
    attention_v_size=None,
):
  """Implementing attention that runs inside each expert.

  Args:
    x: A tensor of shape[batch, depth]. Contains representations from
      different positions, which are lexicographically ordered.
    batch_coordinate: A tensor of shape [batch, 1] containing the batch
      coordinate of each element in x. This is needed to make sure that
      positions from different sequences don't attend to each other.
    mask_right: A bool. If true, we will not attend to positions on the right,
      just as decoder self attention.
    split_batch (bool): If True, each sequence of the batch is processed
      individually on a loop. If False, the sequences are processed all at
      once and a mask is applied to isolate the sequences from each others
    attention_num_head (int): number of attention heads
    attention_kq_size (int): dimension used for the attention key, and query
    attention_v_size (int): dimension used for the attention value

  Returns:
    out: A tensor of shape [batch, depth].
  example use:
  expert_utils.local_moe(
     ...
     expert_fn=functools.partial(self_attention_expert, mask_right=)
     )
  """

  depth = x.get_shape().as_list()[-1]
  length = tf.shape(batch_coordinate)[0]

  # Print a warning message if one of the expert isn't used (useful at
  # inference where summaries aren't used and the gating function don't add
  # noise)
  global _expert_count  # Hack to make each expert have a unique id
  _expert_count += 1
  length = tf.cond(
      tf.equal(length, 0),
      lambda: tf.Print(  # pylint: disable=g-long-lambda
          length, [length], "Expert {} empty: ".format(_expert_count)),
      lambda: length,
  )

  tf.summary.scalar("batch_size", length, family="experts_stats_batch_size")

  attention_kq_size = attention_kq_size or depth
  attention_v_size = attention_v_size or depth

  def length_not_null(x, batch_coordinate):
    """Branch of the graph only evaluated when length isn't null."""

    # Mask between the sequences (not used if map_ids is used)
    bias_batch = attention_bias_coordinates(batch_coordinate)

    def add_or_set_if(prev_bias, new_bias, condition):
      """Add the bias together while considering the None case."""
      if not condition:
        return prev_bias
      elif prev_bias is None:
        return new_bias
      else:
        return prev_bias + new_bias

    def mask_and_call_attention(x):
      """Function applied once for each sequence of the batch."""

      # Mask to prevent sequences of attenting to the future
      length = tf.shape(x)[1]  # x has shape [1, length,...]
      bias_past = tf.reshape(
          attention_bias_lower_triangle(length), [length, length])
      # bias has shape [length, length]

      bias = None
      bias = add_or_set_if(bias, bias_past, mask_right)
      bias = add_or_set_if(bias, bias_batch, not split_batch)
      bias = tf.reshape(bias, [1, 1, length, length])

      return multihead_attention(
          x,
          None,
          bias,
          total_key_depth=attention_kq_size,
          total_value_depth=attention_v_size,
          output_depth=depth,
          num_heads=attention_num_head,
          dropout_rate=0.0)

    if split_batch:
      out = expert_utils.map_ids(x, batch_coordinate, mask_and_call_attention)
    else:
      x = tf.reshape(x, [1, length, depth])
      out = mask_and_call_attention(x)
      out = tf.squeeze(out, 0)
    return out

  # If the length is empty, just forward an empty tensor (avoid having to
  # evaluate multihead_attention with tensor having dim equal to zeros)
  out = tf.cond(
      tf.equal(length, 0),
      lambda: tf.zeros(shape=[0, depth], dtype=tf.float32, name="empty_out"),
      lambda: length_not_null(x, batch_coordinate),
  )
  return out


def local_expert_attention(
    x,
    k,
    loss_coef,
    attention_num_experts,
    train=True,
    batch_coordinate=None,
    **kwargs
):
  """Attention using a mixture of experts.

    Positions sent to the same expert can attend to each other.
    The mixture of experts is "local" in that it is replicated on each
    datashard.

    local_moe flatten all batches so to avoid problems with padding (ex: all
    padding going to the same expert, self attention attending to non null
    padding tokens,...), the padding should be removed before.

  Args:
    x: a Tensor with shape [batch, length, depth] or [1, batch*length, depth]
    k: The number of experts to dispatch each example to
    loss_coef: a scalar. A multiplier for the expert loss
    attention_num_experts: The number of experts to use
    train: a boolean for the current mode
    batch_coordinate (tf.Tensor): int32 tensor of shape [1, batch*length, 1]
      containing the batch ids. If None, deduced from first dim of x.
    **kwargs: Arguments to forward to self_attention_expert

  Returns:
    y: a Tensor with shape [batch, length, depth]
    loss: a Scalar
  """
  if batch_coordinate is None:
    batch_coordinate = tf.expand_dims(
        coordinate_tensor(tf.shape(x)[:-1], axis=0), axis=-1)
  with tf.variable_scope("local_expert_attention"):
    additional_dispatch_params = {
        "batch_coordinate": batch_coordinate
    }
    return expert_utils.local_moe(
        x,
        train,
        functools.partial(self_attention_expert, **kwargs),
        attention_num_experts,
        k=k,
        loss_coef=loss_coef,
        pass_x=True,
        pass_gates=False,
        additional_dispatch_params=additional_dispatch_params,
    )


@expert_utils.add_name_scope()
def expert_dot_product(q, k, v, info_q, info_k):
  """Perform dot product on a subset of the sequence.

  Can add a mask to the attention to prevent sequences to attend to each other
  and to prevent attention to the futur.

  Args:
    q (tf.Tensor): Queries of shape [length_expert_q, depth_k]
    k (tf.Tensor): Keys of shape [length_expert_k, depth_k]
    v (tf.Tensor): Values of shape [length_expert_k, depth_v]
    info_q (BatchInfo): Batch info for queries. If None, no mask is added
    info_k (BatchInfo): Batch info for keys

  Returns:
    tf.Tensor: dot product attention output ([length_expert_q, depth_v])
  """

  length_q = tf.shape(q)[0]
  length_k = tf.shape(k)[0]
  depth_v = v.get_shape().as_list()[-1]

  # Create the mask
  bias = attention_bias_coordinates(info_q.coordinates, info_k.coordinates)
  if info_k.order is not None:
    bias += attention_bias_future(info_q.order, info_k.order)

  # Restore batch and head dimension
  q, k, v = [tf.expand_dims(tf.expand_dims(t, 0), 0) for t in (q, k, v)]

  def is_zero():
    zeros = tf.zeros(shape=[1, 1, length_q, depth_v], dtype=tf.float32)
    zeros = tf.Print(zeros, [length_k, length_q], "length_k/length_q: ")
    return zeros

  def is_not_zero():
    return dot_product_attention(
        q, k, v,
        bias=bias,
        # No image summary to avoid "Retval[0] does not have value" (because
        # inside a condition)
        make_image_summary=False,
    )

  # TODO(epot): Should make sure a query gets at least one key. Because the
  # different sequences of a batch are merged, it's possible that a
  # query from a sequence only receive memory from another sequence, so
  # with the mask, the query will perform a softmax on -infinity values.
  # A hack could be to add at least one sequence of each batch on each group so
  # the query can attend to at least one element.
  # Softmax(Q.K)*V
  v_out = tf.cond(
      tf.logical_or(tf.equal(length_q, 0), tf.equal(length_k, 0)),
      is_zero,
      is_not_zero,
  )

  # Remove batch and head dimension
  v_out = tf.squeeze(v_out, axis=0)
  v_out = tf.squeeze(v_out, axis=0)
  return v_out


@expert_utils.add_name_scope()
def dot_product_single_head(q, k, v, gates_q, gates_k, bi):
  """Perform a dot product attention on a single sequence on a single head.

  This function dispatch the q, k, v and loop over the buckets to compute the
  attention dot product on each subsequences.

  Args:
    q (tf.Tensor): [length_q, depth_q]
    k (tf.Tensor): [length_k, depth_q]
    v (tf.Tensor): [length_k, depth_v]
    gates_q (tf.Tensor): One-hot vector of shape [length_q, nb_buckets]
    gates_k (tf.Tensor): One-hot vector of shape [length_k, nb_buckets]
    bi (BatchInfo): Contains the batch coordinates and sequence order

  Returns:
    tf.Tensor: [length_q, depth_v]
  """

  nb_buckets = gates_q.get_shape().as_list()[-1]

  q_dispatcher = expert_utils.SparseDispatcher(nb_buckets, gates_q)
  k_dispatcher = expert_utils.SparseDispatcher(nb_buckets, gates_k)

  def eventually_dispatch(dispatcher, value):
    if value is not None:
      return dispatcher.dispatch(value)
    return [None] * nb_buckets

  # Iterate over every dispatched group
  list_v_out = []
  for (
      q,
      k,
      v,
      qbc,
      qbo,
      kbc,
      kbo,
  ) in zip(
      # Dispatch queries, keys and values
      q_dispatcher.dispatch(q),
      k_dispatcher.dispatch(k),
      k_dispatcher.dispatch(v),
      # Also dispatch the sequence positions and batch coordinates
      eventually_dispatch(q_dispatcher, bi.coordinates),
      eventually_dispatch(q_dispatcher, bi.order),
      eventually_dispatch(k_dispatcher, bi.coordinates),
      eventually_dispatch(k_dispatcher, bi.order),
  ):
    list_v_out.append(expert_dot_product(
        q, k, v,
        info_q=BatchInfo(coordinates=qbc, order=qbo),
        info_k=BatchInfo(coordinates=kbc, order=kbo)
    ))

  # Combine all buckets together to restore the original length
  return q_dispatcher.combine(list_v_out)


def map_fn_switch(fn, elems, use_map_fn=True, **kwargs):
  """Construct the graph with either tf.map_fn or a python for loop.

  This function is mainly for for benchmarking purpose.

  tf.map_fn is dynamic but is much slower than creating a static graph with
  for loop. However, having a for loop make the graph much longer to build
  and can consume too much RAM on distributed setting.

  Args:
    fn (fct): same that tf.map_fn but for now can only return a single tensor
      value (instead of a tuple of tensor for the general case)
    elems (tuple): same that tf.map_fn
    use_map_fn (bool): If True, tf.map_fn is used, if False, for _ in _: is used
      instead
    **kwargs: Additional tf.map_fn arguments (ignored if use_map_fn is False)

  Returns:
    tf.Tensor: the output of tf.map_fn
  """
  if use_map_fn:
    return tf.map_fn(fn, elems, **kwargs)
  else:
    elems_unpacked = (
        tf.unstack(e) for e in elems
    )
    out_unpacked = [
        fn(e) for e in zip(*elems_unpacked)
    ]
    out = tf.stack(out_unpacked)
    return out


@expert_utils.add_name_scope()
def sparse_dot_product_attention(q, k, v, bi, use_map_fn, experts_params):
  """Sparse multihead self attention.

  Perform an approximation of the full multihead attention by dispatching
  the tokens using their keys/values. Thus the attention matrix are only
  computed each times on a subset of the tokens.

  Notes:
   * The function don't perform scaling here (multihead_attention does
  the /sqrt(depth)).
   * The padding should have been removed (so batch size should be 1 but length
   contains the elements from all different batches)
   * Right now, only self attention is supported so length_q and length_kv
   should be identical and the function will add triangular mask.
   * If bi.order is not None, The bias is added inside this function to
   prevent attention to the future.

  Args:
    q (tf.Tensor): Queries of shape [batch, heads, length_q, depth_k]
    k (tf.Tensor): Keys of shape [batch, heads, length_q, depth_k]
    v (tf.Tensor): Values of shape [batch, heads, length_kv, depth_v]
    bi (BatchInfo): Contains the batch coordinates and sequence order
    use_map_fn (bool): Use either tf.map_fn of python for loop to compute the
      heads separately
    experts_params (dict): Additional params for the local expert

  Returns:
    tf.Tensor: Approximation of Softmax(Q.K) * V, of shape
      [batch, heads, length_q, depth_v]
  """
  batch_size, nb_heads, _, depth = q.get_shape().as_list()
  batch_size = batch_size or tf.shape(q)[0]

  @expert_utils.add_name_scope()
  def flatten_first_dims(x):
    # Case 1: Either constant batch size of size 1 or batch already flattened
    if x.get_shape().as_list()[0] == 1:
      return tf.squeeze(x, axis=0)
    # Case 2: Flatten batch dimension
    else:
      x = tf.transpose(x, perm=[1, 0, 2, 3])
      x = tf.reshape(x, [nb_heads, -1, depth])
      return x

  def flatten_batch(x):
    if x is None:
      return x
    return expert_utils.flatten_all_but_last(x)

  q = flatten_first_dims(q)
  k = flatten_first_dims(k)
  v = flatten_first_dims(v)
  bi = BatchInfo(
      coordinates=flatten_batch(bi.coordinates),
      order=flatten_batch(bi.order),
  )

  # Unstack heads
  list_q = tf.unstack(q)  # list[tf.Tensor(shape=[batch * length, depth])]
  list_k = tf.unstack(k)
  list_v = tf.unstack(v)

  list_gates_q = []
  list_gates_k = []

  total_loss = 0.0
  # There might be a more optimized way to compute all heads at once
  for single_q, single_k, _ in zip(list_q, list_k, list_v):
    # Each head get its own dispatcher
    lhs_gating = LshGating(
        depth=single_q.get_shape().as_list()[-1],
        **experts_params
    )

    list_gates_q.append(lhs_gating.get_gates(single_q))
    list_gates_k.append(lhs_gating.get_gates(single_k))

  gates_q = tf.stack(list_gates_q)
  gates_k = tf.stack(list_gates_k)

  # Process each head separatly
  v_out = map_fn_switch(
      lambda args: dot_product_single_head(bi=bi, *args),
      elems=(q, k, v, gates_q, gates_k),
      dtype=(tf.float32),
      parallel_iterations=2,
      # back_prop=True,
      # swap_memory=False,
      # infer_shape=True,
      # name=None
      use_map_fn=use_map_fn,
  )

  # Restore original shape as expected by multihead_attention
  if isinstance(batch_size, int) and batch_size == 1:
    v_out = tf.expand_dims(v_out, axis=0)  # Restore batch_size = 1
  else:
    v_out = tf.reshape(v_out, [nb_heads, batch_size, -1, depth])
    v_out = tf.transpose(v_out, [1, 0, 2, 3])
  return v_out, total_loss / nb_heads


@expert_utils.add_name_scope()
def dot_product_batched_head(q, k, v, gates_q, gates_k, mask_right=False):
  """Perform a dot product attention on a single sequence on a single head.

  This function dispatch the q, k, v and loop over the buckets to compute the
  attention dot product on each subsequences.

  Args:
    q (tf.Tensor): [batch*heads, length_q, depth_q]
    k (tf.Tensor): [batch*heads, length_k, depth_q]
    v (tf.Tensor): [batch*heads, length_k, depth_v]
    gates_q (tf.Tensor): One-hot of shape [batch*heads, length_q, nb_buckets]
    gates_k (tf.Tensor): One-hot of shape [batch*heads, length_k, nb_buckets]
    mask_right (bool): Add a bias to prevent attention to the future

  Returns:
    tf.Tensor: [length_q, depth_v]
  """
  nb_buckets = tf.shape(gates_q)[-1]

  @expert_utils.add_name_scope()
  def get_dispatcher(gates):
    length = tf.shape(gates)[1]
    # Count the number of ones per batch (and keep the max value)
    nb_elems_to_dispatch = tf.reduce_sum(gates, axis=[1, 2])
    nb_elems_to_dispatch = tf.reduce_max(nb_elems_to_dispatch)
    nb_elems_to_dispatch = tf.to_int32(nb_elems_to_dispatch)
    capacity = nb_elems_to_dispatch // nb_buckets * 2  # Capacity is hardcoded
    capacity = tf.minimum(length, capacity)
    tf.summary.scalar("dispatch_capacity", capacity, family="lsh")
    return expert_utils.TruncatingDispatcher(gates, capacity)

  def add_summary_capacity(x, prefix):
    # Monitor if capacity overflow
    x = x[0, ...]  # Take first batch/head
    x = tf.reduce_sum(x, axis=0)
    tf.summary.scalar(prefix + "_min", tf.reduce_min(x), family="lsh")
    tf.summary.scalar(prefix + "_max", tf.reduce_max(x), family="lsh")
    tf.summary.histogram(prefix + "capacity_distribution", x, family="lsh")
    for i in range(3):  # Show the first 3 buckets
      tf.summary.scalar("{}_{}".format(prefix, i), x[i], family="lsh")
  add_summary_capacity(gates_q, "q")
  add_summary_capacity(gates_k, "k")

  q_dispatcher = get_dispatcher(gates_q)
  k_dispatcher = get_dispatcher(gates_k)

  q = q_dispatcher.dispatch(q)
  k = k_dispatcher.dispatch(k)
  v = k_dispatcher.dispatch(v)

  # Bias of shape [batch*heads, nb_buckets, 1, capacity] broadcasted to every
  # queries
  bias = tf.expand_dims((k_dispatcher.nonpadding() - 1.0) * 1e9, 2)
  if mask_right:
    q_coordinate = tf.to_float(
        tf.expand_dims(q_dispatcher.length_coordinate(), 3))
    k_coordinate = tf.to_float(
        tf.expand_dims(k_dispatcher.length_coordinate(), 2))
    bias += tf.to_float(tf.greater(k_coordinate, q_coordinate)) * -1e9
  # The sequence padding is not masked but is ignored on the next layers

  # q, k, v now have shape [batch*heads, nb_bucket, capacity, depth]
  # The buckets can be seen as different heads
  v_out = dot_product_attention(q, k, v, bias=bias)

  # Combine all buckets together to restore the original length
  return q_dispatcher.combine(v_out)


@expert_utils.add_name_scope()
def sparse_dot_product_attention_truncated(
    q, k, v,
    bi,  # Unused
    experts_params,
    use_map_fn=False,  # Unused
    mask_right=False,
):  # pylint: disable=unused-argument
  """Sparse multihead self attention.

  Perform an approximation of the full multihead attention by dispatching
  the tokens using their keys/values. Thus the attention matrix are only
  computed each times on a subset of the tokens.

  Notes:
   * The function don't perform scaling here (multihead_attention does
  the /sqrt(depth)).
   * The padding should have been removed (so batch size should be 1 but length
   contains the elements from all different batches)
   * Right now, only self attention is supported so length_q and length_kv
   should be identical and the function will add triangular mask.
   * If bi.order is not None, The bias is added inside this function to
   prevent attention to the future.

  Args:
    q (tf.Tensor): Queries of shape [batch, heads, length_q, depth_k]
    k (tf.Tensor): Keys of shape [batch, heads, length_q, depth_k]
    v (tf.Tensor): Values of shape [batch, heads, length_kv, depth_v]
    bi (BatchInfo): Contains the batch coordinates and sequence order
    experts_params (dict): Additional params for the local expert
    use_map_fn (bool): Use either tf.map_fn of python for loop to compute the
      heads separately
    mask_right (bool):
  Returns:
    tf.Tensor: Approximation of Softmax(Q.K) * V, of shape
      [batch, heads, length_q, depth_v]
  """
  # Currently depth is the same for for q and v
  batch_size, nb_heads, _, depth = q.get_shape().as_list()
  batch_size = batch_size or tf.shape(q)[0]

  total_loss = 0.0

  # Each head get its own dispatcher
  list_lsh = [
      LshGating(
          depth=depth,
          **experts_params
      ) for _ in range(nb_heads)
  ]

  @expert_utils.add_name_scope()
  def get_gates_head(x, add_first=False):
    """Return the gates for each heads of the current x.

    Args:
      x (tf.Tensor): of shape [batch, heads, length, depth]
      add_first (bool): if True, add the first element on each bucket

    Returns:
      tf.Tensor: gates of shape [batch, heads, length, num_buckets]
    """
    length = tf.shape(x)[2]

    # Invert heads/batch
    x = tf.transpose(x, perm=[1, 0, 2, 3])
    x = tf.reshape(x, [nb_heads, batch_size*length, depth])

    list_x = tf.unstack(x)  # list[tf.Tensor(shape=[batch * length, depth])]

    # Unstack heads
    list_gates = []
    # There might be a more optimized way to compute all heads at once
    for lsh, single_x in zip(list_lsh, list_x):
      # Each head get its own dispatcher
      gates = lsh.get_gates(single_x)
      nb_buckets = gates.get_shape().as_list()[-1]
      # Reshape to [batch, length, depth] but should consider sequence
      # padding in that case (also dispatch the padding)
      gates = tf.reshape(gates, [batch_size, length, nb_buckets])
      list_gates.append(gates)

    gates = tf.stack(list_gates)

    # Restore original shape
    gates = tf.reshape(gates, [nb_heads, batch_size, length, nb_buckets])
    gates = tf.transpose(gates, [1, 0, 2, 3])

    # Dispatch the first element to every gates to avoid empty buckets
    if add_first:
      gates = tf.maximum(
          gates,
          tf.reshape(tf.one_hot([0], length), [1, 1, length, 1])
      )

    return gates

  gates_q = get_gates_head(q)
  gates_k = get_gates_head(k, add_first=True)

  # [batch, heads, length, depth] => [batch*heads, length, depth]
  q, k, v, gates_q, gates_k = [
      combine_first_two_dimensions(t) for t in (q, k, v, gates_q, gates_k)]

  v_out = dot_product_batched_head(q, k, v, gates_q, gates_k, mask_right)

  # Restore original dimension
  v_out = tf.reshape(v_out, [batch_size, nb_heads, -1, depth])

  return v_out, total_loss / nb_heads


@expert_utils.add_var_scope()
def deconv_elems_1d(x, factor, out_depth=None):
  """Increase the length and change the dimensionality.

  Expand/project each positions of dim depth of the input into
  factor*tokens of dim out_depth

  Args:
    x (tf.Tensor): shape [batch_size, length, depth]
    factor (int): Multiplicative factor of each tokens.
    out_depth (int): Output depth (if None, keep depth constant)

  Returns:
    tf.Tensor: shape [batch_size, length*factor, out_depth]
  """
  out_depth = out_depth or x.get_shape().as_list()[-1]
  x = tf.expand_dims(x, 1)  # [batch_size, 1, length, depth]
  x = tf.layers.conv2d_transpose(
      inputs=x,
      filters=out_depth,
      kernel_size=(1, factor),
      strides=(1, factor),
      padding="valid",
      data_format="channels_last",
  )  # [batch_size, 1, length*factor, out_depth]
  x = tf.squeeze(x, 1)  # [batch_size, length*factor, depth]
  return x


@expert_utils.add_var_scope()
def conv_elems_1d(x, factor, out_depth=None):
  """Decrease the length and change the dimensionality.

  Merge/restore/compress factors positions of dim depth of the input into
  a single position of dim out_depth.
  This is basically just a strided convolution without overlapp
  between each strides.
  The original length has to be divided by factor.

  Args:
    x (tf.Tensor): shape [batch_size, length, depth]
    factor (int): Length compression factor.
    out_depth (int): Output depth

  Returns:
    tf.Tensor: shape [batch_size, length//factor, out_depth]
  """
  out_depth = out_depth or x.get_shape().as_list()[-1]
  # with tf.control_dependencies(  # Dynamic assertion
  #     [tf.assert_equal(tf.shape(x)[1] % factor, 0)]):
  x = tf.expand_dims(x, 1)  # [batch_size, 1, length, depth]
  x = tf.layers.conv2d(
      inputs=x,
      filters=out_depth,
      kernel_size=(1, factor),
      strides=(1, factor),
      padding="valid",
      data_format="channels_last",
  )  # [batch_size, 1, length//factor, out_depth]
  x = tf.squeeze(x, 1)  # [batch_size, length//factor, depth]
  return x


@expert_utils.add_var_scope()
def local_reduction_attention(x, block_length, multihead_params):
  """Reduce the length dimension using self attention.

  Args:
    x (tf.Tensor): float32 of shape [batch, length, depth]
    block_length (int): Block length for local attention (Compression factor)
    multihead_params (dict): parameters for multihead attention

  Returns:
    tf.Tensor: Compressed tensor of shape [batch, length // factor, depth]
  """
  @expert_utils.add_name_scope()
  def dot_product_self_local_attention_flattened(q, k, v):
    """Strided block local self-attention.

    No overlapp between the blocks.

    Args:
      q (tf.Tensor): shape [batch, heads, length, depth_k]
      k (tf.Tensor): shape [batch, heads, length, depth_k]
      v (tf.Tensor): shape [batch, heads, length, depth_v]

    Returns:
      tf.Tensor: shape [batch, heads, length, depth_v]
    """
    _, num_head, _, depth = q.get_shape().as_list()

    # Extract the blocks
    def pad_and_reshape(x):
      """Split the length dim into [num_block, block_length]."""
      length_x = tf.shape(x)[2]
      # Add some padding, but won't matter as the last block will never be
      # attended by the query (after compression)
      x = tf.pad(x, [
          [0, 0],
          [0, 0],
          [0, -length_x % block_length],
          [0, 0]
      ])
      x = tf.reshape(x, [
          tf.shape(x)[0],  # Batch
          num_head,  # Head
          tf.shape(x)[2] // block_length,  # Num blocks
          block_length,  # Block length
          depth,  # Depth
      ])
      return x

    q, k, v = [pad_and_reshape(t) for t in (q, k, v)]

    # Perform attention on the flattened dot product
    logits = tf.matmul(q, k, transpose_b=True)
    logits = tf.reshape(logits, [
        tf.shape(logits)[0],  # Batch
        num_head,  # Head
        tf.shape(logits)[2],  # Num blocks
        block_length**2,  # Flatten last dimension
    ])
    weights = tf.nn.softmax(logits)
    weights = tf.reshape(weights, [
        tf.shape(weights)[0],  # Batch
        num_head,  # Head
        tf.shape(weights)[2],  # Num blocks
        block_length,
        block_length,  # Restore the block length dimension
    ])
    weights = tf.reduce_sum(weights, axis=3, keep_dims=True)  # Compress block
    v_out = tf.matmul(weights, v)  # [1, block_length] @ [block_length, depth]
    v_out = tf.squeeze(v_out, axis=3)
    return v_out

  return multihead_attention(
      x,
      None,
      bias=None,
      output_depth=x.get_shape().as_list()[-1],
      attention_type=dot_product_self_local_attention_flattened,
      **multihead_params
  )


@expert_utils.add_var_scope()
def multihead_self_attention_reduced(
    x,
    memory_antecedent=None,
    bias=None,
    factor=None,
    multihead_params=None,
    nonlinearity="none",
    reduction_type="conv",
):
  """Reduce the length dimension by compressing with conv.

  Args:
    x (tf.Tensor): float32 of shape [batch, length, depth]
    memory_antecedent (tf.Tensor): Unsuported for now
    bias (tf.Tensor): Ignored
    factor (int): compression factor for the memory sequence
    multihead_params (dict): parameters for multihead attention
    nonlinearity (str): Add some non-linearity after the memory block
    reduction_type (str): type of compression

  Returns:
    (tf.Tensor): float32 of shape [batch, length, depth]

  Raises:
    ValueError: If reduction_type or nonlinearity is invalid
  """
  if not factor or not multihead_params:
    raise ValueError("factor and multihead_params should be set")
  if memory_antecedent is not None:
    raise NotImplementedError(
        "multihead_self_attention_reduced only works with self-attention")

  depth = x.get_shape().as_list()[-1]

  # Could try to have some overlapp between the blocks but that would
  # create conv artifacts, would make it difficult to not attend to the future
  # within one group and the padding should be handled specially.

  # Reduce the memory dimension
  if reduction_type == "attention":
    memory_x = local_reduction_attention(x, factor, multihead_params)
  elif reduction_type == "conv":
    # With valid padding, the last block won't be computed (not attended anyway)
    memory_x = conv_elems_1d(x, factor)
  else:
    raise ValueError("Unknown reduction type {}".format(reduction_type))

  if nonlinearity == "silu":
    memory_x *= tf.nn.sigmoid(memory_x)
  elif nonlinearity != "none":
    raise ValueError("Unknown non linearity {}".format(nonlinearity))

  memory_x = tf.concat(
      # Add the first elem to make it attendable by everyone (otherwise the
      # first block cannot attend to anything)
      [x[:, :1, :], memory_x],
      axis=1,
  )

  # Construct the bias
  @expert_utils.add_name_scope()
  def construct_bias_vectors(t, axis):
    length = tf.to_float(tf.shape(t)[1])
    length_coordinates = tf.range(length, dtype=tf.float32)
    length_coordinates = tf.expand_dims(length_coordinates, axis=axis)
    # [1, length_k] or [length_q, 1]
    return length_coordinates

  bias = tf.to_float(tf.greater(
      # Because we add the first elem to the memory block and it can be attended
      # by anyone,we don't need to add +1 anymore to prevent self attention
      # Use * factor to make sure the last tokens  of a block cannot attend the
      # block
      construct_bias_vectors(memory_x, 0) * factor,
      # +epsilon to avoid float equality
      construct_bias_vectors(x, 1) + 1e-3,
  )) * -1e9
  bias = tf.expand_dims(bias, axis=0)
  bias = tf.expand_dims(bias, axis=0)  # [1, 1, length_k, length_q]

  return multihead_attention(
      query_antecedent=x,
      memory_antecedent=memory_x,
      bias=bias,
      output_depth=depth,
      **multihead_params
  )


def scaled_dot_product_attention_simple(q, k, v, bias, name=None):
  """scaled dot-product attention.  One head.  One spatial dimension.

  Args:
    q: a Tensor with shape [batch, length_q, depth_k]
    k: a Tensor with shape [batch, length_kv, depth_k]
    v: a Tensor with shape [batch, length_kv, depth_v]
    bias: optional Tensor broadcastable to [batch, length_q, length_kv]
    name: an optional string

  Returns:
    A Tensor.
  """
  with tf.variable_scope(
      name, default_name="scaled_dot_product_attention_simple"):
    scalar = tf.rsqrt(tf.to_float(tf.shape(q)[2]))
    logits = tf.matmul(q * scalar, k, transpose_b=True)
    if bias is not None:
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    return tf.matmul(weights, v)


_function_cache = {}


def multihead_self_attention_memory_efficient(x,
                                              bias,
                                              num_heads,
                                              head_size=None,
                                              epsilon=1e-6,
                                              forget=True,
                                              test_vars=None,
                                              name=None):
  """Multihead scaled-dot-product self-attention.

  Includes layer norm.

  Returns multihead-self-attention(layer_norm(x))

  Computes one attention head at a time to avoid exhausting memory.

  If forget=True, then forget all forwards activations and recompute on
  the backwards pass.

  Args:
    x: a Tensor with shape [batch, length, input_size]
    bias: an attention bias tensor broadcastable to [batch, 1, length, length]
    num_heads: an integer
    head_size: an optional integer - defaults to input_size/num_heads
    epsilon: a float, for layer norm
    forget: a boolean - forget forwards activations and recompute on backprop
    test_vars: optional tuple of variables for testing purposes
    name: an optional string

  Returns:
    A Tensor.
  """
  io_size = x.get_shape().as_list()[-1]
  if head_size is None:
    assert io_size % num_heads == 0
    head_size = io_size / num_heads

  def forward_internal(x, wqkv, wo, attention_bias, norm_scale, norm_bias):
    """Forward function."""
    n = common_layers.layer_norm_compute_python(
        x, epsilon, norm_scale, norm_bias)
    wqkv_split = tf.unstack(wqkv, num=num_heads)
    wo_split = tf.unstack(wo, num=num_heads)
    y = 0
    for h in xrange(num_heads):
      with tf.control_dependencies([y] if h > 0 else []):
        combined = tf.nn.conv1d(n, wqkv_split[h], 1, "SAME")
        q, k, v = tf.split(combined, 3, axis=2)
        o = scaled_dot_product_attention_simple(q, k, v, attention_bias)
        y += tf.nn.conv1d(o, wo_split[h], 1, "SAME")
    return y

  key = ("multihead_self_attention_memory_efficient %s %s" %
         (num_heads, epsilon))
  if not forget:
    forward_fn = forward_internal
  elif key in _function_cache:
    forward_fn = _function_cache[key]
  else:
    @function.Defun(compiled=True)
    def grad_fn(x, wqkv, wo, attention_bias, norm_scale, norm_bias, dy):
      with tf.control_dependencies([dy]):
        n = common_layers.layer_norm_compute_python(
            x, epsilon, norm_scale, norm_bias)
        wqkv_split = tf.unstack(wqkv, num=num_heads)
        wo_split = tf.unstack(wo, num=num_heads)
        deps = []
        dwqkvs = []
        dwos = []
        dn = 0
        for h in xrange(num_heads):
          with tf.control_dependencies(deps):
            combined = tf.nn.conv1d(n, wqkv_split[h], 1, "SAME")
            q, k, v = tf.split(combined, 3, axis=2)
            o = scaled_dot_product_attention_simple(q, k, v, attention_bias)
            partial_y = tf.nn.conv1d(o, wo_split[h], 1, "SAME")
            pdn, dwqkvh, dwoh = tf.gradients(
                ys=[partial_y],
                xs=[n, wqkv_split[h], wo_split[h]],
                grad_ys=[dy])
            dn += pdn
            dwqkvs.append(dwqkvh)
            dwos.append(dwoh)
            deps = [dn, dwqkvh, dwoh]
        dwqkv = tf.stack(dwqkvs)
        dwo = tf.stack(dwos)
        with tf.control_dependencies(deps):
          dx, dnorm_scale, dnorm_bias = tf.gradients(
              ys=[n], xs=[x, norm_scale, norm_bias], grad_ys=[dn])
        return (dx, dwqkv, dwo, tf.zeros_like(attention_bias),
                dnorm_scale, dnorm_bias)

    @function.Defun(grad_func=grad_fn, compiled=True,
                    separate_compiled_gradients=True)
    def forward_fn(x, wqkv, wo, attention_bias, norm_scale, norm_bias):
      return forward_internal(
          x, wqkv, wo, attention_bias, norm_scale, norm_bias)
    _function_cache[key] = forward_fn

  if bias is not None:
    bias = tf.squeeze(bias, 1)
  with tf.variable_scope(name, default_name="multihead_attention", values=[x]):
    # TODO(noam): it would be nice to save memory by casting x to float16
    # here, but this causes problems with the gradients.  Figure out if there
    # is a way to leave the gradients as float32.
    if test_vars is not None:
      wqkv, wo, norm_scale, norm_bias = list(test_vars)
    else:
      wqkv = tf.get_variable(
          "wqkv", [num_heads, 1, io_size, 3 * head_size],
          initializer=tf.random_normal_initializer(stddev=io_size**-0.5))
      wo = tf.get_variable(
          "wo", [num_heads, 1, head_size, io_size],
          initializer=tf.random_normal_initializer(
              stddev=(head_size * num_heads)**-0.5))
      norm_scale, norm_bias = common_layers.layer_norm_vars(io_size)
    y = forward_fn(x, wqkv, wo, bias, norm_scale, norm_bias)
    y.set_shape(x.get_shape())
    return y


multihead_attention_sparse_dot_prod = functools.partial(
    multihead_attention, attention_type=sparse_dot_product_attention)

multihead_attention_sparse_truncated = functools.partial(
    multihead_attention, attention_type=sparse_dot_product_attention_truncated)
