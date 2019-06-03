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

"""Reversible layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.optimize import linear_sum_assignment
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability import edward2 as ed


class Reverse(tf.keras.layers.Layer):
  """Swaps the forward and reverse transformations of a layer."""

  def __init__(self, reversible_layer, **kwargs):
    super(Reverse, self).__init__(**kwargs)
    if not hasattr(reversible_layer, 'reverse'):
      raise ValueError('Layer passed-in has not implemented "reverse" method: '
                       '{}'.format(reversible_layer))
    self.call = reversible_layer.reverse
    self.reverse = reversible_layer.call


class DiscreteAutoregressiveFlow(tf.keras.layers.Layer):
  """A discrete reversible layer.

  The flow takes as input a one-hot Tensor of shape `[..., length, vocab_size]`.
  The flow returns a Tensor of same shape and dtype. (To enable gradients, the
  input must have float dtype.)

  For the forward pass, the flow computes in serial:

  ```none
  outputs = []
  for t in range(length):
    new_inputs = [outputs, inputs[..., t, :]]
    net = layer(new_inputs)
    loc, scale = tf.split(net, 2, axis=-1)
    loc = tf.argmax(loc, axis=-1)
    scale = tf.argmax(scale, axis=-1)
    new_outputs = (((inputs - loc) * inverse(scale)) % vocab_size)[..., -1, :]
    outputs.append(new_outputs)
  ```

  For the reverse pass, the flow computes in parallel:

  ```none
  net = layer(inputs)
  loc, scale = tf.split(net, 2, axis=-1)
  loc = tf.argmax(loc, axis=-1)
  scale = tf.argmax(scale, axis=-1)
  outputs = (loc + scale * inputs) % vocab_size
  ```

  The modular arithmetic happens in one-hot space.

  If `x` is a discrete random variable, the induced probability mass function on
  the outputs `y = flow(x)` is

  ```none
  p(y) = p(flow.reverse(y)).
  ```

  The location-only transform is always invertible ([integers modulo
  `vocab_size` form an additive group](
  https://en.wikipedia.org/wiki/Modular_arithmetic)). The transform with a scale
  is invertible if the scale and `vocab_size` are coprime (see
  [prime fields](https://en.wikipedia.org/wiki/Finite_field)).
  """

  def __init__(self, layer, temperature, **kwargs):
    """Constructs flow.

    Args:
      layer: Two-headed masked network taking the inputs and returning a
        real-valued Tensor of shape `[..., length, 2*vocab_size]`.
        Alternatively, `layer` may return a Tensor of shape
        `[..., length, vocab_size]` to be used as the location transform; the
        scale transform will be hard-coded to 1.
      temperature: Positive value determining bias of gradient estimator.
      **kwargs: kwargs of parent class.
    """
    super(DiscreteAutoregressiveFlow, self).__init__(**kwargs)
    self.layer = layer
    self.temperature = temperature

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    self.vocab_size = input_shape[-1]
    if isinstance(self.vocab_size, tf.Dimension):
      self.vocab_size = self.vocab_size.value
    if self.vocab_size is None:
      raise ValueError('The last dimension of the inputs to '
                       '`DiscreteAutoregressiveFlow` should be defined. Found '
                       '`None`.')
    self.built = True

  def __call__(self, inputs, *args, **kwargs):
    if not isinstance(inputs, ed.RandomVariable):
      return super(DiscreteAutoregressiveFlow, self).__call__(
          inputs, *args, **kwargs)
    return TransformedRandomVariable(inputs, self)

  def call(self, inputs, **kwargs):
    """Forward pass for left-to-right autoregressive generation."""
    inputs = tf.convert_to_tensor(inputs)
    length = inputs.shape[-2].value
    if length is None:
      raise NotImplementedError('length dimension must be known.')
    # Form initial sequence tensor of shape [..., 1, vocab_size]. In a loop, we
    # incrementally build a Tensor of shape [..., t, vocab_size] as t grows.
    outputs = self._initial_call(inputs[..., 0, :], length, **kwargs)
    # TODO(trandustin): Use tf.while_loop. Unrolling is memory-expensive for big
    # models and not valid for variable lengths.
    for t in range(1, length):
      outputs = self._per_timestep_call(outputs,
                                        inputs[..., t, :],
                                        length,
                                        t,
                                        **kwargs)
    return outputs

  def _initial_call(self, new_inputs, length, **kwargs):
    """Returns Tensor of shape [..., 1, vocab_size].

    Args:
      new_inputs: Tensor of shape [..., vocab_size], the new input to generate
        its output.
      length: Length of final desired sequence.
      **kwargs: Optional keyword arguments to layer.
    """
    inputs = new_inputs[..., tf.newaxis, :]
    # TODO(trandustin): To handle variable lengths, extend MADE to subset its
    # input and output layer weights rather than pad inputs.
    batch_ndims = inputs.shape.ndims - 2
    padded_inputs = tf.pad(
        inputs, [[0, 0]] * batch_ndims + [[0, length - 1], [0, 0]])
    net = self.layer(padded_inputs, **kwargs)
    if net.shape[-1] == 2 * self.vocab_size:
      loc, scale = tf.split(net, 2, axis=-1)
      loc = loc[..., 0:1, :]
      loc = tf.cast(one_hot_argmax(loc, self.temperature), inputs.dtype)
      scale = scale[..., 0:1, :]
      scale = tf.cast(one_hot_argmax(scale, self.temperature), inputs.dtype)
      inverse_scale = multiplicative_inverse(scale, self.vocab_size)
      shifted_inputs = one_hot_minus(inputs, loc)
      outputs = one_hot_multiply(shifted_inputs, inverse_scale)
    elif net.shape[-1] == self.vocab_size:
      loc = net
      loc = loc[..., 0:1, :]
      loc = tf.cast(one_hot_argmax(loc, self.temperature), inputs.dtype)
      outputs = one_hot_minus(inputs, loc)
    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    return outputs

  def _per_timestep_call(self,
                         current_outputs,
                         new_inputs,
                         length,
                         timestep,
                         **kwargs):
    """Returns Tensor of shape [..., timestep+1, vocab_size].

    Args:
      current_outputs: Tensor of shape [..., timestep, vocab_size], the so-far
        generated sequence Tensor.
      new_inputs: Tensor of shape [..., vocab_size], the new input to generate
        its output given current_outputs.
      length: Length of final desired sequence.
      timestep: Current timestep.
      **kwargs: Optional keyword arguments to layer.
    """
    inputs = tf.concat([current_outputs,
                        new_inputs[..., tf.newaxis, :]], axis=-2)
    # TODO(trandustin): To handle variable lengths, extend MADE to subset its
    # input and output layer weights rather than pad inputs.
    batch_ndims = inputs.shape.ndims - 2
    padded_inputs = tf.pad(
        inputs, [[0, 0]] * batch_ndims + [[0, length - timestep - 1], [0, 0]])
    net = self.layer(padded_inputs, **kwargs)
    if net.shape[-1] == 2 * self.vocab_size:
      loc, scale = tf.split(net, 2, axis=-1)
      loc = loc[..., :(timestep+1), :]
      loc = tf.cast(one_hot_argmax(loc, self.temperature), inputs.dtype)
      scale = scale[..., :(timestep+1), :]
      scale = tf.cast(one_hot_argmax(scale, self.temperature), inputs.dtype)
      inverse_scale = multiplicative_inverse(scale, self.vocab_size)
      shifted_inputs = one_hot_minus(inputs, loc)
      new_outputs = one_hot_multiply(shifted_inputs, inverse_scale)
    elif net.shape[-1] == self.vocab_size:
      loc = net
      loc = loc[..., :(timestep+1), :]
      loc = tf.cast(one_hot_argmax(loc, self.temperature), inputs.dtype)
      new_outputs = one_hot_minus(inputs, loc)
    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    outputs = tf.concat([current_outputs, new_outputs[..., -1:, :]], axis=-2)
    if not tf.executing_eagerly():
      outputs.set_shape([None] * batch_ndims + [timestep+1, self.vocab_size])
    return outputs

  def reverse(self, inputs, **kwargs):
    """Reverse pass returning the inverse autoregressive transformation."""
    if not self.built:
      self._maybe_build(inputs)

    net = self.layer(inputs, **kwargs)
    if net.shape[-1] == 2 * self.vocab_size:
      loc, scale = tf.split(net, 2, axis=-1)
      scale = tf.cast(one_hot_argmax(scale, self.temperature), inputs.dtype)
      scaled_inputs = one_hot_multiply(inputs, scale)
    elif net.shape[-1] == self.vocab_size:
      loc = net
      scaled_inputs = inputs
    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    loc = tf.cast(one_hot_argmax(loc, self.temperature), inputs.dtype)
    outputs = one_hot_add(loc, scaled_inputs)
    return outputs

  def log_det_jacobian(self, inputs):
    return tf.cast(0, inputs.dtype)


class DiscreteBipartiteFlow(tf.keras.layers.Layer):
  """A discrete reversible layer.

  The flow takes as input a one-hot Tensor of shape `[..., length, vocab_size]`.
  The flow returns a Tensor of same shape and dtype. (To enable gradients, the
  input must have float dtype.)

  For the forward pass, the flow computes:

  ```none
  net = layer(mask * inputs)
  loc, scale = tf.split(net, 2, axis=-1)
  loc = tf.argmax(loc, axis=-1)
  scale = tf.argmax(scale, axis=-1)
  outputs = ((inputs - (1-mask) * loc) * (1-mask) * inverse(scale)) % vocab_size
  ```

  For the reverse pass, the flow computes:

  ```none
  net = layer(mask * inputs)
  loc, scale = tf.split(net, 2, axis=-1)
  loc = tf.argmax(loc, axis=-1)
  scale = tf.argmax(scale, axis=-1)
  outputs = ((1-mask) * loc + (1-mask) * scale * inputs) % vocab_size
  ```

  The modular arithmetic happens in one-hot space.

  If `x` is a discrete random variable, the induced probability mass function on
  the outputs `y = flow(x)` is

  ```none
  p(y) = p(flow.reverse(y)).
  ```

  The location-only transform is always invertible ([integers modulo
  `vocab_size` form an additive group](
  https://en.wikipedia.org/wiki/Modular_arithmetic)). The transform with a scale
  is invertible if the scale and `vocab_size` are coprime (see
  [prime fields](https://en.wikipedia.org/wiki/Finite_field)).
  """

  def __init__(self, layer, mask, temperature, **kwargs):
    """Constructs flow.

    Args:
      layer: Two-headed masked network taking the inputs and returning a
        real-valued Tensor of shape `[..., length, 2*vocab_size]`.
        Alternatively, `layer` may return a Tensor of shape
        `[..., length, vocab_size]` to be used as the location transform; the
        scale transform will be hard-coded to 1.
      mask: binary Tensor of shape `[length]` forming the bipartite assignment.
      temperature: Positive value determining bias of gradient estimator.
      **kwargs: kwargs of parent class.
    """
    super(DiscreteBipartiteFlow, self).__init__(**kwargs)
    self.layer = layer
    self.mask = mask
    self.temperature = temperature

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    self.vocab_size = input_shape[-1]
    if isinstance(self.vocab_size, tf.Dimension):
      self.vocab_size = self.vocab_size.value
    if self.vocab_size is None:
      raise ValueError('The last dimension of the inputs to '
                       '`DiscreteBipartiteFlow` should be defined. Found '
                       '`None`.')
    self.built = True

  def __call__(self, inputs, *args, **kwargs):
    if not isinstance(inputs, ed.RandomVariable):
      return super(DiscreteBipartiteFlow, self).__call__(
          inputs, *args, **kwargs)
    return TransformedRandomVariable(inputs, self)

  def call(self, inputs, **kwargs):
    """Forward pass for bipartite generation."""
    inputs = tf.convert_to_tensor(inputs)
    batch_ndims = inputs.shape.ndims - 2
    mask = tf.reshape(tf.cast(self.mask, inputs.dtype),
                      [1] * batch_ndims + [-1, 1])
    masked_inputs = mask * inputs
    net = self.layer(masked_inputs, **kwargs)
    if net.shape[-1] == 2 * self.vocab_size:
      loc, scale = tf.split(net, 2, axis=-1)
      loc = tf.cast(one_hot_argmax(loc, self.temperature), inputs.dtype)
      scale = tf.cast(one_hot_argmax(scale, self.temperature), inputs.dtype)
      inverse_scale = multiplicative_inverse(scale, self.vocab_size)
      shifted_inputs = one_hot_minus(inputs, loc)
      masked_outputs = (1. - mask) * one_hot_multiply(shifted_inputs,
                                                      inverse_scale)
    elif net.shape[-1] == self.vocab_size:
      loc = net
      loc = tf.cast(one_hot_argmax(loc, self.temperature), inputs.dtype)
      masked_outputs = (1. - mask) * one_hot_minus(inputs, loc)
    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    outputs = masked_inputs + masked_outputs
    return outputs

  def reverse(self, inputs, **kwargs):
    """Reverse pass for the inverse bipartite transformation."""
    if not self.built:
      self._maybe_build(inputs)

    inputs = tf.convert_to_tensor(inputs)
    batch_ndims = inputs.shape.ndims - 2
    mask = tf.reshape(tf.cast(self.mask, inputs.dtype),
                      [1] * batch_ndims + [-1, 1])
    masked_inputs = mask * inputs
    net = self.layer(masked_inputs, **kwargs)
    if net.shape[-1] == 2 * self.vocab_size:
      loc, scale = tf.split(net, 2, axis=-1)
      scale = tf.cast(one_hot_argmax(scale, self.temperature), inputs.dtype)
      scaled_inputs = one_hot_multiply(inputs, scale)
    elif net.shape[-1] == self.vocab_size:
      loc = net
      scaled_inputs = inputs
    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    loc = tf.cast(one_hot_argmax(loc, self.temperature), inputs.dtype)
    masked_outputs = (1. - mask) * one_hot_add(loc, scaled_inputs)
    outputs = masked_inputs + masked_outputs
    return outputs

  def log_det_jacobian(self, inputs):
    return tf.cast(0, inputs.dtype)


class SinkhornAutoregressiveFlow(tf.keras.layers.Layer):
  """A discrete reversible layer using Sinkhorn normalization for permutations.

  The flow takes as input a one-hot Tensor of shape `[..., length, vocab_size]`.
  The flow returns a Tensor of same shape and dtype. (To enable gradients, the
  input must have float dtype.)
  """

  def __init__(self, layer, temperature, **kwargs):
    """Constructs flow.

    Args:
      layer: Masked network taking inputs with shape `[..., length, vocab_size]`
        and returning a real-valued Tensor of shape
        `[..., length, vocab_size ** 2]`. Sinkhorn iterations are applied to
        each `layer` output to produce permutation matrices.
      temperature: Positive value determining bias of gradient estimator.
      **kwargs: kwargs of parent class.
    """
    super(SinkhornAutoregressiveFlow, self).__init__(**kwargs)
    self.layer = layer
    self.temperature = temperature

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    self.vocab_size = input_shape[-1]
    if isinstance(self.vocab_size, tf.Dimension):
      self.vocab_size = self.vocab_size.value
    if self.vocab_size is None:
      raise ValueError('The last dimension of the inputs to '
                       '`DiscreteAutoregressiveFlow` should be defined. Found '
                       '`None`.')
    self.built = True

  def __call__(self, inputs, *args, **kwargs):
    if not isinstance(inputs, ed.RandomVariable):
      return super(SinkhornAutoregressiveFlow, self).__call__(
          inputs, *args, **kwargs)
    return TransformedRandomVariable(inputs, self)

  def call(self, inputs, **kwargs):
    """Forward pass for left-to-right autoregressive generation."""
    inputs = tf.convert_to_tensor(inputs)
    length = inputs.shape[-2].value
    if length is None:
      raise NotImplementedError('length dimension must be known.')
    # Form initial sequence tensor of shape [..., 1, vocab_size]. In a loop, we
    # incrementally build a Tensor of shape [..., t, vocab_size] as t grows.
    outputs = self._initial_call(inputs[..., 0, :], length, **kwargs)
    for t in range(1, length):
      outputs = self._per_timestep_call(outputs,
                                        inputs[..., t, :],
                                        length,
                                        t,
                                        **kwargs)
    return outputs

  def _initial_call(self, new_inputs, length, **kwargs):
    """Returns Tensor of shape [..., 1, vocab_size].

    Args:
      new_inputs: Tensor of shape [..., vocab_size], the new input to generate
        its output.
      length: Length of final desired sequence.
      **kwargs: Optional keyword arguments to layer.
    """
    inputs = new_inputs[..., tf.newaxis, :]
    # TODO(trandustin): To handle variable lengths, extend MADE to subset its
    # input and output layer weights rather than pad inputs.
    batch_ndims = inputs.shape.ndims - 2
    padded_inputs = tf.pad(
        inputs, [[0, 0]] * batch_ndims + [[0, length - 1], [0, 0]])
    temperature = 1.
    logits = self.layer(padded_inputs / temperature, **kwargs)
    logits = logits[..., 0:1, :]
    logits = tf.reshape(
        logits,
        logits.shape[:-1].concatenate([self.vocab_size, self.vocab_size]))
    soft = sinkhorn(logits)
    hard = tf.cast(soft_to_hard_permutation(soft), inputs.dtype)
    hard = tf.reshape(hard, logits.shape)
    # Inverse of permutation matrix is its transpose.
    # inputs is [batch_size, timestep + 1, vocab_size].
    # hard is [batch_size, timestep + 1, vocab_size, vocab_size].
    outputs = tf.matmul(inputs[..., tf.newaxis, :],
                        hard,
                        transpose_b=True)[..., 0, :]
    return outputs

  def _per_timestep_call(self,
                         current_outputs,
                         new_inputs,
                         length,
                         timestep,
                         **kwargs):
    """Returns Tensor of shape [..., timestep+1, vocab_size].

    Args:
      current_outputs: Tensor of shape [..., timestep, vocab_size], the so-far
        generated sequence Tensor.
      new_inputs: Tensor of shape [..., vocab_size], the new input to generate
        its output given current_outputs.
      length: Length of final desired sequence.
      timestep: Current timestep.
      **kwargs: Optional keyword arguments to layer.
    """
    inputs = tf.concat([current_outputs,
                        new_inputs[..., tf.newaxis, :]], axis=-2)
    # TODO(trandustin): To handle variable lengths, extend MADE to subset its
    # input and output layer weights rather than pad inputs.
    batch_ndims = inputs.shape.ndims - 2
    padded_inputs = tf.pad(
        inputs, [[0, 0]] * batch_ndims + [[0, length - timestep - 1], [0, 0]])
    logits = self.layer(padded_inputs, **kwargs)
    logits = logits[..., :(timestep+1), :]
    logits = tf.reshape(
        logits,
        logits.shape[:-1].concatenate([self.vocab_size, self.vocab_size]))
    soft = sinkhorn(logits / self.temperature)
    hard = tf.cast(soft_to_hard_permutation(soft), inputs.dtype)
    hard = tf.reshape(hard, logits.shape)
    # Inverse of permutation matrix is its transpose.
    # inputs is [batch_size, timestep + 1, vocab_size].
    # hard is [batch_size, timestep + 1, vocab_size, vocab_size].
    new_outputs = tf.matmul(inputs[..., tf.newaxis, :],
                            hard,
                            transpose_b=True)[..., 0, :]
    outputs = tf.concat([current_outputs, new_outputs[..., -1:, :]], axis=-2)
    if not tf.executing_eagerly():
      outputs.set_shape([None] * batch_ndims + [timestep+1, self.vocab_size])
    return outputs

  def reverse(self, inputs, **kwargs):
    """Reverse pass returning the inverse autoregressive transformation."""
    if not self.built:
      self._maybe_build(inputs)

    logits = self.layer(inputs, **kwargs)
    logits = tf.reshape(
        logits,
        logits.shape[:-1].concatenate([self.vocab_size, self.vocab_size]))
    soft = sinkhorn(logits / self.temperature, n_iters=20)
    hard = soft_to_hard_permutation(soft)
    hard = tf.reshape(hard, logits.shape)
    # Recover the permutation by right-multiplying by the permutation matrix.
    outputs = tf.matmul(inputs[..., tf.newaxis, :], hard)[..., 0, :]
    return outputs

  def log_det_jacobian(self, inputs):
    return tf.cast(0, inputs.dtype)


def soft_to_hard_permutation(inputs):
  """Returns permutation matrices by solving a matching problem.

  Solves linear sum assignment to convert doubly-stochastic matrices to
  permutation matrices. It uses scipy.optimize.linear_sum_assignment to solve
  the optimization problem max_P sum_i,j M_i,j P_i,j with P a permutation
  matrix. Notice the negative sign; the reason, the original function solves a
  minimization problem.

  Code is adapted from Mena et al. [1].

  [1] Gonzalo Mena, David Belanger, Scott Linderman, Jasper Snoek.
  Learning latent permutations with Gumbel-Sinkhorn networks. International
  Conference on Learning Representations, 2018.

  Args:
    inputs: A `Tensor` with shape `[:, vocab_size, vocab_size]` that is
      doubly-stochastic in its last two dimensions.

  Returns:
    outputs: A hard permutation `Tensor` with the same shape as `inputs` (in
      other words the last two dimensions are doubly-stochastic and each element
      is 0 or 1).
  """

  def hungarian(x):
    if x.ndim == 2:
      x = np.reshape(x, [1, x.shape[0], x.shape[1]])
    sol = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)
    for i in range(x.shape[0]):
      sol[i, :] = linear_sum_assignment(-x[i, :])[1].astype(np.int32)
    return sol

  vocab_size = inputs.shape[-1]
  # Note: tf.py_func isn't currently supported on headless GPUs.
  # TODO(vafa): Fix tf.py_func headless GPU bug.
  permutation_lists = tf.py_func(hungarian, [inputs], tf.int32)
  hard = tf.one_hot(permutation_lists, depth=vocab_size)
  outputs = tf.stop_gradient(hard - inputs) + inputs
  return outputs


def one_hot_argmax(inputs, temperature, axis=-1):
  """Returns one-hot of argmax with backward pass set to softmax-temperature."""
  vocab_size = inputs.shape[-1]
  hard = tf.one_hot(tf.argmax(inputs, axis=axis),
                    depth=vocab_size,
                    axis=axis,
                    dtype=inputs.dtype)
  soft = tf.nn.softmax(inputs / temperature, axis=axis)
  outputs = soft + tf.stop_gradient(hard - soft)
  return outputs


def one_hot_add(inputs, shift):
  """Performs (inputs + shift) % vocab_size in the one-hot space.

  Args:
    inputs: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
      Tensor.
    shift: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
      Tensor specifying how much to shift the corresponding one-hot vector in
      inputs. Soft values perform a "weighted shift": for example,
      shift=[0.2, 0.3, 0.5] performs a linear combination of 0.2 * shifting by
      zero; 0.3 * shifting by one; and 0.5 * shifting by two.

  Returns:
    Tensor of same shape and dtype as inputs.
  """
  # Compute circular 1-D convolution with shift as the kernel.
  inputs = tf.cast(inputs, tf.complex64)
  shift = tf.cast(shift, tf.complex64)
  return tf.real(tf.signal.ifft(tf.signal.fft(inputs) * tf.signal.fft(shift)))


def one_hot_minus(inputs, shift):
  """Performs (inputs - shift) % vocab_size in the one-hot space.

  Args:
    inputs: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
      Tensor.
    shift: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
      Tensor specifying how much to shift the corresponding one-hot vector in
      inputs. Soft values perform a "weighted shift": for example,
      shift=[0.2, 0.3, 0.5] performs a linear combination of 0.2 * shifting by
      zero; 0.3 * shifting by one; and 0.5 * shifting by two.

  Returns:
    Tensor of same shape and dtype as inputs.
  """
  # TODO(trandustin): Implement with circular conv1d.
  inputs = tf.convert_to_tensor(inputs)
  shift = tf.cast(shift, inputs.dtype)
  vocab_size = inputs.shape[-1].value
  # Form a [..., vocab_size, vocab_size] matrix. Each batch element of
  # inputs will vector-matrix multiply the vocab_size x vocab_size matrix. This
  # "shifts" the inputs batch element by the corresponding shift batch element.
  shift_matrix = tf.stack([tf.roll(shift, i, axis=-1)
                           for i in range(vocab_size)], axis=-2)
  outputs = tf.einsum('...v,...uv->...u', inputs, shift_matrix)
  return outputs


def one_hot_multiply(inputs, scale):
  """Performs (inputs * scale) % vocab_size in the one-hot space.

  Args:
    inputs: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
      Tensor.
    scale: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
      Tensor specifying how much to scale the corresponding one-hot vector in
      inputs. Soft values perform a "weighted scale": for example,
      scale=[0.2, 0.3, 0.5] performs a linear combination of
      0.2 * scaling by zero; 0.3 * scaling by one; and 0.5 * scaling by two.

  Returns:
    Tensor of same shape and dtype as inputs.
  """
  # TODO(trandustin): Implement with circular conv1d.
  inputs = tf.convert_to_tensor(inputs)
  scale = tf.cast(scale, inputs.dtype)
  batch_shape = inputs.shape[:-1].as_list()
  vocab_size = inputs.shape[-1].value
  # Form a [..., vocab_size, vocab_size] tensor. The ith row of the
  # batched vocab_size x vocab_size matrix represents scaling inputs by i.
  permutation_matrix = tf.floormod(
      tf.tile(tf.range(vocab_size)[:, tf.newaxis], [1, vocab_size]) *
      tf.range(vocab_size)[tf.newaxis], vocab_size)
  permutation_matrix = tf.one_hot(permutation_matrix, depth=vocab_size, axis=-1)
  # Scale the inputs according to the permutation matrix of all possible scales.
  scaled_inputs = tf.einsum('...v,avu->...au', inputs, permutation_matrix)
  scaled_inputs = tf.concat([tf.zeros(batch_shape + [1, vocab_size]),
                             scaled_inputs[..., 1:, :]], axis=-2)
  # Reduce rows of the scaled inputs by the scale values. This forms a
  # weighted linear combination of scaling by zero, scaling by one, and so on.
  outputs = tf.einsum('...v,...vu->...u', scale, scaled_inputs)
  return outputs


def py_multiplicative_inverse(a, n):
  """Multiplicative inverse of a modulo n (in Python).

  Implements extended Euclidean algorithm.

  Args:
    a: int-like np.ndarray.
    n: int.

  Returns:
    Multiplicative inverse as an int32 np.ndarray with same shape as a.
  """
  batched_a = np.asarray(a, dtype=np.int32)
  batched_inverse = []
  for a in np.nditer(batched_a):
    inverse = 0
    new_inverse = 1
    remainder = n
    new_remainder = a
    while new_remainder != 0:
      quotient = remainder // new_remainder
      (inverse, new_inverse) = (new_inverse, inverse - quotient * new_inverse)
      (remainder, new_remainder) = (new_remainder,
                                    remainder - quotient * new_remainder)
    if remainder > 1:
      return ValueError(
          'Inverse for {} modulo {} does not exist.'.format(a, n))
    if inverse < 0:
      inverse += n
    batched_inverse.append(inverse)
  return np.asarray(batched_inverse, dtype=np.int32).reshape(batched_a.shape)


def multiplicative_inverse(a, n):
  """Multiplicative inverse of a modulo n.

  Args:
    a: Tensor of shape [..., vocab_size]. It denotes an integer in the one-hot
      space.
    n: int Tensor of shape [...].

  Returns:
    Tensor of same shape and dtype as a.
  """
  a = tf.convert_to_tensor(a)
  n = tf.convert_to_tensor(n)
  vocab_size = a.shape[-1].value
  a_dtype = a.dtype
  sparse_a = tf.argmax(a, axis=-1)
  sparse_outputs = tf.py_func(
      py_multiplicative_inverse, [sparse_a, n], tf.int32)
  sparse_outputs.set_shape(sparse_a.shape)
  outputs = tf.one_hot(sparse_outputs, depth=vocab_size, dtype=a_dtype)
  return outputs


class ActNorm(tf.keras.layers.Layer):
  """Actnorm, an affine reversible layer (Prafulla and Kingma, 2018).

  Weights use data-dependent initialization in which outputs have zero mean
  and unit variance per channel (last dimension). The mean/variance statistics
  are computed from the first batch of inputs.
  """

  def __init__(self, epsilon=tf.keras.backend.epsilon(), **kwargs):
    super(ActNorm, self).__init__(**kwargs)
    self.epsilon = epsilon

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    last_dim = input_shape[-1]
    if isinstance(last_dim, tf.Dimension):
      last_dim = last_dim.value
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `ActNorm` '
                       'should be defined. Found `None`.')
    bias = self.add_weight('bias', [last_dim], dtype=self.dtype)
    log_scale = self.add_weight('log_scale', [last_dim], dtype=self.dtype)
    # Set data-dependent initializers.
    bias = bias.assign(self.bias_initial_value)
    with tf.control_dependencies([bias]):
      self.bias = bias
    log_scale = log_scale.assign(self.log_scale_initial_value)
    with tf.control_dependencies([log_scale]):
      self.log_scale = log_scale
    self.built = True

  def __call__(self, inputs, *args, **kwargs):
    if not self.built:
      mean, variance = tf.nn.moments(
          inputs, axes=[i for i in range(inputs.shape.ndims - 1)])
      self.bias_initial_value = -mean
      # TODO(trandustin): Optionally, actnorm multiplies log_scale by a fixed
      # log_scale factor (e.g., 3.) and initializes by
      # initial_value / log_scale_factor.
      self.log_scale_initial_value = tf.log(
          1. / (tf.sqrt(variance) + self.epsilon))

    if not isinstance(inputs, ed.RandomVariable):
      return super(ActNorm, self).__call__(inputs, *args, **kwargs)
    return TransformedRandomVariable(inputs, self)

  def call(self, inputs):
    return (inputs + self.bias) * tf.exp(self.log_scale)

  def reverse(self, inputs):
    return inputs * tf.exp(-self.log_scale) - self.bias

  def log_det_jacobian(self, inputs):
    """Returns log det | dx / dy | = num_events * sum log | scale |."""
    del inputs  # unused
    # Number of events is number of all elements excluding the batch and
    # channel dimensions.
    num_events = tf.reduce_prod(tf.shape(inputs)[1:-1])
    log_det_jacobian = num_events * tf.reduce_sum(self.log_scale)
    return log_det_jacobian


class MADE(tf.keras.Model):
  """Masked autoencoder for distribution estimation (Germain et al., 2015).

  MADE takes as input a real Tensor of shape [..., length, channels] and returns
  a Tensor of shape [..., length, units] and same dtype. It masks layer weights
  to satisfy autoregressive constraints with respect to the length dimension. In
  particular, for a given ordering, each input dimension of length can be
  reconstructed from previous dimensions.

  The output's units dimension captures per-time-step representations. For
  example, setting units to 2 can parameterize the location and log-scale of an
  autoregressive Gaussian distribution.
  """

  def __init__(self,
               units,
               hidden_dims,
               input_order='left-to-right',
               hidden_order='left-to-right',
               activation=None,
               use_bias=True,
               **kwargs):
    """Constructs network.

    Args:
      units: Positive integer, dimensionality of the output space.
      hidden_dims: list with the number of hidden units per layer. It does not
        include the output layer; those number of units will always be set to
        the input dimension multiplied by `num_heads`. Each hidden unit size
        must be at least the size of length (otherwise autoregressivity is not
        possible).
      input_order: Order of degrees to the input units: 'random',
        'left-to-right', 'right-to-left', or an array of an explicit order.
        For example, 'left-to-right' builds an autoregressive model
        p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
      hidden_order: Order of degrees to the hidden units: 'random',
        'left-to-right'. If 'left-to-right', hidden units are allocated equally
        (up to a remainder term) to each degree.
      activation: Activation function.
      use_bias: Whether to use a bias.
      **kwargs: Keyword arguments of parent class.
    """
    super(MADE, self).__init__(**kwargs)
    self.units = int(units)
    self.hidden_dims = hidden_dims
    self.input_order = input_order
    self.hidden_order = hidden_order
    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias
    self.network = tf.keras.Sequential([])

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    length = input_shape[-2]
    channels = input_shape[-1]
    if isinstance(length, tf.Dimension):
      length = length.value
    if isinstance(channels, tf.Dimension):
      channels = channels.value
    if length is None or channels is None:
      raise ValueError('The two last dimensions of the inputs to '
                       '`MADE` should be defined. Found `None`.')
    masks = create_masks(input_dim=length,
                         hidden_dims=self.hidden_dims,
                         input_order=self.input_order,
                         hidden_order=self.hidden_order)

    # Input-to-hidden layer: [..., length, channels] -> [..., hidden_dims[0]].
    self.network.add(tf.keras.layers.Reshape([length * channels]))
    # Tile the mask so each element repeats contiguously; this is compatible
    # with the autoregressive contraints unlike naive tiling.
    mask = masks[0]
    mask = tf.tile(mask[:, tf.newaxis, :], [1, channels, 1])
    mask = tf.reshape(mask, [mask.shape[0] * channels, mask.shape[-1]])
    if self.hidden_dims:
      layer = tf.keras.layers.Dense(
          self.hidden_dims[0],
          kernel_initializer=make_masked_initializer(mask),
          kernel_constraint=make_masked_constraint(mask),
          activation=self.activation,
          use_bias=self.use_bias)
      self.network.add(layer)

    # Hidden-to-hidden layers: [..., hidden_dims[l-1]] -> [..., hidden_dims[l]].
    for l in range(1, len(self.hidden_dims)):
      layer = tf.keras.layers.Dense(
          self.hidden_dims[l],
          kernel_initializer=make_masked_initializer(masks[l]),
          kernel_constraint=make_masked_constraint(masks[l]),
          activation=self.activation,
          use_bias=self.use_bias)
      self.network.add(layer)

    # Hidden-to-output layer: [..., hidden_dims[-1]] -> [..., length, units].
    # Tile the mask so each element repeats contiguously; this is compatible
    # with the autoregressive contraints unlike naive tiling.
    if self.hidden_dims:
      mask = masks[-1]
    mask = tf.tile(mask[..., tf.newaxis], [1, 1, self.units])
    mask = tf.reshape(mask, [mask.shape[0], mask.shape[1] * self.units])
    layer = tf.keras.layers.Dense(
        length * self.units,
        kernel_initializer=make_masked_initializer(mask),
        kernel_constraint=make_masked_constraint(mask),
        activation=None,
        use_bias=self.use_bias)
    self.network.add(layer)
    self.network.add(tf.keras.layers.Reshape([length, self.units]))
    self.built = True

  def call(self, inputs):
    return self.network(inputs)


def create_degrees(input_dim,
                   hidden_dims,
                   input_order='left-to-right',
                   hidden_order='left-to-right'):
  """Returns a list of degree vectors, one for each input and hidden layer.

  A unit with degree d can only receive input from units with degree < d. Output
  units always have the same degree as their associated input unit.

  Args:
    input_dim: Number of inputs.
    hidden_dims: list with the number of hidden units per layer. It does not
      include the output layer. Each hidden unit size must be at least the size
      of length (otherwise autoregressivity is not possible).
    input_order: Order of degrees to the input units: 'random', 'left-to-right',
      'right-to-left', or an array of an explicit order. For example,
      'left-to-right' builds an autoregressive model
      p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
    hidden_order: Order of degrees to the hidden units: 'random',
      'left-to-right'. If 'left-to-right', hidden units are allocated equally
      (up to a remainder term) to each degree.
  """
  if (isinstance(input_order, str) and
      input_order not in ('random', 'left-to-right', 'right-to-left')):
    raise ValueError('Input order is not valid.')
  if hidden_order not in ('random', 'left-to-right'):
    raise ValueError('Hidden order is not valid.')

  degrees = []
  if isinstance(input_order, str):
    input_degrees = np.arange(1, input_dim + 1)
    if input_order == 'right-to-left':
      input_degrees = np.flip(input_degrees, 0)
    elif input_order == 'random':
      np.random.shuffle(input_degrees)
  else:
    input_order = np.array(input_order)
    if np.all(np.sort(input_order) != np.arange(1, input_dim + 1)):
      raise ValueError('invalid input order')
    input_degrees = input_order
  degrees.append(input_degrees)

  for units in hidden_dims:
    if hidden_order == 'random':
      min_prev_degree = min(np.min(degrees[-1]), input_dim - 1)
      hidden_degrees = np.random.randint(
          low=min_prev_degree, high=input_dim, size=units)
    elif hidden_order == 'left-to-right':
      hidden_degrees = (np.arange(units) % max(1, input_dim - 1) +
                        min(1, input_dim - 1))
    degrees.append(hidden_degrees)
  return degrees


def create_masks(input_dim,
                 hidden_dims,
                 input_order='left-to-right',
                 hidden_order='left-to-right'):
  """Returns a list of binary mask matrices respecting autoregressive ordering.

  Args:
    input_dim: Number of inputs.
    hidden_dims: list with the number of hidden units per layer. It does not
      include the output layer; those number of units will always be set to
      input_dim downstream. Each hidden unit size must be at least the size of
      length (otherwise autoregressivity is not possible).
    input_order: Order of degrees to the input units: 'random', 'left-to-right',
      'right-to-left', or an array of an explicit order. For example,
      'left-to-right' builds an autoregressive model
      p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
    hidden_order: Order of degrees to the hidden units: 'random',
      'left-to-right'. If 'left-to-right', hidden units are allocated equally
      (up to a remainder term) to each degree.
  """
  degrees = create_degrees(input_dim, hidden_dims, input_order, hidden_order)
  masks = []
  # Create input-to-hidden and hidden-to-hidden masks.
  for input_degrees, output_degrees in zip(degrees[:-1], degrees[1:]):
    mask = tf.cast(input_degrees[:, np.newaxis] <= output_degrees, tf.float32)
    masks.append(mask)

  # Create hidden-to-output mask.
  mask = tf.cast(degrees[-1][:, np.newaxis] < degrees[0], tf.float32)
  masks.append(mask)
  return masks


def make_masked_initializer(mask):
  initializer = tf.keras.initializers.glorot_uniform()
  def masked_initializer(shape, dtype=None, partition_info=None):
    return mask * initializer(shape, dtype, partition_info)
  return masked_initializer


def make_masked_constraint(mask):
  constraint = tf.identity
  def masked_constraint(x):
    return mask * constraint(x)
  return masked_constraint


def sinkhorn(inputs, n_iters=20):
  """Performs incomplete Sinkhorn normalization to inputs.

  By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved  matrix
  with positive entries can be turned into a doubly-stochastic matrix
  (i.e. its rows and columns add up to one) via the succesive row and column
  normalization.
  -To ensure positivity, the effective input to sinkhorn has to be
  exp(inputs) (elementwise).
  -However, for stability, sinkhorn works in the log-space. It is only at
   return time that entries are exponentiated.

  Code is adapted from Mena et al. [2].

  [1] Richard Sinkhorn and Paul Knopp. Concerning nonnegative matrices and
  doubly stochastic matrices. Pacific Journal of Mathematics, 1967.

  [2] Gonzalo Mena, David Belanger, Scott Linderman, Jasper Snoek.
  Learning latent permutations with Gumbel-Sinkhorn networks. International
  Conference on Learning Representations, 2018.

  Args:
    inputs: A `Tensor` with shape `[..., vocab_size, vocab_size]`.
    n_iters: Number of sinkhorn iterations (in practice, as little as 20
      iterations are needed to achieve decent convergence for `vocab_size` ~100)

  Returns:
    outputs: A `Tensor` of close-to-doubly-stochastic matrices with shape
      `[:, vocab_size, vocab_size]`.
  """
  vocab_size = tf.shape(inputs)[-1]
  log_alpha = tf.reshape(inputs, [-1, vocab_size, vocab_size])

  for _ in range(n_iters):
    log_alpha -= tf.reshape(tf.reduce_logsumexp(log_alpha, axis=2),
                            [-1, vocab_size, 1])
    log_alpha -= tf.reshape(tf.reduce_logsumexp(log_alpha, axis=1),
                            [-1, 1, vocab_size])
  outputs = tf.exp(log_alpha)
  return outputs


class TransformedDistribution(tfp.distributions.Distribution):
  """Distribution of f(x), where x ~ p(x) and f is reversible."""

  def __init__(self, base, reversible_layer, name=None):
    """Constructs a transformed distribution.

    Args:
      base: Base distribution.
      reversible_layer: Callable with methods `reverse` and `log_det_jacobian`.
      name: Name for scoping operations in the class.
    """
    self.base = base
    self.reversible_layer = reversible_layer
    if name is None:
      name = reversible_layer.name + base.name
    super(TransformedDistribution, self).__init__(
        base.dtype,
        base.reparameterization_type,
        base.validate_args,
        base.allow_nan_stats,
        parameters=dict(locals()),
        name=name)

  def _event_shape_tensor(self):
    return self.base.event_shape_tensor()

  def _event_shape(self):
    return self.base.event_shape

  def _batch_shape_tensor(self):
    return self.base.batch_shape_tensor()

  def _batch_shape(self):
    return self.base.batch_shape

  def __getitem__(self, slices):
    overrides = {'base': self.base[slices]}
    return self.copy(**overrides)

  def _call_sample_n(self, sample_shape, seed, name, **kwargs):
    x = self.base.sample(sample_shape, seed, **kwargs)
    y = self.reversible_layer(x)
    return y

  def _log_prob(self, value):
    x = self.reversible_layer.reverse(value)
    log_det_jacobian = self.reversible_layer.log_det_jacobian(value)
    return self.base.log_prob(x) + log_det_jacobian

  def _prob(self, value):
    if not hasattr(self.base, '_prob'):
      return tf.exp(self.log_prob(value))
    x = self.reversible_layer.reverse(value)
    log_det_jacobian = self.reversible_layer.log_det_jacobian(value)
    return self.base.prob(x) * tf.exp(log_det_jacobian)

  def _log_cdf(self, value):
    x = self.reversible_layer.reverse(value)
    return self.base.log_cdf(x)

  def _cdf(self, value):
    x = self.reversible_layer.reverse(value)
    return self.base.cdf(x)

  def _log_survival_function(self, value):
    x = self.reversible_layer.reverse(value)
    return self.base.log_survival_function(x)

  def _survival_function(self, value):
    x = self.reversible_layer.reverse(value)
    return self.base.survival_function(x)

  def _quantile(self, value):
    inverse_cdf = self.base.quantile(value)
    return self.reversible_layer(inverse_cdf)

  def _entropy(self):
    dummy = tf.zeros(
        tf.concat([self.batch_shape_tensor(), self.event_shape_tensor()], 0),
        dtype=self.dtype)
    log_det_jacobian = self.reversible_layer.log_det_jacobian(dummy)
    entropy = self.base.entropy() - log_det_jacobian
    return entropy


@ed.interceptable
def TransformedRandomVariable(random_variable,  # pylint: disable=invalid-name
                              reversible_layer,
                              name=None,
                              sample_shape=(),
                              value=None):
  """Random variable for f(x), where x ~ p(x) and f is reversible."""
  return ed.RandomVariable(
      distribution=TransformedDistribution(random_variable.distribution,
                                           reversible_layer,
                                           name=name),
      sample_shape=sample_shape,
      value=value)
