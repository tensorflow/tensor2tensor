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

"""Models for semi-parallel and parallel decoding with the transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_model
class TransformerBlockParallel(transformer.Transformer):
  """Transformer that predicts blocks of the output in parallel."""

  def body(self, features):
    assert self._hparams.block_size > 0
    assert not common_layers.is_xla_compiled()
    assert "targets_segmentation" not in features

    decoder_output = super(TransformerBlockParallel, self).body(features)
    assert not isinstance(decoder_output, tuple)
    assert len(decoder_output.shape) == 4

    relu_dropout_broadcast_dims = (
        common_layers.comma_separated_string_to_integer_list(
            getattr(self._hparams, "relu_dropout_broadcast_dims", "")))

    with tf.variable_scope("block_size_%d" % self._hparams.block_size):
      block_output = common_layers.dense_relu_dense(
          decoder_output,
          self._hparams.block_size * self._hparams.filter_size,
          self._hparams.block_size * self._hparams.hidden_size,
          dropout=self._hparams.relu_dropout,
          dropout_broadcast_dims=relu_dropout_broadcast_dims)

    batch_size, length = common_layers.shape_list(decoder_output)[:2]
    block_output = tf.reshape(block_output, [
        batch_size,
        length,
        self._hparams.block_size,
        self._hparams.hidden_size
    ])

    block_output = common_layers.layer_postprocess(
        decoder_output, block_output, self._hparams)

    return block_output

  def top(self, body_output, features):
    assert self._hparams.block_size > 0

    if (self._hparams.mode == tf.estimator.ModeKeys.TRAIN or
        self._hparams.mode == tf.estimator.ModeKeys.EVAL):
      if self._hparams.mode == tf.estimator.ModeKeys.TRAIN:
        features["block_index"] = tf.random_uniform(
            shape=[], minval=0, maxval=self._hparams.block_size, dtype=tf.int64)
      else:
        features["block_index"] = 0
      k = features["block_index"]
      body_output = body_output[:, :, k:k + 1, :]

    return super(TransformerBlockParallel, self).top(body_output, features)

  def loss(self, logits, features):
    assert self._hparams.block_size > 0

    def shift_left_4d(x, k):
      return tf.pad(x, [[0, 0], [0, k], [0, 0], [0, 0]])[:, k:, :, :]

    targets = features["targets"]
    assert len(targets.shape) == 4

    targets = tf.concat([
        shift_left_4d(targets, i)
        for i in range(self._hparams.block_size)
    ], axis=2)

    if (self._hparams.mode == tf.estimator.ModeKeys.TRAIN or
        self._hparams.mode == tf.estimator.ModeKeys.EVAL):
      assert "block_index" in features
      k = features["block_index"]
      targets = targets[:, :, k:k + 1, :]

    features["targets"] = targets

    loss = super(TransformerBlockParallel, self).loss(logits, features)

    if self._hparams.mode == tf.estimator.ModeKeys.TRAIN:
      loss_num, loss_den = loss
      loss_val = loss_num / loss_den
      for i in range(self._hparams.block_size):
        # Hack: if you report a loss of NaN, TensorBoard will plot a point at
        # the previous value without a connecting line. This is used here to
        # separate out the training losses by block index.
        one_or_nan = tf.cond(tf.equal(k, i), lambda: 1.0, lambda: float("nan"))
        tf.summary.scalar(
            "block_index_%d" % i, one_or_nan * loss_val, family="losses")

    return loss

  def _greedy_infer(self, features, decode_length, use_tpu=False):
    assert not use_tpu
    return self._slow_greedy_infer_guess_and_check(features, decode_length)

  def _beam_decode(self, features, decode_length, beam_size, top_beams, alpha):
    raise NotImplementedError

  def _slow_greedy_infer_guess_and_check(self, features, decode_length):
    assert self._hparams.block_size > 0
    assert self._hparams.force_full_predict
    assert self._hparams.sampling_method == "argmax"
    assert self._decode_hparams.batch_size == 1
    assert self._decode_hparams.block_size > 0
    assert self._decode_hparams.block_size <= self._hparams.block_size
    assert self._decode_hparams.guess_and_check_top_k > 0

    inputs_old = features["inputs"]
    assert "targets" not in features

    assert len(features["inputs"].shape) in [3, 4]
    if len(features["inputs"].shape) < 4:
      features["inputs"] = tf.expand_dims(features["inputs"], 2)

    block_size = self._decode_hparams.block_size
    decode_length += tf.shape(features["inputs"])[1]

    def while_exit_cond(result, length):  # pylint: disable=unused-argument
      return tf.logical_and(
          length < decode_length,
          tf.reduce_all(
              tf.not_equal(result[:, :length, :, :], text_encoder.EOS_ID))
      )

    def infer_step(result, length):
      """Inference step."""

      def print_info(result, length, new_length):
        vocab = self.problem_hparams.vocabulary["targets"]
        tf.logging.info(
            "length=%s new_length=%s length_diff=%s new_suffix=%s",
            length,
            new_length,
            new_length - length,
            str([
                vocab._subtoken_id_to_subtoken_string(index)  # pylint: disable=protected-access
                for index in result[0, -block_size:, 0, 0][:new_length - length]
            ]).decode("unicode-escape"),
        )

      features["targets"] = tf.pad(result, [[0, 0], [0, 1], [0, 0], [0, 0]])
      samples, logits, losses = self.sample(features)  # pylint: disable=unused-variable

      _, top_k_indices = tf.nn.top_k(
          logits[:, :-1, :1, :, :],
          k=self._decode_hparams.guess_and_check_top_k)
      in_top_k = tf.reduce_any(
          tf.equal(tf.to_int64(top_k_indices), tf.expand_dims(result, 4)),
          axis=4)

      eos_cumsum = tf.cumsum(
          tf.to_int32(tf.equal(result, text_encoder.EOS_ID)), axis=1)
      after_eos = tf.greater(common_layers.shift_right(eos_cumsum), 0)

      correct = tf.logical_and(in_top_k, tf.logical_not(after_eos))
      correct_cumsum = tf.cumsum(tf.to_int32(correct), axis=1)
      perfect_cumsum = 1 + tf.range(tf.shape(correct)[1])
      for axis in [0, 2, 3]:
        perfect_cumsum = tf.expand_dims(perfect_cumsum, axis=axis)

      new_length = tf.reduce_sum(
          tf.to_int32(tf.equal(correct_cumsum, perfect_cumsum)), axis=1)
      new_length = tf.squeeze(new_length, axis=[0, 1, 2])
      new_length = tf.minimum(new_length, decode_length)

      new_result = tf.concat([
          result[:, :new_length, :, :],
          tf.reshape(
              samples[:, new_length, :block_size, :], [1, block_size, 1, 1])
      ], axis=1)

      with tf.control_dependencies([
          tf.py_func(print_info, [result, length, new_length], [])
      ]):
        new_result = tf.identity(new_result)

      return new_result, new_length

    result = tf.zeros((1, 0, 1, 1), dtype=tf.int64)
    length = tf.squeeze(tf.zeros(1, dtype=tf.int32))

    result, length = tf.while_loop(
        while_exit_cond,
        infer_step,
        [result, length],
        shape_invariants=[
            tf.TensorShape([1, None, 1, 1]),
            tf.TensorShape([]),
        ],
        back_prop=False,
        parallel_iterations=1)

    result = result[:, :length, :, :]

    features["inputs"] = inputs_old

    return {
        "outputs": result,
        "scores": None,
    }


@registry.register_hparams
def transformer_base_bs1():
  hparams = transformer.transformer_base()
  hparams.add_hparam("block_size", 1)
  return hparams


@registry.register_hparams
def transformer_base_bs2():
  hparams = transformer.transformer_base()
  hparams.add_hparam("block_size", 2)
  return hparams


@registry.register_hparams
def transformer_base_bs3():
  hparams = transformer.transformer_base()
  hparams.add_hparam("block_size", 3)
  return hparams


@registry.register_hparams
def transformer_base_bs4():
  hparams = transformer.transformer_base()
  hparams.add_hparam("block_size", 4)
  return hparams


@registry.register_hparams
def transformer_base_bs5():
  hparams = transformer.transformer_base()
  hparams.add_hparam("block_size", 5)
  return hparams


@registry.register_hparams
def transformer_base_bs6():
  hparams = transformer.transformer_base()
  hparams.add_hparam("block_size", 6)
  return hparams


@registry.register_hparams
def transformer_base_bs7():
  hparams = transformer.transformer_base()
  hparams.add_hparam("block_size", 7)
  return hparams


@registry.register_hparams
def transformer_base_bs8():
  hparams = transformer.transformer_base()
  hparams.add_hparam("block_size", 8)
  return hparams


@registry.register_hparams
def transformer_base_bs9():
  hparams = transformer.transformer_base()
  hparams.add_hparam("block_size", 9)
  return hparams


@registry.register_hparams
def transformer_base_bs10():
  hparams = transformer.transformer_base()
  hparams.add_hparam("block_size", 10)
  return hparams


@registry.register_hparams
def transformer_big_bs1():
  hparams = transformer.transformer_big()
  hparams.add_hparam("block_size", 1)
  return hparams


@registry.register_hparams
def transformer_tiny_bs1():
  hparams = transformer.transformer_tiny()
  hparams.add_hparam("block_size", 1)
  return hparams


@registry.register_hparams
def transformer_tiny_bs2():
  hparams = transformer.transformer_tiny()
  hparams.add_hparam("block_size", 2)
  return hparams


@registry.register_hparams
def transformer_tiny_bs3():
  hparams = transformer.transformer_tiny()
  hparams.add_hparam("block_size", 3)
  return hparams
