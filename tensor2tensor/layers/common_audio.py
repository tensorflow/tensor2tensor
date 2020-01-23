# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""Utils for audio."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import scipy.signal
import tensorflow.compat.v1 as tf


def add_delta_deltas(filterbanks, name=None):
  """Compute time first and second-order derivative channels.

  Args:
    filterbanks: float32 tensor with shape [batch_size, len, num_bins, 1]
    name: scope name

  Returns:
    float32 tensor with shape [batch_size, len, num_bins, 3]
  """
  delta_filter = np.array([2, 1, 0, -1, -2])
  delta_delta_filter = scipy.signal.convolve(delta_filter, delta_filter, "full")

  delta_filter_stack = np.array(
      [[0] * 4 + [1] + [0] * 4, [0] * 2 + list(delta_filter) + [0] * 2,
       list(delta_delta_filter)],
      dtype=np.float32).T[:, None, None, :]

  delta_filter_stack /= np.sqrt(
      np.sum(delta_filter_stack**2, axis=0, keepdims=True))

  filterbanks = tf.nn.conv2d(
      filterbanks, delta_filter_stack, [1, 1, 1, 1], "SAME", data_format="NHWC",
      name=name)
  return filterbanks


def compute_mel_filterbank_features(
    waveforms,
    sample_rate=16000, dither=1.0 / np.iinfo(np.int16).max, preemphasis=0.97,
    frame_length=25, frame_step=10, fft_length=None,
    window_fn=functools.partial(tf.signal.hann_window, periodic=True),
    lower_edge_hertz=80.0, upper_edge_hertz=7600.0, num_mel_bins=80,
    log_noise_floor=1e-3, apply_mask=True):
  """Implement mel-filterbank extraction using tf ops.

  Args:
    waveforms: float32 tensor with shape [batch_size, max_len]
    sample_rate: sampling rate of the waveform
    dither: stddev of Gaussian noise added to waveform to prevent quantization
      artefacts
    preemphasis: waveform high-pass filtering constant
    frame_length: frame length in ms
    frame_step: frame_Step in ms
    fft_length: number of fft bins
    window_fn: windowing function
    lower_edge_hertz: lowest frequency of the filterbank
    upper_edge_hertz: highest frequency of the filterbank
    num_mel_bins: filterbank size
    log_noise_floor: clip small values to prevent numeric overflow in log
    apply_mask: When working on a batch of samples, set padding frames to zero
  Returns:
    filterbanks: a float32 tensor with shape [batch_size, len, num_bins, 1]
  """
  # `stfts` is a complex64 Tensor representing the short-time Fourier
  # Transform of each signal in `signals`. Its shape is
  # [batch_size, ?, fft_unique_bins]
  # where fft_unique_bins = fft_length // 2 + 1

  # Find the wave length: the largest index for which the value is !=0
  # note that waveforms samples that are exactly 0.0 are quite common, so
  # simply doing sum(waveforms != 0, axis=-1) will not work correctly.
  wav_lens = tf.reduce_max(
      tf.expand_dims(tf.range(tf.shape(waveforms)[1]), 0) *
      tf.to_int32(tf.not_equal(waveforms, 0.0)),
      axis=-1) + 1
  if dither > 0:
    waveforms += tf.random_normal(tf.shape(waveforms), stddev=dither)
  if preemphasis > 0:
    waveforms = waveforms[:, 1:] - preemphasis * waveforms[:, :-1]
    wav_lens -= 1
  frame_length = int(frame_length * sample_rate / 1e3)
  frame_step = int(frame_step * sample_rate / 1e3)
  if fft_length is None:
    fft_length = int(2**(np.ceil(np.log2(frame_length))))

  stfts = tf.signal.stft(
      waveforms,
      frame_length=frame_length,
      frame_step=frame_step,
      fft_length=fft_length,
      window_fn=window_fn,
      pad_end=True)

  stft_lens = (wav_lens + (frame_step - 1)) // frame_step
  masks = tf.to_float(tf.less_equal(
      tf.expand_dims(tf.range(tf.shape(stfts)[1]), 0),
      tf.expand_dims(stft_lens, 1)))

  # An energy spectrogram is the magnitude of the complex-valued STFT.
  # A float32 Tensor of shape [batch_size, ?, 257].
  magnitude_spectrograms = tf.abs(stfts)

  # Warp the linear-scale, magnitude spectrograms into the mel-scale.
  num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
  linear_to_mel_weight_matrix = (
      tf.signal.linear_to_mel_weight_matrix(
          num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
          upper_edge_hertz))
  mel_spectrograms = tf.tensordot(
      magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
  # Note: Shape inference for tensordot does not currently handle this case.
  mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))

  log_mel_sgram = tf.log(tf.maximum(log_noise_floor, mel_spectrograms))

  if apply_mask:
    log_mel_sgram *= tf.expand_dims(tf.to_float(masks), -1)

  return tf.expand_dims(log_mel_sgram, -1, name="mel_sgrams")
