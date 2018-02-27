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

"""Common classes for automatic speech recogntion (ASR) datasets.

The audio import uses sox to generate normalized waveforms, please install
it as appropriate (e.g. using apt-get or yum).
"""

import functools
import os
from subprocess import call
import tempfile

# Dependency imports

import numpy as np
from scipy.io import wavfile
import scipy.signal

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import metrics
from tensor2tensor.utils import modality
from tensor2tensor.utils import registry

import tensorflow as tf


#
# ASR Feature pipeline in TF.
#
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
    window_fn=functools.partial(tf.contrib.signal.hann_window, periodic=True),
    lower_edge_hertz=80.0, upper_edge_hertz=7600.0, num_mel_bins=80,
    log_noise_floor=1e-3, apply_mask=True):
  """Implement mel-filterbank extraction using tf ops.

  Args:
    waveforms: float32 tensor with shape [batch_size, max_len]
    sample_rate: sampling rate of the waveform
    dither: stddev of Gaussian noise added to waveform to prevent quantization
      artefacts
    preemphasis: waveform high-pass filtering costant
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

  stfts = tf.contrib.signal.stft(
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
      tf.contrib.signal.linear_to_mel_weight_matrix(
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


#
# Audio problem definition
#
class AudioEncoder(object):
  """Encoder class for saving and loading waveforms."""

  def __init__(self, num_reserved_ids=0, sample_rate=16000):
    assert num_reserved_ids == 0
    self._sample_rate = sample_rate

  @property
  def num_reserved_ids(self):
    return 0

  def encode(self, s):
    """Transform a string with a filename into a list of float32.

    Args:
      s: path to the file with a waveform.

    Returns:
      samples: list of int16s
    """
    # Make sure that the data is a single channel, 16bit, 16kHz wave.
    # TODO(chorowski): the directory may not be writable, this should fallback
    # to a temp path, and provide instructions for instaling sox.
    if not s.endswith(".wav"):
      out_filepath = s + ".wav"
      if not os.path.exists(out_filepath):
        call(["sox", "-r", "16k", "-b", "16", "-c", "1", s, out_filepath])
      s = out_filepath
    rate, data = wavfile.read(s)
    assert rate == self._sample_rate
    assert len(data.shape) == 1
    if data.dtype not in [np.float32, np.float64]:
      data = data.astype(np.float32) / np.iinfo(data.dtype).max
    return data.tolist()

  def decode(self, ids):
    """Transform a sequence of float32 into a waveform.

    Args:
      ids: list of integers to be converted.

    Returns:
      Path to the temporary file where the waveform was saved.

    Raises:
      ValueError: if the ids are not of the appropriate size.
    """
    _, tmp_file_path = tempfile.mkstemp()
    wavfile.write(tmp_file_path, self._sample_rate, np.asarray(ids))
    return tmp_file_path

  def decode_list(self, ids):
    """Transform a sequence of int ids into an image file.

    Args:
      ids: list of integers to be converted.

    Returns:
      Singleton list: path to the temporary file where the wavfile was saved.
    """
    return [self.decode(ids)]

  @property
  def vocab_size(self):
    return 256


class ByteTextEncoderWithEos(text_encoder.ByteTextEncoder):
  """Encodes each byte to an id and appends the EOS token."""

  def encode(self, s):
    return super(ByteTextEncoderWithEos, self).encode(s) + [text_encoder.EOS_ID]


class SpeechRecognitionProblem(problem.Problem):
  """Base class for speech recognition problems."""

  def hparams(self, defaults, model_hparams):
    p = model_hparams
    # Filterbank extraction in bottom instead of preprocess_example is faster.
    p.add_hparam("audio_preproc_in_bottom", False)
    # The trainer seems to reserve memory for all members of the input dict
    p.add_hparam("audio_keep_example_waveforms", False)
    p.add_hparam("audio_sample_rate", 16000)
    p.add_hparam("audio_preemphasis", 0.97)
    p.add_hparam("audio_dither", 1.0 / np.iinfo(np.int16).max)
    p.add_hparam("audio_frame_length", 25.0)
    p.add_hparam("audio_frame_step", 10.0)
    p.add_hparam("audio_lower_edge_hertz", 20.0)
    p.add_hparam("audio_upper_edge_hertz", 8000.0)
    p.add_hparam("audio_num_mel_bins", 80)
    p.add_hparam("audio_add_delta_deltas", True)

    p = defaults
    # p.stop_at_eos = int(False)
    p.input_modality = {"inputs": ("audio:speech_recognition_modality", None)}
    p.target_modality = (registry.Modalities.SYMBOL, 256)

  @property
  def is_character_level(self):
    return True

  @property
  def input_space_id(self):
    return problem.SpaceID.AUDIO_SPECTRAL

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_CHR

  def feature_encoders(self, _):
    return {
        "inputs": None,  # Put None to make sure that the logic in
                         # decoding.py doesn't try to convert the floats
                         # into text...
        "waveforms": AudioEncoder(),
        "targets": ByteTextEncoderWithEos(),
    }

  def example_reading_spec(self):
    data_fields = {
        "waveforms": tf.VarLenFeature(tf.float32),
        "targets": tf.VarLenFeature(tf.int64),
    }

    data_items_to_decoders = None

    return data_fields, data_items_to_decoders

  def preprocess_example(self, example, mode, hparams):
    p = hparams
    if p.audio_preproc_in_bottom:
      example["inputs"] = tf.expand_dims(
          tf.expand_dims(example["waveforms"], -1), -1)
    else:
      waveforms = tf.expand_dims(example["waveforms"], 0)
      mel_fbanks = compute_mel_filterbank_features(
          waveforms,
          sample_rate=p.audio_sample_rate,
          dither=p.audio_dither,
          preemphasis=p.audio_preemphasis,
          frame_length=p.audio_frame_length,
          frame_step=p.audio_frame_step,
          lower_edge_hertz=p.audio_lower_edge_hertz,
          upper_edge_hertz=p.audio_upper_edge_hertz,
          num_mel_bins=p.audio_num_mel_bins,
          apply_mask=False)
      if p.audio_add_delta_deltas:
        mel_fbanks = add_delta_deltas(mel_fbanks)
      fbank_size = common_layers.shape_list(mel_fbanks)
      assert fbank_size[0] == 1

      # This replaces CMVN estimation on data

      mean = tf.reduce_mean(mel_fbanks, keepdims=True, axis=1)
      variance = tf.reduce_mean((mel_fbanks-mean)**2, keepdims=True, axis=1)
      mel_fbanks = (mel_fbanks - mean) / variance

      # Later models like to flatten the two spatial dims. Instead, we add a
      # unit spatial dim and flatten the frequencies and channels.
      example["inputs"] = tf.reshape(
          mel_fbanks, [fbank_size[1], fbank_size[2], fbank_size[3]])

    if not p.audio_keep_example_waveforms:
      del example["waveforms"]
    return super(SpeechRecognitionProblem, self
                ).preprocess_example(example, mode, hparams)

  def eval_metrics(self):
    defaults = super(SpeechRecognitionProblem, self).eval_metrics()
    return defaults + [metrics.Metrics.EDIT_DISTANCE]


@registry.register_audio_modality
class SpeechRecognitionModality(modality.Modality):
  """Common ASR filterbank processing."""

  def bottom(self, inputs):
    """Use batchnorm instead of CMVN and shorten the stft with strided convs.

    Args:
      inputs: float32 tensor with shape [batch_size, len, 1, freqs * channels]

    Returns:
      float32 tensor with shape [batch_size, shorter_len, 1, hidden_size]
    """
    p = self._model_hparams

    num_mel_bins = p.audio_num_mel_bins
    num_channels = 3 if p.audio_add_delta_deltas else 1

    with tf.variable_scope(self.name):
      if p.audio_preproc_in_bottom:
        # Compute filterbanks
        with tf.variable_scope("fbanks"):
          waveforms = tf.squeeze(inputs, [2, 3])
          mel_fbanks = compute_mel_filterbank_features(
              waveforms,
              sample_rate=p.audio_sample_rate,
              dither=p.audio_dither,
              preemphasis=p.audio_preemphasis,
              frame_length=p.audio_frame_length,
              frame_step=p.audio_frame_step,
              lower_edge_hertz=p.audio_lower_edge_hertz,
              upper_edge_hertz=p.audio_upper_edge_hertz,
              num_mel_bins=p.audio_num_mel_bins,
              apply_mask=True)
          if p.audio_add_delta_deltas:
            mel_fbanks = add_delta_deltas(mel_fbanks)
          x = tf.reshape(mel_fbanks,
                         common_layers.shape_list(mel_fbanks)[:2] +
                         [num_mel_bins, num_channels])

          nonpadding_mask = 1. - common_attention.embedding_to_padding(x)
          num_of_nonpadding_elements = tf.reduce_sum(
              nonpadding_mask) * num_mel_bins * num_channels

          # This replaces CMVN estimation on data
          mean = tf.reduce_sum(
              x, axis=[1], keepdims=True) / num_of_nonpadding_elements
          variance = (num_of_nonpadding_elements * mean**2. -
                      2. * mean * tf.reduce_sum(x, axis=[1], keepdims=True) +
                      tf.reduce_sum(x**2, axis=[1], keepdims=True)
                     ) / num_of_nonpadding_elements
          x = (x - mean) / variance * tf.expand_dims(nonpadding_mask, -1)
      else:
        x = inputs

      # The convention is that the models are flattened along the spatial,
      # dimensions, thus the speech preprocessor treats frequencies and
      # channels as image colors (last axis)
      x.set_shape([None, None, num_mel_bins, num_channels])

      # TODO(chorowski): how to specify bottom's hparams and avoid hardcoding?
      x = tf.pad(x, [[0, 0], [0, 8], [0, 0], [0, 0]])
      for _ in range(2):
        x = tf.layers.conv2d(
            x, 128, (3, 3), (2, 2), use_bias=False)
        x = common_layers.layer_norm(x)
        x = tf.nn.relu(x)

      xshape = common_layers.shape_list(x)
      # apply a conv that will remove all frequencies and at the same time
      # project the output into desired hidden_size
      x = tf.pad(x, [[0, 0], [0, 2], [0, 0], [0, 0]])
      x = tf.layers.conv2d(x, p.hidden_size, (3, xshape[2]), use_bias=False)

      assert common_layers.shape_list(x)[2] == 1
      x = common_layers.layer_norm(x)
      x = tf.nn.relu(x)
    return x
