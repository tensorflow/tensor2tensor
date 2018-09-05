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
"""Encoder for audio data."""

import os
from subprocess import call
import tempfile
import numpy as np
from scipy.io import wavfile


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
    # to a temp path, and provide instructions for installing sox.
    if s.endswith(".mp3"):
      # TODO(dliebling) On Linux, check if libsox-fmt-mp3 is installed.
      out_filepath = s[:-4] + ".wav"
      call([
          "sox", "--guard", s, "-r", "16k", "-b", "16", "-c", "1", out_filepath
      ])
      s = out_filepath
    elif not s.endswith(".wav"):
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
