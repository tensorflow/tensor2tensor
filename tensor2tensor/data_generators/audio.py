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

"""TIMIT data generator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from subprocess import call
import tarfile
import wave

# Dependency imports

from tensor2tensor.data_generators import generator_utils

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("timit_paths", "",
                    "Comma-separated list of tarfiles containing TIMIT "
                    "datasets")

_TIMIT_TRAIN_DATASETS = [
    ["timit/TIMIT/TRAIN", (".WAV", ".WRD")],
]
_TIMIT_TEST_DATASETS = [
    ["timit/TIMIT/TEST", (".WAV", ".WRD")],
]


def _get_timit(directory):
  """Extract TIMIT datasets to directory unless directory/timit exists."""
  if os.path.exists(os.path.join(directory, "timit")):
    return

  assert FLAGS.timit_paths
  for path in FLAGS.timit_paths.split(","):
    with tf.gfile.GFile(path) as f:
      with tarfile.open(fileobj=f, mode="r:gz") as timit_compressed:
        timit_compressed.extractall(directory)


def _collect_data(directory, input_ext, target_ext):
  """Traverses directory collecting input and target files."""
  # Directory from string to tuple pair of strings
  # key: the filepath to a datafile including the datafile's basename. Example,
  #   if the datafile was "/path/to/datafile.wav" then the key would be
  #   "/path/to/datafile"
  # value: a pair of strings (input_filepath, target_filepath)
  data_files = dict()
  for root, _, filenames in os.walk(directory):
    input_files = [filename for filename in filenames if input_ext in filename]
    for input_filename in input_files:
      basename = input_filename.strip(input_ext)
      input_file = os.path.join(root, input_filename)
      target_file = os.path.join(root, basename + target_ext)
      key = os.path.join(root, basename)
      assert os.path.exists(target_file)
      assert key not in data_files
      data_files[key] = (input_file, target_file)
  return data_files


def _get_audio_data(filepath):
  # Construct a true .wav file.
  out_filepath = filepath.strip(".WAV") + ".wav"
  # Assumes sox is installed on system. Sox converts from NIST SPHERE to WAV.
  call(["sox", filepath, out_filepath])
  wav_file = wave.open(open(out_filepath))
  frame_count = wav_file.getnframes()
  byte_array = wav_file.readframes(frame_count)
  data = [int(b.encode("hex"), base=16) for b in byte_array]
  return data, frame_count, wav_file.getsampwidth(), wav_file.getnchannels()


def _get_text_data(filepath):
  with tf.gfile.GFile(filepath, mode="r") as text_file:
    words = []
    for line in text_file:
      word = line.strip().split()[2]
      words.append(word)
    return " ".join(words)


def timit_generator(data_dir,
                    tmp_dir,
                    training,
                    how_many,
                    start_from=0,
                    eos_list=None,
                    vocab_filename=None,
                    vocab_size=0):
  """Data generator for TIMIT transcription problem.

  Args:
    data_dir: path to the data directory.
    tmp_dir: path to temporary storage directory.
    training: a Boolean; if true, we use the train set, otherwise the test set.
    how_many: how many inputs and labels to generate.
    start_from: from which input to start.
    eos_list: optional list of end of sentence tokens, otherwise use default
      value `1`.
    vocab_filename: file within `tmp_dir` to read vocabulary from. If this is
      not provided then the target sentence will be encoded by character.
    vocab_size: integer target to generate vocabulary size to.

  Yields:
    A dictionary representing the images with the following fields:
    * inputs: a float sequence containing the audio data
    * audio/channel_count: an integer
    * audio/sample_count: an integer
    * audio/sample_width: an integer
    * targets: an integer sequence representing the encoded sentence
  """
  eos_list = [1] if eos_list is None else eos_list
  if vocab_filename is not None:
    vocab_symbolizer = generator_utils.get_or_generate_vocab(
        data_dir, tmp_dir, vocab_filename, vocab_size)
  _get_timit(tmp_dir)
  datasets = (_TIMIT_TRAIN_DATASETS if training else _TIMIT_TEST_DATASETS)
  i = 0
  for data_dir, (audio_ext, transcription_ext) in datasets:
    data_dir = os.path.join(tmp_dir, data_dir)
    data_files = _collect_data(data_dir, audio_ext, transcription_ext)
    data_pairs = data_files.values()
    for input_file, target_file in sorted(data_pairs)[start_from:]:
      if i == how_many:
        return
      i += 1
      audio_data, sample_count, sample_width, num_channels = _get_audio_data(
          input_file)
      text_data = _get_text_data(target_file)
      if vocab_filename is None:
        label = [ord(c) for c in text_data] + eos_list
      else:
        label = vocab_symbolizer.encode(text_data) + eos_list
      yield {
          "inputs": audio_data,
          "audio/channel_count": [num_channels],
          "audio/sample_count": [sample_count],
          "audio/sample_width": [sample_width],
          "targets": label
      }
