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

"""Librispeech dataset."""

import os
from subprocess import call
import tarfile
import wave

# Dependency imports

import numpy as np

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import modality
from tensor2tensor.utils import registry

import tensorflow as tf


_LIBRISPEECH_TRAIN_DATASETS = [
    [
        "http://www.openslr.org/resources/12/train-clean-100.tar.gz",  # pylint: disable=line-too-long
        "train-clean-100"
    ],
    [
        "http://www.openslr.org/resources/12/train-clean-360.tar.gz",
        "train-clean-360"
    ],
    [
        "http://www.openslr.org/resources/12/train-other-500.tar.gz",
        "train-other-500"
    ],
]
_LIBRISPEECH_TEST_DATASETS = [
    [
        "http://www.openslr.org/resources/12/dev-clean.tar.gz",
        "dev-clean"
    ],
    [
        "http://www.openslr.org/resources/12/dev-other.tar.gz",
        "dev-other"
    ],
]


def _collect_data(directory, input_ext, transcription_ext):
  """Traverses directory collecting input and target files."""
  # Directory from string to tuple pair of strings
  # key: the filepath to a datafile including the datafile's basename. Example,
  #   if the datafile was "/path/to/datafile.wav" then the key would be
  #   "/path/to/datafile"
  # value: a pair of strings (media_filepath, label)
  data_files = dict()
  for root, _, filenames in os.walk(directory):
    transcripts = [filename for filename in filenames
                   if transcription_ext in filename]
    for transcript in transcripts:
      transcript_path = os.path.join(root, transcript)
      with open(transcript_path, "r") as transcript_file:
        for transcript_line in transcript_file:
          line_contents = transcript_line.split(" ", 1)
          assert len(line_contents) == 2
          media_base, label = line_contents
          key = os.path.join(root, media_base)
          assert key not in data_files
          media_name = "%s.%s"%(media_base, input_ext)
          media_path = os.path.join(root, media_name)
          data_files[key] = (media_path, label)
  return data_files


def _get_audio_data(filepath):
  # Construct a true .wav file.
  out_filepath = filepath.strip(".flac") + ".wav"
  # Assumes sox is installed on system. Sox converts from FLAC to WAV.
  call(["sox", filepath, out_filepath])
  wav_file = wave.open(open(out_filepath))
  frame_count = wav_file.getnframes()
  byte_array = wav_file.readframes(frame_count)

  data = np.fromstring(byte_array, np.uint8).tolist()
  return data, frame_count, wav_file.getsampwidth(), wav_file.getnchannels()


class LibrispeechTextEncoder(text_encoder.TextEncoder):

  def encode(self, s):
    return [self._num_reserved_ids + ord(c) for c in s]

  def decode(self, ids):
    """Transform a sequence of int ids into a human-readable string.

    EOS is not expected in ids.

    Args:
      ids: list of integers to be converted.
    Returns:
      s: human-readable string.
    """
    decoded_ids = []
    for id_ in ids:
      if 0 <= id_ < self._num_reserved_ids:
        decoded_ids.append(text_encoder.RESERVED_TOKENS[int(id_)])
      else:
        decoded_ids.append(id_ - self._num_reserved_ids)
    return "".join([chr(d) for d in decoded_ids])


@registry.register_audio_modality
class LibrispeechModality(modality.Modality):
  """Performs strided conv compressions for audio spectral data."""

  def bottom(self, inputs):
    """Transform input from data space to model space.

    Args:
      inputs: A Tensor with shape [batch, ...]
    Returns:
      body_input: A Tensor with shape [batch, ?, ?, body_input_depth].
    """
    with tf.variable_scope(self.name):
      # TODO(aidangomez): Will need to sort out a better audio pipeline
      def xnet_resblock(x, filters, res_relu, name):
        with tf.variable_scope(name):
          # We only stride along the length dimension to preserve the spectral
          # bins (which are tiny in dimensionality relative to length)
          y = common_layers.separable_conv_block(
              x,
              filters, [((1, 1), (3, 3)), ((1, 1), (3, 3))],
              first_relu=True,
              padding="SAME",
              force2d=True,
              name="sep_conv_block")
          y = common_layers.pool(y, (3, 3), "MAX", "SAME", strides=(2, 1))
          return y + common_layers.conv_block(
              x,
              filters, [((1, 1), (1, 1))],
              padding="SAME",
              strides=(2, 1),
              first_relu=res_relu,
              force2d=True,
              name="res_conv0")

      # Rescale from UINT8 to floats in [-1,-1]
      signals = (tf.to_float(inputs)-127)/128.
      signals = tf.squeeze(signals, [2, 3])

      # `stfts` is a complex64 Tensor representing the short-time Fourier
      # Transform of each signal in `signals`. Its shape is
      # [batch_size, ?, fft_unique_bins]
      # where fft_unique_bins = fft_length // 2 + 1 = 513.
      stfts = tf.contrib.signal.stft(signals, frame_length=1024, frame_step=512,
                                     fft_length=1024)

      # An energy spectrogram is the magnitude of the complex-valued STFT.
      # A float32 Tensor of shape [batch_size, ?, 513].
      magnitude_spectrograms = tf.abs(stfts)

      # Warp the linear-scale, magnitude spectrograms into the mel-scale.
      num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
      lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 64
      sample_rate = 16000
      linear_to_mel_weight_matrix = (
          tf.contrib.signal.linear_to_mel_weight_matrix(
              num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
              upper_edge_hertz))
      mel_spectrograms = tf.tensordot(
          magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
      # Note: Shape inference for tensordot does not currently handle this case.
      mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
          linear_to_mel_weight_matrix.shape[-1:]))

      x = tf.expand_dims(mel_spectrograms, 2)
      x.set_shape([None, None, None, num_mel_bins])
      for i in xrange(self._model_hparams.audio_compression):
        x = xnet_resblock(x, 2**(i + 1), True, "compress_block_%d" % i)
      return xnet_resblock(x, self._body_input_depth, False,
                           "compress_block_final")


@registry.register_problem()
class Librispeech(problem.Problem):
  """Problem spec for English word to dictionary definition."""

  @property
  def is_character_level(self):
    return True

  @property
  def input_space_id(self):
    return problem.SpaceID.AUDIO_SPECTRAL

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def num_shards(self):
    return 100

  @property
  def use_subword_tokenizer(self):
    return False

  @property
  def num_dev_shards(self):
    return 1

  @property
  def use_train_shards_for_dev(self):
    """If true, we only generate training data and hold out shards for dev."""
    return False

  def feature_encoders(self, _):
    return {
        "inputs": text_encoder.TextEncoder(),
        "targets": LibrispeechTextEncoder(),
    }

  def example_reading_spec(self):
    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets": tf.VarLenFeature(tf.int64),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)

  def generator(self, data_dir, tmp_dir, training,
                eos_list=None, start_from=0, how_many=0):
    eos_list = [1] if eos_list is None else eos_list
    datasets = (_LIBRISPEECH_TRAIN_DATASETS if training
                else _LIBRISPEECH_TEST_DATASETS)
    num_reserved_ids = self.feature_encoders(None)["targets"].num_reserved_ids
    i = 0
    for url, subdir in datasets:
      filename = os.path.basename(url)
      compressed_file = generator_utils.maybe_download(tmp_dir, filename, url)

      read_type = "r:gz" if filename.endswith("tgz") else "r"
      with tarfile.open(compressed_file, read_type) as corpus_tar:
        # Create a subset of files that don't already exist.
        #   tarfile.extractall errors when encountering an existing file
        #   and tarfile.extract is extremely slow
        members = []
        for f in corpus_tar:
          if not os.path.isfile(os.path.join(tmp_dir, f.name)):
            members.append(f)
        corpus_tar.extractall(tmp_dir, members=members)

      data_dir = os.path.join(tmp_dir, "LibriSpeech", subdir)
      data_files = _collect_data(data_dir, "flac", "txt")
      data_pairs = data_files.values()
      for media_file, text_data in sorted(data_pairs)[start_from:]:
        if how_many > 0 and i == how_many:
          return
        i += 1
        audio_data, sample_count, sample_width, num_channels = _get_audio_data(
            media_file)
        label = [num_reserved_ids + ord(c) for c in text_data] + eos_list
        yield {
            "inputs": audio_data,
            "audio/channel_count": [num_channels],
            "audio/sample_count": [sample_count],
            "audio/sample_width": [sample_width],
            "targets": label
        }

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    train_paths = self.training_filepaths(
        data_dir, self.num_shards, shuffled=False)
    dev_paths = self.dev_filepaths(
        data_dir, self.num_dev_shards, shuffled=False)
    if self.use_train_shards_for_dev:
      all_paths = train_paths + dev_paths
      generator_utils.generate_files(
          self.generator(data_dir, tmp_dir, True), all_paths)
      generator_utils.shuffle_dataset(all_paths)
    else:
      generator_utils.generate_dataset_and_shuffle(
          self.generator(data_dir, tmp_dir, True), train_paths,
          self.generator(data_dir, tmp_dir, False), dev_paths)

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.stop_at_eos = int(False)
    p.input_modality = {"inputs": ("audio:librispeech_modality", None)}
    p.target_modality = (registry.Modalities.SYMBOL, 256)

  def preprocess_example(self, example, mode, hparams):
    return example


# TODO(lukaszkaiser): clean up hparams or remove from here.
def add_librispeech_hparams(hparams):
  """Adding to base hparams the attributes for for librispeech."""
  hparams.batch_size = 36
  hparams.audio_compression = 8
  hparams.hidden_size = 2048
  hparams.max_input_seq_length = 600000
  hparams.max_target_seq_length = 350
  hparams.max_length = hparams.max_input_seq_length
  hparams.min_length_bucket = hparams.max_input_seq_length // 2
  hparams.learning_rate = 0.05
  hparams.train_steps = 5000000
  hparams.num_hidden_layers = 4
  return hparams
