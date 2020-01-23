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

"""Mozilla Common Voice dataset.

Note: Generating the full set of examples can take upwards of 5 hours.
As the Common Voice data are distributed in MP3 format, experimenters will need
to have both SoX (http://sox.sourceforge.net) and on Linux, the libsox-fmt-mp3
package installed. The original samples will be downsampled by the encoder.
"""

import csv
import os
import tarfile
import tqdm  # pylint: disable=g-bad-import-order
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import speech_recognition
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf

_COMMONVOICE_URL = "https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz"  # pylint: disable=line-too-long

_COMMONVOICE_TRAIN_DATASETS = ["cv-valid-train", "cv-other-train"]
_COMMONVOICE_DEV_DATASETS = ["cv-valid-dev", "cv-other-dev"]
_COMMONVOICE_TEST_DATASETS = ["cv-valid-test", "cv-other-test"]


def _collect_data(directory):
  """Traverses directory collecting input and target files.

  Args:
   directory: base path to extracted audio and transcripts.
  Returns:
   list of (media_base, media_filepath, label) tuples
  """
  # Returns:
  data_files = []
  transcripts = [
      filename for filename in os.listdir(directory)
      if filename.endswith(".csv")
  ]
  for transcript in transcripts:
    transcript_path = os.path.join(directory, transcript)
    with open(transcript_path, "r") as transcript_file:
      transcript_reader = csv.reader(transcript_file)
      # skip header
      _ = next(transcript_reader)
      for transcript_line in transcript_reader:
        media_name, label = transcript_line[0:2]
        filename = os.path.join(directory, media_name)
        data_files.append((media_name, filename, label))
  return data_files


def _file_exists(path, filename):
  """Checks if the filename exists under the path."""
  return os.path.isfile(os.path.join(path, filename))


def _is_relative(path, filename):
  """Checks if the filename is relative, not absolute."""
  return os.path.abspath(os.path.join(path, filename)).startswith(path)


@registry.register_problem()
class CommonVoice(speech_recognition.SpeechRecognitionProblem):
  """Problem spec for Commonvoice using clean and noisy data."""

  # Select only the clean data
  TRAIN_DATASETS = _COMMONVOICE_TRAIN_DATASETS[:1]
  DEV_DATASETS = _COMMONVOICE_DEV_DATASETS[:1]
  TEST_DATASETS = _COMMONVOICE_TEST_DATASETS[:1]

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
  def num_test_shards(self):
    return 1

  @property
  def use_train_shards_for_dev(self):
    """If true, we only generate training data and hold out shards for dev."""
    return False

  def generator(self,
                data_dir,
                tmp_dir,
                datasets,
                eos_list=None,
                start_from=0,
                how_many=0):
    del eos_list
    i = 0

    filename = os.path.basename(_COMMONVOICE_URL)
    compressed_file = generator_utils.maybe_download(tmp_dir, filename,
                                                     _COMMONVOICE_URL)

    read_type = "r:gz" if filename.endswith(".tgz") else "r"
    with tarfile.open(compressed_file, read_type) as corpus_tar:
      # Create a subset of files that don't already exist.
      #   tarfile.extractall errors when encountering an existing file
      #   and tarfile.extract is extremely slow. For security, check that all
      #   paths are relative.
      members = [
          f for f in corpus_tar if _is_relative(tmp_dir, f.name) and
          not _file_exists(tmp_dir, f.name)
      ]
      corpus_tar.extractall(tmp_dir, members=members)

    raw_data_dir = os.path.join(tmp_dir, "cv_corpus_v1")
    data_tuples = _collect_data(raw_data_dir)
    encoders = self.feature_encoders(data_dir)
    audio_encoder = encoders["waveforms"]
    text_encoder = encoders["targets"]
    for dataset in datasets:
      data_tuples = (tup for tup in data_tuples if tup[0].startswith(dataset))
      for utt_id, media_file, text_data in tqdm.tqdm(
          sorted(data_tuples)[start_from:]):
        if how_many > 0 and i == how_many:
          return
        i += 1
        wav_data = audio_encoder.encode(media_file)
        yield {
            "waveforms": wav_data,
            "waveform_lens": [len(wav_data)],
            "targets": text_encoder.encode(text_data),
            "raw_transcript": [text_data],
            "utt_id": [utt_id],
            "spk_id": ["unknown"],
        }

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    train_paths = self.training_filepaths(
        data_dir, self.num_shards, shuffled=False)
    dev_paths = self.dev_filepaths(
        data_dir, self.num_dev_shards, shuffled=False)
    test_paths = self.test_filepaths(
        data_dir, self.num_test_shards, shuffled=True)

    generator_utils.generate_files(
        self.generator(data_dir, tmp_dir, self.TEST_DATASETS), test_paths)

    if self.use_train_shards_for_dev:
      all_paths = train_paths + dev_paths
      generator_utils.generate_files(
          self.generator(data_dir, tmp_dir, self.TRAIN_DATASETS), all_paths)
      generator_utils.shuffle_dataset(all_paths)
    else:
      generator_utils.generate_dataset_and_shuffle(
          self.generator(data_dir, tmp_dir, self.TRAIN_DATASETS), train_paths,
          self.generator(data_dir, tmp_dir, self.DEV_DATASETS), dev_paths)


@registry.register_problem()
class CommonVoiceTrainFullTestClean(CommonVoice):
  """Problem to train on full set, but evaluate on clean data only."""

  def training_filepaths(self, data_dir, num_shards, shuffled):
    return CommonVoice.training_filepaths(self, data_dir, num_shards, shuffled)

  def dev_filepaths(self, data_dir, num_shards, shuffled):
    return CommonVoiceClean.dev_filepaths(self, data_dir, num_shards, shuffled)

  def test_filepaths(self, data_dir, num_shards, shuffled):
    return CommonVoiceClean.test_filepaths(self, data_dir, num_shards, shuffled)

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    raise Exception("Generate Commonvoice and Commonvoice_clean data.")

  def filepattern(self, data_dir, mode, shard=None):
    """Get filepattern for data files for mode.

    Matches mode to a suffix.
    * DatasetSplit.TRAIN: train
    * DatasetSplit.EVAL: dev
    * DatasetSplit.TEST: test
    * tf.estimator.ModeKeys.PREDICT: dev

    Args:
      data_dir: str, data directory.
      mode: DatasetSplit
      shard: int, if provided, will only read data from the specified shard.

    Returns:
      filepattern str
    """
    shard_str = "-%05d" % shard if shard is not None else ""
    if mode == problem.DatasetSplit.TRAIN:
      path = os.path.join(data_dir, "common_voice")
      suffix = "train"
    elif mode in [problem.DatasetSplit.EVAL, tf.estimator.ModeKeys.PREDICT]:
      path = os.path.join(data_dir, "common_voice_clean")
      suffix = "dev"
    else:
      assert mode == problem.DatasetSplit.TEST
      path = os.path.join(data_dir, "common_voice_clean")
      suffix = "test"

    return "%s-%s%s*" % (path, suffix, shard_str)


@registry.register_problem()
class CommonVoiceClean(CommonVoice):
  """Problem spec for Common Voice using clean train and clean eval data."""

  # Select only the "clean" data (crowdsourced quality control).
  TRAIN_DATASETS = _COMMONVOICE_TRAIN_DATASETS[:1]
  DEV_DATASETS = _COMMONVOICE_DEV_DATASETS[:1]
  TEST_DATASETS = _COMMONVOICE_TEST_DATASETS[:1]


@registry.register_problem()
class CommonVoiceNoisy(CommonVoice):
  """Problem spec for Common Voice using noisy train and noisy eval data."""

  # Select only the "other" data.
  TRAIN_DATASETS = _COMMONVOICE_TRAIN_DATASETS[1:]
  DEV_DATASETS = _COMMONVOICE_DEV_DATASETS[1:]
  TEST_DATASETS = _COMMONVOICE_TEST_DATASETS[1:]


def set_common_voice_length_hparams(hparams):
  hparams.max_length = 1650 * 80
  hparams.max_input_seq_length = 1650
  hparams.max_target_seq_length = 350
  return hparams
