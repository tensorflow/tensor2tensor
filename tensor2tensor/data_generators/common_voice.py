# coding=utf-8
# Copyright 2022 The Tensor2Tensor Authors.
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

_EXT_ARCHIVE = ".tar.gz"
_CORPUS_VERSION = "cv-corpus-5-2020-06-22"
_BASE_URL = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/" + _CORPUS_VERSION  # pylint: disable=line-too-long

_COMMONVOICE_TRAIN_DATASETS = ["validated", "train"]
_COMMONVOICE_DEV_DATASETS = ["dev", "other"]
_COMMONVOICE_TEST_DATASETS = ["test"]

_LANGUAGES = {
  "tt",
  "en",
  "de",
  "fr",
  "cy",
  "br",
  "cv",
  "tr",
  "ky",
  "ga-IE",
  "kab",
  "ca",
  "zh-TW",
  "sl",
  "it",
  "nl",
  "cnh",
  "eo",
  "et",
  "fa",
  "pt",
  "eu",
  "es",
  "zh-CN",
  "mn",
  "sah",
  "dv",
  "rw",
  "sv-SE",
  "ru",
  "id",
  "ar",
  "ta",
  "ia",
  "lv",
  "ja",
  "vot",
  "ab",
  "zh-HK",
  "rm-sursilv"
}

def _file_exists(path, filename):
  """Checks if the filename exists under the path."""
  return os.path.isfile(os.path.join(path, filename))


def _is_relative(path, filename):
  """Checks if the filename is relative, not absolute."""
  return not os.path.abspath(os.path.join(path, filename)).startswith(path)


@registry.register_problem()
class CommonVoice(speech_recognition.SpeechRecognitionProblem):
  """Problem spec for Commonvoice using clean and noisy data."""

  # Select only the clean data
  TRAIN_DATASETS = _COMMONVOICE_TRAIN_DATASETS
  DEV_DATASETS = _COMMONVOICE_DEV_DATASETS[:1]
  TEST_DATASETS = _COMMONVOICE_TEST_DATASETS

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
                language,
                datasets):
    if language in _LANGUAGES:
      _CODE = language
    else:
      _CODE = "en"
    _URL = _BASE_URL + _CODE + _EXT_ARCHIVE
    filename = os.path.basename(_URL)
    compressed_file = generator_utils.maybe_download(tmp_dir, filename,
                                                     _URL)

    read_type = "r:gz" if filename.endswith(".tar.gz") else "r"
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

    raw_data_dir = os.path.join(tmp_dir, _CORPUS_VERSION + _CODE + "/")
    encoders = self.feature_encoders(data_dir)
    audio_encoder = encoders["waveforms"]
    text_encoder = encoders["targets"]

    for dataset in datasets:
      full_dataset_filename = dataset + ".tsv"
      with tf.io.gfile.GFile(os.path.join(raw_data_dir, full_dataset_filename)) as file_:
        dataset = csv.DictReader(file_, delimiter="\t")
        f_length = len(file_.readlines())
        file_.seek(0, 0)
        for i, row in tqdm.tqdm(enumerate(dataset), total=f_length):
          file_path = os.path.join(os.path.join(raw_data_dir, "clips/"), row["path"])
          if tf.io.gfile.exists(file_path):
            try:
              wav_data = audio_encoder.encode(file_path)
              yield {
                "waveforms": wav_data,
                "waveform_lens": [len(wav_data)],
                "targets": text_encoder.encode(row["sentence"]),
                "raw_transcript": [row["sentence"]],
                "utt_id": row["client_id"]
              }
            except Exception as e:
              print(e)

  def generate_data(self, data_dir, tmp_dir, language, task_id=-1):
    train_paths = self.training_filepaths(
        data_dir, self.num_shards, shuffled=False)
    dev_paths = self.dev_filepaths(
        data_dir, self.num_dev_shards, shuffled=False)
    test_paths = self.test_filepaths(
        data_dir, self.num_test_shards, shuffled=True)

    generator_utils.generate_files(
        self.generator(data_dir, tmp_dir, language, self.TEST_DATASETS), test_paths)

    if self.use_train_shards_for_dev:
      all_paths = train_paths + dev_paths
      generator_utils.generate_files(
          self.generator(data_dir, tmp_dir, language, self.TRAIN_DATASETS), all_paths)
      generator_utils.shuffle_dataset(all_paths)
    else:
      generator_utils.generate_dataset_and_shuffle(
          self.generator(data_dir, tmp_dir, language, self.TRAIN_DATASETS), train_paths,
          self.generator(data_dir, tmp_dir, language, self.DEV_DATASETS), dev_paths)

def set_common_voice_length_hparams(hparams):
  hparams.max_length = 1650 * 80
  hparams.max_input_seq_length = 1650
  hparams.max_target_seq_length = 350
  return hparams
