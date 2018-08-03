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
"""Librispeech dataset."""

import os
import tarfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import speech_recognition
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
_LIBRISPEECH_DEV_DATASETS = [
    [
        "http://www.openslr.org/resources/12/dev-clean.tar.gz",
        "dev-clean"
    ],
    [
        "http://www.openslr.org/resources/12/dev-other.tar.gz",
        "dev-other"
    ],
]
_LIBRISPEECH_TEST_DATASETS = [
    [
        "http://www.openslr.org/resources/12/test-clean.tar.gz",
        "test-clean"
    ],
    [
        "http://www.openslr.org/resources/12/test-other.tar.gz",
        "test-other"
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
          line_contents = transcript_line.strip().split(" ", 1)
          media_base, label = line_contents
          key = os.path.join(root, media_base)
          assert key not in data_files
          media_name = "%s.%s"%(media_base, input_ext)
          media_path = os.path.join(root, media_name)
          data_files[key] = (media_base, media_path, label)
  return data_files


@registry.register_problem()
class Librispeech(speech_recognition.SpeechRecognitionProblem):
  """Problem spec for Librispeech using clean and noisy data."""

  # Select only the clean data
  TRAIN_DATASETS = _LIBRISPEECH_TRAIN_DATASETS
  DEV_DATASETS = _LIBRISPEECH_DEV_DATASETS
  TEST_DATASETS = _LIBRISPEECH_TEST_DATASETS

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

  def generator(self, data_dir, tmp_dir, datasets,
                eos_list=None, start_from=0, how_many=0):
    del eos_list
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

      encoders = self.feature_encoders(None)
      audio_encoder = encoders["waveforms"]
      text_encoder = encoders["targets"]

      for utt_id, media_file, text_data in sorted(data_pairs)[start_from:]:
        if how_many > 0 and i == how_many:
          return
        i += 1
        wav_data = audio_encoder.encode(media_file)
        spk_id, unused_book_id, _ = utt_id.split("-")
        yield {
            "waveforms": wav_data,
            "waveform_lens": [len(wav_data)],
            "targets": text_encoder.encode(text_data),
            "raw_transcript": [text_data],
            "utt_id": [utt_id],
            "spk_id": [spk_id],
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
class LibrispeechTrainFullTestClean(Librispeech):
  """Problem to train on full 960h, but evaluate on clean data only."""

  def training_filepaths(self, data_dir, num_shards, shuffled):
    return Librispeech.training_filepaths(self, data_dir, num_shards, shuffled)

  def dev_filepaths(self, data_dir, num_shards, shuffled):
    return LibrispeechClean.dev_filepaths(self, data_dir, num_shards, shuffled)

  def test_filepaths(self, data_dir, num_shards, shuffled):
    return LibrispeechClean.test_filepaths(self, data_dir, num_shards, shuffled)

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    raise Exception("Generate librispeech and librispeech_clean data.")

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
      path = os.path.join(data_dir, "librispeech")
      suffix = "train"
    elif mode in [problem.DatasetSplit.EVAL, tf.estimator.ModeKeys.PREDICT]:
      path = os.path.join(data_dir, "librispeech_clean")
      suffix = "dev"
    else:
      assert mode == problem.DatasetSplit.TEST
      path = os.path.join(data_dir, "librispeech_clean")
      suffix = "test"

    return "%s-%s%s*" % (path, suffix, shard_str)


@registry.register_problem()
class LibrispeechCleanSmall(Librispeech):
  """Problem spec for Librispeech using 100h clean train and clean eval data."""

  # Select only the clean data
  TRAIN_DATASETS = _LIBRISPEECH_TRAIN_DATASETS[:1]
  DEV_DATASETS = _LIBRISPEECH_DEV_DATASETS[:1]
  TEST_DATASETS = _LIBRISPEECH_TEST_DATASETS[:1]


@registry.register_problem()
class LibrispeechClean(Librispeech):
  """Problem spec for Librispeech using 460h clean train and clean eval data."""

  # Select only the clean data
  TRAIN_DATASETS = _LIBRISPEECH_TRAIN_DATASETS[:2]
  DEV_DATASETS = _LIBRISPEECH_DEV_DATASETS[:1]
  TEST_DATASETS = _LIBRISPEECH_TEST_DATASETS[:1]


@registry.register_problem()
class LibrispeechNoisy(Librispeech):
  """Problem spec for Librispeech using 400h noisy train and noisy eval data."""

  # Select only the clean data
  TRAIN_DATASETS = _LIBRISPEECH_TRAIN_DATASETS[2:]
  DEV_DATASETS = _LIBRISPEECH_DEV_DATASETS[1:]
  TEST_DATASETS = _LIBRISPEECH_TEST_DATASETS[1:]


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


def set_librispeech_length_hparams(hparams):
  hparams.max_length = 1650 * 80  # this limits inputs[1] * inputs[2]
  hparams.max_input_seq_length = 1650
  hparams.max_target_seq_length = 350
  return hparams
