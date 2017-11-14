import os
from subprocess import call
import tarfile
import wave
import numpy as np
import six
from tensor2tensor.data_generators import generator_utils

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
    transcripts = [filename for filename in filenames if transcription_ext in filename]
    for transcript in transcripts:
      basename = transcript.strip(transcription_ext)
      transcript_path = os.path.join(root, transcript)
      with open(transcript_path, 'r') as transcript_file:
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
   

def librispeech_generator(data_dir, tmp_dir, training, eos_list=None, start_from=0, how_many=0):
  eos_list = [1] if eos_list is None else eos_list
  datasets = (_LIBRISPEECH_TRAIN_DATASETS if training else _LIBRISPEECH_TEST_DATASETS)
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
      label = [ord(c) for c in text_data] + eos_list
      yield {
          "inputs": audio_data,
          "audio/channel_count": [num_channels],
          "audio/sample_count": [sample_count],
          "audio/sample_width": [sample_width],
          "targets": label
      }