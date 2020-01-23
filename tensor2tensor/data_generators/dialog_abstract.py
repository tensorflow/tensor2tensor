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

"""Abstract class for dialog problems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import tarfile
import zipfile

import requests
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators.text_problems import VocabType
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics
import tensorflow.compat.v1 as tf

# End-of-sentence marker.
EOS = text_encoder.EOS_ID


# An abstract base class for word based chatbot problems.
class DialogAbstract(text_problems.Text2TextProblem):
  """Abstract class for dialog problems."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.TOKEN

  @property
  def is_generate_per_split(self):
    return True

  @property
  def vocab_file(self):
    return self.vocab_filename

  @property
  def vocab_filename(self):
    return 'vocab.chatbot.' + str(self.targeted_vocab_size)

  @property
  def oov_token(self):
    return '<unk>'

  @property
  def use_subword_tokenizer(self):
    return False

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def targeted_vocab_size(self):
    return 2**14

  @property
  def targeted_dataset_size(self):
    # Number of utterance pairs in the full dataset.
    # If it's 0, then the full size of the dataset is used.
    return 0

  @property
  def dataset_split(self):
    return {'train': 80, 'val': 10, 'test': 10}

  @property
  def dataset_splits(self):
    return [{
        'split': problem.DatasetSplit.TRAIN,
        'shards': 1,
    }, {
        'split': problem.DatasetSplit.EVAL,
        'shards': 1,
    }, {
        'split': problem.DatasetSplit.TEST,
        'shards': 1,
    }]

  @property
  def data_dir(self):
    return ''

  @property
  def raw_data_dir(self):
    return ''

  @property
  def raw_data(self):
    return ''

  @property
  def zipped_data(self):
    return ''

  @property
  def url(self):
    return ''

  @data_dir.setter
  def data_dir(self, value):
    self._data_dir = value

  @raw_data_dir.setter
  def raw_data_dir(self, value):
    self._raw_data_dir = value

  @raw_data.setter
  def raw_data(self, value):
    self._raw_data = value

  @zipped_data.setter
  def zipped_data(self, value):
    self._zipped_data = value

  @url.setter
  def url(self, value):
    self._url = value

  # Main function where the preprocessing of the data starts.
  def preprocess_data(self, train_mode):
    return NotImplementedError

  # This should also be overriden if the data_pipeline_status is used.
  def create_data(self, train_mode):
    pass

  def data_pipeline_status(self, train_mode):
    """Check at which part of the pipeline are we at.

    This function first checks recursively at which point in the
    data processing point are we (what files can be found on the disk),
    and then proceeds from there.

    Args:
      train_mode: string, whether we are in train, dev or test mode
    """

    # Build the source and target paths.
    sourcepath = os.path.join(self._data_dir, train_mode + 'Source.txt')
    targetpath = os.path.join(self._data_dir, train_mode + 'Target.txt')

    # If raw data dir doesn't exist, create it.
    if not os.path.exists(self._raw_data_dir):
      os.makedirs(self._raw_data_dir)

    # Check whether sourcePath.txt exists.
    if (os.path.isfile(sourcepath) and os.path.isfile(targetpath) and
        os.path.isfile(os.path.join(self._data_dir, self.vocab_file))):
      print('problem_log: Source, target and vocab files exist in ' +
            self._data_dir + ', proceeding with data generation. ' +
            'If you want to rebuild these files, delete them first.')
      return

    # Check whether the raw data is extracted to the raw_data_dir folder.
    elif os.path.exists(self._raw_data):
      print('problem_log: No source, target or vocab files found in ' +
            self._data_dir + '.')
      print('problem_log: Extracted raw data is in ' + self._raw_data_dir +
            '. Proceeding with creating source, target and vocab files.')
      self.create_data(train_mode)

    # Check whether the data is downloaded in the raw_data_dir_folder.
    elif os.path.exists(self._zipped_data):
      print('problem_log: No source, target or vocab files found in ' +
            self._data_dir + '.')
      print('problem_log: No extracted raw data found in ' +
            self._raw_data_dir + '.')
      print('problem_log: Unextracted raw data is in ' + self._raw_data_dir +
            '. Extracting and creating source, target and vocab files.')
      self.extract_data(train_mode)

    else:
      print('problem_log: No source, target or vocab files found in ' +
            self._data_dir + '.')
      print('problem_log: No raw data found in ' + self._raw_data_dir +
            '. Proceeding with downloading the data, extracting it, ' +
            'and creating source, target and vocab files.')
      self.download_data(train_mode)

  def download_data(self, train_mode):
    """Download data from official sources.

    Args:
      train_mode: string, whether we are in train, dev or test mode
    """

    # Open the url and download the data with progress bars.
    data_stream = requests.get(self._url, stream=True)
    with open(self._zipped_data, 'wb') as f:
      for chunk in data_stream.iter_content(1024):
        if chunk:
          f.write(chunk)
          f.flush()

    # Next step is extracting the data.
    print('problem_log: Extracting data to ' + self._zipped_data + '.')
    self.extract_data(train_mode)

  def extract_data(self, train_mode):
    """Extract data and go to the next step.

    Args:
      train_mode:  string, whether we are in train, dev or test mode
    """

    if self._zipped_data[-2:] == 'gz':
      zip_file = tarfile.open(self._zipped_data, 'r:gz')
    elif self._zipped_data[-3:] == 'zip':
      zip_file = zipfile.ZipFile(self._zipped_data, 'r')
    else:
      print('problem_log: ' + self._zipped_data +
            ' is not a .zip or .gz file, so I can\'t extract it.')

    zip_file.extractall(self._raw_data_dir)
    zip_file.close()

    # Next step is creating the source, target and vocab files.
    print('problem_log: Creating ' +
          train_mode + ' files in ' + self._data_dir)
    self.create_data(train_mode)

  # hparams for the problem.
  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.stop_at_eos = int(True)

    p.modality = {'targets': modalities.ModalityType.SYMBOL}
    if self.has_inputs:
      p.modality['inputs'] = modalities.ModalityType.SYMBOL
      p.vocab_size = {'inputs': self._encoders['inputs'].vocab_size}
    p.vocab_size['targets'] = self._encoders['inputs'].vocab_size

    if self.vocab_type == VocabType.CHARACTER:
      p.loss_multiplier = 2.0

    if self.packed_length:
      if self.has_inputs:
        p.modality['inputs_segmentation'] = modalities.ModalityType.IDENTITY
        p.modality['inputs_position'] = modalities.ModalityType.IDENTITY
        p.vocab_size['inputs_segmentation'] = None
        p.vocab_size['inputs_position'] = None
      p.modality['targets_segmentation'] = modalities.ModalityType.IDENTITY
      p.modality['targets_position'] = modalities.ModalityType.IDENTITY
      p.vocab_size['targets_segmentation'] = None
      p.vocab_size['targets_position'] = None

  # What evaluation metrics to use with this problem.
  def eval_metrics(self):
    return [metrics.Metrics.ACC, metrics.Metrics.ACC_TOP5,
            metrics.Metrics.ACC_PER_SEQ,
            metrics.Metrics.NEG_LOG_PERPLEXITY,
            metrics.Metrics.APPROX_BLEU]

  # Override this, to start with preprocessing.
  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    self.data_dir = data_dir
    # Determine whether we are in training or validation mode.
    self.mode = {problem.DatasetSplit.TRAIN: 'train',
                 problem.DatasetSplit.EVAL: 'dev',
                 problem.DatasetSplit.TEST: 'test'}
    filepath_fns = {problem.DatasetSplit.TRAIN: self.training_filepaths,
                    problem.DatasetSplit.EVAL: self.dev_filepaths,
                    problem.DatasetSplit.TEST: self.test_filepaths}

    split_paths = [(split['split'], filepath_fns[split['split']](
        data_dir, split['shards'], shuffled=self.already_shuffled))
                   for split in self.dataset_splits]
    all_paths = []
    for _, paths in split_paths:
      all_paths.extend(paths)

    if self.is_generate_per_split:
      for split, paths in split_paths:
        # Create the source and target txt files from the raw data.
        self.preprocess_data(self.mode[split])
        generator_utils.generate_files(
            self.generate_encoded_samples(data_dir, tmp_dir, split), paths)
    else:
      self.preprocess_data(self.mode[problem.DatasetSplit.TRAIN])
      generator_utils.generate_files(
          self.generate_encoded_samples(
              data_dir, tmp_dir, problem.DatasetSplit.TRAIN), all_paths)

    generator_utils.shuffle_dataset(all_paths, extra_fn=self._pack_fn())

  def generate_samples(self, data_dir, tmp_dir, data_split):
    """This function generates train and validation pairs in t2t-datagen style.

    The function assumes that if you have data at one level of the pipeline,
    you don't want to re-generate it, so for example if the 4 txt files exist,
    the function continues by generating the t2t-datagen format files.
    So if you want to re-download or re-generate data,
    you have to delete it first from the appropriate directories.

    Args:
      data_dir: string, Directory where the data will be generated. The raw
                        data has to be downloaded one directory level higher.
      tmp_dir: string, temp directory.
      data_split: string, which data split to generate samples for

    Yields:
      dict
    """

    self.data_dir = data_dir
    print('problem_log: ' +
          self.mode[data_split] + ' data generation activated.')

    s_path = os.path.join(data_dir, self.mode[data_split] + 'Source.txt')
    t_path = os.path.join(data_dir, self.mode[data_split] + 'Target.txt')

    # Open the files and yield source-target lines.
    with tf.gfile.GFile(s_path, mode='r') as source_file:
      with tf.gfile.GFile(t_path, mode='r') as target_file:
        source, target = source_file.readline(), target_file.readline()
        while source and target:
          yield {'inputs': source.strip(), 'targets': target.strip()}
          source, target = source_file.readline(), target_file.readline()

  def save_vocab(self, vocab):
    """Save the vocabulary to a file.

    Args:
      vocab: dict
    """
    voc_file = open(os.path.join(self._data_dir, self.vocab_file), 'w')

    # Put the reserved tokens in.
    voc_file.write('<pad>\n')
    voc_file.write('<EOS>\n')
    for word, _ in vocab.most_common(self.targeted_vocab_size - 3):
      voc_file.write(word + '\n')
    voc_file.write('<unk>')

    voc_file.close()

  # Open the 6 files to write the processed data into.
  def open_6_files(self):
    trainsource = open(os.path.join(self._data_dir, 'trainSource.txt'), 'w')
    traintarget = open(os.path.join(self._data_dir, 'trainTarget.txt'), 'w')
    devsource = open(os.path.join(self._data_dir, 'devSource.txt'), 'w')
    devtarget = open(os.path.join(self._data_dir, 'devTarget.txt'), 'w')
    testsource = open(os.path.join(self._data_dir, 'testSource.txt'), 'w')
    testtarget = open(os.path.join(self._data_dir, 'testTarget.txt'), 'w')

    return trainsource, traintarget, devsource, \
        devtarget, testsource, testtarget

  # Close the 6 files to write the processed data into.
  def close_n_files(self, files):
    for f in files:
      f.close()

  def clean_line(self, line):
    """Clean a line with some regex rules.

    Args:
      line: string, line to be processed and returned

    Returns:
      string
    """

    # 2 functions for more complex replacing.
    def replace(matchobj):
      return re.sub("'", " '", str(matchobj.group(0)))

    def replace_null(matchobj):
      return re.sub("'", '', str(matchobj.group(0)))

    # Keep some special tokens.
    line = re.sub("[^a-z .?!'0-9]", '', line)
    line = re.sub('[.]', ' . ', line)
    line = re.sub('[?]', ' ? ', line)
    line = re.sub('[!]', ' ! ', line)

    # Take care of apostrophes.
    line = re.sub("[ ]'[ ]", ' ', line)
    line = re.sub(" '[a-z]", replace_null, line)
    line = re.sub("n't", " n't", line)
    line = re.sub("[^ n]'[^ t]", replace, line)

    return line
