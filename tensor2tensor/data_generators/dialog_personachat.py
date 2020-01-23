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

"""Persona-chat dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import tarfile
import zipfile

from tensor2tensor.data_generators import dialog_abstract
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry


# End-of-sentence marker.
EOS = text_encoder.EOS_ID


@registry.register_problem
class DialogPersonachat16k(dialog_abstract.DialogAbstract):
  """Implements a simple chatbot for the original Persona-chat dataset.

  The personas are not used in this class, only the raw dialogs.
  https://github.com/facebookresearch/ParlAI/tree/master/projects/personachat
  """

  def preprocess_data(self, train_mode):
    """Main function where the preprocessing of the data starts.

    Args:
      train_mode: string, whether we are in train, dev or test mode
    """

    # Set the raw data directory and data.
    self.raw_data_dir = os.path.join('/'.join(self._data_dir.split('/')[:-1]),
                                     'raw_data')
    self.raw_data = os.path.join(self._raw_data_dir, 'ConvAI2')
    self.zipped_data = os.path.join(self._raw_data_dir, 'convai2.tar.gz')

    # Create the download url.
    self.url = 'http://parl.ai/downloads/convai2/convai2_fix_723.tgz'

    # Check at which part of the pipeline are we at.
    self.data_pipeline_status(train_mode)

  def extract_data(self, train_mode):
    """Extract data and go to the next step.

    Args:
      train_mode: string, whether we are in train, dev or test mode
    """

    if self._zipped_data[-2:] == 'gz':
      zip_file = tarfile.open(self._zipped_data, 'r:gz')
    elif self._zipped_data[-3:] == 'zip':
      zip_file = zipfile.ZipFile(self._zipped_data, 'r')
    else:
      print('problem_log: ' + self._zipped_data +
            ' is not a .zip or .gz file, so I can\'t extract it.')

    zip_file.extractall(self._raw_data)
    zip_file.close()

    # Next step is creating the source, target and vocab files.
    print('problem_log: Creating ' +
          train_mode + ' files in ' + self._data_dir + '.')
    self.create_data(train_mode)

  def create_data(self, train_mode):
    """Create the source, target and vocab files.

    Args:
      train_mode: string, whether we are in train, dev or test mode
    """

    # Open the 6 files.
    trainsource, traintarget, devsource, devtarget, testsource, testtarget = \
        self.open_6_files()

    # Open the raw data.
    train_dialogs = open(
        os.path.join(self._raw_data, 'train_none_original_no_cands.txt'),
        errors='ignore')
    valid_dialogs = open(
        os.path.join(self._raw_data, 'valid_none_original_no_cands.txt'),
        errors='ignore')
    filenames = [train_dialogs, valid_dialogs]

    # Copy the data to a new file.
    with open(os.path.join(self._raw_data,
                           'full_none_original_no_cands.txt'), 'w') as outfile:
      for fname in filenames:
        with fname as infile:
          outfile.write(infile.read())
    train_dialogs.close()
    valid_dialogs.close()

    # Open the big file.
    dialogs = open(
        os.path.join(self._raw_data, 'full_none_original_no_cands.txt'),
        errors='ignore')

    number_of_lines = 0
    current_dialog = ''
    dialog_list = []
    dialog_silenced = False
    # Iterate through the file and build list of dialogs separated by __eou__.
    for line in dialogs:
      if number_of_lines % 10000 == 0:
        print('problem_log: Parsed ' + str(number_of_lines) + ' lines.')

      dialog_id = line.split()[0]
      # Check if this is a refurbished line.
      if ('__SILENCE__' not in line and
          ((dialog_silenced and dialog_id == '1') or not dialog_silenced)):
        dialog_silenced = False
        number_of_lines += 1

        # Get the utterances.
        source = ' '.join(line.split('\t')[0].split()[1:])
        target = line.split('\t')[1].strip('\n')
        source = self.clean_line(source.lower())
        target = self.clean_line(target.lower())

        # Whether this is a new dialog.
        if dialog_id == '1' and current_dialog:
          dialog_list.append(current_dialog)
          current_dialog = source + '__eou__' + target + '__eou__'
        else:
          current_dialog += source + '__eou__' + target + '__eou__'
      else:
        dialog_silenced = True

      if (self.targeted_dataset_size != 0 and
          self.targeted_dataset_size < number_of_lines):
        break
    dialogs.close()

    vocabulary = collections.Counter()
    number_of_dialogs = 0
    dataset_split_counter = 0
    # Build the dataset.
    for dialog in dialog_list:
      if number_of_dialogs % 1000 == 0:
        print('problem_log: Parsed ' + str(number_of_dialogs) + ' dialogs.')

      # Check which file we should write to.
      if dataset_split_counter <= self.dataset_split['train']:
        source_file = trainsource
        target_file = traintarget
      elif dataset_split_counter <= (self.dataset_split['train'] +
                                     self.dataset_split['val']):
        source_file = devsource
        target_file = devtarget
      else:
        source_file = testsource
        target_file = testtarget

      utterances = dialog.split('__eou__')[:-1]
      i = 0
      # Loop through the dialog.
      for utterance in utterances:
        i += 1
        # Build vocabulary.
        if dataset_split_counter <= self.dataset_split['train']:
          words = utterance.split()
          for word in words:
            if word in vocabulary:
              vocabulary[word] += 1
            else:
              vocabulary[word] = 1

        # Write to files.
        if i != len(utterances):
          source_file.write(utterance + '\n')
        if i != 1:
          target_file.write(utterance + '\n')

      dataset_split_counter += 1
      number_of_dialogs += 1
      # Reset the split counter if we reached 100%.
      if dataset_split_counter == 100:
        dataset_split_counter = 0

    # Close the files.
    self.close_n_files([trainsource,
                        traintarget,
                        devsource,
                        devtarget,
                        testsource,
                        testtarget])
    # Save the vocabulary.
    self.save_vocab(vocabulary)
