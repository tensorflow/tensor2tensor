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

"""DailyDialog dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from tensor2tensor.data_generators import dialog_abstract
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry


# End-of-sentence marker.
EOS = text_encoder.EOS_ID


@registry.register_problem
class DialogDailydialog16k(dialog_abstract.DialogAbstract):
  """A class implementing a simple chatbot problem for the DailyDialog dataset.

  https://arxiv.org/abs/1710.03957
  This version doesn't use any auxiliary information.
  """

  def preprocess_data(self, train_mode):
    """Main function where the preprocessing of the data starts.

    Args:
      train_mode: string, whether we are in train, dev or test mode
    """

    # Set the raw data directory and data.
    self.raw_data_dir = os.path.join('/'.join(self._data_dir.split('/')[:-1]),
                                     'raw_data')
    self.raw_data = os.path.join(self._raw_data_dir, 'ijcnlp_dailydialog')
    self.zipped_data = os.path.join(self._raw_data_dir,
                                    'ijcnlp_dailydialog.zip')

    # Create the download url.
    self.url = 'http://yanran.li/files/ijcnlp_dailydialog.zip'

    # Check at which part of the pipeline are we at.
    self.data_pipeline_status(train_mode)

  def create_data(self, train_mode):
    """Create the source, target and vocab files.

    Args:
      train_mode: string, whether we are in train, dev or test mode
    """

    # Open the 6 files.
    trainsource, traintarget, devsource, devtarget, testsource, testtarget = \
        self.open_6_files()

    # Open the raw data.
    dialogs = open(
        os.path.join(self._raw_data, 'dialogues_text.txt'), errors='ignore')

    vocabulary = collections.Counter()
    number_of_dialogs = 0
    line_counter = 0
    dataset_split_counter = 0
    # Iterate through the file.
    for dialog in dialogs:
      dataset_split_counter += 1
      if number_of_dialogs % 1000 == 0:
        print('problem_log: Parsed ' + str(number_of_dialogs) + ' dialogs.')

      # Utterances are separated by the __eou__ token.
      utterances = dialog.split('__eou__')[:-1]

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

      # Clean the utterances.
      i = 0
      for utterance in utterances:
        line_counter += 1
        utterance = self.clean_line(utterance.lower())
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

      number_of_dialogs += 1
      # Reset the split counter if we reached 100%.
      if dataset_split_counter == 100:
        dataset_split_counter = 0

      # Check if we reached the desired dataset size.
      if (self.targeted_dataset_size != 0 and
          self.targeted_dataset_size < line_counter):
        break

    # Close the files.
    self.close_n_files([trainsource,
                        traintarget,
                        devsource,
                        devtarget,
                        testsource,
                        testtarget])
    dialogs.close()

    # Save the vocabulary.
    self.save_vocab(vocabulary)
