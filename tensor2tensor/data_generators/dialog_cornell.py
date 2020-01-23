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

"""Cornell Movie Dialog Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re

from tensor2tensor.data_generators import dialog_abstract
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry


# End-of-sentence marker.
EOS = text_encoder.EOS_ID


@registry.register_problem
class DialogCornell32k(dialog_abstract.DialogAbstract):
  """Implements the chatbot problem with Cornell Movie Dialog dataset.

  https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
  """

  @property
  def targeted_vocab_size(self):
    return 2**15

  def preprocess_data(self, train_mode):
    """Main function where the preprocessing of the data starts.

    Args:
      train_mode: string, whether we are in train, dev or test mode
    """

    # Set the raw data directory and data.
    self.raw_data_dir = os.path.join('/'.join(self._data_dir.split('/')[:-1]),
                                     'raw_data')
    self.raw_data = os.path.join(self._raw_data_dir,
                                 'cornell movie-dialogs corpus')
    self.zipped_data = os.path.join(self._raw_data_dir,
                                    'cornell_movie_dialogs_corpus.zip')

    # Create the download url.
    self.url = ('http://www.cs.cornell.edu/~cristian/data/' +
                'cornell_movie_dialogs_corpus.zip')

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
    movie_lines = open(
        os.path.join(self._raw_data, 'movie_lines.txt'), errors='ignore')
    dialog_list = self.extract_dialog_ids()

    vocabulary = collections.Counter()
    line_dict = {}
    number_of_lines = 0
    # Iterate through file.
    for line in movie_lines:
      if number_of_lines % 10000 == 0:
        print('problem_log: Parsed ' + str(number_of_lines) + ' lines.')

      line = line.split(' +++$+++ ')
      dialog_id = line[0]
      line = line[4].lower()

      # Do some cleaning.
      line = self.clean_line(line)
      line_dict[dialog_id] = line

      number_of_lines += 1
      # Check if we reached the desired dataset size.
      if (self.targeted_dataset_size != 0 and
          self.targeted_dataset_size < number_of_lines):
        break

    counter = 0
    dataset_split_counter = 0
    # Save the actual dialogs.
    for dialog in dialog_list:
      if counter % 10000 == 0:
        print('problem_log: Saved ' +
              str(counter) + '/' + str(len(dialog_list)) + ' dialogs.')

      dataset_split_counter += 1
      i = 0
      # Save one utterance.
      for utterance in dialog:
        if (utterance != dialog[-1] and
            dialog[i + 1] != 'L211194' and
            dialog[i + 1] != 'L1045'):
          source_line = line_dict[utterance] + '\n'
          target_line = line_dict[dialog[i + 1]] + '\n'

          # Save to the files according to dataset split.
          if dataset_split_counter <= self.dataset_split['train']:
            # Build vocabulary.
            words = source_line.split()
            for word in words:
              vocabulary[word] = vocabulary.get(word, 0) + 1

            trainsource.write(source_line)
            traintarget.write(target_line)

          elif dataset_split_counter <= (self.dataset_split['train'] +
                                         self.dataset_split['val']):
            devsource.write(source_line)
            devtarget.write(target_line)
          else:
            testsource.write(source_line)
            testtarget.write(target_line)
        i += 1

      # Reset the split counter if we reached 100%.
      if dataset_split_counter == 100:
        dataset_split_counter = 0
      counter += 1

    # Close the files.
    self.close_n_files([trainsource,
                        traintarget,
                        devsource,
                        devtarget,
                        testsource,
                        testtarget])
    movie_lines.close()

    # Save the vocabulary.
    self.save_vocab(vocabulary)

  # Extract the dialog ids from the dialog file.
  def extract_dialog_ids(self):
    dialogs = open(os.path.join(self._raw_data, 'movie_conversations.txt'),
                   errors='ignore')

    dialog_list = []
    # Each line contains a dialog.
    for line in dialogs:
      line = line.split(' +++$+++ ')
      line = line[3].split(',')

      i = 0
      for item in line:
        line[i] = re.sub('[^A-Z0-9]', '', item)
        i += 1
      dialog_list.append(line)

    dialogs.close()
    return dialog_list
