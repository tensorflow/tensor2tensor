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

"""OpenSubtitles dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re
import zipfile

from tensor2tensor.data_generators import dialog_abstract
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry


# End-of-sentence marker.
EOS = text_encoder.EOS_ID


@registry.register_problem
class DialogOpensubtitles64k2009(dialog_abstract.DialogAbstract):
  """A class implementing the chatbot problem for the OpenSubtitles dataset.

  http://opus.nlpl.eu/OpenSubtitles-v2018.php
  """

  @property
  def targeted_vocab_size(self):
    return 2**16

  @property
  def dataset_version(self):
    # Year of the opensubtitles dataset creation.
    return 2009

  def extract_data(self, train_mode):
    """Extract data and go to the next step.

    Args:
      train_mode: string, whether we are in train, dev or test mode
    """

    if self._zipped_data[-3:] == 'zip' or self._zipped_data[-2:] == 'gz':
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

  def preprocess_data(self, train_mode):
    """Main function where the preprocessing of the data starts.

    Args:
      train_mode: string, whether we are in train, dev or test mode
    """

    year = '' if self.dataset_version == 2009 else str(self.dataset_version)
    # Set the raw data directory and data.
    self.raw_data_dir = os.path.join('/'.join(self._data_dir.split('/')[:-1]),
                                     'raw_data_' + str(self.dataset_version))
    self.raw_data = os.path.join(self._raw_data_dir, 'OpenSubtitles' + year)
    self.zipped_data = os.path.join(self._raw_data_dir, 'en.tar.gz')

    # Create the download url.
    self.url = ('http://opus.nlpl.eu/download.php?f=OpenSubtitles' +
                str(year) + '/en.tar.gz')

    # Check at which part of the pipeline are we at.
    self.data_pipeline_status(train_mode)

  def create_data(self, train_mode):
    """Create the source, target and vocab files.

    Args:
      train_mode: string, whether we are in train, dev or test mode
    """

    # open the 6 files
    trainsource, traintarget, devsource, devtarget, testsource, testtarget = \
        self.open_6_files()

    conv_id = 0
    number_of_lines = 0
    dataset_split_counter = 0
    vocabulary = collections.Counter()
    # Dind all the files.
    for root, _, files in os.walk(self._raw_data_dir):
      for f in files:
        if conv_id % 100 == 0:
          print('problem_log: Parsed ' + str(conv_id) + ' files.')

        source_lines = ''
        target_lines = ''
        conv_id += 1
        dataset_split_counter += 1

        # Open one .xml file and parse it.
        with open(os.path.join(root, f), 'r', errors='ignore') as txt_file:
          words = ''
          line_id = 1

          # Parse one line.
          for line in txt_file:
            line = str(line)

            # Check if it's a new sentence.
            if line.find('<s id="') != -1:
              if words:
                # Do some cleaning.
                words = self.clean_line(words)

                # Build the vocabulary.
                if dataset_split_counter <= self.dataset_split['train']:
                  word_list = words.split()
                  for word in word_list:
                    vocabulary[word] = vocabulary.get(word, 0) + 1

                # Add the previous line.
                source_lines += words + '\n'
                if line_id != 1:
                  target_lines += words + '\n'
                line_id += 1
              words = ''

            else:
              index = line.find('<w id="')
              if index >= 0:
                line = line[index:]
                word = line[line.find('>') + 1:line.find('</w')]
                words = words + ' ' + word.replace('\t', ' ')

          # Delete the final source sentence, since it doesn't have a target.
          source_lines = '\n'.join(source_lines.split('\n')[:-2]) + '\n'

        # Save the dialog according to the dataset split.
        if dataset_split_counter <= self.dataset_split['train']:
          trainsource.write(source_lines)
          traintarget.write(target_lines)
        elif dataset_split_counter <= (self.dataset_split['train'] +
                                       self.dataset_split['val']):
          devsource.write(source_lines)
          devtarget.write(target_lines)
        else:
          testsource.write(source_lines)
          testtarget.write(target_lines)

        # Reset the split counter if we reached 100%.
        if dataset_split_counter == 100:
          dataset_split_counter = 0

        # Check if we reached the desired dataset size.
        number_of_lines += line_id
        if (self.targeted_dataset_size != 0 and
            self.targeted_dataset_size < number_of_lines):
          break
      else:
        continue
      break

    # Close the files.
    self.close_n_files([trainsource,
                        traintarget,
                        devsource,
                        devtarget,
                        testsource,
                        testtarget])
    # Save the vocabulary.
    self.save_vocab(vocabulary)

  def clean_line(self, line):
    """Clean a line with some regex rules.

    Args:
      line: string, line to be processed and returned

    Returns:
      string
    """

    line = line.lower()
    line = re.sub("[^a-z .!?'\t\\\\]", '', line)
    line = re.sub("\\\\['] ", " '", line)
    line = re.sub('[\\\\]', ' ', line)
    line = re.sub('[.]', ' . ', line)
    line = re.sub('[?]', ' ? ', line)
    line = re.sub('[!]', ' ! ', line)
    line = re.sub("[ ]'[ ]", ' ', line)
    line = re.sub("n't", " n't", line)

    return line


@registry.register_problem
class DialogOpensubtitles64k2011(DialogOpensubtitles64k2009):

  @property
  def dataset_version(self):
    # Year of the opensubtitles dataset creation.
    return 2011


@registry.register_problem
class DialogOpensubtitles64k2012(DialogOpensubtitles64k2009):

  @property
  def dataset_version(self):
    # Year of the opensubtitles dataset creation.
    return 2012


@registry.register_problem
class DialogOpensubtitles64k2013(DialogOpensubtitles64k2009):

  @property
  def dataset_version(self):
    # Year of the opensubtitles dataset creation.
    return 2013


@registry.register_problem
class DialogOpensubtitles64k2016(DialogOpensubtitles64k2009):

  @property
  def dataset_version(self):
    # Year of the opensubtitles dataset creation.
    return 2016


@registry.register_problem
class DialogOpensubtitles64k2018(DialogOpensubtitles64k2009):

  @property
  def dataset_version(self):
    # Year of the opensubtitles dataset creation.
    return 2018
