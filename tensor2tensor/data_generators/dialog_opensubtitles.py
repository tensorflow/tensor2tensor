from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import re
import gzip
from collections import Counter

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import dialog_abstract


# End-of-sentence marker.
EOS = text_encoder.EOS_ID


@registry.register_problem
class DialogOpensubtitles64k(dialog_abstract.DialogAbstract):
  '''
  A class implementing the chatbot problem for the OpenSubtitles dataset.
  http://opus.nlpl.eu/OpenSubtitles-v2018.php
  '''

  @property
  def targeted_vocab_size(self):
    return 2**16

  @property
  def dataset_version(self):
    # Year of the opensubtitles dataset creation.
    return 2012

  # Main function where the preprocessing of the data starts.
  def preprocess_data(self, train_mode):
    '''
    Params:
      :train_mode: Whether we are in train or dev mode.
    '''

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

  # Create the source, target and vocab files.
  def create_data(self, train_mode):
    '''
    Params:
      :train_mode: Whether we are in train or dev mode.
    '''

    # open the 6 files
    trainSource, trainTarget, devSource, devTarget, testSource, testTarget = \
        self.open_6_files()

    conv_id = 0
    number_of_lines = 0
    dataset_split_counter = 0
    vocabulary = Counter()
    # Dind all the files.
    for root, subfolders, files in os.walk(self._raw_data_dir):
      for file in files:
        if conv_id % 100 == 0:
          print('problem_log: Parsed ' + str(conv_id) + ' files.')
        if file.endswith('.gz'):
          source_lines = ''
          target_lines = ''
          conv_id += 1
          dataset_split_counter += 1

          # Open one .gz file and parse it.
          with gzip.open(os.path.join(root, file), 'r') as txt_file:
            words = ''
            line_id = 1

            # Parse one line.
            for line in txt_file:
              line = str(line)

              # Check if it's a new sentence.
              if line.find('<s id="') != -1:
                if len(words) > 0:
                  # Do some cleaning.
                  words = self.clean_line(words)

                  # Build the vocabulary.
                  if dataset_split_counter <= self.dataset_split['train']:
                    word_list = words.split()
                    for word in word_list:
                      if word in vocabulary:
                        vocabulary[word] += 1
                      else:
                        vocabulary[word] = 1

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
            trainSource.write(source_lines)
            trainTarget.write(target_lines)
          elif dataset_split_counter <= (self.dataset_split['train'] +
                                         self.dataset_split['val']):
            devSource.write(source_lines)
            devTarget.write(target_lines)
          else:
            testSource.write(source_lines)
            testTarget.write(target_lines)

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
    self.close_n_files([trainSource,
                        trainTarget,
                        devSource,
                        devTarget,
                        testSource,
                        testTarget])
    # Save the vocabulary.
    self.save_vocab(vocabulary)

  # Clean a line with some re rules.
  def clean_line(self, line):
    '''
    Params:
      :line: Line to be processed and returned.
    '''
    line = line.lower()
    line = re.sub("[^a-z .!?'\t\\\]", '', line)
    line = re.sub("\\\['] ", " '", line)
    line = re.sub('[\\\]', ' ', line)
    line = re.sub('[.]', ' . ', line)
    line = re.sub('[?]', ' ? ', line)
    line = re.sub('[!]', ' ! ', line)
    line = re.sub("[ ]'[ ]", ' ', line)
    line = re.sub("n't", " n't", line)

    return line
