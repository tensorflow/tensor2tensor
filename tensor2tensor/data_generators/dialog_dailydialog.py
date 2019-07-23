from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from collections import Counter

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import dialog_abstract
from tensor2tensor.utils import registry


# End-of-sentence marker.
EOS = text_encoder.EOS_ID


@registry.register_problem
class DialogDailydialog16k(dialog_abstract.DialogAbstract):
  '''
  https://arxiv.org/abs/1710.03957
  A class implementing a simple chatbot problem for the DailyDialog dataset.
  This version doesn't use any auxiliary information.
  '''

  # Main function where the preprocessing of the data starts.
  def preprocess_data(self, train_mode):
    '''
    Params:
      :train_mode: Whether we are in train or dev mode.
    '''

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

  # Create the source, target and vocab files.
  def create_data(self, train_mode):
    '''
    Params:
      :train_mode: Whether we are in train or dev mode.
    '''

    # Open the 6 files.
    trainSource, trainTarget, devSource, devTarget, testSource, testTarget = \
        self.open_6_files()

    # Open the raw data.
    dialogs = open(
        os.path.join(self._raw_data, 'dialogues_text.txt'), errors='ignore')

    vocabulary = Counter()
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
        source_file = trainSource
        target_file = trainTarget
      elif dataset_split_counter <= (self.dataset_split['train'] +
                                     self.dataset_split['val']):
        source_file = devSource
        target_file = devTarget
      else:
        source_file = testSource
        target_file = testTarget

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
    self.close_n_files([trainSource,
                       trainTarget,
                       devSource,
                       devTarget,
                       testSource,
                       testTarget])
    dialogs.close()

    # Save the vocabulary.
    self.save_vocab(vocabulary)
