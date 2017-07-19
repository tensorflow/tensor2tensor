# Copyright 2017 The Tensor2Tensor Authors.
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

"""Genetics problems.

Inputs are bases ACTG (with indices assigned in that order).

Requires the h5py library.

File format expected:
  * h5 file
  * h5 datasets should include {train, valid, test}_{in, na, out}, which will
    map to inputs, targets mask, and targets for the train, dev, and test
    datasets.
  * Each record in *_in is a bool 2-D numpy array with one-hot encoded base
    pairs with shape [num_input_timesteps, 4]. The base order is ACTG.
  * Each record in *_na is a bool 1-D numpy array with shape
    [num_output_timesteps].
  * Each record in *_out is a float 2-D numpy array with shape
    [num_output_timesteps, num_predictions].
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing as mp
import os

# Dependency imports

import h5py
import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry

_bases = list("ACTG")
BASE_TO_ID = dict(zip(_bases, range(len(_bases))))
ID_TO_BASE = dict(zip(range(len(_bases)), _bases))
UNK_ID = len(_bases)


# TODO(rsepassi):
# * DataEncoder for genetic bases
# * GeneticModality and problem hparams
# * Training preprocessing


class GeneticsProblem(problem.Problem):

  @property
  def download_url(self):
    raise NotImplementedError()

  @property
  def h5_file(self):
    raise NotImplementedError()

  def generate_data(self, data_dir, tmp_dir, num_shards=None):
    if num_shards is None:
      num_shards = 100

    # Download source data
    h5_filepath = generator_utils.maybe_download(tmp_dir, self.h5_file,
                                                 self.download_url)
    with h5py.File(h5_filepath, "r") as h5_file:
      num_train_examples = h5_file["train_in"].len()
      num_dev_examples = h5_file["valid_in"].len()
      num_test_examples = h5_file["test_in"].len()

    # Collect all_filepaths to later shuffle
    all_filepaths = []
    # Collect created shard processes to start and join
    processes = []

    datasets = [(self.training_filepaths, num_shards, "train",
                 num_train_examples), (self.dev_filepaths, 1, "valid",
                                       num_dev_examples),
                (self.test_filepaths, 1, "test", num_test_examples)]
    for fname_fn, nshards, key_prefix, num_examples in datasets:
      outfiles = fname_fn(data_dir, nshards, shuffled=False)
      all_filepaths.extend(outfiles)
      for start_idx, end_idx, outfile in generate_shard_args(
          outfiles, num_examples):
        p = mp.Process(
            target=generate_dataset,
            args=(h5_filepath, key_prefix, [outfile], start_idx, end_idx))
        processes.append(p)

    # Start and wait for processes
    assert len(processes) == num_shards + 2  # 1 per training shard + dev + test
    for p in processes:
      p.start()
    for p in processes:
      p.join()

    # Shuffle
    generator_utils.shuffle_dataset(all_filepaths)


@registry.register_problem("genetics_cage10")
class GeneticsCAGE10(GeneticsProblem):

  @property
  def download_url(self):
    return "https://storage.googleapis.com/262k_binned/cage10_l262k_w128.h5"

  @property
  def h5_file(self):
    return "cage10.h5"


@registry.register_problem("genetics_gm12878")
class GeneticsGM12878(GeneticsProblem):

  @property
  def download_url(self):
    return "https://storage.googleapis.com/262k_binned/gm12878_l262k_w128.h5"

  @property
  def h5_file(self):
    return "gm12878.h5"


def generate_shard_args(outfiles, num_examples):
  """Generate start and end indices per outfile."""
  num_shards = len(outfiles)
  num_examples_per_shard = num_examples // num_shards
  start_idxs = [i * num_examples_per_shard for i in xrange(num_shards)]
  end_idxs = list(start_idxs)
  end_idxs.pop(0)
  end_idxs.append(num_examples)
  return zip(start_idxs, end_idxs, outfiles)


def generate_dataset(h5_filepath,
                     key_prefix,
                     out_filepaths,
                     start_idx=None,
                     end_idx=None):
  print("PID: %d, Key: %s, (Start, End): (%s, %s)" % (os.getpid(), key_prefix,
                                                      start_idx, end_idx))
  generator_utils.generate_files(
      dataset_generator(h5_filepath, key_prefix, start_idx, end_idx),
      out_filepaths)


def dataset_generator(filepath, dataset, start_idx=None, end_idx=None):
  with h5py.File(filepath, "r") as h5_file:
    # Get input keys from h5_file
    src_keys = [s % dataset for s in ["%s_in", "%s_na", "%s_out"]]
    src_values = [h5_file[k] for k in src_keys]
    inp_data, mask_data, out_data = src_values
    assert len(set([v.len() for v in src_values])) == 1

    if start_idx is None:
      start_idx = 0
    if end_idx is None:
      end_idx = inp_data.len()

    for i in xrange(start_idx, end_idx):
      if i % 100 == 0:
        print("Generating example %d for %s" % (i, dataset))
      inputs, mask, outputs = inp_data[i], mask_data[i], out_data[i]
      yield to_example_dict(inputs, mask, outputs)


def to_example_dict(inputs, mask, outputs):
  """Convert single h5 record to an example dict."""
  # Inputs
  input_ids = []
  last_idx = -1
  for row in np.argwhere(inputs):
    idx, base_id = row
    idx, base_id = int(idx), int(base_id)
    assert idx > last_idx  # if not, means 2 True values in 1 row
    # Some rows are all False. Those rows are mapped to UNK_ID.
    while idx != last_idx + 1:
      input_ids.append(UNK_ID + text_encoder.NUM_RESERVED_TOKENS)
      last_idx += 1
    input_ids.append(base_id + text_encoder.NUM_RESERVED_TOKENS)
    last_idx = idx
  assert len(inputs) == len(input_ids)
  input_ids.append(text_encoder.EOS_ID)

  # Targets: mask and output
  targets_mask = [float(v) for v in mask]
  # The output is (n, m); store targets_shape so that it can be reshaped
  # properly on the other end.
  targets = [float(v) for v in outputs.flatten()]
  targets_shape = [int(dim) for dim in outputs.shape]
  assert mask.shape[0] == outputs.shape[0]

  example_keys = ["inputs", "targets_mask", "targets", "targets_shape"]
  ex_dict = dict(
      zip(example_keys, [input_ids, targets_mask, targets, targets_shape]))
  return ex_dict
