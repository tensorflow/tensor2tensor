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

"""Gene expression problems.

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

import math
import multiprocessing as mp
import os
import h5py
import numpy as np

from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import dna_encoder
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf

MAX_CONCURRENT_PROCESSES = 10


class GeneExpressionProblem(problem.Problem):
  """Base Problem for gene expression datasets."""

  @property
  def download_url(self):
    raise NotImplementedError()

  @property
  def h5_file(self):
    raise NotImplementedError()

  @property
  def num_output_predictions(self):
    """Number of float predictions per timestep."""
    return 10

  @property
  def chunk_size(self):
    return 4

  def feature_encoders(self, data_dir):
    del data_dir
    return {
        "inputs": dna_encoder.DNAEncoder(chunk_size=self.chunk_size),
        # TODO(rsepassi): RealEncoder?
        "targets": text_encoder.TextEncoder()
    }

  @property
  def num_shards(self):
    return 100

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    try:
      # Download source data if download_url specified
      h5_filepath = generator_utils.maybe_download(tmp_dir, self.h5_file,
                                                   self.download_url)
    except NotImplementedError:
      # Otherwise, look for it locally
      h5_filepath = os.path.join(tmp_dir, self.h5_file)

    with h5py.File(h5_filepath, "r") as h5_file:
      num_train_examples = h5_file["train_in"].len()
      num_dev_examples = h5_file["valid_in"].len()
      num_test_examples = h5_file["test_in"].len()

    # Collect all_filepaths to later shuffle
    all_filepaths = []
    # Collect created shard processes to start and join
    processes = []

    datasets = [(self.training_filepaths, self.num_shards, "train",
                 num_train_examples), (self.dev_filepaths, 10, "valid",
                                       num_dev_examples),
                (self.test_filepaths, 10, "test", num_test_examples)]
    for fname_fn, nshards, key_prefix, num_examples in datasets:
      outfiles = fname_fn(data_dir, nshards, shuffled=False)
      all_filepaths.extend(outfiles)
      for start_idx, end_idx, outfile in generate_shard_args(
          outfiles, num_examples):
        p = mp.Process(
            target=generate_dataset,
            args=(h5_filepath, key_prefix, [outfile], self.chunk_size,
                  start_idx, end_idx))
        processes.append(p)

    # 1 per training shard + 10 for dev + 10 for test
    assert len(processes) == self.num_shards + 20

    # Start and wait for processes in batches
    num_batches = int(
        math.ceil(float(len(processes)) / MAX_CONCURRENT_PROCESSES))
    for i in range(num_batches):
      start = i * MAX_CONCURRENT_PROCESSES
      end = start + MAX_CONCURRENT_PROCESSES
      current = processes[start:end]
      for p in current:
        p.start()
      for p in current:
        p.join()

    # Shuffle
    generator_utils.shuffle_dataset(all_filepaths)

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.modality = {"inputs": modalities.ModalityType.SYMBOL,
                  "targets": modalities.ModalityType.REAL_LOG_POISSON_LOSS}
    p.vocab_size = {"inputs": self._encoders["inputs"].vocab_size,
                    "targets": self.num_output_predictions}
    p.input_space_id = problem.SpaceID.DNA
    p.target_space_id = problem.SpaceID.REAL

  def example_reading_spec(self):
    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets": tf.VarLenFeature(tf.float32),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)

  def preprocess_example(self, example, mode, unused_hparams):
    del mode

    # Reshape targets to contain num_output_predictions per output timestep
    example["targets"] = tf.reshape(example["targets"],
                                    [-1, 1, self.num_output_predictions])
    # Slice off EOS - not needed, and messes up the GeneExpressionConv model
    # which expects the input length to be a multiple of the target length.
    example["inputs"] = example["inputs"][:-1]

    return example

  def eval_metrics(self):
    return [metrics.Metrics.LOG_POISSON, metrics.Metrics.R2]


@registry.register_problem
class GenomicsExpressionCage10(GeneExpressionProblem):

  @property
  def download_url(self):
    return "https://storage.googleapis.com/262k_binned/cage10_l262k_w128.h5"

  @property
  def h5_file(self):
    return "cage10.h5"


@registry.register_problem
class GenomicsExpressionGm12878(GeneExpressionProblem):

  @property
  def download_url(self):
    return "https://storage.googleapis.com/262k_binned/gm12878_l262k_w128.h5"

  @property
  def h5_file(self):
    return "gm12878.h5"


@registry.register_problem
class GenomicsExpressionL262k(GeneExpressionProblem):

  @property
  def h5_file(self):
    return "l262k_w128.h5"


def generate_shard_args(outfiles, num_examples):
  """Generate start and end indices per outfile."""
  num_shards = len(outfiles)
  num_examples_per_shard = num_examples // num_shards
  start_idxs = [i * num_examples_per_shard for i in range(num_shards)]
  end_idxs = list(start_idxs)
  end_idxs.pop(0)
  end_idxs.append(num_examples)
  return zip(start_idxs, end_idxs, outfiles)


def generate_dataset(h5_filepath,
                     key_prefix,
                     out_filepaths,
                     chunk_size=1,
                     start_idx=None,
                     end_idx=None):
  print("PID: %d, Key: %s, (Start, End): (%s, %s)" % (os.getpid(), key_prefix,
                                                      start_idx, end_idx))
  generator_utils.generate_files(
      dataset_generator(h5_filepath, key_prefix, chunk_size, start_idx,
                        end_idx), out_filepaths)


def dataset_generator(filepath,
                      dataset,
                      chunk_size=1,
                      start_idx=None,
                      end_idx=None):
  """Generate example dicts."""
  encoder = dna_encoder.DNAEncoder(chunk_size=chunk_size)
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

    for i in range(start_idx, end_idx):
      if i % 100 == 0:
        print("Generating example %d for %s" % (i, dataset))
      inputs, mask, outputs = inp_data[i], mask_data[i], out_data[i]
      ex_dict = to_example_dict(encoder, inputs, mask, outputs)
      # Original data has one output for every 128 input bases. Ensure that the
      # ratio has been maintained given the chunk size and removing EOS.
      assert (len(ex_dict["inputs"]) - 1) == ((
          128 // chunk_size) * ex_dict["targets_shape"][0])
      yield ex_dict


def to_example_dict(encoder, inputs, mask, outputs):
  """Convert single h5 record to an example dict."""
  # Inputs
  bases = []
  input_ids = []
  last_idx = -1
  for row in np.argwhere(inputs):
    idx, base_id = row
    idx, base_id = int(idx), int(base_id)
    assert idx > last_idx  # if not, means 2 True values in 1 row
    # Some rows are all False. Those rows are mapped to UNK_ID.
    while idx != last_idx + 1:
      bases.append(encoder.UNK)
      last_idx += 1
    bases.append(encoder.BASES[base_id])
    last_idx = idx
  assert len(inputs) == len(bases)

  input_ids = encoder.encode(bases)
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
