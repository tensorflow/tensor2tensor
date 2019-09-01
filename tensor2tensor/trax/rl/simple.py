# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

"""SimPLe helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import random

import cloudpickle as pickle
import numpy as np
from tensorflow.io import gfile


def load_trajectories(trajectory_dir, eval_frac):
  """Loads trajectories from a possibly nested directory of pickles."""
  train_trajectories = []
  eval_trajectories = []
  # Search the entire directory subtree for trajectories.
  for (subdir, _, filenames) in gfile.walk(trajectory_dir):
    for filename in filenames:
      shard_path = os.path.join(subdir, filename)
      with gfile.GFile(shard_path, "rb") as f:
        trajectories = pickle.load(f)
        pivot = int(len(trajectories) * (1 - eval_frac))
        train_trajectories.extend(trajectories[:pivot])
        eval_trajectories.extend(trajectories[pivot:])
  assert train_trajectories, "Haven't found any training data."
  assert eval_trajectories, "Haven't found any evaluation data."
  return (train_trajectories, eval_trajectories)


def generate_examples(trajectories, trajectory_to_training_examples_fn):
  """Generates an infinite stream of shuffled examples out of trajectories."""
  examples = [
      example  # pylint: disable=g-complex-comprehension
      for trajectory_examples in map(
          trajectory_to_training_examples_fn, trajectories)
      for example in trajectory_examples
  ]
  assert examples
  while True:
    random.shuffle(examples)
    for example in examples:
      yield example


def mix_streams(stream1, stream2, mix_prob):
  """Mixes two streams together with a fixed probability."""
  while True:
    # In the corner cases (mix_prob = 0 or 1) mixing the other stream never
    # happens, because random() samples from the semi-open interval [0, 1).
    if random.random() < mix_prob:
      yield next(stream1)
    else:
      yield next(stream2)


def batch_stream(stream, batch_size):
  """Batches a stream of training examples."""
  def make_batch(examples):
    """Stacks a structure of numpy arrays nested in lists/tuples."""
    assert examples
    if isinstance(examples[0], (list, tuple)):
      return type(examples[0])(
          make_batch([example[i] for example in examples])
          for i in range(len(examples[0]))
      )
    else:
      return np.stack(examples, axis=0)

  # Take consecutive batches from an infinite stream. This way there are no
  # incomplete batches. We might get duplicate examples in the same batch, but
  # that should be very rare.
  while True:
    yield make_batch(list(itertools.islice(stream, batch_size)))
