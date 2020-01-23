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

"""Multi-problem scheduling in T2T.

Data sampling schedules are specified by an interpolation method i and a
sequence of tuples (t, pmf), where i can either be 'linear' or 'step',
t is the global_step at training, and pmf is the distribution from which
training examples from each problem are sampled.

Linear interpolation constructs a piecewise linear training schedule, connecting
pmfs with linear segments. Step interpolation abruptly shifts the sampling
distribution to pmf at global_step t. Both interpolation methods can approximate
any continuous sampling process with sufficient points of interpolation.

Continuation of the interpolant is constant outside the domain specified by
the schedule. That is, we sample from pmfs[0] for global_step < ts[0] and
pmfs[-1] for global_step > ts[-1].

Examples of schedule strings include:

(1) 'step @0 0.7, 0.3': Sample from problem 0 w.p. 0.7 and problem 1 w.p. 0.3
    for the entirety of training. Since there is only one point, the choice of
    interpolation method and global_step does not matter.

(2) 'step @0 1.0 0.0 @100 0.0 1.0': Train on problem 0 for the first 100 steps
    then train on problem 1 for the rest of training.

(3) 'step @0 0.5 0.5 0.0 @100 1.0 0.0 0.0': Pretrain on problems 0 and 1 for the
    first 100 steps then fine tune on problem 2 for the rest of training.

(4) 'linear @0 1.0 0.0 @100 0.0 1.0' Linear transition from training on problem
    0 to problem 1 over 100 steps, then train on problem 1 for the rest of
    training.

(5) 'linear @0 1.0 0.0 @100 0.9 0.1  @200 0.4 0.6  @300 0.0 1.0': Approximate
    inverse exponential decay from problem 0 to problem 1 over 300 steps, then
    train on problem 1 for the rest of training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

import numpy as np

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
import tensorflow.compat.v1 as tf


class MultiProblemV2(problem.Problem):
  """Dataset scheduling for multiple problems."""

  def __init__(self, problems, schedule, **kwargs):
    """Creates a MultiProblem object.

    Args:
      problems: A list of problem.Problem objects.
      schedule: A schedule tuple, see encode_schedule for details.
      **kwargs: Keywords for problem.Problem.__init__.
    """
    super(MultiProblemV2, self).__init__(**kwargs)
    self.problems = problems
    self.schedule = schedule

  def filepattern(self, *args, **kwargs):
    """Returns a list of filepatterns, one for each problem."""
    return [p.filepattern(*args, **kwargs) for p in self.problems]

  def generate_data(self, *args, **kwargs):
    """Generates data for each problem."""
    for p in self.problems:
      p.generate_data(*args, **kwargs)

  @property
  def only_eval_first_problem(self):
    """Only run validation on examples from the first problem."""
    return False

  def normalize_example(self, example, hparams):
    """Preprocesses examples from different problems before mixing."""
    del hparams  # Unused.
    return example

  def dataset(self, mode, hparams=None, global_step=None, **kwargs):
    """Returns a dataset containing examples from multiple problems.

    Args:
      mode: A member of problem.DatasetSplit.
      hparams: A tf.HParams object, the model hparams.
      global_step: A scalar tensor used to compute the sampling distribution.
        If global_step is None, we call tf.train.get_or_create_global_step by
        default.
      **kwargs: Keywords for problem.Problem.Dataset.

    Returns:
      A dataset containing examples from multiple problems.
    """
    datasets = [p.dataset(mode, **kwargs) for p in self.problems]
    datasets = [
        d.map(lambda x, i=j: self.normalize_example(  # pylint: disable=g-long-lambda
            dict(x, problem_id=tf.constant([i])), hparams))
        for j, d in enumerate(datasets)  # Tag examples with a problem_id.
    ]
    if mode is problem.DatasetSplit.TRAIN:
      if global_step is None:
        global_step = tf.train.get_or_create_global_step()
      pmf = get_schedule_distribution(self.schedule, global_step)
      return get_multi_dataset(datasets, pmf)
    elif self.only_eval_first_problem:
      return datasets[0]
    else:
      datasets = [d.repeat() for d in datasets]
      return tf.data.Dataset.zip(tuple(datasets)).flat_map(
          lambda *x: functools.reduce(  # pylint: disable=g-long-lambda
              tf.data.Dataset.concatenate,
              map(tf.data.Dataset.from_tensors, x)))


class MultiText2TextProblem(MultiProblemV2, text_problems.Text2TextProblem):
  """Dataset scheduling for multiple text-to-text problems."""

  def normalize_example(self, example, hparams):
    """Assumes that example contains both inputs and targets."""

    length = self.max_length(hparams)
    def _to_constant_shape(tensor):
      tensor = tensor[:length]
      tensor = tf.pad(tensor, [(0, length - tf.shape(tensor)[0])])
      return tf.reshape(tensor, [length])

    if self.has_inputs:
      example['inputs'] = _to_constant_shape(example['inputs'])
      example['targets'] = _to_constant_shape(example['targets'])
    elif 'inputs' in example:
      if self.packed_length:
        raise ValueError('cannot concatenate packed examples on the fly.')
      inputs = example.pop('inputs')[:-1]  # Remove EOS token.
      targets = tf.concat([inputs, example['targets']], 0)
      example['targets'] = _to_constant_shape(targets)
    else:
      example['targets'] = _to_constant_shape(example['targets'])
    if self.packed_length:
      if self.has_inputs:
        if 'inputs_segmentation' in example:
          example['inputs_segmentation'] = _to_constant_shape(
              example['inputs_segmentation'])
          example['inputs_position'] = _to_constant_shape(
              example['inputs_position'])
        else:
          example['inputs_segmentation'] = tf.to_int64(
              tf.not_equal(example['inputs'], 0))
          example['inputs_position'] = (
              example['inputs_segmentation'] * tf.range(length, dtype=tf.int64))
      if 'targets_segmentation' in example:
        example['targets_segmentation'] = _to_constant_shape(
            example['targets_segmentation'])
        example['targets_position'] = _to_constant_shape(
            example['targets_position'])
      else:
        example['targets_segmentation'] = tf.to_int64(
            tf.not_equal(example['targets'], 0))
        example['targets_position'] = (
            example['targets_segmentation'] * tf.range(length, dtype=tf.int64))
    return example

  def generate_data_with_shared_vocab(self, data_dir, tmp_dir, task_id=-1):
    """Generates TF-Records for problems using a global vocabulary file."""
    global_vocab_filename = os.path.join(data_dir, self.vocab_filename)
    if not tf.gfile.Exists(global_vocab_filename):
      raise ValueError(
          'Global vocabulary file: %s does not exist, '
          'please create one using build_vocab.py' % global_vocab_filename)
    # Before generating data, we copy the global vocabulary file to the children
    # locations. Although this is not the most disk efficient strategy, it
    # imposes the fewest changes to the text-to-text API.
    for p in self.problems:
      local_vocab_filename = os.path.join(data_dir, p.vocab_filename)
      if not tf.gfile.Exists(local_vocab_filename):
        tf.gfile.Copy(global_vocab_filename, local_vocab_filename)
      p.generate_data(data_dir, tmp_dir, task_id)

  @property
  def packed_length(self):
    """Set this to a positive integer if some of the problems are packed."""
    return None


def get_multi_dataset(datasets, pmf=None):
  """Returns a Dataset that samples records from one or more Datasets.

  Args:
    datasets: A list of one or more Dataset objects to sample from.
    pmf: A tensor of shape [len(datasets)], the probabilities to sample each
      dataset with. This tensor is often constructed with the global_step. If
      this is None, we sample from the datasets uniformly at random.

  Returns:
    A Dataset object containing records from multiple datasets. Note that
    because this dataset iterates through other datasets it is stateful, thus
    you will need to call make_initializable_iterator instead of
    make_one_shot_iterator.
  """
  pmf = tf.fill([len(datasets)], 1.0 / len(datasets)) if pmf is None else pmf
  samplers = [d.repeat().make_one_shot_iterator().get_next for d in datasets]
  sample = lambda _: categorical_case(pmf, samplers)
  return tf.data.Dataset.from_tensors([]).repeat().map(sample)


def get_schedule_distribution(schedule, global_step=None):
  """Computes the pmf of a schedule given the global_step.

  Args:
    schedule: A schedule tuple, see encode_schedule for details.
    global_step: A scalar tensor, the step to query the schedule.

  Returns:
    A 1-D tensor of probs, the sampling distribution of the global_step.
  """
  interpolation, steps, pmfs = schedule
  if len(pmfs) == 1:
    # py_func doesn't seem to work on TPU - at least get the constant case to
    # run.
    # TODO(noam): get the general case working.
    return pmfs[0]
  if global_step is None:
    global_step = tf.train.get_or_create_global_step()
  if interpolation == 'step':
    interpolation_fn = step_interpolation
  elif interpolation == 'linear':
    interpolation_fn = linear_interpolation
  else:
    raise ValueError('Invalid interpolation strategy: %s' % interpolation)
  return tf.reshape(
      tf.py_func(
          func=lambda x: interpolation_fn(x, np.array(steps), np.array(pmfs)),
          inp=[global_step], Tout=tf.float32), [len(pmfs[0])])


def categorical_case(pmf, fns, rand=None):
  """Returns the outputs of fns[i] with probability pmf[i].

  Args:
    pmf: A 1-D tensor of probabilities, the probability mass function.
    fns: A list of callables that return tensors, same length as pmf.
    rand: An optional scalar between 0.0 and 1.0, the output of an RNG.

  Returns:
    A tensor, the output of fns[i] with probability pmf[i].
  """
  rand = tf.random_uniform([]) if rand is None else rand
  cmf = tf.pad(tf.cumsum(pmf), [(1, 0)])
  cmf = [cmf[i] for i in range(len(fns) + 1)]
  preds = [(rand >= a) & (rand < b) for a, b in zip(cmf[:-1], cmf[1:])]
  return tf.case(list(zip(preds, fns)), exclusive=True)


def linear_interpolation(x, xp, fp, **kwargs):
  """Multi-dimensional linear interpolation.

  Returns the multi-dimensional piecewise linear interpolant to a function with
  given discrete data points (xp, fp), evaluated at x.

  Note that *N and *M indicate zero or more dimensions.

  Args:
    x: An array of shape [*N], the x-coordinates of the interpolated values.
    xp: An np.array of shape [D], the x-coordinates of the data points, must be
      increasing.
    fp: An np.array of shape [D, *M], the y-coordinates of the data points.
    **kwargs: Keywords for np.interp.

  Returns:
    An array of shape [*N, *M], the interpolated values.
  """
  yp = fp.reshape([fp.shape[0], -1]).transpose()
  y = np.stack([np.interp(x, xp, zp, **kwargs) for zp in yp]).transpose()
  return y.reshape(x.shape[:1] + fp.shape[1:]).astype(np.float32)


def step_interpolation(x, xp, fp, **kwargs):
  """Multi-dimensional step interpolation.

  Returns the multi-dimensional step interpolant to a function with
  given discrete data points (xp, fp), evaluated at x.

  Note that *N and *M indicate zero or more dimensions.

  Args:
    x: An array of shape [*N], the x-coordinates of the interpolated values.
    xp: An np.array of shape [D], the x-coordinates of the data points, must be
      increasing.
    fp: An np.array of shape [D, *M], the y-coordinates of the data points.
    **kwargs: Unused.

  Returns:
    An array of shape [*N, *M], the interpolated values.
  """
  del kwargs  # Unused.
  xp = np.expand_dims(xp, -1)
  lower, upper = xp[:-1], xp[1:]
  conditions = (x >= lower) & (x < upper)
  # Underflow and overflow conditions and values. Values default to fp[0] and
  # fp[-1] respectively.
  conditions = np.concatenate([[x < xp[0]], conditions, [x >= xp[-1]]])
  values = np.concatenate([[fp[0]], fp])
  assert np.all(np.sum(conditions, 0) == 1), 'xp must be increasing.'
  indices = np.argmax(conditions, 0)
  return values[indices].astype(np.float32)


def constant_schedule(pmf):
  """Returns a schedule tuple for constant sampling distribution.

  Args:
    pmf: An array of shape [N] of probabilities. The sampling distribution to
      use throughout training. Probabilities must sum to one.

  Returns:
    A schedule tuple, see encode_schedule for details.
  """
  return ('step', (0,), (tuplize(pmf),))


def example_rates_to_pmf(example_rates):
  """Creates a probability-mass-function based on relative example rates.

  Args:
    example_rates: a list or tuple
  Returns:
    a list of floats
  """
  total = sum(example_rates)
  return [r / total for r in example_rates]


def epoch_rates_to_pmf(problems, epoch_rates=None):
  """Create a probability-mass-function based on relative epoch rates.

  if epoch_rates=None, then we use uniform epoch rates [1.0] * len(problems)
  i.e. it takes each problem the same time to go through one epoch.

  If epoch_rates is given, then these are the relative numbers of epochs
  of each problem to go through in a given amount of time.

  Each must have problem.num_training_examples implemented.

  Args:
    problems: a list of Problem instances.
    epoch_rates: an optional list of float

  Returns:
    a list of floating point values.
  """
  if epoch_rates is None:
    epoch_rates = [1.0] * len(problems)
  example_rates = [epoch_rate * p.num_training_examples
                   for p, epoch_rate in zip(problems, epoch_rates)]
  return example_rates_to_pmf(example_rates)


def encode_schedule(schedule):
  """Encodes a schedule tuple into a string.

  Args:
    schedule: A tuple containing (interpolation, steps, pmfs), where
      interpolation is a string specifying the interpolation strategy, steps
      is an int array_like of shape [N] specifying the global steps, and pmfs is
      an array_like of shape [N, M] where pmf[i] is the sampling distribution
      at global step steps[i]. N is the number of schedule requirements to
      interpolate and M is the size of the probability space.

  Returns:
    The string encoding of the schedule tuple.
  """
  interpolation, steps, pmfs = schedule
  return interpolation + ' ' + ' '.join(
      '@' + str(s) + ' ' + ' '.join(map(str, p)) for s, p in zip(steps, pmfs))


def decode_schedule(string):
  """Decodes a string into a schedule tuple.

  Args:
    string: The string encoding of a schedule tuple.

  Returns:
    A schedule tuple, see encode_schedule for details.
  """
  splits = string.split()
  steps = [int(x[1:]) for x in splits[1:] if x[0] == '@']
  pmfs = np.reshape(
      [float(x) for x in splits[1:] if x[0] != '@'], [len(steps), -1])
  return splits[0], tuplize(steps), tuplize(pmfs)


def tuplize(nested):
  """Recursively converts iterables into tuples.

  Args:
    nested: A nested structure of items and iterables.

  Returns:
    A nested structure of items and tuples.
  """
  if isinstance(nested, str):
    return nested
  try:
    return tuple(map(tuplize, nested))
  except TypeError:
    return nested
