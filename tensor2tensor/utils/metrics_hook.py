# coding=utf-8
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

"""Summary-based SessionRunHooks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports

import tensorflow as tf

from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import event_multiplexer


class MetricsBasedHook(tf.train.SessionRunHook):
  """Base class for hooks based on summary metrics.

  Subclasses should override _process_metrics.

  If _process_metrics returns True, calls run_context.request_stop().

  This can be used to something like "Stop after the loss has stopped decreasing
  for 5000 steps.
  """
  _RUN_NAME = "run%d"

  def __init__(self, events_dir, subdirs=None, tags=None, every_n_steps=1000):
    """Construct MetricsBasedHook.

    Args:
      events_dir: str, top-level directory containing events files.
      subdirs: list<str>, subdirectories of events_dir that also contain
        events files. Use "" to specify the top-level directory. Defaults to
        [""].
      tags: list<str>, names of metrics to collect. Default will collect all
        metrics.
      every_n_steps: int, collect metrics every n steps.
    """
    self._events_dir = events_dir
    self._subdirs = subdirs or [""]
    self._tags = tags
    self._every_n_steps = every_n_steps
    self._start_step = None
    self._event_multiplexer = self._init_multiplexer()

  def _init_multiplexer(self):
    dirs = [os.path.join(self._events_dir, subdir) for subdir in self._subdirs]
    run_path_map = dict([(self._RUN_NAME % i, d) for i, d in enumerate(dirs)])
    return event_multiplexer.EventMultiplexer(run_path_map)

  def begin(self):
    self._global_step_tensor = tf.train.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError("Global step must be created to use MetricsBasedHook.")

  def after_create_session(self, session, coord):
    del coord
    if self._start_step is None:
      self._start_step = session.run(self._global_step_tensor)

  def before_run(self, run_context):
    del run_context
    return tf.train.SessionRunArgs([self._global_step_tensor])

  def after_run(self, run_context, run_values):
    global_step = run_values.results[0]
    if (global_step - self._start_step) % self._every_n_steps != 0:
      return
    metrics = self._collect_metrics()
    self._after_run(run_context, run_values, global_step, metrics)

  def _after_run(self, run_context, run_values, global_step, metrics):
    if self._process_metrics(global_step, metrics):
      run_context.request_stop()

  def _collect_metrics(self):
    self._event_multiplexer.Reload()
    subdir_data = {}
    for i, subdir in enumerate(self._subdirs):
      subdir_metrics = {}

      accum = self._event_multiplexer.GetAccumulator(self._RUN_NAME % i)
      for tag in accum.Tags()[event_accumulator.SCALARS]:
        steps, vals = zip(*[
            (event.step, event.value) for event in accum.Scalars(tag)])
        subdir_metrics[tag] = (steps, vals)

      subdir_data[subdir] = subdir_metrics
    return subdir_data

  def _process_metrics(self, global_step, metrics):
    """Process the collected metrics.

    Args:
      global_step: int, the current global step value.
      metrics: dict<str subdirectory, dict subdir_metrics>. The collected
        metrics. subdir_metrics is a dict from tag name to tuple of lists. The
        lists are a list of global steps and a list of values.
        i.e. subdir_metrics:
          `dict<str tag, tuple<list<int> global steps, list<float> values>>>`

    Returns:
      should_stop: bool. If True, will request that the session stops.
    """
    return False


class EarlyStoppingHook(MetricsBasedHook):
  """EarlyStoppingHook will stop training when a given metric has plateaued."""

  def __init__(self,
               events_dir,
               tag,
               num_plateau_steps=1000,
               plateau_delta=0.1,
               plateau_decrease=True,
               every_n_steps=1000):
    """Create an EarlyStoppingHook.

    This hook will stop training when the metric identified by tag has
    plateaued. Plateaued is defined by the metric having stopped
    increasing/decreasing (based on plateau_decrease) by plateau_delta for
    num_plateau_steps.

    Args:
      events_dir: Directory with events files.
      tag: Name of metric in TensorBoard.
      num_plateau_steps: Number of steps over which to check the plateau.
      plateau_delta: delta to define a "plateau".
      plateau_decrease: whether to check decrease or increase in the metric.
      every_n_steps: how often to run this hook.

    Returns:
      An instance of EarlyStoppingHook.
    """
    super(EarlyStoppingHook, self).__init__(
        events_dir=events_dir, tags=[tag], every_n_steps=every_n_steps)
    self._num_plateau_steps = num_plateau_steps
    self._plateau_delta = plateau_delta
    self._plateau_decrease = plateau_decrease

  def _process_metrics(self, global_step, metrics):
    if not metrics:
      return

    if not list(metrics.values())[0]:
      return

    # Metrics should have just a single subdir and a single tag
    steps, vals = list(metrics.values())[0][self._tags[0]]
    return has_metric_plateaued(
        steps,
        vals,
        num_steps=self._num_plateau_steps,
        delta=self._plateau_delta,
        decrease=self._plateau_decrease)


class PlateauOpHook(MetricsBasedHook):
  """Runs an op when a metric has plateaued."""

  def __init__(self,
               events_dir,
               tag,
               plateau_op,
               num_plateau_steps=1000,
               plateau_delta=0.1,
               plateau_decrease=True,
               every_n_steps=1000,
               only_once=False):
    """See EarlyStoppingHook for args. Runs plateau_op if plateaued."""
    super(PlateauOpHook, self).__init__(
        events_dir=events_dir, tags=[tag], every_n_steps=every_n_steps)
    self._num_plateau_steps = num_plateau_steps
    self._plateau_delta = plateau_delta
    self._plateau_decrease = plateau_decrease
    self._plateau_op = plateau_op
    self._only_once = only_once
    self._should_run_op = False
    self._ever_ran = False
    self._last_metric_step_seen = 0

  @property
  def keep_alive(self):
    if self._only_once and self._ever_ran:
      return False
    return True

  def before_run(self, run_context):
    del run_context

    fetches = [self._global_step_tensor]
    if self._should_run_op and self.keep_alive:
      fetches.append(self._plateau_op)
      self._should_run_op = False
      self._ever_ran = True

    return tf.train.SessionRunArgs(fetches)

  def _after_run(self, run_context, run_values, global_step, metrics):
    del run_context
    del run_values
    del global_step

    if not self.keep_alive:
      return

    if not metrics:
      return

    if not list(metrics.values())[0]:
      return

    # There should be only a single subdir and a single tag
    steps, vals = list(metrics.values())[0][self._tags[0]]

    if not steps:
      return

    last_step = steps[-1]
    if last_step == self._last_metric_step_seen:
      return
    self._last_metric_step_seen = last_step

    if has_metric_plateaued(
        steps,
        vals,
        num_steps=self._num_plateau_steps,
        delta=self._plateau_delta,
        decrease=self._plateau_decrease):
      self._should_run_op = True


def has_metric_plateaued(steps, values, num_steps=100, delta=0.1,
                         decrease=True):
  """Check if metric has plateaued.

  A metric has plateaued if the value has not increased/decreased (depending on
  `decrease`) by `delta` for at least `num_steps`.

  Args:
    steps: list<int> list of global steps for values.
    values: list<float> list of metric values.
    num_steps: int, number of steps the metric has to have been plateaued for.
    delta: float, how much the metric should have changed by over num_steps.
    decrease: bool, whether to check if the metric has decreased by delta or
      increased by delta.

  Returns:
    bool, whether the metric has plateaued.
  """
  assert num_steps > 0
  if len(steps) < 2:
    return False

  steps_at_least_num_steps_ago = [
      s for s in steps if s <= (steps[-1] - num_steps)
  ]
  if not steps_at_least_num_steps_ago:
    # Not enough steps yet
    return False
  delta_step_idx = len(steps_at_least_num_steps_ago) - 1

  start_val = values[delta_step_idx]
  values_to_check = values[delta_step_idx:]
  observed_deltas = []
  for val in values_to_check:
    if decrease:
      observed_delta = start_val - val
    else:
      observed_delta = val - start_val
    observed_deltas.append(observed_delta)

  within_range = [obs < delta for obs in observed_deltas]
  return all(within_range)
