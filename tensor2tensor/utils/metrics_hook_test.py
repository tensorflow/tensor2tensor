# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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

"""Tests for metrics_hook."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import shutil

# Dependency imports

from tensor2tensor.utils import metrics_hook

import tensorflow as tf


class DummyHook(metrics_hook.MetricsBasedHook):

  def _process_metrics(self, global_step, metrics):
    if metrics:
      assert "" in metrics
      assert isinstance(metrics[""], dict)
      if metrics[""]:
        assert "global_step_1" in metrics[""]
    self.test_metrics = metrics
    if global_step >= 40:
      return True


class MetricsHookTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.base_checkpoint_dir = tf.test.get_temp_dir()
    shutil.rmtree(cls.base_checkpoint_dir, ignore_errors=True)

  def ckpt_dir(self, name):
    return os.path.join(self.base_checkpoint_dir, name)

  @contextlib.contextmanager
  def sess(self, hook, ckpt_dir):
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=ckpt_dir,
        save_checkpoint_secs=0,
        save_summaries_steps=10,
        hooks=[hook]) as sess:
      self._sess = sess
      yield sess

  def flush(self):
    self._sess._hooks[1]._summary_writer.flush()

  def testStop(self):
    global_step = tf.train.create_global_step()
    tf.summary.scalar("global_step", global_step)
    incr_global_step = tf.assign_add(global_step, 1)

    ckpt_dir = self.ckpt_dir("stop")
    dummy = DummyHook(ckpt_dir, every_n_steps=10)
    with self.sess(dummy, ckpt_dir) as sess:
      for _ in range(20):
        sess.run(incr_global_step)

      # Summary files should now have 2 global step values in them
      self.flush()

      # Run for 10 more so that the hook gets triggered again
      for _ in range(10):
        sess.run(incr_global_step)

      # Check that the metrics have actually been collected.
      self.assertTrue("" in dummy.test_metrics)
      metrics = dummy.test_metrics[""]
      self.assertTrue("global_step_1" in metrics)
      steps, vals = metrics["global_step_1"]
      self.assertTrue(len(steps) == len(vals))
      self.assertTrue(len(steps) >= 2)

      # Run for 10 more so that the hook triggers stoppage
      for _ in range(10):
        sess.run(incr_global_step)

      with self.assertRaisesRegexp(RuntimeError, "after should_stop requested"):
        sess.run(incr_global_step)

  def testEarlyStoppingHook(self):
    global_step = tf.train.create_global_step()
    counter = tf.get_variable("count", initializer=0, dtype=tf.int32)
    tf.summary.scalar("count", counter)
    incr_global_step = tf.assign_add(global_step, 1)
    incr_counter = tf.assign_add(counter, 1)

    # Stop if the global step has not gone up by more than 1 in 20 steps.

    ckpt_dir = self.ckpt_dir("early")
    stop_hook = metrics_hook.EarlyStoppingHook(
        ckpt_dir,
        "count_1",
        num_plateau_steps=20,
        plateau_delta=1.,
        plateau_decrease=False,
        every_n_steps=10)
    with self.sess(stop_hook, ckpt_dir) as sess:
      for _ in range(20):
        sess.run((incr_global_step, incr_counter))

      # Summary files should now have 2 values in them
      self.flush()

      # Run for more steps so that the hook gets triggered and we verify that we
      # don't stop.
      for _ in range(30):
        sess.run((incr_global_step, incr_counter))

      self.flush()

      # Run without incrementing the counter
      for _ in range(40):
        sess.run(incr_global_step)

      # Metrics should be written such that now the counter has gone >20 steps
      # without being incremented.
      self.flush()

      # Check that we ask for stop
      with self.assertRaisesRegexp(RuntimeError, "after should_stop requested"):
        for _ in range(30):
          sess.run(incr_global_step)

  def testPlateauOpHook(self):
    global_step = tf.train.create_global_step()
    counter = tf.get_variable("count", initializer=0, dtype=tf.int32)
    indicator = tf.get_variable("indicator", initializer=0, dtype=tf.int32)
    tf.summary.scalar("count", counter)
    incr_global_step = tf.assign_add(global_step, 1)
    incr_counter = tf.assign_add(counter, 1)
    incr_indicator = tf.assign_add(indicator, 1)

    # Stop if the global step has not gone up by more than 1 in 20 steps.

    ckpt_dir = self.ckpt_dir("plateauop")
    stop_hook = metrics_hook.PlateauOpHook(
        ckpt_dir,
        "count_1",
        incr_indicator,
        num_plateau_steps=20,
        plateau_delta=1.,
        plateau_decrease=False,
        every_n_steps=10)
    with self.sess(stop_hook, ckpt_dir) as sess:
      for _ in range(20):
        sess.run((incr_global_step, incr_counter))

      # Summary files should now have 2 values in them
      self.flush()

      # Run for more steps so that the hook gets triggered and we verify that we
      # don't stop.
      for _ in range(30):
        sess.run((incr_global_step, incr_counter))

      self.flush()

      # Run without incrementing the counter
      for _ in range(30):
        sess.run(incr_global_step)
      self.flush()

      self.assertTrue(sess.run(indicator) < 1)

      # Metrics should be written such that now the counter has gone >20 steps
      # without being incremented.
      # Check that we run the incr_indicator op several times
      for _ in range(3):
        for _ in range(10):
          sess.run(incr_global_step)
        self.flush()

      self.assertTrue(sess.run(indicator) > 1)

if __name__ == "__main__":
  tf.test.main()
