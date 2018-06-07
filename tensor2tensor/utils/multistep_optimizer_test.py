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
"""Multi-step Optimizer Test Module for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensor2tensor.utils import multistep_optimizer
import tensorflow as tf


class MultistepAdamOptimizerTest(tf.test.TestCase):

  def testMultistep(self):
    ver = tf.__version__.split('.')
    # TODO(rsepassi): Remove version check once 1.5 is not tested anymore
    if int(ver[0]) <= 1 and int(ver[1]) < 6:
      # MultistepAdamOptimizer requires TF >= 1.6
      return
    dtype = tf.float32
    beta1 = 0.2
    beta2 = 0.99
    alpha = 10.0
    grads0_np_lst = [
        np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype),
        np.array([0.2, -0.1], dtype=dtype.as_numpy_dtype),
        np.array([0.3, 0.1], dtype=dtype.as_numpy_dtype),
        np.array([0.4, -0.1], dtype=dtype.as_numpy_dtype)
    ]
    grads1_np_lst = [
        np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype),
        np.array([0.02, 0.02], dtype=dtype.as_numpy_dtype),
        np.array([-0.04, 0.04], dtype=dtype.as_numpy_dtype),
        np.array([-0.04, 0.06], dtype=dtype.as_numpy_dtype)
    ]
    var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
    var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
    # Test accumulating gradients for n=1..4 steps
    for n in range(1, 5):
      with tf.Graph().as_default():
        with tf.Session():
          singlestep_var0 = tf.Variable(var0_np)
          singlestep_var1 = tf.Variable(var1_np)

          multistep_var0 = tf.Variable(var0_np)
          multistep_var1 = tf.Variable(var1_np)

          singlestep_opt = tf.train.AdamOptimizer(
              beta1=beta1, beta2=beta2, learning_rate=alpha)
          multistep_opt = multistep_optimizer.MultistepAdamOptimizer(
              n=n, beta1=beta1, beta2=beta2, learning_rate=alpha)

          singlestep_update = singlestep_opt.apply_gradients([
              (tf.constant(sum(grads0_np_lst[:n]) / n), singlestep_var0),
              (tf.constant(sum(grads1_np_lst[:n]) / n), singlestep_var1)])
          multistep_updates = [
              multistep_opt.apply_gradients([(tf.constant(g0), multistep_var0),
                                             (tf.constant(g1), multistep_var1)])
              for g0, g1 in zip(grads0_np_lst, grads1_np_lst)][:n]

          self.evaluate(tf.global_variables_initializer())
          (singlestep_beta1_power,
           singlestep_beta2_power) = singlestep_opt._get_beta_accumulators()
          (multistep_beta1_power,
           multistep_beta2_power) = multistep_opt._get_beta_accumulators()

          # Run 3 steps of Adam
          for _ in range(1, 4):
            self.evaluate(singlestep_update)
            for multistep_update in multistep_updates:
              self.evaluate(multistep_update)

            self.assertAllCloseAccordingToType(
                self.evaluate(singlestep_beta1_power),
                self.evaluate(multistep_beta1_power))
            self.assertAllCloseAccordingToType(
                self.evaluate(singlestep_beta2_power),
                self.evaluate(multistep_beta2_power))
            # Validate updated params
            self.assertAllCloseAccordingToType(
                self.evaluate(singlestep_var0),
                self.evaluate(multistep_var0))
            self.assertAllCloseAccordingToType(
                self.evaluate(singlestep_var1),
                self.evaluate(multistep_var1))


if __name__ == '__main__':
  tf.test.main()
