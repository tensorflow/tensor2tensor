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

# Dependency imports

import numpy as np

from tensor2tensor.utils.multistep_optimizer import MultistepAdamOptimizer

import tensorflow as tf


def adam_update_numpy(param,
                      g_t,
                      t,
                      m,
                      v,
                      alpha=0.001,
                      beta1=0.9,
                      beta2=0.999,
                      epsilon=1e-8):
  alpha_t = alpha * np.sqrt(1 - beta2**t) / (1 - beta1**t)

  m_t = beta1 * m + (1 - beta1) * g_t
  v_t = beta2 * v + (1 - beta2) * g_t * g_t

  param_t = param - alpha_t * m_t / (np.sqrt(v_t) + epsilon)
  return param_t, m_t, v_t


class MultistepAdamOptimizerTest(tf.test.TestCase):

  def testMultistep(self):
    ver = tf.__version__.split('.')
    # TODO: Remove version check once 1.5 is not tested anymore
    if int(ver[0]) <= 1 and int(ver[1]) < 6:
      # MultistepAdamOptimizer requires TF >= 1.6
      return
    dtype = tf.float32
    beta1=0.2
    beta2=0.99
    alpha=10.0
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
    # Test accumulating gradients for n=1..4 steps
    for n in range(1, 5):
      with self.test_session():
        with self.test_session(graph=tf.Graph()):
          # Initialize variables for numpy implementation.
          m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
          var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
          var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)

          var0 = tf.Variable(var0_np)
          var1 = tf.Variable(var1_np)

          opt = MultistepAdamOptimizer(
              n=n, beta1=beta1, beta2=beta2, learning_rate=alpha)
          updates = [
              opt.apply_gradients([(tf.constant(g0), var0),
                                   (tf.constant(g1), var1)])
              for g0, g1 in zip(grads0_np_lst, grads1_np_lst)][:n]

          self.evaluate(tf.global_variables_initializer())
          beta1_power, beta2_power = opt._get_beta_accumulators()
          # Fetch params to validate initial values
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))

          # Average gradients for numpy implementation
          avg_grads0_np = sum(grads0_np_lst[:n]) / n
          avg_grads1_np = sum(grads1_np_lst[:n]) / n
          # Run 3 steps of Adam
          for t in range(1, 4):
            for update in updates:  # Do n updates
              self.evaluate(update)

            self.assertAllCloseAccordingToType(beta1**(t + 1),
                                               self.evaluate(beta1_power))
            self.assertAllCloseAccordingToType(beta2**(t + 1),
                                               self.evaluate(beta2_power))
            var0_np, m0, v0 = adam_update_numpy(
                var0_np, avg_grads0_np, t, m0, v0,
                beta1=beta1, beta2=beta2, alpha=alpha)
            var1_np, m1, v1 = adam_update_numpy(
                var1_np, avg_grads1_np, t, m1, v1,
                beta1=beta1, beta2=beta2, alpha=alpha)

            # Validate updated params
            self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
            self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))


if __name__ == "__main__":
  tf.test.main()

