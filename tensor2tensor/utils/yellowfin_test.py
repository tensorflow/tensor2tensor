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
"""YellowFin Test Module for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from tensor2tensor.utils.yellowfin import YellowFinOptimizer

import tensorflow as tf


n_dim = 1000000
n_iter = 0


class YellowFinTest(tf.test.TestCase):

  def tune_everything(self, x0squared, c, t, gmin, gmax):
    del t
    # First tune based on dynamic range
    if c == 0:
      dr = gmax / gmin
      mustar = ((np.sqrt(dr) - 1) / (np.sqrt(dr) + 1))**2
      alpha_star = (1 + np.sqrt(mustar))**2/gmax

      return alpha_star, mustar

    dist_to_opt = x0squared
    grad_var = c
    max_curv = gmax
    min_curv = gmin
    const_fact = dist_to_opt * min_curv**2 / 2 / grad_var
    coef = [-1, 3, -(3 + const_fact), 1]
    roots = np.roots(coef)
    roots = roots[np.real(roots) > 0]
    roots = roots[np.real(roots) < 1]
    root = roots[np.argmin(np.imag(roots))]

    assert root > 0 and root < 1 and np.absolute(root.imag) < 1e-6

    dr = max_curv / min_curv
    assert max_curv >= min_curv
    mu = max(((np.sqrt(dr) - 1) / (np.sqrt(dr) + 1))**2, root**2)

    lr_min = (1 - np.sqrt(mu))**2 / min_curv

    alpha_star = lr_min
    mustar = mu

    return alpha_star, mustar

  def testMeasurement(self):
    opt = YellowFinOptimizer(zero_debias=False)
    w = tf.Variable(np.ones([n_dim,]),
                    dtype=tf.float32,
                    name="w",
                    trainable=True)
    b = tf.Variable(np.ones([1,], dtype=np.float32),
                    dtype=tf.float32,
                    name="b",
                    trainable=True)
    x = tf.constant(np.ones([n_dim,], dtype=np.float32),
                    dtype=tf.float32)
    _ = tf.multiply(w, x) + b  # loss
    tvars = tf.trainable_variables()

    w_grad_val = tf.placeholder(tf.float32, shape=(n_dim,))
    b_grad_val = tf.placeholder(tf.float32, shape=(1,))
    apply_op = opt.apply_gradients(zip([w_grad_val, b_grad_val], tvars))

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init_op)
      target_h_max = 0.0
      target_h_min = 0.0
      g_norm_squared_avg = 0.0
      g_norm_avg = 0.0
      g_avg = 0.0
      target_dist = 0.0
      for i in range(n_iter):
        feed_dict = {w_grad_val: (i + 1) * np.ones([n_dim,], dtype=np.float32),
                     b_grad_val: (i + 1) * np.ones([1,], dtype=np.float32)}
        res = sess.run([opt._curv_win,
                        opt._h_max,
                        opt._h_min,
                        opt._grad_var,
                        opt._dist_to_opt_avg,
                        apply_op], feed_dict=feed_dict)

        g_norm_squared_avg = (
            0.999 * g_norm_squared_avg +
            0.001 * np.sum(((i + 1) * np.ones([n_dim + 1,]))**2))
        g_norm_avg = (0.999 * g_norm_avg +
                      0.001 * np.linalg.norm((i + 1)*np.ones([n_dim + 1,])))
        g_avg = 0.999 * g_avg + 0.001 * (i + 1)

        target_h_max = 0.999 * target_h_max + 0.001 * (i + 1)**2*(n_dim + 1)
        target_h_min = (0.999 * target_h_min +
                        0.001 * max(1, i + 2 - 20)**2 * (n_dim + 1))
        target_var = g_norm_squared_avg - g_avg**2 * (n_dim + 1)
        target_dist = (0.999 * target_dist +
                       0.001 * g_norm_avg / g_norm_squared_avg)

        assert np.abs(target_h_max - res[1]) < np.abs(target_h_max) * 1e-3
        assert np.abs(target_h_min - res[2]) < np.abs(target_h_min) * 1e-3
        assert np.abs(target_var - res[3]) < np.abs(res[3]) * 1e-3
        assert np.abs(target_dist - res[4]) < np.abs(res[4]) * 1e-3

  def testLrMu(self):
    opt = YellowFinOptimizer(learning_rate=0.5, momentum=0.5, zero_debias=False)
    w = tf.Variable(np.ones([n_dim,]),
                    dtype=tf.float32,
                    name="w",
                    trainable=True)
    b = tf.Variable(np.ones([1,],
                            dtype=np.float32),
                    dtype=tf.float32,
                    name="b",
                    trainable=True)
    x = tf.constant(np.ones([n_dim,], dtype=np.float32), dtype=tf.float32)
    _ = tf.multiply(w, x) + b  # loss
    tvars = tf.trainable_variables()

    w_grad_val = tf.Variable(np.zeros([n_dim,]),
                             dtype=tf.float32,
                             trainable=False)
    b_grad_val = tf.Variable(np.zeros([1,]),
                             dtype=tf.float32,
                             trainable=False)
    apply_op = opt.apply_gradients(zip([w_grad_val, b_grad_val], tvars))

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init_op)
      target_h_max = 0.0
      target_h_min = 0.0
      g_norm_squared_avg = 0.0
      g_norm_avg = 0.0
      g_avg = 0.0
      target_dist = 0.0
      target_lr = 0.5
      target_mu = 0.5
      for i in range(n_iter):

        sess.run(tf.assign(w_grad_val, (i + 1) * np.ones([n_dim,],
                                                         dtype=np.float32)))
        sess.run(tf.assign(b_grad_val, (i + 1) * np.ones([1,],
                                                         dtype=np.float32)))

        res = sess.run([opt._curv_win,
                        opt._h_max,
                        opt._h_min,
                        opt._grad_var,
                        opt._dist_to_opt_avg,
                        opt._lr_var,
                        opt._mu_var,
                        apply_op])

        res[5] = opt._lr_var.eval()
        res[6] = opt._mu_var.eval()

        g_norm_squared_avg = (
            0.999 * g_norm_squared_avg +
            0.001 * np.sum(((i + 1) * np.ones([n_dim + 1,]))**2))
        g_norm_avg = (0.999 * g_norm_avg +
                      0.001 * np.linalg.norm((i + 1)*np.ones([n_dim + 1,])))
        g_avg = 0.999 * g_avg + 0.001 * (i + 1)

        target_h_max = 0.999 * target_h_max + 0.001 * (i + 1)**2 * (n_dim + 1)
        target_h_min = (0.999 * target_h_min +
                        0.001 * max(1, i + 2 - 20)**2 * (n_dim + 1))
        target_var = g_norm_squared_avg - g_avg**2 * (n_dim + 1)
        target_dist = (0.999 * target_dist +
                       0.001 * g_norm_avg / g_norm_squared_avg)

        if i > 0:
          lr, mu = self.tune_everything(target_dist**2,
                                        target_var,
                                        1,
                                        target_h_min,
                                        target_h_max)
          target_lr = 0.999 * target_lr + 0.001 * lr
          target_mu = 0.999 * target_mu + 0.001 * mu

        assert np.abs(target_h_max - res[1]) < np.abs(target_h_max) * 1e-3
        assert np.abs(target_h_min - res[2]) < np.abs(target_h_min) * 1e-3
        assert np.abs(target_var - res[3]) < np.abs(res[3]) * 1e-3
        assert np.abs(target_dist - res[4]) < np.abs(res[4]) * 1e-3
        assert (target_lr == 0.0 or
                (np.abs(target_lr - res[5]) < np.abs(res[5]) * 1e-3))
        assert (target_mu == 0.0 or
                (np.abs(target_mu - res[6]) < np.abs(res[6]) * 5e-3))


if __name__ == "__main__":
  tf.test.main()
