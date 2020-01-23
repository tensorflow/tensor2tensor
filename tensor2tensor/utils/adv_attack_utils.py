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

"""Utilities to assist in performing adversarial attack using Cleverhans."""

from cleverhans import attacks
from cleverhans import model
from cleverhans import utils_tf

import numpy as np

from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


@registry.register_attack
def fgsm():
  return attacks.FastGradientMethod


@registry.register_attack
def madry():
  return attacks.MadryEtAl


@registry.register_attack
def random():
  return RandomAttack


class T2TAttackModel(model.Model):
  """Wrapper of Cleverhans Model object."""

  def __init__(self, model_fn, features, params, config, scope=None):
    self._model_fn = model_fn
    self._params = params
    self._config = config
    self._logits_dict = {}
    self._additional_features = features
    self._scope = scope

  def fprop(self, x):
    if x.name in self._logits_dict:
      return self._logits_dict[x.name]

    x = tf.map_fn(tf.image.per_image_standardization, x)
    self._additional_features['inputs'] = x

    if self._scope is None:
      scope = tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE)
    else:
      scope = tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE)

    with scope:
      logits = self._model_fn(
          self._additional_features,
          None,
          'attack',
          params=self._params,
          config=self._config)
    self._logits_dict[x.name] = logits

    return {model.Model.O_LOGITS: tf.reshape(logits, [-1, logits.shape[-1]])}


class RandomAttack(attacks.FastGradientMethod):
  """Blackbox random sample attack."""

  def __init__(self, m, back='tf', sess=None):
    if not isinstance(m, model.Model):
      m = model.CallableModelWrapper(m, 'probs')

    super(RandomAttack, self).__init__(m, back, sess)
    self.feedable_kwargs = {
        'eps': np.float32,
        'num_samples': np.float32,
        'num_batches': np.float32,
        'y': np.float32,
        'y_target': np.float32,
        'clip_min': np.float32,
        'clip_max': np.float32
    }
    self.structural_kwargs = ['ord']

  def generate(self, x, **kwargs):
    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    labels, _ = self.get_or_guess_labels(x, kwargs)

    x_shape = x.shape.as_list()
    deltas_shape = [x_shape[0], self.num_samples] + x_shape[1:]

    def cond(i, old_adv_x, old_loss):
      del old_adv_x, old_loss
      return tf.less(i, self.num_batches)

    def body(i, old_adv_x, old_loss, labels=labels):
      """Find example with max loss value amongst batch of perturbations."""
      deltas = tf.random_uniform(deltas_shape)

      # generate uniform samples from the l^p unit ball interior
      if self.ord == np.inf:
        deltas *= 2. * self.eps
        deltas -= self.eps
      elif self.ord == 1:
        # ref: https://mathoverflow.net/questions/9185/how-to-generate-random-points-in-ell-p-balls  pylint: disable=line-too-long
        exp = -tf.log(deltas)
        shift = -tf.log(tf.random_uniform(deltas_shape[:2]))
        norm = tf.reduce_sum(tf.abs(exp), range(2, len(deltas_shape) - 2))
        scale = tf.reshape(shift + norm,
                           deltas_shape[:2] + [1] * (len(deltas_shape) - 2))
        deltas = exp / scale
      elif self.ord == 2:
        # ref: https://blogs.sas.com/content/iml/2016/04/06/generate-points-uniformly-in-ball.html  pylint: disable=line-too-long
        dims = tf.reduce_prod(deltas_shape[2:])
        deltas = tf.pow(deltas, 1. / dims)
        normal = tf.random_normal(deltas)
        normal /= tf.sqrt(
            tf.reduce_sum(normal**2, axis=range(2,
                                                len(deltas_shape) - 2)),
            keepdims=True)
        deltas *= normal
      else:
        raise NotImplementedError('Only L-inf, L1 and L2 norms are '
                                  'currently implemented.')

      adv_x = tf.expand_dims(x, 1) + deltas
      labels = tf.expand_dims(labels, 1)
      labels = tf.tile(labels, [1, self.num_samples, 1])

      if (self.clip_min is not None) and (self.clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

      adv_x_r = tf.reshape(adv_x, [-1] + deltas_shape[2:])
      preds = self.model.get_probs(adv_x_r)
      preds_shape = preds.shape.as_list()
      preds = tf.reshape(preds, deltas_shape[:2] + preds_shape[1:])

      if labels is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, -1, keep_dims=True)
        labels = tf.to_float(tf.equal(preds, preds_max))
        labels = tf.stop_gradient(labels)
      labels = labels / tf.reduce_sum(labels, -1, keep_dims=True)

      # Compute loss
      loss = utils_tf.model_loss(labels, preds, mean=False)
      if self.y_target is not None:
        loss = -loss

      # find the maximum loss value
      input_idx = tf.one_hot(tf.argmax(loss, axis=1), self.num_samples, axis=1)
      loss = tf.reduce_sum(loss * input_idx, axis=1)
      input_idx = tf.reshape(input_idx,
                             deltas_shape[:2] + [1] * (len(deltas_shape) - 2))
      adv_x = tf.reduce_sum(adv_x * input_idx, axis=1)

      condition = tf.greater(old_loss, loss)
      new_loss = tf.where(condition, old_loss, loss)
      new_adv_x = tf.where(condition, old_adv_x, adv_x)
      print(new_loss, new_adv_x)

      return i + 1, new_adv_x, new_loss

    _, adv_x, _ = tf.while_loop(
        cond, body,
        [tf.zeros([]),
         tf.zeros_like(x), -1e10 * tf.ones(x_shape[0])], back_prop=False)

    return adv_x

  def parse_params(
      self,
      eps=0.3,
      num_samples=100,
      num_batches=100,
      ord=np.inf,  # pylint: disable=redefined-builtin
      y=None,
      y_target=None,
      clip_min=None,
      clip_max=None,
      **kwargs):
    self.num_samples = num_samples
    self.num_batches = num_batches
    return super(RandomAttack, self).parse_params(eps, ord, y, y_target,
                                                  clip_min, clip_max, **kwargs)
