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
"""Utilities to assist in performing adversarial attack using Cleverhans."""

from cleverhans import attacks
from cleverhans import model

from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_attack
def fgsm():
  return attacks.FastGradientMethod


@registry.register_attack
def madry():
  return attacks.MadryEtAl


class T2TAttackModel(model.Model):
  """Wrapper of Cleverhans Model object."""

  def __init__(self, model_fn, params, config):
    self._model_fn = model_fn
    self._params = params
    self._config = config
    self._logits_dict = {}

  def get_logits(self, x):
    if x.name in self._logits_dict:
      return self._logits_dict[x.name]

    x = tf.map_fn(tf.image.per_image_standardization, x)

    logits = self._model_fn(
        {
            "inputs": x
        },
        None,
        "attack",
        params=self._params,
        config=self._config)
    self._logits_dict[x.name] = logits

    return tf.squeeze(logits)
