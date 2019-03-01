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

r"""Jaxboard Summary Types Demo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags

import warnings  # pylint: disable=g-bad-import-order
import matplotlib as mpl
with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  mpl.use('Agg')
# pylint: disable=g-import-not-at-top
from matplotlib import pyplot as plt
import numpy as onp

from tensor2tensor.jax import jaxboard

flags.DEFINE_string('tb_log_dir', '/tmp/tb_logs',
                    'Path where we store summaries.')
FLAGS = flags.FLAGS


def demo():
  """Run Summary Types Demo."""
  sw = jaxboard.SummaryWriter(
      os.path.join(FLAGS.tb_log_dir, 'demo', 'summarydemo'))

  # Scalars.  We pass in step explicitly.
  for i, v in enumerate(onp.sin(onp.linspace(0.0, 1.0, 100))):
    sw.scalar('summarydemo_loss', v + 0.1 * onp.random.random(), step=i)

  # SummaryWriter stores last step variable passed in, we can also set it
  # explicitly for a set of exports to avoid providing the kwarg.
  sw.step = 2

  # Images. [H,W] or [H,W,C] with C = 1 or 3
  sw.image('pic_c0', onp.random.random((100, 100)))
  sw.image('pic_c1', onp.random.random((100, 100, 1)))
  sw.image('pic_c3', onp.random.random((100, 100, 3)))

  # Tiled sets of images. Must be [N,H,W,C] with C = 1 or 3
  bw_tiles = onp.stack([
      0.1 * onp.random.random((100, 100, 1)), 0.2 * onp.random.random(
          (100, 100, 1)), 0.4 * onp.random.random((100, 100, 1)),
      0.8 * onp.random.random((100, 100, 1))
  ])
  sw.images('pics_tiled_c1', bw_tiles, rows=2, cols=2)
  clr_tiles = onp.stack([
      0.1 * onp.random.random((100, 100, 3)), 0.2 * onp.random.random(
          (100, 100, 3)), 0.4 * onp.random.random((100, 100, 3)),
      0.8 * onp.random.random((100, 100, 3))
  ])
  sw.images('pics_tiled_c3', clr_tiles, rows=2, cols=2)

  # Matplotlib plots. Just pass in prepared stateful pyplot object.
  # -- scatter
  plt.figure(figsize=(4, 4))
  plt.scatter(
      onp.random.randint(size=(10,), low=0, high=10),
      onp.random.randint(size=(10,), low=0, high=10))
  sw.plot('plot1', plt)

  # -- imshow
  plt.figure(figsize=(4, 4))
  plt.imshow(
      onp.random.randint(size=(50, 50, 3), low=0, high=255),
      cmap='viridis',
      interpolation='nearest')
  sw.plot('plot2', plt)

  # Audio.
  t = onp.linspace(0, 1.0, 44100)
  sinwave = (
      0.1 * onp.sin(440. * onp.pi * t) *
      # slow ramp-up to prevent 'pop'
      onp.where(t > 0.2, 1.0, t / 0.2))
  sw.audio('audio', sinwave)

  # Text.
  # -- tensorboard text plugin supports some markdown formatting!
  sw.text('text', 'Colorless _green_ __ideas__ sleep furiously.')

  # -- 1d/2d arrays of strings rendered as tables by plugin:
  sw.text('text1d', ['Colorless', 'green', 'ideas', 'sleep', 'furiously.'])
  sw.text('text2d', onp.array([['foo', 'bar'], ['baz', 'qup']]))

  # Histograms / Distributions.
  # (bins can be int or array - passed into onp.histogram bins arg)
  sw.histogram('histo', onp.random.normal(size=(1000,)), 25, step=3)
  sw.histogram('histo', onp.random.normal(size=(1000,)), 25, step=4)
  sw.histogram('histo', onp.random.normal(size=(1000,)), 25, step=5)

  # Fin.
  sw.close()


def main(argv):
  del argv
  demo()


if __name__ == '__main__':
  app.run(main)
