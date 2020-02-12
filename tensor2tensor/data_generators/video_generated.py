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

"""Data generators for video problems with artificially generated frames."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from tensor2tensor.data_generators import video_utils
from tensor2tensor.layers import modalities
from tensor2tensor.utils import contrib
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf

try:
  import matplotlib  # pylint: disable=g-import-not-at-top
  matplotlib.use("agg")
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
except ImportError:
  pass


@registry.register_problem
class VideoStochasticShapes10k(video_utils.VideoProblem):
  """Shapes moving in a stochastic way."""

  @property
  def is_generate_per_split(self):
    """Whether we have a train/test split or just hold out data."""
    return False  # Just hold out some generated data for evals.

  @property
  def frame_height(self):
    return 64

  @property
  def frame_width(self):
    return 64

  @property
  def total_number_of_frames(self):
    # 10k videos
    return 10000 * self.video_length

  @property
  def video_length(self):
    return 5

  @property
  def random_skip(self):
    return False

  @property
  def only_keep_videos_from_0th_frame(self):
    return True

  @property
  def use_not_breaking_batching(self):
    return True

  def eval_metrics(self):
    return []

  @property
  def extra_reading_spec(self):
    """Additional data fields to store on disk and their decoders."""
    data_fields = {
        "frame_number": tf.FixedLenFeature([1], tf.int64),
    }
    decoders = {
        "frame_number":
            contrib.slim().tfexample_decoder.Tensor(tensor_key="frame_number"),
    }
    return data_fields, decoders

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.modality = {
        "inputs": modalities.ModalityType.VIDEO,
        "targets": modalities.ModalityType.VIDEO,
    }
    p.vocab_size = {
        "inputs": 256,
        "targets": 256,
    }

  @staticmethod
  def get_circle(x, y, z, c, s):
    """Draws a circle with center(x, y), color c, size s and z-order of z."""
    cir = plt.Circle((x, y), s, fc=c, zorder=z)
    return cir

  @staticmethod
  def get_rectangle(x, y, z, c, s):
    """Draws a rectangle with center(x, y), color c, size s and z-order of z."""
    rec = plt.Rectangle((x-s, y-s), s*2.0, s*2.0, fc=c, zorder=z)
    return rec

  @staticmethod
  def get_triangle(x, y, z, c, s):
    """Draws a triangle with center (x, y), color c, size s and z-order of z."""
    points = np.array([[0, 0], [s, s*math.sqrt(3.0)], [s*2.0, 0]])
    tri = plt.Polygon(points + [x-s, y-s], fc=c, zorder=z)
    return tri

  def generate_stochastic_shape_instance(self):
    """Yields one video of a shape moving to a random direction.

       The size and color of the shapes are random but
       consistent in a single video. The speed is fixed.

    Raises:
       ValueError: The frame size is not square.
    """
    if self.frame_height != self.frame_width or self.frame_height % 2 != 0:
      raise ValueError("Generator only supports square frames with even size.")

    lim = 10.0
    direction = np.array([[+1.0, +1.0],
                          [+1.0, +0.0],
                          [+1.0, -1.0],
                          [+0.0, +1.0],
                          [+0.0, -1.0],
                          [-1.0, +1.0],
                          [-1.0, +0.0],
                          [-1.0, -1.0]
                         ])

    sp = np.array([lim/2.0, lim/2.0])
    rnd = np.random.randint(len(direction))
    di = direction[rnd]

    colors = ["b", "g", "r", "c", "m", "y"]
    color = np.random.choice(colors)

    shape = np.random.choice([
        VideoStochasticShapes10k.get_circle,
        VideoStochasticShapes10k.get_rectangle,
        VideoStochasticShapes10k.get_triangle])
    speed = 1.0

    size = np.random.uniform(0.5, 1.5)

    back_color = str(0.0)
    plt.ioff()

    xy = np.array(sp)

    for _ in range(self.video_length):
      fig = plt.figure()
      fig.set_dpi(self.frame_height//2)
      fig.set_size_inches(2, 2)
      ax = plt.axes(xlim=(0, lim), ylim=(0, lim))

      # Background
      ax.add_patch(VideoStochasticShapes10k.get_rectangle(
          0.0, 0.0, -1.0, back_color, 25.0))
      # Foreground
      ax.add_patch(shape(xy[0], xy[1], 0.0, color, size))

      plt.axis("off")
      plt.tight_layout(pad=-2.0)
      fig.canvas.draw()
      image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
      image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
      image = np.copy(np.uint8(image))

      plt.close()
      xy += speed * di

      yield image

  def generate_samples(self, data_dir, tmp_dir, unused_dataset_split):
    counter = 0
    done = False
    while not done:
      for frame_number, frame in enumerate(
          self.generate_stochastic_shape_instance()):
        if counter >= self.total_number_of_frames:
          done = True
          break

        yield {"frame": frame, "frame_number": [frame_number]}
        counter += 1
