# coding=utf-8
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils. for Allen Brain Atlas dataset, download and subimages."""

import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf


def try_importing_pil_image():
  """Import a PIL Image object if the function is called."""
  try:
    from PIL import Image
  except ImportError:
    tf.logging.error("Can't import Image from PIL (Pillow). Please install it, "
                     "such as by running `pip install Pillow`.")
    exit(1)

  return Image


def mock_raw_image(x_dim=1024, y_dim=1024, num_channels=3,
                   output_path=None, write_image=True):
  """Generate random `x_dim` by `y_dim`, optionally to `output_path`.

  Args:
    output_path: str, path to which to write image.
    x_dim: int, the x dimension of generated raw image.
    y_dim: int, the x dimension of generated raw image.
    return_raw_image: bool, whether to return the generated image (as a
      numpy array).

  Returns:
    numpy.array: The random `x_dim` by `y_dim` image (i.e. array).

  """

  rand_shape = (x_dim, y_dim, num_channels)
  tf.logging.debug(rand_shape)

  if num_channels != 3:
    raise NotImplementedError("mock_raw_image for channels != 3 not yet "
                              "implemented.")

  img = np.random.random(rand_shape)
  img = np.uint8(img*255)

  if write_image:
    if not isinstance(output_path, str):
      raise ValueError("Output path must be of type str if write_image=True, "
                       "saw %s." % output_path)

    image_obj = try_importing_pil_image()
    pil_img = image_obj.fromarray(img, mode="RGB")
    with tf.gfile.Open(output_path, "w") as f:
      pil_img.save(f, "jpeg")

  return img


def mock_raw_data(tmp_dir, raw_dim=1024, num_channels=3, num_images=1):
  """Mock a raw data download directory with meta and raw subdirs.

  Notes:

    * This utility is shared by tests in both allen_brain_utils and
      allen_brain so kept here instead of in one of *_test.

  Args:
    tmp_dir: str, temporary dir in which to mock data.
    raw_dim: int, the x and y dimension of generated raw imgs.

  """

  tf.gfile.MakeDirs(tmp_dir)

  for image_id in range(0, num_images):

    raw_image_path = os.path.join(tmp_dir, "%s.jpg" % image_id)

    mock_raw_image(x_dim=raw_dim, y_dim=raw_dim,
                   num_channels=num_channels,
                   output_path=raw_image_path)


class TemporaryDirectory(object):
  """For py2 support of `with tempfile.TemporaryDirectory() as name:`"""

  def __enter__(self):
    self.name = tempfile.mkdtemp()
    return self.name

  def __exit__(self, exc_type, exc_value, traceback):
    shutil.rmtree(self.name)
