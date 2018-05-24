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

"""Tests of data download and subimage generation utilities."""

import logging
import os
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from tensor2tensor.data_generators import allen_downloader


def mock_raw_image(output_path, dim=1024):
  """Generate random `dim`x`dim` image to `output_path`.

  Args:
    output_path (str): Path to which to write image.
    raw_dim (int): The x and y dimension of generated raw imgs.

  """

  with open(output_path, "w") as f:
    img = np.random.random((dim, dim))
    img = np.float32(img)
    img = Image.fromarray(img)
    img = img.convert("RGB")
    img.save(f, "jpeg")


def mock_raw_data(tmp_dir, raw_dim=1024):
  """Mock a raw data download directory with meta and raw subdirs.

  E.g.
    {data_root}/
      meta/
      raw/
        dataset_id/
          image_id/
            raw_{image_id}.jpg [random image]

  Args:
    tmp_dir (str): Temporary dir in which to mock data.
    raw_dim (int): The x and y dimension of generated raw imgs.

  Returns:
    tmp_dir (str): Path to root of generated data dir.

  """

  meta = os.path.join(tmp_dir, "meta")
  raw = os.path.join(tmp_dir, "raw")
  os.mkdir(meta)
  os.mkdir(raw)
  mock_dataset_id = "1234"
  mock_image_id = "4321"
  dataset = os.path.join(raw, mock_dataset_id)
  os.mkdir(dataset)
  image_dir = os.path.join(dataset, mock_image_id)
  os.mkdir(image_dir)
  raw_image_path = os.path.join(image_dir, "raw_%s.jpg" % mock_image_id)

  mock_raw_image(raw_image_path, raw_dim)


class TemporaryDirectory(object):
  """For py2 support of `with tempfile.TemporaryDirectory() as name:`"""

  def __enter__(self):
    self.name = tempfile.mkdtemp()
    return self.name

  def __exit__(self, exc_type, exc_value, traceback):
    shutil.rmtree(self.name)


class TestDownloader(unittest.TestCase):

  def test_runs(self):
    """Test that maybe_download_image_datasets runs."""

    # By default, such as on the CI, don't run this test since tests that
    # access the network aren't allowed. But retain the test as a
    # development tool.
    skip = True
    if not skip:
      with TemporaryDirectory() as tmp_dir:
        num_sections = 1
        img_per_section = 1
        allen_downloader.maybe_download_image_datasets(
            data_root=tmp_dir, section_offset=0,
            num_sections=num_sections, images_per_section=img_per_section)


class TestSubimageGenerator(unittest.TestCase):

  def test_generates_expected_output_files(self):
    """Test that we can run the downloader and subimage steps e2e."""

    with TemporaryDirectory() as tmp_dir:

      mock_raw_data(tmp_dir, 2048)
      image_dir = os.path.join(tmp_dir, "raw", "1234", "4321")
      self.assertTrue(os.path.exists(image_dir))

      allen_downloader.subimages_for_image_files(tmp_dir)

      expected_fnames = [
          'raw_4321.jpg',
          '1024x1024_3_4321.jpg',
          '1024x1024_2_4321.jpg',
          '1024x1024_1_4321.jpg',
          '1024x1024_0_4321.jpg'
      ]

      self.assertEqual(os.listdir(image_dir), expected_fnames)


if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  unittest.main()
