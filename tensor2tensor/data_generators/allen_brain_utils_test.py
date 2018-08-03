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

"""Tests of utilities supporting Allen Brain Atlas problems."""

import os

import tensorflow as tf

from tensor2tensor.data_generators.allen_brain_utils import mock_raw_data
from tensor2tensor.data_generators.allen_brain_utils import mock_raw_image
from tensor2tensor.data_generators.allen_brain_utils import TemporaryDirectory


class TestTemporaryDirectory(tf.test.TestCase):
  """Tests of py2/py3 tmpdir context pattern compatibility class."""

  def test_makes_tmpdir(self):
    """Test that a tmpdir is created."""
    with TemporaryDirectory() as tmp_dir:

      # Within the temporary context the tmpdir has been created
      self.assertTrue(tf.gfile.Exists(tmp_dir))

    # The tmpdir no longer exists outside of the temporary context
    self.assertFalse(tf.gfile.Exists(tmp_dir))


class TestImageMock(tf.test.TestCase):
  """Tests of image mocking utility."""

  def test_image_mock_produces_expected_shape(self):
    """Test that the image mocking utility produces expected shape output."""

    with TemporaryDirectory() as tmp_dir:

      cases = [
          {
              "x_dim": 8,
              "y_dim": 8,
              "num_channels": 3,
              "output_path": "/foo",
              "write_image": True
          }
      ]

      for cid, case in enumerate(cases):
        output_path = os.path.join(tmp_dir, "dummy%s.jpg" % cid)
        img = mock_raw_image(x_dim=case["x_dim"],
                             y_dim=case["y_dim"],
                             num_channels=case["num_channels"],
                             output_path=output_path,
                             write_image=case["write_image"])

        self.assertEqual(img.shape, (case["x_dim"], case["y_dim"],
                                     case["num_channels"]))
        if case["write_image"]:
          self.assertTrue(tf.gfile.Exists(output_path))


class TestMockRawData(tf.test.TestCase):
  """Tests of raw data mocking utility."""

  def test_runs(self):
    """Test that data mocking utility runs for cases expected to succeed."""

    with TemporaryDirectory() as tmp_dir:

      mock_raw_data(tmp_dir, raw_dim=256, num_channels=3, num_images=40)


if __name__ == "__main__":
  tf.test.main()
