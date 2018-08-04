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
"""LSUN datasets (bedrooms only for now)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import zipfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.utils import registry
import tensorflow as tf


_LSUN_URL = "http://lsun.cs.princeton.edu/htbin/download.cgi?tag=latest&category=%s&set=%s"
_LSUN_DATA_FILENAME = "lsun-%s-%s.zip"


def pil_image():
  import PIL  # pylint: disable=g-import-not-at-top
  return PIL.Image


def _get_lsun(directory, category, split_name):
  """Downloads all lsun files to directory unless they are there."""
  generator_utils.maybe_download(directory,
                                 _LSUN_DATA_FILENAME % (category, split_name),
                                 _LSUN_URL % (category, split_name))


@registry.register_problem
class ImageLsunBedrooms(image_utils.ImageProblem):
  """LSUN Bedrooms."""

  @property
  def num_channels(self):
    """Number of color channels."""
    return 3

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    """Generates LSUN bedrooms dataset and writes it in data_dir."""
    generator_utils.generate_dataset_and_shuffle(
        self.read_and_convert_to_png(tmp_dir, "train"),
        self.training_filepaths(data_dir, 100, shuffled=False),
        self.read_and_convert_to_png(tmp_dir, "val"),
        self.dev_filepaths(data_dir, 1, shuffled=False))

  def read_and_convert_to_png(self, tmp_dir, split_name):
    """Downloads the datasets, extracts from zip and yields in PNG format."""
    category = "bedroom"
    _get_lsun(tmp_dir, category, split_name)
    filename = _LSUN_DATA_FILENAME % (category, split_name)
    data_path = os.path.join(tmp_dir, filename)
    print("Extracting zip file.")
    zip_ref = zipfile.ZipFile(data_path, "r")
    zip_ref.extractall(tmp_dir)
    zip_ref.close()

    print("Opening database.")
    data_file = os.path.join(tmp_dir,
                             "%s_%s_lmdb/data.mdb" % (category, split_name))

    filename_queue = tf.train.string_input_producer([data_file], num_epochs=1)
    reader = tf.LMDBReader()
    _, webp_image_tensor = reader.read(filename_queue)

    object_count = 0
    with tf.train.MonitoredTrainingSession() as session:
      while True:
        webp_image = session.run(webp_image_tensor)
        object_count += 1
        if object_count % 1000 == 0:
          print("Extracted %d objects." % object_count)
        # Unfortunately Tensorflow doesn't support reading or parsing
        # WebP images, so we have to do it via Image PIL library.
        image = pil_image().open(io.BytesIO(webp_image))
        buf = io.BytesIO()
        width, height = image.size
        image.save(buf, "PNG")
        yield {
            "image/encoded": [buf.getvalue()],
            "image/format": ["png"],
            "image/class/label": [0],
            "image/height": [height],
            "image/width": [width]
        }
