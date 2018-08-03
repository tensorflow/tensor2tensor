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
"""Problem definitions for Allen Brain Atlas problems.

Notes:

  * TODO(cwbeitel): Want to be able to increase up-sampling ratio and/or
    in-paint fraction over the course of training. This could be done by
    defining a range of problems or perhaps more aptly with an hparam
    that is dialed up depending on training performance.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from io import BytesIO
import math
import os

import numpy as np
import requests

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf

_BASE_EXAMPLE_IMAGE_SIZE = 64


# A 100 image random subset of non-failed acquisitions of Mouse imaging
# products from Allen Brain Institute (api.brain-map.org) dataset. The
# full set (or a desired subset) of image IDs can be obtained following
# the steps described here: http://help.brain-map.org/display/api,
# e.g. https://gist.github.com/cwbeitel/5dffe90eb561637e35cdf6aa4ee3e704
_IMAGE_IDS = [
    "74887117", "71894997", "69443979", "79853548", "101371232", "77857182",
    "70446772", "68994990", "69141561", "70942310", "70942316", "68298378",
    "69690156", "74364867", "77874134", "75925043", "73854431", "69206601",
    "71771457", "101311379", "74777533", "70960269", "71604493", "102216720",
    "74776437", "75488723", "79815814", "77857132", "77857138", "74952778",
    "69068486", "648167", "75703410", "74486118", "77857098", "637407",
    "67849516", "69785503", "71547630", "69068504", "69184074", "74853078",
    "74890694", "74890698", "75488687", "71138602", "71652378", "68079764",
    "70619061", "68280153", "73527042", "69764608", "68399025", "244297",
    "69902658", "68234159", "71495521", "74488395", "73923026", "68280155",
    "75488747", "69589140", "71342189", "75119214", "79455452", "71774294",
    "74364957", "68031779", "71389422", "67937572", "69912671", "73854471",
    "75008183", "101371376", "75703290", "69533924", "79853544", "77343882",
    "74887133", "332587", "69758622", "69618413", "77929999", "244293",
    "334792", "75825136", "75008103", "70196678", "71883965", "74486130",
    "74693566", "76107119", "76043858", "70252433", "68928364", "74806345",
    "67848661", "75900326", "71773690", "75008171"]


def PIL_Image():  # pylint: disable=invalid-name
  from PIL import Image  # pylint: disable=g-import-not-at-top
  return Image


def _get_case_file_paths(tmp_dir, case, training_fraction=0.95):
  """Obtain a list of image paths corresponding to training or eval case.

  Args:
    tmp_dir: str, the root path to which raw images were written, at the
      top level having meta/ and raw/ subdirs.
    case: bool, whether obtaining file paths for training (true) or eval
      (false).
    training_fraction: float, the fraction of the sub-image path list to
      consider as the basis for training examples.

  Returns:
    list: A list of file paths.

  Raises:
    ValueError: if images not found in tmp_dir, or if training_fraction would
      leave no examples for eval.
  """

  paths = tf.gfile.Glob("%s/*.jpg" % tmp_dir)

  if not paths:
    raise ValueError("Search of tmp_dir (%s) " % tmp_dir,
                     "for subimage paths yielded an empty list, ",
                     "can't proceed with returning training/eval split.")

  split_index = int(math.floor(len(paths)*training_fraction))

  if split_index >= len(paths):
    raise ValueError("For a path list of size %s "
                     "and a training_fraction of %s "
                     "the resulting split_index of the paths list, "
                     "%s, would leave no elements for the eval "
                     "condition." % (len(paths),
                                     training_fraction,
                                     split_index))

  if case:
    return paths[:split_index]
  else:
    return paths[split_index:]


def maybe_download_image_dataset(image_ids, target_dir):
  """Download a set of images from api.brain-map.org to `target_dir`.

  Args:
    image_ids: list, a list of image ids.
    target_dir: str, a directory to which to download the images.
  """

  tf.gfile.MakeDirs(target_dir)

  num_images = len(image_ids)

  for i, image_id in enumerate(image_ids):

    destination = os.path.join(target_dir, "%s.jpg" % i)
    tmp_destination = "%s.temp" % destination

    source_url = ("http://api.brain-map.org/api/v2/"
                  "section_image_download/%s" % image_id)

    if tf.gfile.Exists(destination):
      tf.logging.info("Image with ID already present, "
                      "skipping download (%s of %s)." % (
                          i+1, num_images
                      ))
      continue

    tf.logging.info("Downloading image with id %s (%s of %s)" % (
        image_id, i+1, num_images
    ))

    response = requests.get(source_url, stream=True)

    response.raise_for_status()

    with tf.gfile.Open(tmp_destination, "w") as f:
      for block in response.iter_content(1024):
        f.write(block)

    tf.gfile.Rename(tmp_destination, destination)


def random_square_mask(shape, fraction):
  """Create a numpy array with specified shape and masked fraction.

  Args:
    shape: tuple, shape of the mask to create.
    fraction: float, fraction of the mask area to populate with `mask_scalar`.

  Returns:
    numpy.array: A numpy array storing the mask.
  """

  mask = np.ones(shape)

  patch_area = shape[0]*shape[1]*fraction
  patch_dim = np.int(math.floor(math.sqrt(patch_area)))
  if patch_area == 0 or patch_dim == 0:
    return mask

  x = np.random.randint(shape[0] - patch_dim)
  y = np.random.randint(shape[1] - patch_dim)

  mask[x:(x + patch_dim), y:(y + patch_dim), :] = 0

  return mask


def _generator(tmp_dir, training, size=_BASE_EXAMPLE_IMAGE_SIZE,
               training_fraction=0.95):
  """Base problem example generator for Allen Brain Atlas problems.

  Args:

    tmp_dir: str, a directory where raw example input data has been stored.
    training: bool, whether the mode of operation is training (or,
      alternatively, evaluation), determining whether examples in tmp_dir
      prefixed with train or dev will be used.
    size: int, the image size to add to the example annotation.
    training_fraction: float, the fraction of the sub-image path list to
      consider as the basis for training examples.

  Yields:
    A dictionary representing the images with the following fields:
      * image/encoded: The string encoding the image as JPEG.
      * image/format: The string "jpeg" indicating the image format.
      * image/height: The integer indicating the image height.
      * image/width: The integer indicating the image height.

  """

  maybe_download_image_dataset(_IMAGE_IDS, tmp_dir)

  image_files = _get_case_file_paths(tmp_dir=tmp_dir,
                                     case=training,
                                     training_fraction=training_fraction)

  image_obj = PIL_Image()

  tf.logging.info("Loaded case file paths (n=%s)" % len(image_files))
  height = size
  width = size

  for input_path in image_files:

    img = image_obj.open(input_path)
    img = np.float32(img)
    shape = np.shape(img)

    for h_index in range(0, int(math.floor(shape[0]/size))):

      h_offset = h_index * size
      h_end = h_offset + size - 1

      for v_index in range(0, int(math.floor(shape[1]/size))):

        v_offset = v_index * size
        v_end = v_offset + size - 1

        # Extract a sub-image tile.
        subimage = np.uint8(img[h_offset:h_end, v_offset:v_end])  # pylint: disable=invalid-sequence-index

        # Filter images that are likely background (not tissue).
        if np.amax(subimage) < 230:
          continue

        subimage = image_obj.fromarray(subimage)
        buff = BytesIO()
        subimage.save(buff, format="JPEG")
        subimage_encoded = buff.getvalue()

        yield {
            "image/encoded": [subimage_encoded],
            "image/format": ["jpeg"],
            "image/height": [height],
            "image/width": [width]
        }


@registry.register_problem
class Img2imgAllenBrain(problem.Problem):
  """Allen Brain Atlas histology dataset.

  See also: http://help.brain-map.org/

  Notes:

    * 64px to 64px identity mapping problem, no in-painting.

  """

  @property
  def train_shards(self):
    return 100

  @property
  def dev_shards(self):
    return 10

  @property
  def training_fraction(self):
    return 0.95

  @property
  def num_channels(self):
    """Number of color channels."""
    return 3

  @property
  def input_dim(self):
    """The x and y dimension of the input image."""
    # By default, there is no input image, only a target.
    return 64

  @property
  def output_dim(self):
    """The x and y dimension of the target image."""
    return 64

  @property
  def inpaint_fraction(self):
    """The fraction of the input image to be in-painted."""
    # By default, no in-painting is performed.
    return None

  def preprocess_example(self, example, mode, hparams):

    # Crop to target shape instead of down-sampling target, leaving target
    # of maximum available resolution.
    target_shape = (self.output_dim, self.output_dim, self.num_channels)
    example["targets"] = tf.random_crop(example["targets"], target_shape)

    example["inputs"] = image_utils.resize_by_area(example["targets"],
                                                   self.input_dim)

    if self.inpaint_fraction is not None and self.inpaint_fraction > 0:

      mask = random_square_mask((self.input_dim,
                                 self.input_dim,
                                 self.num_channels),
                                self.inpaint_fraction)

      example["inputs"] = tf.multiply(
          tf.convert_to_tensor(mask, dtype=tf.int64),
          example["inputs"])

      if self.input_dim is None:
        raise ValueError("Cannot train in-painting for examples with "
                         "only targets (i.e. input_dim is None, "
                         "implying there are only targets to be "
                         "generated).")

    return example

  def feature_encoders(self, data_dir):
    del data_dir
    return {
        "inputs": text_encoder.ImageEncoder(channels=self.num_channels),
        "targets": text_encoder.ImageEncoder(channels=self.num_channels)
    }

  def example_reading_spec(self):
    data_fields = {
        "image/encoded": tf.FixedLenFeature((), tf.string),
        "image/format": tf.FixedLenFeature((), tf.string),
    }

    data_items_to_decoders = {
        "targets":
            tf.contrib.slim.tfexample_decoder.Image(
                image_key="image/encoded",
                format_key="image/format",
                channels=self.num_channels),
    }

    return data_fields, data_items_to_decoders

  def eval_metrics(self):
    eval_metrics = [
        metrics.Metrics.ACC,
        metrics.Metrics.ACC_PER_SEQ,
        metrics.Metrics.NEG_LOG_PERPLEXITY
    ]
    return eval_metrics

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    generator_utils.generate_dataset_and_shuffle(
        self.generator(tmp_dir, True),
        self.training_filepaths(data_dir, self.train_shards, shuffled=True),
        self.generator(tmp_dir, False),
        self.dev_filepaths(data_dir, self.dev_shards, shuffled=True))

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.input_modality = {"inputs": ("image:identity", 256)}
    p.target_modality = ("image:identity", 256)
    p.batch_size_multiplier = 256
    p.input_space_id = problem.SpaceID.IMAGE
    p.target_space_id = problem.SpaceID.IMAGE

  def generator(self, tmp_dir, is_training):
    if is_training:
      return _generator(tmp_dir, True, size=_BASE_EXAMPLE_IMAGE_SIZE,
                        training_fraction=self.training_fraction)
    else:
      return _generator(tmp_dir, False, size=_BASE_EXAMPLE_IMAGE_SIZE,
                        training_fraction=self.training_fraction)


@registry.register_problem
class Img2imgAllenBrainDim48to64(Img2imgAllenBrain):
  """48px to 64px resolution up-sampling problem."""

  def dataset_filename(self):
    return "img2img_allen_brain"  # Reuse base problem data

  @property
  def input_dim(self):
    return 48

  @property
  def output_dim(self):
    return 64


@registry.register_problem
class Img2imgAllenBrainDim8to32(Img2imgAllenBrain):
  """8px to 32px resolution up-sampling problem."""

  def dataset_filename(self):
    return "img2img_allen_brain"  # Reuse base problem data

  @property
  def input_dim(self):
    return 8

  @property
  def output_dim(self):
    return 32


@registry.register_problem
class Img2imgAllenBrainDim16to16Paint1(Img2imgAllenBrain):
  """In-painting problem (1%) with no resolution upsampling."""

  def dataset_filename(self):
    return "img2img_allen_brain"  # Reuse base problem data

  @property
  def input_dim(self):
    return 16

  @property
  def output_dim(self):
    return 16

  @property
  def inpaint_fraction(self):
    return 0.01

