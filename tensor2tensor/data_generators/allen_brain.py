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

"""Problem definitions for Allen Brain Atlas problems.

Notes:

  * TODO: Support in-painting of non-square regions.

  * TODO: Want to be able to increase up-sampling ratio and/or in-paint
    fraction over the course of training. This could be done by defining a
    range of problems or perhaps more aptly with an hparam that is dialed
    up depending on training performance.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
from io import BytesIO

from PIL import Image

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics

from tensor2tensor.data_generators.allen_brain_utils import maybe_download_image_datasets

import tensorflow as tf

_BASE_EXAMPLE_IMAGE_SIZE = 64


def _get_case_file_paths(tmp_dir, case, training_fraction=0.95):
  """Obtain a list of image paths corresponding to training or eval case.

  Args:
    tmp_dir (str): The root path to which raw images were written, at the
      top level having meta/ and raw/ subdirs.
    size (int): The size of sub-images to consider (`size`x`size`).
    case (int): 0 or 1, the former corresponding to eval, the latter to
      training.
    training_fraction (float): The fraction of the sub-image path list to
      consider as the basis for training examples.

  Returns:
    list: A list of file paths.

  """

  paths = tf.gfile.Glob("%s/raw/*/*/raw_*.jpg" % tmp_dir)

  tf.logging.debug(paths)

  if not paths:
    raise ValueError("Search of tmp_dir (%s) " % tmp_dir,
                     "for subimage paths yielded an empty list, ",
                     "can't proceed with returning training/eval split.")

  split_index = int(math.floor(len(paths)*training_fraction))
  tf.logging.debug(split_index)

  if split_index >= len(paths):
    raise ValueError("For a path list of size %s " % len(paths),
                     "and a training_fraction of %s " % training_fraction,
                     "the resulting split_index of the paths list, ",
                     "%s, would leave no elements for the eval " % split_index,
                     "condition.")

  if case == 1:
    return paths[:split_index]
  else:
    return paths[split_index:]


def _generator(tmp_dir, training, size=_BASE_EXAMPLE_IMAGE_SIZE,
               training_fraction=0.95):
  """Base problem example generator for Allen Brain Atlas problems.

  Args:

    tmp_dir (str): A directory where raw example input data has been stored.
    training (bool): Whether the mode of operation is training (or,
      alternatively, evaluation), determining whether examples in tmp_dir
      prefixed with train or dev will be used.
    size (int): The image size to add to the example annotation.

  Yields:
    A dictionary representing the images with the following fields:
      * image/encoded: The string encoding the image as JPEG.
      * image/format: The string "jpeg" indicating the image format.
      * image/height: The integer indicating the image height.
      * image/width: The integer indicating the image height.

  """

  maybe_download_image_datasets(data_root=tmp_dir,
                                section_offset=0,
                                num_sections="all")

  image_files = _get_case_file_paths(tmp_dir=tmp_dir,
                                     case=training,
                                     training_fraction=training_fraction)

  tf.logging.info("Loaded case file paths (n=%s)" % len(image_files))
  height = size
  width = size
  for input_path in image_files:

    img = Image.open(input_path)
    img = np.float32(img)
    shape = np.shape(img)

    for h_index in range(0, int(math.floor(shape[0]/size))):

      h_offset = h_index * size
      h_end = h_offset + size - 1

      for v_index in range(0, int(math.floor(shape[1]/size))):

        v_offset = v_index * size
        v_end = v_offset + size - 1

        # Extract a sub-image tile and convert to float in range
        # [0, 1-ish].
        # pylint: disable=invalid-sequence-index
        std_sub = img[h_offset:h_end, v_offset:v_end]/255.0

        # Clip the ish, convert from [0,1] to [0, 255], then to
        # uint8 type.
        subimage = np.uint8(np.clip(std_sub, 0, 1)*255)

        # TODO: I'm guessing there is a better way to do this.
        subimage = Image.fromarray(subimage)
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
class AllenBrainImage2image(problem.Problem):
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
    example["targets"] = image_utils.resize_by_area(example["targets"],
                                                    self.output_dim)
    example["inputs"] = image_utils.resize_by_area(example["targets"],
                                                   self.input_dim)

    if self.inpaint_fraction is not None and self.inpaint_fraction > 0:
      if self.input_dim is None:
        raise ValueError("Cannot train in-painting for examples with "
                         "only targets (i.e. input_dim is None, "
                         "implying there are only targets to be "
                         "generated).")
      raise NotImplementedError("In-painting is not yet supported.")

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
      return _generator(tmp_dir, int(True), size=_BASE_EXAMPLE_IMAGE_SIZE,
                        training_fraction=self.training_fraction)
    else:
      return _generator(tmp_dir, int(False), size=_BASE_EXAMPLE_IMAGE_SIZE,
                        training_fraction=self.training_fraction)


@registry.register_problem
class AllenBrainImage2imageDim48to64(AllenBrainImage2image):
  """48px to 64px resolution up-sampling problem.

  Notes:

    * 1.25x resolution up-sampling to 64px target.

    * See AllenBrainImage2image for more details.

  """

  @property
  def input_dim(self):
    return 48

  @property
  def output_dim(self):
    return 64
