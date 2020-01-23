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

"""CelebA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import zipfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


@registry.register_problem
class ImageCeleba(image_utils.ImageProblem):
  """CelebA dataset, aligned and cropped images."""
  IMG_DATA = ("img_align_celeba.zip",
              "https://drive.google.com/uc?export=download&"
              "id=0B7EVK8r0v71pZjFTYXZWM3FlRnM")
  LANDMARKS_DATA = ("celeba_landmarks_align",
                    "https://drive.google.com/uc?export=download&"
                    "id=0B7EVK8r0v71pd0FJY3Blby1HUTQ")
  ATTR_DATA = ("celeba_attr", "https://drive.google.com/uc?export=download&"
               "id=0B7EVK8r0v71pblRyaVFSWGxPY0U")

  LANDMARK_HEADINGS = ("lefteye_x lefteye_y righteye_x righteye_y "
                       "nose_x nose_y leftmouth_x leftmouth_y rightmouth_x "
                       "rightmouth_y").split()
  ATTR_HEADINGS = (
      "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs "
      "Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair "
      "Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair "
      "Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache "
      "Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline "
      "Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings "
      "Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
  ).split()

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": 256,
                    "targets": 256}
    p.batch_size_multiplier = 256
    p.input_space_id = 1
    p.target_space_id = 1

  def generator(self, tmp_dir, how_many, start_from=0):
    """Image generator for CELEBA dataset.

    Args:
      tmp_dir: path to temporary storage directory.
      how_many: how many images and labels to generate.
      start_from: from which image to start.

    Yields:
      A dictionary representing the images with the following fields:
      * image/encoded: the string encoding the image as JPEG,
      * image/format: the string "jpeg" representing image format,
    """
    out_paths = []
    for fname, url in [self.IMG_DATA, self.LANDMARKS_DATA, self.ATTR_DATA]:
      path = generator_utils.maybe_download_from_drive(tmp_dir, fname, url)
      out_paths.append(path)

    img_path, landmarks_path, attr_path = out_paths  # pylint: disable=unbalanced-tuple-unpacking
    unzipped_folder = img_path[:-4]
    if not tf.gfile.Exists(unzipped_folder):
      zipfile.ZipFile(img_path, "r").extractall(tmp_dir)

    with tf.gfile.Open(landmarks_path) as f:
      landmarks_raw = f.read()

    with tf.gfile.Open(attr_path) as f:
      attr_raw = f.read()

    def process_landmarks(raw_data):
      landmarks = {}
      lines = raw_data.split("\n")
      headings = lines[1].strip().split()
      for line in lines[2:-1]:
        values = line.strip().split()
        img_name = values[0]
        landmark_values = [int(v) for v in values[1:]]
        landmarks[img_name] = landmark_values
      return landmarks, headings

    def process_attrs(raw_data):
      attrs = {}
      lines = raw_data.split("\n")
      headings = lines[1].strip().split()
      for line in lines[2:-1]:
        values = line.strip().split()
        img_name = values[0]
        attr_values = [int(v) for v in values[1:]]
        attrs[img_name] = attr_values
      return attrs, headings

    img_landmarks, _ = process_landmarks(landmarks_raw)
    img_attrs, _ = process_attrs(attr_raw)

    image_files = list(sorted(tf.gfile.Glob(unzipped_folder + "/*.jpg")))
    for filename in image_files[start_from:start_from + how_many]:
      img_name = os.path.basename(filename)
      landmarks = img_landmarks[img_name]
      attrs = img_attrs[img_name]

      with tf.gfile.Open(filename, "rb") as f:
        encoded_image_data = f.read()
        yield {
            "image/encoded": [encoded_image_data],
            "image/format": ["jpeg"],
            "attributes": attrs,
            "landmarks": landmarks,
        }

  @property
  def train_shards(self):
    return 100

  @property
  def dev_shards(self):
    return 10

  @property
  def test_shards(self):
    return 10

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    train_gen = self.generator(tmp_dir, 162770)
    train_paths = self.training_filepaths(
        data_dir, self.train_shards, shuffled=False)
    generator_utils.generate_files(train_gen, train_paths)

    dev_gen = self.generator(tmp_dir, 19867, 162770)
    dev_paths = self.dev_filepaths(data_dir, self.dev_shards, shuffled=False)
    generator_utils.generate_files(dev_gen, dev_paths)

    test_gen = self.generator(tmp_dir, 19962, 162770+19867)
    test_paths = self.test_filepaths(data_dir, self.test_shards, shuffled=False)
    generator_utils.generate_files(test_gen, test_paths)

    generator_utils.shuffle_dataset(train_paths + dev_paths + test_paths)


@registry.register_problem
class ImageCelebaMultiResolution(ImageCeleba):
  """CelebA at multiple resolutions.

  The resolutions are specified as a hyperparameter during preprocessing.
  """

  def dataset_filename(self):
    return "image_celeba"

  def preprocess_example(self, example, mode, hparams):
    image = example["inputs"]
    # Get resize method. Include a default if not specified, or if it's not in
    # TensorFlow's collection of pre-implemented resize methods.
    resize_method = getattr(hparams, "resize_method", "BICUBIC")
    resize_method = getattr(tf.image.ResizeMethod, resize_method, resize_method)

    # Remove boundaries in CelebA images. Remove 40 pixels each side
    # vertically and 20 pixels each side horizontally.
    image = tf.image.crop_to_bounding_box(image, 40, 20, 218 - 80, 178 - 40)

    highest_res = hparams.resolutions[-1]
    if resize_method == "DILATED":
      # Resize image so that dilated subsampling is properly divisible.
      scaled_image = image_utils.resize_by_area(image, highest_res)
      scaled_images = image_utils.make_multiscale_dilated(
          scaled_image, hparams.resolutions, num_channels=self.num_channels)
    else:
      scaled_images = image_utils.make_multiscale(
          image, hparams.resolutions,
          resize_method=resize_method, num_channels=self.num_channels)

    # Pack tuple of scaled images into one tensor. We do this by enforcing the
    # columns to match for every resolution.
    example["inputs"] = image
    example["targets"] = tf.concat([
        tf.reshape(scaled_image,
                   [res**2 // highest_res, highest_res, self.num_channels])
        for scaled_image, res in zip(scaled_images, hparams.resolutions)],
                                   axis=0)
    return example


@registry.register_problem
class Img2imgCeleba(ImageCeleba):
  """8px to 32px problem."""

  def dataset_filename(self):
    return "image_celeba"

  def preprocess_example(self, example, unused_mode, unused_hparams):
    image = example["inputs"]
    # Remove boundaries in CelebA images. Remove 40 pixels each side
    # vertically and 20 pixels each side horizontally.
    image = tf.image.crop_to_bounding_box(image, 40, 20, 218 - 80, 178 - 40)
    image_8 = image_utils.resize_by_area(image, 8)
    image_32 = image_utils.resize_by_area(image, 32)

    example["inputs"] = image_8
    example["targets"] = image_32
    return example


@registry.register_problem
class Img2imgCeleba64(Img2imgCeleba):
  """8px to 64px problem."""

  def preprocess_example(self, example, unused_mode, unused_hparams):
    image = example["inputs"]
    # Remove boundaries in CelebA images. Remove 40 pixels each side
    # vertically and 20 pixels each side horizontally.
    image = tf.image.crop_to_bounding_box(image, 40, 20, 218 - 80, 178 - 40)
    image_8 = image_utils.resize_by_area(image, 8)
    image_64 = image_utils.resize_by_area(image, 64)

    example["inputs"] = image_8
    example["targets"] = image_64
    return example


@registry.register_problem
class ImageCeleba32(Img2imgCeleba):
  """CelebA resized to spatial dims [32, 32]."""

  def preprocess_example(self, example, unused_mode, unused_hparams):
    image = example["inputs"]
    # Remove boundaries in CelebA images. Remove 40 pixels each side
    # vertically and 20 pixels each side horizontally.
    image = tf.image.crop_to_bounding_box(image, 40, 20, 218 - 80, 178 - 40)
    image = image_utils.resize_by_area(image, 32)

    example["inputs"] = image
    example["targets"] = image
    return example


@registry.register_problem
class ImageCeleba64(Img2imgCeleba):
  """CelebA resized to spatial dims [64, 64]."""

  def preprocess_example(self, example, unused_mode, unused_hparams):
    image = example["inputs"]
    # Remove boundaries in CelebA images. Remove 40 pixels each side
    # vertically and 20 pixels each side horizontally.
    image = tf.image.crop_to_bounding_box(image, 40, 20, 218 - 80, 178 - 40)
    image = image_utils.resize_by_area(image, 64)

    example["inputs"] = image
    example["targets"] = image
    return example


