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

"""FSNS."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import modalities
from tensor2tensor.utils import contrib
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


@registry.register_problem
class ImageFSNS(image_utils.ImageProblem):
  """Problem spec for French Street Name recognition."""

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    list_url = ("https://raw.githubusercontent.com/tensorflow/models/master/"
                "street/python/fsns_urls.txt")
    fsns_urls = generator_utils.maybe_download(tmp_dir, "fsns_urls.txt",
                                               list_url)
    fsns_files = [
        f.strip() for f in open(fsns_urls, "r") if f.startswith("http://")
    ]
    for url in fsns_files:
      if "/train/train" in url:
        generator_utils.maybe_download(
            data_dir, "image_fsns-train" + url[-len("-00100-of-00512"):], url)
      elif "/validation/validation" in url:
        generator_utils.maybe_download(
            data_dir, "image_fsns-dev" + url[-len("-00100-of-00512"):], url)
      elif "charset" in url:
        generator_utils.maybe_download(data_dir, "charset_size134.txt", url)

  def feature_encoders(self, data_dir):
    # This vocab file must be present within the data directory.
    vocab_filename = os.path.join(data_dir, "charset_size134.txt")
    return {
        "inputs": text_encoder.ImageEncoder(),
        "targets": text_encoder.SubwordTextEncoder(vocab_filename)
    }

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.modality = {"inputs": modalities.ModalityType.IMAGE,
                  "targets": modalities.ModalityType.SYMBOL}
    p.vocab_size = {"inputs": 256,
                    "targets": self._encoders["targets"].vocab_size}
    p.batch_size_multiplier = 256
    p.input_space_id = problem.SpaceID.IMAGE
    p.target_space_id = problem.SpaceID.EN_TOK

  def example_reading_spec(self):
    label_key = "image/unpadded_label"
    data_fields, data_items_to_decoders = (
        super(ImageFSNS, self).example_reading_spec())
    data_fields[label_key] = tf.VarLenFeature(tf.int64)
    data_items_to_decoders["targets"] = contrib.slim().tfexample_decoder.Tensor(
        label_key)
    return data_fields, data_items_to_decoders
