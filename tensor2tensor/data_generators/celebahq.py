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

"""CelebA-HQ."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


@registry.register_problem
class ImageCelebahq128(image_utils.ImageProblem):
  """CelebA-HQ dataset, downsampled as 128x128."""

  def dataset_filename(self):
    return "image_celebahq-128"

  def example_reading_spec(self):
    data_fields = {
        "image/encoded": tf.FixedLenFeature((), tf.string),
        "image/format": tf.FixedLenFeature((), tf.string, default_value="png"),
    }
    _, data_items_to_decoders = super(
        ImageCelebahq128, self).example_reading_spec()
    return data_fields, data_items_to_decoders

  def filepattern(self, data_dir, mode, shard=None):
    """Get filepattern for data files for mode.

    Args:
      data_dir: str, data directory.
      mode: DatasetSplit
      shard: int, if provided, will only read data from the specified shard.

    Returns:
      filepattern str
    """
    path = os.path.join(data_dir, self.dataset_filename())
    if shard is not None:
      shard_str = "%05d" % shard
    elif mode == problem.DatasetSplit.TRAIN:
      # Use the first 90 shards.
      shard_str = "000[0-8]"
    else:
      assert mode in [problem.DatasetSplit.EVAL,
                      tf.estimator.ModeKeys.PREDICT,
                      problem.DatasetSplit.TEST]
      # Use the last 10 shards.
      shard_str = "0009"

    return "%s-%s*" % (path, shard_str)

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    raise NotImplementedError("Data preprocessing for CelebA-HQ is not "
                              "currently available. Please follow the steps "
                              "in https://github.com/tkarras/progressive_growin"
                              "g_of_gans.")

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.batch_size_multiplier = 1
    p.modality = {"inputs": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": 256}
    p.input_space_id = 1

  def preprocess_example(self, example, mode, hparams):
    del mode, hparams  # unused
    example["inputs"].set_shape((128, 128, 3))
    return example


@registry.register_problem
class ImageCelebahq128Dmol(ImageCelebahq128):
  """CelebA-HQ dataset with discretized mixture of logistics for evaluation."""

  def eval_metrics(self):
    return [
        metrics.Metrics.DMOL_PERPLEXITY
    ]


@registry.register_problem
class ImageCelebahq256(ImageCelebahq128):
  """CelebA-HQ dataset, downsampled as 256x256."""

  def dataset_filename(self):
    return "image_celebahq-256"

  def preprocess_example(self, example, mode, hparams):
    del mode, hparams  # unused
    example["inputs"].set_shape((256, 256, 3))
    return example


@registry.register_problem
class ImageCelebahq256Dmol(ImageCelebahq256):
  """CelebA-HQ dataset with discretized mixture of logistics for evaluation."""

  def eval_metrics(self):
    return [
        metrics.Metrics.DMOL_PERPLEXITY
    ]
