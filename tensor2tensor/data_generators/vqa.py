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
"""Data generators for VQA data sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import csv
import json
import os
import random
import sys
import tarfile
import zipfile
import numpy as np

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import vqa_utils
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf


def _get_vqa_v2_annotations(directory,
                            annotation_url,
                            annotation_filename="vqa_v2.tar.gz"):
  """Extract the VQA V2 annotation files to directory unless it's there."""
  annotation_file = generator_utils.maybe_download_from_drive(
      directory, annotation_filename, annotation_url)
  with tarfile.open(annotation_file, "r:gz") as annotation_tar:
    annotation_tar.extractall(directory)


def _get_vqa_v2_image_raw_dataset(directory, image_root_url, image_urls):
  """Extract the VQA V2 image data set to directory unless it's there."""
  for url in image_urls:
    filename = os.path.basename(url)
    download_url = os.path.join(image_root_url, url)
    path = generator_utils.maybe_download(directory, filename, download_url)
    unzip_dir = os.path.join(directory, filename.strip(".zip"))
    if not tf.gfile.Exists(unzip_dir):
      zipfile.ZipFile(path, "r").extractall(directory)


def _get_vqa_v2_image_feature_dataset(
    directory, feature_url, feature_filename="mscoco_feat.tar.gz"):
  """Extract the VQA V2 feature data set to directory unless it's there."""
  feature_file = generator_utils.maybe_download_from_drive(
      directory, feature_filename, feature_url)
  with tarfile.open(feature_file, "r:gz") as feature_tar:
    feature_tar.extractall(directory)


class ImageQuestion2MultilabelProblem(image_utils.ImageProblem):
  """Base class for image question answer problem."""

  @property
  def target_space_id(self):
    raise NotImplementedError()

  @property
  def vocab_size(self):
    raise NotImplementedError

  @property
  def num_classes(self):
    raise NotImplementedError()

  @property
  def vocab_filename(self):
    raise NotImplementedError()

  @property
  def label_filename(self):
    raise NotImplementedError()

  @property
  def train_shards(self):
    raise NotImplementedError()

  @property
  def dev_shards(self):
    raise NotImplementedError()

  def source_data_files(self, dataset_split):
    raise NotImplementedError()

  def generator(self, data_dir, tmp_dir, dataset_split):
    raise NotImplementedError()

  def eval_metrics(self):
    return [
        metrics.Metrics.ACC_MULTILABEL_MATCH3,
    ]

  def feature_encoders(self, data_dir):
    input_encoder = text_encoder.ImageEncoder(channels=self.num_channels)
    vocab_file = os.path.join(data_dir, self.vocab_filename)
    question_encoder = text_encoder.TokenTextEncoder(
        vocab_file, replace_oov="UNK")
    label_file = os.path.join(data_dir, self.label_filename)
    target_encoder = text_encoder.ClassLabelEncoder(
        class_labels_fname=label_file)
    return {"inputs": input_encoder,
            "question": question_encoder,
            "targets": target_encoder}

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    question_encoder = self._encoders["question"]
    targets_encoder = self._encoders["targets"]

    p.input_modality = {
        "inputs": (registry.Modalities.IMAGE + ":identity", None),
        "question": (registry.Modalities.SYMBOL, question_encoder.vocab_size)
    }
    p.target_modality = (registry.Modalities.CLASS_LABEL + ":multi_label",
                         targets_encoder.vocab_size)
    p.input_space_id = problem.SpaceID.IMAGE  # multiple input features?
    p.target_space_id = self.target_space_id

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    generator_utils.generate_dataset_and_shuffle(
        self.generator(data_dir, tmp_dir, problem.DatasetSplit.TRAIN),
        self.training_filepaths(data_dir, self.train_shards, shuffled=False),
        self.generator(data_dir, tmp_dir, problem.DatasetSplit.EVAL),
        self.dev_filepaths(data_dir, self.dev_shards, shuffled=False))


@registry.register_problem
class ImageVqav2Tokens10kLabels3k(ImageQuestion2MultilabelProblem):
  """VQA V2, raw images, 10k question vocab, 3k answer label."""
  _MSCOCO_ROOT_URL = "http://msvocds.blob.core.windows.net/"
  _MSCOCO_IMAGE_URLS = [
      "coco2014/train2014.zip", "coco2014/val2014.zip", "coco2014/test2014.zip",
  ]
  _VQA_V2_ANNOTATION_URL = ("https://drive.google.com/uc?export=download&id="
                            "1xfMU54ObCLvMRAekT3cfcIg-AgY39fWB")

  _VQA_V2_TRAIN_DATASETS = [
      ("trainval_resnet101_faster_rcnn_genome_36.tsv",
       "v2_train2014_annotations.json"),
  ]
  _VQA_V2_DEV_DATASETS = [
      ("trainval_resnet101_faster_rcnn_genome_36.tsv",
       "v2_val2014_annotations.json"),
  ]
  _VQA_V2_TEST_DATASETS = [
      ("test2015_resnet101_faster_rcnn_genome_36.tsv",
       "v2_test2015_annotations.json"),
  ]

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return self._VQA_V2_TRAIN_DATASETS if train else self._VQA_V2_DEV_DATASETS

  @property
  def target_space_id(self):
    return problem.SpaceID.GENERIC

  @property
  def vocab_size(self):
    return 10000

  @property
  def num_classes(self):
    return 3000

  @property
  def vocab_filename(self):
    return "question.vocab.%d" % self.vocab_size

  @property
  def label_filename(self):
    return "answer.label.%d" % self.num_classes

  @property
  def train_shards(self):
    return 128

  @property
  def dev_shards(self):
    return 64

  def example_reading_spec(self):
    data_fields, data_items_to_decoders = (
        super(ImageVqav2Tokens10kLabels3k, self).example_reading_spec())
    data_fields["image/image_id"] = tf.FixedLenFeature((), tf.int64)
    data_fields["image/question_id"] = tf.FixedLenFeature((), tf.int64)
    data_fields["image/question"] = tf.FixedLenSequenceFeature(
        (), tf.int64, allow_missing=True)
    data_fields["image/answer"] = tf.FixedLenSequenceFeature(
        (), tf.int64, allow_missing=True)

    data_items_to_decoders[
        "question"] = tf.contrib.slim.tfexample_decoder.Tensor(
            "image/question")
    data_items_to_decoders[
        "targets"] = tf.contrib.slim.tfexample_decoder.Tensor(
            "image/answer")
    return data_fields, data_items_to_decoders

  def preprocess_example(self, example, mode, hparams):
    # hparams is model_hparams
    image = example["inputs"]
    example["inputs"] = vqa_utils.vqa_v2_preprocess_image(
        image, hparams.height, hparams.width, mode,
        resize_side=hparams.resize_side, distort=hparams.distort,
        image_model_fn=hparams.image_model_fn)
    return example

  def generator(self, data_dir, tmp_dir, dataset_split):
    datasets = self.source_data_files(dataset_split)
    return self.vqa_v2_generator(data_dir, tmp_dir, datasets)

  def vqa_v2_generator(self, data_dir, tmp_dir, datasets):
    """VQA v2 generator using raw images."""
    _get_vqa_v2_annotations(tmp_dir, self._VQA_V2_ANNOTATION_URL)
    _get_vqa_v2_image_raw_dataset(tmp_dir, self._MSCOCO_ROOT_URL,
                                  self._MSCOCO_IMAGE_URLS)
    vocab_path = os.path.join(data_dir, self.vocab_filename)
    if not tf.gfile.Exists(vocab_path):
      vocab_tmp_path = os.path.join(tmp_dir, self.vocab_filename)
      tf.gfile.Copy(vocab_tmp_path, vocab_path)
      with tf.gfile.GFile(vocab_path, mode="r") as f:
        vocab_data = "<pad>\n<EOS>\n" + f.read() + "UNK\n"
      with tf.gfile.GFile(vocab_path, mode="w") as f:
        f.write(vocab_data)
    label_path = os.path.join(data_dir, self.label_filename)
    if not tf.gfile.Exists(label_path):
      label_tmp_path = os.path.join(tmp_dir, self.label_filename)
      tf.gfile.Copy(label_tmp_path, label_path)

    vocab_encoder = text_encoder.TokenTextEncoder(vocab_path, replace_oov="UNK")
    label_encoder = text_encoder.ClassLabelEncoder(
        class_labels_fname=label_path)

    prefix_annotation = []
    for prefix, annotation_file in datasets:
      annotation_path = os.path.join(tmp_dir, annotation_file)
      with tf.gfile.Open(annotation_path) as f:
        annotation_json = json.loads(f.read())
      prefix_annotation += [(prefix, anno) for anno in annotation_json]
    random.shuffle(prefix_annotation)
    annotation_count = len(prefix_annotation)
    tf.logging.info("Processing %d annotations for vqa v2" %(annotation_count))

    for prefix, anno in prefix_annotation:
      image_id = anno["image_id"]
      question = vocab_encoder.encode(anno["question"])
      answer = [label_encoder.encode(ans) for ans in anno["answer"]]
      answer = answer if answer else [0]  # 0 indicates padding
      image_filename = "COCO_" + prefix + "_" + str(image_id).zfill(12) + ".jpg"
      image_filepath = os.path.join(tmp_dir, prefix, image_filename)
      with tf.gfile.Open(image_filepath, "r") as f:
        encoded_image_data = f.read()
        yield {
            "image/encoded": [encoded_image_data],
            "image/format": ["jpeg"],
            "image/image_id": [image_id],
            "image/question_id": [anno["question_id"]],
            "image/question": question,
            "image/answer": answer,
        }


@registry.register_problem
class ImageVqav2RcnnFeatureTokens10kLabels3k(ImageVqav2Tokens10kLabels3k):
  """VQA V2, image feature, 10k question vocab, 3k answer label."""
  _VQA_V2_FEATURE_URL = ("https://drive.google.com/uc?export=download&id="
                         "1yTTFUWqx1SScC-Whs2vRbF3tDsEEjrtt")

  @property
  def num_boxes(self):
    return 36

  @property
  def feature_dimension(self):
    return 2048

  @property
  def spatial_feature_dimension(self):
    return 6

  @property
  def feature_file_field_names(self):
    return ["image_id",
            "image_w",
            "image_h",
            "num_boxes",
            "boxes",
            "features"]

  def preprocess_example(self, example, mode, hparams):
    # reshape some features
    example["inputs"] = tf.reshape(
        example["inputs"], [self.num_boxes, 1, self.feature_dimension])
    example["spatial_feature"] = tf.reshape(
        example["spatial_feature"],
        [self.num_boxes, 1, self.spatial_feature_dimension])
    return example

  def example_reading_spec(self):
    data_fields, data_items_to_decoders = {}, {}
    data_fields["image/feature"] = tf.FixedLenSequenceFeature(
        (), tf.float32, allow_missing=True)
    data_fields["image/spatial_feature"] = tf.FixedLenSequenceFeature(
        (), tf.float32, allow_missing=True)
    data_fields["image/image_id"] = tf.FixedLenFeature((), tf.int64)
    data_fields["image/question_id"] = tf.FixedLenFeature((), tf.int64)
    data_fields["image/question"] = tf.FixedLenSequenceFeature(
        (), tf.int64, allow_missing=True)
    data_fields["image/answer"] = tf.FixedLenSequenceFeature(
        (), tf.int64, allow_missing=True)

    data_items_to_decoders[
        "inputs"] = tf.contrib.slim.tfexample_decoder.Tensor(
            "image/feature")
    data_items_to_decoders[
        "question_id"] = tf.contrib.slim.tfexample_decoder.Tensor(
            "image/question_id")
    data_items_to_decoders[
        "image_id"] = tf.contrib.slim.tfexample_decoder.Tensor(
            "image/image_id")

    data_items_to_decoders[
        "spatial_feature"] = tf.contrib.slim.tfexample_decoder.Tensor(
            "image/spatial_feature")
    data_items_to_decoders[
        "question"] = tf.contrib.slim.tfexample_decoder.Tensor(
            "image/question")
    data_items_to_decoders[
        "targets"] = tf.contrib.slim.tfexample_decoder.Tensor(
            "image/answer")

    return data_fields, data_items_to_decoders

  def vqa_v2_generator(self, data_dir, tmp_dir, datasets):
    """VQA v2 generator using image features."""
    _get_vqa_v2_annotations(tmp_dir, self._VQA_V2_ANNOTATION_URL)
    _get_vqa_v2_image_feature_dataset(tmp_dir, self._VQA_V2_FEATURE_URL)
    vocab_path = os.path.join(data_dir, self.vocab_filename)
    if not tf.gfile.Exists(vocab_path):
      vocab_tmp_path = os.path.join(tmp_dir, self.vocab_filename)
      tf.gfile.Copy(vocab_tmp_path, vocab_path)
      with tf.gfile.GFile(vocab_path, mode="r") as f:
        vocab_data = "<pad>\n<EOS>\n" + f.read() + "UNK\n"
      with tf.gfile.GFile(vocab_path, mode="w") as f:
        f.write(vocab_data)
    label_path = os.path.join(data_dir, self.label_filename)
    if not tf.gfile.Exists(label_path):
      label_tmp_path = os.path.join(tmp_dir, self.label_filename)
      tf.gfile.Copy(label_tmp_path, label_path)

    vocab_encoder = text_encoder.TokenTextEncoder(vocab_path, replace_oov="UNK")
    label_encoder = text_encoder.ClassLabelEncoder(
        class_labels_fname=label_path)

    # merge annotations
    annotation_json = []
    for _, annotation_file in datasets:
      annotation_path = os.path.join(tmp_dir, annotation_file)
      with tf.gfile.Open(annotation_path) as f:
        annotation_json += json.loads(f.read())
    annotation_count = len(annotation_json)
    tf.logging.info("Processing %d annotations for vqa v2" %(annotation_count))

    imageid2annotation = {}
    for anno in annotation_json:
      if anno["image_id"] not in imageid2annotation:
        imageid2annotation[anno["image_id"]] = [anno]
      else:
        imageid2annotation[anno["image_id"]].append(anno)

    csv.field_size_limit(sys.maxsize)
    for feature_file, _ in datasets:
      feature_file_path = os.path.join(tmp_dir, feature_file)
      with open(feature_file_path, "r+b") as tsv_file:
        csv_reader = csv.DictReader(
            tsv_file, delimiter="\t", fieldnames=self.feature_file_field_names)
        for item in csv_reader:
          item["num_boxes"] = int(item["num_boxes"])
          image_id = int(item["image_id"])
          image_w = float(item["image_w"])
          image_h = float(item["image_h"])
          bboxes = np.frombuffer(base64.decodestring(item["boxes"]),
                                 dtype=np.float32).reshape(
                                     (item["num_boxes"], -1))

          box_width = bboxes[:, 2] - bboxes[:, 0]
          box_height = bboxes[:, 3] - bboxes[:, 1]
          scaled_width = box_width / image_w
          scaled_height = box_height / image_h
          scaled_x = bboxes[:, 0] / image_w
          scaled_y = bboxes[:, 1] / image_h

          box_width = box_width[..., np.newaxis]
          box_height = box_height[..., np.newaxis]
          scaled_width = scaled_width[..., np.newaxis]
          scaled_height = scaled_height[..., np.newaxis]
          scaled_x = scaled_x[..., np.newaxis]
          scaled_y = scaled_y[..., np.newaxis]

          spatial_features = np.concatenate(
              (scaled_x,
               scaled_y,
               scaled_x + scaled_width,
               scaled_y + scaled_height,
               scaled_width,
               scaled_height),
              axis=1)

          if image_id in imageid2annotation:
            for anno in imageid2annotation[image_id]:
              question = vocab_encoder.encode(anno["question"])
              answer = [label_encoder.encode(ans) for ans in anno["answer"]]
              answer = answer if answer else [0]  # 0 indicates padding
              yield {
                  "image/feature":
                  np.frombuffer(base64.decodestring(item["features"]),
                                dtype=np.float32).tolist(),
                  "image/spatial_feature": spatial_features.flatten().tolist(),
                  "image/height": [image_h],
                  "image/width": [image_w],
                  "image/bboxes": bboxes.flatten().tolist(),
                  "image/image_id": [image_id],
                  "image/question_id": [anno["question_id"]],
                  "image/question": question,
                  "image/answer": answer,
              }

            del imageid2annotation[image_id]

    # assert all annotations are included
    assert not imageid2annotation
