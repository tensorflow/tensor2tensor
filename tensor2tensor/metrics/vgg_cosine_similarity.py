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

"""VGG Cosine similarity metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.slim.python.slim.nets import vgg
from tensorflow_models.slim.preprocessing import vgg_preprocessing


def vgg_features(x):
  """Computes VGG features of input x.

  Args:
    x: 4-D Tensor, shape=(batch_size, height, width, channels)
  Returns:
    features: A list of tensors of VGG-features corresponding to x.
  """
  preprocess_single = functools.partial(
      vgg_preprocessing.preprocess_image, output_height=224, output_width=224,
      is_training=False)
  x = tf.map_fn(preprocess_single, x)
  _, features = vgg.vgg_16(x, num_classes=1000, is_training=False)

  # filter fully connected end-points
  return [t for n, t in features.items() if "fc" not in n]


def vgg_cosine_similarity(images1, images2):
  """VGG cosine similarity between images1[i] and images2[i].

  For every feature obtained from VGG, the cosine similarity is computed across
  the channels and averaged spatially. This is then averaged across
  all VGG features.

  Args:
    images1: 4-D Tensor, shape=(batch_size, height, width, n_channels)
    images2: 4-D Tensor, shape=(batch_size, height, width, n_channels)
  Returns:
    similarity: 1-D Tensor, shape=(batch_size,)
  """
  with arg_scope(vgg.vgg_arg_scope()):
    img1_features = vgg_features(images1)
    tf.get_variable_scope().reuse_variables()
    img2_features = vgg_features(images2)

  all_dists = []
  for img1_feat, img2_feat in zip(img1_features, img2_features):

    # Computes cosine similarity across channels, i.e dot-product and sum.
    img1_feat = tf.nn.l2_normalize(img1_feat, axis=-1)
    img2_feat = tf.nn.l2_normalize(img2_feat, axis=-1)
    distance = img1_feat * img2_feat

    # Computes mean of the distance spatially.
    curr_cosine_dist = tf.reduce_mean(
        tf.reduce_sum(distance, axis=-1), axis=[1, 2])
    all_dists.append(curr_cosine_dist)
  all_dists = tf.stack(all_dists)

  # Average across VGG features.
  return tf.reduce_mean(all_dists, axis=0)


def vgg_cosine_similarity_from_ckpt(images1, images2, ckpt_path):
  """VGG Cosine similarity using a trained VGG ckpt.

  Args:
    images1: 4-D NumPy array, shape=(batch_size, height, width, n_channels)
    images2: 4-D NumPy array, shape=(batch_size, height, width, n_channels)
    ckpt_path: Path to trained VGG ckpt.
  Returns:
    similarity: 1-D NumPy array, shape=(batch_size,)
  """
  with tf.Graph().as_default():
    images1 = tf.convert_to_tensor(images1, dtype=tf.float32)
    images2 = tf.convert_to_tensor(images2, dtype=tf.float32)
    vgg_sim = vgg_cosine_similarity(images1, images2)


    saver = tf.train.Saver()
    with tf.Session() as sess:
      saver.restore(sess, ckpt_path)
      vgg_sim_np = sess.run(vgg_sim)
    return vgg_sim_np

