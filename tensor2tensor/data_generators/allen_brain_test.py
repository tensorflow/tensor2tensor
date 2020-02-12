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

"""Tests of the Allen Brain Atlas problems."""

import os
import shutil
import tempfile

import numpy as np

from tensor2tensor.data_generators import allen_brain
from tensor2tensor.models import image_transformer_2d
from tensor2tensor.utils import contrib

import tensorflow.compat.v1 as tf


tfe = contrib.eager()
tfe.enable_eager_execution()
Modes = tf.estimator.ModeKeys  # pylint: disable=invalid-name


def mock_raw_image(x_dim=1024, y_dim=1024, num_channels=3,
                   output_path=None, write_image=True):
  """Generate random `x_dim` by `y_dim`, optionally to `output_path`.

  Args:
    x_dim: int, the x dimension of generated raw image.
    y_dim: int, the x dimension of generated raw image.
    num_channels: int, number of channels in image.
    output_path: str, path to which to write image.
    write_image: bool, whether to write the image to output_path.

  Returns:
    numpy.array: The random `x_dim` by `y_dim` image (i.e. array).
  """

  rand_shape = (x_dim, y_dim, num_channels)

  if num_channels != 3:
    raise NotImplementedError("mock_raw_image for channels != 3 not yet "
                              "implemented.")

  img = np.random.random(rand_shape)
  img = np.uint8(img*255)

  if write_image:
    image_obj = allen_brain.PIL_Image()
    pil_img = image_obj.fromarray(img, mode="RGB")
    with tf.gfile.Open(output_path, "w") as f:
      pil_img.save(f, "jpeg")

  return img


def mock_raw_data(tmp_dir, raw_dim=1024, num_channels=3, num_images=1):
  """Mock a raw data download directory with meta and raw subdirs.

  Notes:

    * This utility is shared by tests in both allen_brain_utils and
      allen_brain so kept here instead of in one of *_test.

  Args:
    tmp_dir: str, temporary dir in which to mock data.
    raw_dim: int, the x and y dimension of generated raw imgs.
    num_channels: int, number of channels in image.
    num_images: int, number of images to mock.
  """

  tf.gfile.MakeDirs(tmp_dir)

  for image_id in range(num_images):

    raw_image_path = os.path.join(tmp_dir, "%s.jpg" % image_id)

    mock_raw_image(x_dim=raw_dim, y_dim=raw_dim,
                   num_channels=num_channels,
                   output_path=raw_image_path)


class TemporaryDirectory(object):
  """For py2 support of `with tempfile.TemporaryDirectory() as name:`"""

  def __enter__(self):
    self.name = tempfile.mkdtemp()
    return self.name

  def __exit__(self, exc_type, exc_value, traceback):
    shutil.rmtree(self.name)


class TestAllenBrain(tf.test.TestCase):
  """Tests that are common to all Allen Brain Atlas problems."""

  def setUp(self):

    self.all_problems = [
        allen_brain.Img2imgAllenBrainDim16to16Paint1
    ]

  def test_generator_produces_examples(self):
    """Basic test that the generator produces examples with expected keys."""

    for is_training in [True, False]:
      with TemporaryDirectory() as tmp_dir:
        mock_raw_data(tmp_dir, raw_dim=256, num_images=100)
        for example in allen_brain._generator(tmp_dir, is_training):
          for key in ["image/encoded", "image/format",
                      "image/height", "image/width"]:
            self.assertTrue(key in example.keys())

  def test_generate_data_produces_examples_of_correct_shape(self):
    """Test examples have correct input and output shapes.

    Notes:

      * Loops over all AllenBrainImage2image* problems.

    """

    with TemporaryDirectory() as tmp_dir:
      mock_raw_data(tmp_dir, raw_dim=256, num_images=100)
      with TemporaryDirectory() as data_dir:
        for problem_obj in self.all_problems:
          problem_object = problem_obj()

          problem_object.generate_data(data_dir, tmp_dir)

          for mode in [Modes.TRAIN, Modes.EVAL]:

            dataset = problem_object.dataset(mode, data_dir)
            example = tfe.Iterator(dataset).next()

            num_channels = problem_object.num_channels

            # Check that the input tensor has the right shape
            input_dim = problem_object.input_dim
            self.assertEqual(example["inputs"].numpy().shape,
                             (input_dim, input_dim, num_channels))

            # Check that the targets tensor has the right shape
            output_dim = problem_object.output_dim
            self.assertEqual(example["targets"].numpy().shape,
                             (output_dim, output_dim, num_channels))

  def test_transformer2d_single_step_e2e(self):
    """Minimal end-to-end test of training and eval on allen_brain_image2image.

    Notes:

      * Runs problem generate_data

      * Runs a single step of training

      * Runs model in eval mode to obtain a prediction and confirms the
        resulting shape.

        * TODO: Running this in predict mode crashes in my environment.
          Separately have seen predict mode not produce the right shape
          output tensors, as if .infer is still a wip.

    """

    problem_object = allen_brain.Img2imgAllenBrainDim8to32()

    with TemporaryDirectory() as tmp_dir:

      mock_raw_data(tmp_dir, raw_dim=256, num_images=100)

      with TemporaryDirectory() as data_dir:

        problem_object.generate_data(data_dir, tmp_dir)

        input_xy_dim = problem_object.input_dim
        target_xy_dim = problem_object.output_dim
        num_channels = problem_object.num_channels

        hparams = image_transformer_2d.img2img_transformer2d_tiny()
        hparams.data_dir = data_dir

        p_hparams = problem_object.get_hparams(hparams)

        model = image_transformer_2d.Img2imgTransformer(
            hparams, tf.estimator.ModeKeys.TRAIN, p_hparams
        )

        @tfe.implicit_value_and_gradients
        def loss_fn(features):
          _, losses = model(features)
          return losses["training"]

        batch_size = 1
        train_dataset = problem_object.dataset(Modes.TRAIN, data_dir)
        train_dataset = train_dataset.repeat(None).batch(batch_size)

        optimizer = tf.train.AdamOptimizer()

        example = tfe.Iterator(train_dataset).next()
        example["targets"] = tf.reshape(example["targets"],
                                        [batch_size,
                                         target_xy_dim,
                                         target_xy_dim,
                                         num_channels])
        _, gv = loss_fn(example)
        optimizer.apply_gradients(gv)

        model.set_mode(Modes.EVAL)
        dataset = problem_object.dataset(Modes.EVAL, data_dir)

        example = tfe.Iterator(dataset).next()
        example["inputs"] = tf.reshape(example["inputs"],
                                       [1,
                                        input_xy_dim,
                                        input_xy_dim,
                                        num_channels])
        example["targets"] = tf.reshape(example["targets"],
                                        [1,
                                         target_xy_dim,
                                         target_xy_dim,
                                         num_channels])

        predictions, _ = model(example)

        self.assertEqual(predictions.numpy().shape,
                         (1,
                          target_xy_dim,
                          target_xy_dim,
                          num_channels,
                          256))


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
