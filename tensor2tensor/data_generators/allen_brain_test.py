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

"""Tests of the Allen Brain Atlas problems."""

import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

from tensor2tensor.data_generators import allen_brain
from tensor2tensor.data_generators.allen_brain import _generator
from tensor2tensor.data_generators.allen_brain_utils import mock_raw_data
from tensor2tensor.data_generators.allen_brain_utils import TemporaryDirectory
from tensor2tensor.models import image_transformer_2d

tfe.enable_eager_execution()
Modes = tf.estimator.ModeKeys


class TestAllenBrain(tf.test.TestCase):
  """Tests that are common to all Allen Brain Atlas problems."""

  def setUp(self):

    self.all_problems = [
        #allen_brain.Img2imgAllenBrain,
        #allen_brain.Img2imgAllenBrainDim48to64,
        #allen_brain.Img2imgAllenBrainDim8to32,
        #allen_brain.Img2imgAllenBrainDim16to32,
        allen_brain.Img2imgAllenBrainDim16to16Paint1
    ]

  def test_generator_produces_examples(self):
    """Basic test that the generator produces examples with expected keys."""

    for is_training in [True, False]:
      with TemporaryDirectory() as tmp_dir:
        mock_raw_data(tmp_dir, raw_dim=256, num_images=100)
        for example in _generator(tmp_dir, is_training):
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


if __name__ == "__main__":
  tf.test.main()
