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
"""Tests for tensor2tensor.registry."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.utils import modality
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

# pylint: disable=unused-variable


class ModelRegistryTest(tf.test.TestCase):

  def setUp(self):
    registry._reset()

  def testT2TModelRegistration(self):

    @registry.register_model
    class MyModel1(t2t_model.T2TModel):
      pass

    model = registry.model("my_model1")
    self.assertTrue(model is MyModel1)

  def testNamedRegistration(self):

    @registry.register_model("model2")
    class MyModel1(t2t_model.T2TModel):
      pass

    model = registry.model("model2")
    self.assertTrue(model is MyModel1)

  def testNonT2TModelRegistration(self):

    @registry.register_model
    def model_fn():
      pass

    model = registry.model("model_fn")
    self.assertTrue(model is model_fn)

  def testUnknownModel(self):
    with self.assertRaisesRegexp(LookupError, "never registered"):
      registry.model("not_registered")

  def testDuplicateRegistration(self):

    @registry.register_model
    def m1():
      pass

    with self.assertRaisesRegexp(LookupError, "already registered"):

      @registry.register_model("m1")
      def m2():
        pass

  def testListModels(self):

    @registry.register_model
    def m1():
      pass

    @registry.register_model
    def m2():
      pass

    self.assertSetEqual(set(["m1", "m2"]), set(registry.list_models()))

  def testSnakeCase(self):
    convert = registry._convert_camel_to_snake

    self.assertEqual("typical_camel_case", convert("TypicalCamelCase"))
    self.assertEqual("numbers_fuse2gether", convert("NumbersFuse2gether"))
    self.assertEqual("numbers_fuse2_gether", convert("NumbersFuse2Gether"))
    self.assertEqual("lstm_seq2_seq", convert("LSTMSeq2Seq"))
    self.assertEqual("starts_lower", convert("startsLower"))
    self.assertEqual("starts_lower_caps", convert("startsLowerCAPS"))
    self.assertEqual("caps_fuse_together", convert("CapsFUSETogether"))
    self.assertEqual("startscap", convert("Startscap"))
    self.assertEqual("s_tartscap", convert("STartscap"))


class HParamRegistryTest(tf.test.TestCase):

  def setUp(self):
    registry._reset()

  def testHParamSet(self):

    @registry.register_hparams
    def my_hparams_set():
      return 3

    @registry.register_ranged_hparams
    def my_hparams_range(_):
      pass

    self.assertEqual(registry.hparams("my_hparams_set"), my_hparams_set())
    self.assertTrue(
        registry.ranged_hparams("my_hparams_range") is my_hparams_range)

  def testNamedRegistration(self):

    @registry.register_hparams("a")
    def my_hparams_set():
      return 7

    @registry.register_ranged_hparams("a")
    def my_hparams_range(_):
      pass

    self.assertEqual(registry.hparams("a"), my_hparams_set())
    self.assertTrue(registry.ranged_hparams("a") is my_hparams_range)

  def testUnknownHparams(self):
    with self.assertRaisesRegexp(LookupError, "never registered"):
      registry.hparams("not_registered")
    with self.assertRaisesRegexp(LookupError, "never registered"):
      registry.ranged_hparams("not_registered")

  def testNoneHparams(self):

    @registry.register_hparams
    def hp():
      pass

    with self.assertRaisesRegexp(TypeError, "is None"):
      registry.hparams("hp")

  def testDuplicateRegistration(self):

    @registry.register_hparams
    def hp1():
      pass

    with self.assertRaisesRegexp(LookupError, "already registered"):

      @registry.register_hparams("hp1")
      def hp2():
        pass

    @registry.register_ranged_hparams
    def rhp1(_):
      pass

    with self.assertRaisesRegexp(LookupError, "already registered"):

      @registry.register_ranged_hparams("rhp1")
      def rhp2(_):
        pass

  def testListHparams(self):

    @registry.register_hparams
    def hp1():
      pass

    @registry.register_hparams("hp2_named")
    def hp2():
      pass

    @registry.register_ranged_hparams
    def rhp1(_):
      pass

    @registry.register_ranged_hparams("rhp2_named")
    def rhp2(_):
      pass

    self.assertSetEqual(set(["hp1", "hp2_named"]), set(registry.list_hparams()))
    self.assertSetEqual(
        set(["rhp1", "rhp2_named"]), set(registry.list_ranged_hparams()))

  def testRangeSignatureCheck(self):

    with self.assertRaisesRegexp(ValueError, "must take a single argument"):

      @registry.register_ranged_hparams
      def rhp_bad():
        pass

    with self.assertRaisesRegexp(ValueError, "must take a single argument"):

      @registry.register_ranged_hparams
      def rhp_bad2(a, b):  # pylint: disable=unused-argument
        pass


class ModalityRegistryTest(tf.test.TestCase):

  def setUp(self):
    registry._reset()

  def testModalityRegistration(self):

    @registry.register_symbol_modality
    class MySymbolModality(modality.Modality):
      pass

    @registry.register_audio_modality
    class MyAudioModality(modality.Modality):
      pass

    @registry.register_image_modality
    class MyImageModality(modality.Modality):
      pass

    @registry.register_class_label_modality
    class MyClassLabelModality(modality.Modality):
      pass

    self.assertTrue(
        registry.symbol_modality("my_symbol_modality") is MySymbolModality)
    self.assertTrue(
        registry.audio_modality("my_audio_modality") is MyAudioModality)
    self.assertTrue(
        registry.image_modality("my_image_modality") is MyImageModality)
    self.assertTrue(
        registry.class_label_modality("my_class_label_modality") is
        MyClassLabelModality)

  def testDefaultNameLookup(self):

    @registry.register_symbol_modality("default")
    class MyDefaultModality(modality.Modality):
      pass

    self.assertTrue(registry.symbol_modality() is MyDefaultModality)

  def testList(self):

    @registry.register_symbol_modality
    class MySymbolModality(modality.Modality):
      pass

    @registry.register_audio_modality
    class MyAudioModality(modality.Modality):
      pass

    @registry.register_image_modality
    class MyImageModality(modality.Modality):
      pass

    @registry.register_class_label_modality
    class MyClassLabelModality(modality.Modality):
      pass

    expected = [
        "symbol:my_symbol_modality", "audio:my_audio_modality",
        "image:my_image_modality", "class_label:my_class_label_modality"
    ]

    self.assertSetEqual(set(registry.list_modalities()), set(expected))


if __name__ == "__main__":
  tf.test.main()
