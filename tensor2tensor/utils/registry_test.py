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

"""Tests for tensor2tensor.registry."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow.compat.v1 as tf


# pylint: disable=unused-variable,unused-argument


class RegistryClassTest(tf.test.TestCase):
  """Test of base registry.Registry class."""

  def testGetterSetter(self):
    r = registry.Registry("test_registry")
    r["hello"] = lambda: "world"
    r["a"] = lambda: "b"
    self.assertEqual(r["hello"](), "world")
    self.assertEqual(r["a"](), "b")

  def testDefaultKeyFn(self):
    r = registry.Registry("test", default_key_fn=lambda x: x().upper())
    r.register()(lambda: "hello")
    self.assertEqual(r["HELLO"](), "hello")

  def testNoKeyProvided(self):
    r = registry.Registry("test")
    def f():
      return 3
    r.register(f)
    self.assertEqual(r["f"](), 3)

  def testMembership(self):
    r = registry.Registry("test_registry")
    r["a"] = lambda: None
    r["b"] = lambda: 4
    self.assertTrue("a" in r)
    self.assertTrue("b" in r)

  def testIteration(self):
    r = registry.Registry("test_registry")
    r["a"] = lambda: None
    r["b"] = lambda: 4
    self.assertEqual(sorted(r), ["a", "b"])

  def testLen(self):
    r = registry.Registry("test_registry")
    self.assertEqual(len(r), 0)
    r["a"] = lambda: None
    self.assertEqual(len(r), 1)
    r["b"] = lambda: 4
    self.assertEqual(len(r), 2)

  def testTransformer(self):
    r = registry.Registry(
        "test_registry", value_transformer=lambda x, y: x + y())
    r.register(3)(lambda: 5)
    r.register(10)(lambda: 12)
    self.assertEqual(r[3], 8)
    self.assertEqual(r[10], 22)
    self.assertEqual(set(r.values()), set((8, 22)))
    self.assertEqual(set(r.items()), set(((3, 8), (10, 22))))

  def testGet(self):
    r = registry.Registry("test_registry", value_transformer=lambda k, v: v())
    r["a"] = lambda: "xyz"
    self.assertEqual(r.get("a"), "xyz")
    self.assertEqual(r.get("a", 3), "xyz")
    self.assertIsNone(r.get("b"))
    self.assertEqual(r.get("b", 3), 3)


class EnvProblemRegistryTest(tf.test.TestCase):

  def setUp(self):
    registry.Registries.env_problems._clear()

  def testEnvProblem(self):
    # Register this class and expect to get it back.

    @registry.register_env_problem
    class EnvProb(object):

      batch_size = None

      def initialize(self, batch_size):
        self.batch_size = batch_size

    # Get it with given batch_size.
    batch_size = 100
    ep = registry.env_problem("env_prob", batch_size=batch_size)

    # name property is set.
    self.assertEqual("env_prob", ep.name)

    # initialize was called and therefore batch_size was set.
    self.assertEqual(batch_size, ep.batch_size)

    # assert on the type.
    self.assertIsInstance(ep, EnvProb)


class ModelRegistryTest(tf.test.TestCase):

  def setUp(self):
    registry.Registries.models._clear()

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
    with self.assertRaisesRegexp(KeyError, "never registered"):
      registry.model("not_registered")

  def testDuplicateRegistration(self):

    @registry.register_model
    def m1():
      pass

    with self.assertRaisesRegexp(KeyError, "already registered"):

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


class OptimizerRegistryTest(tf.test.TestCase):

  def setUp(self):
    registry.Registries.optimizers._clear()

  def testRegistration(self):
    @registry.register_optimizer
    def my_optimizer(learning_rate, hparams):
      return 3

    @registry.register_optimizer("my_other_optimizer")
    def another_optimizer(learning_rate, hparams):
      return 5

    self.assertEqual(registry.optimizer("my_optimizer"), my_optimizer)
    self.assertEqual(
        registry.optimizer("my_other_optimizer"), another_optimizer)

  def testMembership(self):
    @registry.register_optimizer
    def my_optimizer(learning_rate, hparams):
      return 3

    @registry.register_optimizer("my_other_optimizer")
    def another_optimizer(learning_rate, hparams):
      return 5

    self.assertTrue("my_optimizer" in registry.Registries.optimizers)
    self.assertTrue("my_other_optimizer" in registry.Registries.optimizers)
    self.assertFalse("another_optimizer" in registry.Registries.optimizers)
    self.assertEqual(len(registry.Registries.optimizers), 2)

  def testArgErrorCheck(self):
    with self.assertRaisesRegexp(ValueError, "must take .* arguments"):
      registry.Registries.optimizers.register("OneArgs")(lambda x: 4)
    with self.assertRaisesRegexp(ValueError, "must take .* arguments"):
      registry.Registries.optimizers.register("ThreeArgs")(
          lambda x, y, z: 4)
    with self.assertRaisesRegexp(ValueError, "must take .* arguments"):
      registry.Registries.optimizers.register("NArgs")(lambda *args: 4)
    with self.assertRaisesRegexp(ValueError, "must take .* arguments"):
      registry.Registries.optimizers.register("Kwargs")(lambda **kargs: 4)
    with self.assertRaisesRegexp(ValueError, "must take .* arguments"):
      registry.Registries.optimizers.register("TwoAndKwargs")(
          lambda a, b, **kargs: 4)

  def testMultipleRegistration(self):
    @registry.register_optimizer
    def my_optimizer(learning_rate, hparams):
      return 3

    with self.assertRaisesRegexp(KeyError, "already registered"):

      @registry.register_optimizer("my_optimizer")
      def another_fn(learning_rate, hparams):
        return 5

  def testUnknownOptimizer(self):
    with self.assertRaisesRegexp(KeyError, "never registered"):
      registry.optimizer("not_registered_optimizer")

  def testGetterSetterInterface(self):
    def f(x, y):
      return 3

    k = "blah"
    registry.Registries.optimizers[k] = f
    self.assertEqual(registry.optimizer(k), f)
    self.assertEqual(registry.Registries.optimizers[k], f)
    self.assertEqual(registry.Registries.optimizers[k], registry.optimizer(k))


class HParamRegistryTest(tf.test.TestCase):

  def setUp(self):
    registry.Registries.hparams._clear()
    registry.Registries.ranged_hparams._clear()

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
    with self.assertRaisesRegexp(KeyError, "never registered"):
      registry.hparams("not_registered")
    with self.assertRaisesRegexp(KeyError, "never registered"):
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


class RegistryHelpTest(tf.test.TestCase):
  """Test class for common functions."""

  def testRegistryHelp(self):
    help_str = registry.help_string()
    self.assertIsNotNone(help_str)
    self.assertGreater(len(help_str), 0)

if __name__ == "__main__":
  tf.test.main()
