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

"""Registry for models, hyperparameter settings, problem types, and datasets.

Define a new model by subclassing T2TModel and register it:

```
@registry.register_model
class MyModel(T2TModel):
  ...
```

Access by snake-cased name: `registry.model("my_model")`. If you're using
`t2t_trainer.py`, you can pass on the command-line: `--model=my_model`.

See all the models registered: `registry.list_models()`.

For hyperparameter sets:
  * Register: `registry.register_hparams`
  * List: `registry.list_hparams`
  * Retrieve by name: `registry.hparams`
  * Command-line flag in `t2t_trainer.py`: `--hparams_set=name`

For hyperparameter ranges:
  * Register: `registry.register_ranged_hparams`
  * List: `registry.list_ranged_hparams`
  * Retrieve by name: `registry.ranged_hparams`
  * Command-line flag in `t2t_trainer.py`: `--hparams_range=name`
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.contrib.framework as framework
from tensorflow.python.util import tf_inspect as inspect
import collections

from tensor2tensor.utils import misc_utils


def default_name(class_or_fn):
  """Default name for a class or fn.

  This is the naming function by default for registers expecting classes or
  functions.

  Args:
    class_or_fn: class or function to be named.

  Returns:
    Default name for registration.
  """
  return misc_utils.camelcase_to_snakecase(class_or_fn.__name__)


def default_object_name(obj):
  """Default name for an object.

  This is the naming function by default for registers expectings objects for
  values.

  Args:
    obj: an object instance

  Returns:
    The registry's default name for the class of the object.
  """
  return default_name(obj.__class__)


def _default_value_transformer(k, v):
  return v


class Registry(object):
  """Dict-like class for managing registrations."""
  def __init__(
      self, register_name, default_key_fn=default_name, validator=None,
      on_set_callback=None, value_transformer=_default_value_transformer):
    """
    Args:
      register_name: str identifier for the given register. Used in error msgs.
      default_key_fn (optional): function mapping value -> key for registration
        when a key are not provided
      validator (optional): if given, this is run before setting a given
        (key, value) pair. Accepts (key, value) and should raise if there is a
        problem. Overwriting existing keys is not allowed and is checked
        separately.
      on_set_callback (optional): callback function accepting (key, value) pair
        which is run after an item is successfully set.
      value_transformer (optional): if run, `__getitem__` will return
        value_transformer(key, registered_value).
    """
    self._register = {}
    self._name = register_name
    self._default_key_fn = default_key_fn
    self._validator = validator
    self._on_set_callback = on_set_callback
    self._value_transformer = value_transformer

  def default_key(self, key):
    return self._default_key_fn(key)

  @property
  def name(self):
    return self._name

  def validate(self, key, value):
    if self._validator is not None:
      self._validator(key, value)

  def __setitem__(self, key, value):
    if key in self:
      raise KeyError("key %s already registered in registry %s"
                     % (key, self._name))
    self.validate(key, value)
    self._register[key] = value
    callback = self._on_set_callback
    if callback is not None:
      callback(key, value)

  def register(self, key=None):
    def decorator(value, key=None):
      if key is None:
        key = self.default_key(value)
      self[key] = value
      return value

    # Handle if decorator was used without parens
    if callable(key):
      return decorator(value=key, key=self.default_key(key))
    else:
      return lambda value: decorator(value, key=key)

  def _get(self, key):
    # convenience function for maintaining old API.
    # e.g. model = model_registry._get
    # not to be confused with self.get, which has a default value
    return self[key]

  def __getitem__(self, key):
    if key not in self:
      raise KeyError("%s never registered with register %s. Available:\n %s" %
                     (key, self.name,
                      display_list_by_prefix(sorted(self), 4)))
    value = self._register[key]
    return self._value_transformer(key, value)

  def __contains__(self, key):
    return key in self._register

  def __keys__(self):
    return self._register.keys()

  def values(self):
    return (self[k] for k in self)       # complicated because of transformer

  def items(self):
    return ((k, self[k]) for k in self)  # complicated because of transformer

  def __iter__(self):
    return iter(self._register)

  def __len__(self):
    return len(self._register)

  def clear(self):
    """Clear the internal register of previously registered values."""
    self._register.clear()

  def __delitem__(self, k):
    del self._register[k]

  def pop(self, k):
    return self._value_transformer(k, self._register.pop(k))

  def get(self, key, d=None):
    return self[key] if key in self else d



def _on_model_set(k, v):
  v.REGISTERED_NAME = k


def _nargs_validator(nargs, message):
  def f(key, value):
    args, varargs, keywords, _ = inspect.getargspec(value)
    if len(args) != nargs or varargs is not None or keywords is not None:
      raise ValueError(message)

  return f


ProblemSpec = collections.namedtuple(
    "ProblemSpec", ["base_name", "was_reversed", "was_copy"])


def parse_problem_name(name):
  """Determines if problem_name specifies a copy and/or reversal.

  Args:
    name: str, problem name, possibly with suffixes.

  Returns:
    base_name: A string with the base problem name.
    was_reversed: A boolean.
    was_copy: A boolean.
  """
  # Recursively strip tags until we reach a base name.
  if name.endswith("_rev"):
    base, was_rev, was_copy = parse_problem_name(name[:-4])
    if was_rev:
      # duplicate rev
      raise ValueError(
          "Invalid problem name %s: multiple '_rev' instances" % name)
    return ProblemSpec(base, True, was_copy)
  elif name.endswith("_copy"):
    base, was_reversed, was_copy = parse_problem_name(name[:-5])
    if was_copy:
      raise ValueError(
          "Invalid problem_name %s: multiple '_copy' instances" % name)
    return ProblemSpec(base, was_reversed, True)
  else:
    return ProblemSpec(name, False, False)


def get_problem_name(base_name, was_reversed=False, was_copy=False):
  name = base_name
  if was_copy:
    name = "%s_copy" % name
  if was_reversed:
    name = "%s_rev" % name
  return name


def _problem_name_validator(k, v):
  if parse_problem_name(k).base_name != k:
    raise KeyError(
        "Invalid problem name: cannot end in %s or %s" % ("_rev", "_copy"))


def _call_value(k, v):
  return v()


def _hparams_value_transformer(key, value):
  out = value()
  if out is None:
    raise TypeError("HParams %s is None. Make sure the registered function "
                    "returns the HParams object" % key)
  return out


model_registry = Registry("models", on_set_callback=_on_model_set)
optimizer_registry = Registry(
    "optimizers",
    default_key_fn=lambda fn: misc_utils.snakecase_to_camelcase(fn.__name__),
    validator=_nargs_validator(
        2,
        "Optimizer registration function must take exactly two arguments: "
        "learning_rate (float) and hparams (HParams)."))
hparams_registry = Registry(
    "hparams", value_transformer=_hparams_value_transformer)
ranged_hparams_registry = Registry(
    "ranged_hparams", validator=_nargs_validator(
        1,
        "RangedHParams set function must take a single argument, "
        "the RangedHParams object."))
base_problem_registry = Registry("problems", validator=_problem_name_validator)
attack_registry = Registry(
    "attacks", value_transformer=_call_value)
attack_params_registry = Registry(
    "attack_params", value_transformer=_call_value)
pruning_params_registry = Registry(
    "pruning_params", value_transformer=_call_value)
pruning_strategy_registry = Registry("pruning_strategies")

# consistent version of old API
model = model_registry._get
list_models = lambda: sorted(model_registry)
register_model = model_registry.register

optimizer = optimizer_registry._get
list_optimizers = lambda: sorted(optimizer_registry)
register_optimizer = optimizer_registry.register

hparams = hparams_registry._get
list_hparams = lambda: sorted(hparams_registry)
register_hparams = hparams_registry.register

ranged_hparams = ranged_hparams_registry._get
list_ranged_hparams = lambda: sorted(ranged_hparams_registry)
register_ranged_hparams = ranged_hparams_registry.register

base_problem = base_problem_registry._get
list_base_problems = lambda: sorted(base_problem_registry)
register_base_problem = base_problem_registry.register

# list_problems won't list all rev/copy combinations,
# so the name is slightly confusing. Similarly, register_problem will raise an
# error if attempting to register a value with a non-base key.
# Keeping for back-compatibility
list_problems = list_base_problems
register_problem = register_base_problem


def problem(problem_name, base_registry=base_problem_registry):
  """Get possibly copied/reversed problem registered in `base_registry`.

  Args:
    problem_name: string problem name. See `parse_problem_name`.

  Returns:
    possibly reversed/copied version of base problem registered in the given
    registry.
  """
  spec = parse_problem_name(problem_name)
  return base_registry[spec.base_name](
      was_copy=spec.was_copy, was_reversed=spec.was_reversed)


attack = attack_registry._get
list_attacks = lambda: sorted(attack_registry)
register_attack = attack_registry.register

attack_params = attack_params_registry._get
list_attack_params = lambda: sorted(attack_params_registry)
register_attack_params = attack_params_registry.register

pruning_params = pruning_params_registry._get
list_pruning_params = lambda: sorted(pruning_params_registry)
register_pruning_params = pruning_params_registry.register

pruning_strategy = pruning_strategy_registry._get
list_pruning_strategies = lambda: sorted(pruning_strategy_registry)
register_pruning_strategy = pruning_strategy_registry.register


# deprecated functions - plurals inconsistent with rest
# deprecation decorators added 2019-01-25
attacks = framework.deprecated(None, "Use registry.attack")(attack)
pruning_strategies = framework.deprecated(
    None, "Use registry.pruning_strategy")(pruning_strategy)


def display_list_by_prefix(names_list, starting_spaces=0):
  """Creates a help string for names_list grouped by prefix."""
  cur_prefix, result_lines = None, []
  space = " " * starting_spaces
  for name in sorted(names_list):
    split = name.split("_", 1)
    prefix = split[0]
    if cur_prefix != prefix:
      result_lines.append(space + prefix + ":")
      cur_prefix = prefix
    result_lines.append(space + "  * " + name)
  return "\n".join(result_lines)


def help_string():
  """Generate help string with contents of registry."""
  help_str = """
Registry contents:
------------------

  Models:
%s

  HParams:
%s

  RangedHParams:
%s

  Problems:
%s

  Optimizers:
%s

  Attacks:
%s

  Attack HParams:
%s

  Pruning HParams:
%s

  Pruning Strategies:
%s
"""
  lists = tuple(
      display_list_by_prefix(entries, starting_spaces=4) for entries in [
          list_models(),
          list_hparams(),
          list_ranged_hparams(),
          list_problems(),
          list_optimizers(),
          list_attacks(),
          list_attack_params(),
          list_pruning_params(),
          list_pruning_strategies(),
      ]
  )
  return help_str % lists
