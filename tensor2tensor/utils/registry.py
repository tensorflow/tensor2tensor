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

"""Object registration.

Registries are instances of `Registry`.

See `Registries` for a centralized list of object registries
(models, problems, hyperparameter sets, etc.).

New functions and classes can be registered using `.register`. The can be
accessed/queried similar to dictionaries, keyed by default by `snake_case`
equivalents.

```
@Registries.models.register
class MyModel(T2TModel):
  ...

'my_model' in Registries.models  # True
for k in Registries.models:
  print(k)  # prints 'my_model'
model = Registries.models['my_model'](constructor_arg)
```

#### Legacy Support

Define a new model by subclassing T2TModel and register it:

```
@register_model
class MyModel(T2TModel):
  ...
```

Access by snake-cased name: `model("my_model")`. If you're using
`t2t_trainer.py`, you can pass on the command-line: `--model=my_model`.

See all the models registered: `list_models()`.

For hyperparameter sets:
  * Register: `register_hparams`
  * List: `list_hparams`
  * Retrieve by name: `hparams`
  * Command-line flag in `t2t_trainer.py`: `--hparams_set=name`

For hyperparameter ranges:
  * Register: `register_ranged_hparams`
  * List: `list_ranged_hparams`
  * Retrieve by name: `ranged_hparams`
  * Command-line flag in `t2t_trainer.py`: `--hparams_range=name`
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensor2tensor.utils import misc_utils
import tensorflow.compat.v1 as tf

from tensorflow.python.util import tf_inspect as inspect  # pylint: disable=g-direct-tensorflow-import


def default_name(class_or_fn):
  """Default name for a class or function.

  This is the naming function by default for registries expecting classes or
  functions.

  Args:
    class_or_fn: class or function to be named.

  Returns:
    Default name for registration.
  """
  return misc_utils.camelcase_to_snakecase(class_or_fn.__name__)


default_object_name = lambda obj: default_name(type(obj))


class Registry(object):
  """Dict-like class for managing function registrations.

  ```python
  my_registry = Registry("custom_name")

  @my_registry.register
  def my_func():
    pass

  @my_registry.register()
  def another_func():
    pass

  @my_registry.register("non_default_name")
  def third_func(x, y, z):
    pass

  def foo():
    pass

  my_registry.register()(foo)
  my_registry.register("baz")(lambda (x, y): x + y)
  my_register.register("bar")

  print(list(my_registry))
  # ["my_func", "another_func", "non_default_name", "foo", "baz"]
  # (order may vary)
  print(my_registry["non_default_name"] is third_func)  # True
  print("third_func" in my_registry)                    # False
  print("bar" in my_registry)                           # False
  my_registry["non-existent_key"]                       # raises KeyError
  ```

  Optional validation, on_set callback and value transform also supported.
  See `__init__` doc.
  """

  def __init__(self,
               registry_name,
               default_key_fn=default_name,
               validator=None,
               on_set=None,
               value_transformer=(lambda k, v: v)):
    """Construct a new registry.

    Args:
      registry_name: str identifier for the given registry. Used in error msgs.
      default_key_fn (optional): function mapping value -> key for registration
        when a key is not provided
      validator (optional): if given, this is run before setting a given (key,
        value) pair. Accepts (key, value) and should raise if there is a
        problem. Overwriting existing keys is not allowed and is checked
        separately. Values are also checked to be callable separately.
      on_set (optional): callback function accepting (key, value) pair which is
        run after an item is successfully set.
      value_transformer (optional): if run, `__getitem__` will return
        value_transformer(key, registered_value).
    """
    self._registry = {}
    self._name = registry_name
    self._default_key_fn = default_key_fn
    self._validator = validator
    self._on_set = on_set
    self._value_transformer = value_transformer

  def default_key(self, value):
    """Default key used when key not provided. Uses function from __init__."""
    return self._default_key_fn(value)

  @property
  def name(self):
    return self._name

  def validate(self, key, value):
    """Validation function run before setting. Uses function from __init__."""
    if self._validator is not None:
      self._validator(key, value)

  def on_set(self, key, value):
    """Callback called on successful set. Uses function from __init__."""
    if self._on_set is not None:
      self._on_set(key, value)

  def __setitem__(self, key, value):
    """Validate, set, and (if successful) call `on_set` for the given item.

    Args:
      key: key to store value under. If `None`, `self.default_key(value)` is
        used.
      value: callable stored under the given key.

    Raises:
      KeyError: if key is already in registry.
    """
    if key is None:
      key = self.default_key(value)
    if key in self:
      raise KeyError(
          "key %s already registered in registry %s" % (key, self._name))
    if not callable(value):
      raise ValueError("value must be callable")
    self.validate(key, value)
    self._registry[key] = value
    self.on_set(key, value)

  def register(self, key_or_value=None):
    """Decorator to register a function, or registration itself.

    This is primarily intended for use as a decorator, either with or without
    a key/parentheses.
    ```python
    @my_registry.register('key1')
    def value_fn(x, y, z):
      pass

    @my_registry.register()
    def another_fn(x, y):
      pass

    @my_registry.register
    def third_func():
      pass
    ```

    Note if key_or_value is provided as a non-callable, registration only
    occurs once the returned callback is called with a callable as its only
    argument.
    ```python
    callback = my_registry.register('different_key')
    'different_key' in my_registry  # False
    callback(lambda (x, y): x + y)
    'different_key' in my_registry  # True
    ```

    Args:
      key_or_value (optional): key to access the registered value with, or the
        function itself. If `None` (default), `self.default_key` will be called
        on `value` once the returned callback is called with `value` as the only
        arg. If `key_or_value` is itself callable, it is assumed to be the value
        and the key is given by `self.default_key(key)`.

    Returns:
      decorated callback, or callback generated a decorated function.
    """

    def decorator(value, key):
      self[key] = value
      return value

    # Handle if decorator was used without parens
    if callable(key_or_value):
      return decorator(value=key_or_value, key=None)
    else:
      return lambda value: decorator(value, key=key_or_value)

  def __getitem__(self, key):
    if key not in self:
      raise KeyError("%s never registered with registry %s. Available:\n %s" %
                     (key, self.name, display_list_by_prefix(sorted(self), 4)))
    value = self._registry[key]
    return self._value_transformer(key, value)

  def __contains__(self, key):
    return key in self._registry

  def keys(self):
    return self._registry.keys()

  def values(self):
    return (self[k] for k in self)  # complicated because of transformer

  def items(self):
    return ((k, self[k]) for k in self)  # complicated because of transformer

  def __iter__(self):
    return iter(self._registry)

  def __len__(self):
    return len(self._registry)

  def _clear(self):
    self._registry.clear()

  def get(self, key, default=None):
    return self[key] if key in self else default


def _on_model_set(k, v):
  v.REGISTERED_NAME = k


def _nargs_validator(nargs, message):
  """Makes validator for function to ensure it takes nargs args."""
  if message is None:
    message = "Registered function must take exactly %d arguments" % nargs

  def f(key, value):
    del key
    spec = inspect.getfullargspec(value)
    if (len(spec.args) != nargs or spec.varargs is not None or
        spec.varkw is not None):
      raise ValueError(message)

  return f


ProblemSpec = collections.namedtuple("ProblemSpec",
                                     ["base_name", "was_reversed", "was_copy"])


def parse_problem_name(name):
  """Determines if problem_name specifies a copy and/or reversal.

  Args:
    name: str, problem name, possibly with suffixes.

  Returns:
    ProblemSpec: namedtuple with ["base_name", "was_reversed", "was_copy"]

  Raises:
    ValueError if name contains multiple suffixes of the same type
      ('_rev' or '_copy'). One of each is ok.
  """
  # Recursively strip tags until we reach a base name.
  if name.endswith("_rev"):
    base, was_reversed, was_copy = parse_problem_name(name[:-4])
    if was_reversed:
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
  """Construct a problem name from base and reversed/copy options.

  Inverse of `parse_problem_name`.

  Args:
    base_name: base problem name. Should not end in "_rev" or "_copy"
    was_reversed: if the problem is to be reversed
    was_copy: if the problem is to be copied

  Returns:
    string name consistent with use with `parse_problem_name`.

  Raises:
    ValueError if `base_name` ends with "_rev" or "_copy"
  """
  if any(base_name.endswith(suffix) for suffix in ("_rev", "_copy")):
    raise ValueError("`base_name` cannot end in '_rev' or '_copy'")
  name = base_name
  if was_copy:
    name = "%s_copy" % name
  if was_reversed:
    name = "%s_rev" % name
  return name


def _problem_name_validator(k, v):
  del v
  if parse_problem_name(k).base_name != k:
    raise KeyError(
        "Invalid problem name: cannot end in %s or %s" % ("_rev", "_copy"))


def _on_problem_set(k, v):
  v.name = k


def _call_value(k, v):
  del k
  return v()


def _hparams_value_transformer(key, value):
  out = value()
  if out is None:
    raise TypeError("HParams %s is None. Make sure the registered function "
                    "returns the HParams object" % key)
  return out


class Registries(object):
  """Object holding `Registry` objects."""

  def __init__(self):
    raise RuntimeError("Registries is not intended to be instantiated")

  models = Registry("models", on_set=_on_model_set)

  optimizers = Registry(
      "optimizers",
      validator=_nargs_validator(
          2, "Registered optimizer functions must take exactly two arguments: "
          "learning_rate (float) and hparams (HParams)."))

  hparams = Registry("hparams", value_transformer=_hparams_value_transformer)

  ranged_hparams = Registry(
      "ranged_hparams",
      validator=_nargs_validator(
          1, "Registered ranged_hparams functions must take a single argument, "
          "the RangedHParams object."))

  problems = Registry(
      "problems", validator=_problem_name_validator, on_set=_on_problem_set)

  attacks = Registry("attacks", value_transformer=_call_value)

  attack_params = Registry("attack_params", value_transformer=_call_value)

  pruning_params = Registry("pruning_params", value_transformer=_call_value)

  pruning_strategies = Registry("pruning_strategies")

  mtf_layers = Registry(
      "mtf_layers",
      validator=_nargs_validator(
          2, "Registered layer functions must take exaction two arguments: "
          "hparams (HParams) and prefix (str)."))

  env_problems = Registry("env_problems", on_set=_on_problem_set)


# consistent version of old API
model = Registries.models.__getitem__
list_models = lambda: sorted(Registries.models)
register_model = Registries.models.register


def optimizer(name):
  """Get pre-registered optimizer keyed by name.

  `name` should be snake case, though SGD -> sgd, RMSProp -> rms_prop and
  UpperCamelCase -> snake_case conversions included for legacy support.

  Args:
    name: name of optimizer used in registration. This should be a snake case
      identifier, though others supported for legacy reasons.

  Returns:
    optimizer
  """
  warn_msg = ("Please update `registry.optimizer` callsite "
              "(likely due to a `HParams.optimizer` value)")
  if name == "SGD":
    name = "sgd"
    tf.logging.warning("'SGD' optimizer now keyed by 'sgd'. %s" % warn_msg)
  elif name == "RMSProp":
    name = "rms_prop"
    tf.logging.warning(
        "'RMSProp' optimizer now keyed by 'rms_prop'. %s" % warn_msg)
  else:
    snake_name = misc_utils.camelcase_to_snakecase(name)
    if name != snake_name:
      tf.logging.warning(
          "optimizer names now keyed by snake_case names. %s" % warn_msg)
      name = snake_name
  return Registries.optimizers[name]


list_optimizers = lambda: sorted(Registries.optimizers)
register_optimizer = Registries.optimizers.register

hparams = Registries.hparams.__getitem__
register_hparams = Registries.hparams.register

list_env_problems = lambda: sorted(Registries.env_problems)
register_env_problem = Registries.env_problems.register


def list_hparams(prefix=None):
  hp_names = sorted(Registries.hparams)
  if prefix:
    hp_names = [name for name in hp_names if name.startswith(prefix)]
  return hp_names


ranged_hparams = Registries.ranged_hparams.__getitem__
list_ranged_hparams = lambda: sorted(Registries.ranged_hparams)
register_ranged_hparams = Registries.ranged_hparams.register

base_problem = Registries.problems.__getitem__
list_base_problems = lambda: sorted(Registries.problems)
register_base_problem = Registries.problems.register

# Keeping for back-compatibility
list_problems = list_base_problems
register_problem = register_base_problem


def problem(problem_name, **kwargs):
  """Get possibly copied/reversed problem in `base_registry` or `env_registry`.

  Args:
    problem_name: string problem name. See `parse_problem_name`.
    **kwargs: forwarded to env problem's initialize method.

  Returns:
    possibly reversed/copied version of base problem registered in the given
    registry.
  """
  spec = parse_problem_name(problem_name)
  try:
    return Registries.problems[spec.base_name](
        was_copy=spec.was_copy, was_reversed=spec.was_reversed)
  except KeyError:
    # If name is not found in base problems then try creating an env problem
    return env_problem(problem_name, **kwargs)


def env_problem(env_problem_name, **kwargs):
  """Get and initialize the `EnvProblem` with the given name and batch size.

  Args:
    env_problem_name: string name of the registered env problem.
    **kwargs: forwarded to env problem's initialize method.

  Returns:
    an initialized EnvProblem with the given batch size.
  """

  ep_cls = Registries.env_problems[env_problem_name]
  ep = ep_cls()
  ep.initialize(**kwargs)
  return ep


attack = Registries.attacks.__getitem__
list_attacks = lambda: sorted(Registries.attacks)
register_attack = Registries.attacks.register

attack_params = Registries.attack_params.__getitem__
list_attack_params = lambda: sorted(Registries.attack_params)
register_attack_params = Registries.attack_params.register

pruning_params = Registries.pruning_params.__getitem__
list_pruning_params = lambda: sorted(Registries.pruning_params)
register_pruning_params = Registries.pruning_params.register

pruning_strategy = Registries.pruning_strategies.__getitem__
list_pruning_strategies = lambda: sorted(Registries.pruning_strategies)
register_pruning_strategy = Registries.pruning_strategies.register


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

  Env Problems:
%s
"""
  lists = tuple(
      display_list_by_prefix(entries, starting_spaces=4) for entries in [  # pylint: disable=g-complex-comprehension
          list_models(),
          list_hparams(),
          list_ranged_hparams(),
          list_base_problems(),
          list_optimizers(),
          list_attacks(),
          list_attack_params(),
          list_pruning_params(),
          list_pruning_strategies(),
          list_env_problems(),
      ])
  return help_str % lists
