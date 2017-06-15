# Copyright 2017 Google Inc.
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
`trainer.py`, you can pass on the command-line: `--model=my_model`.

See all the models registered: `registry.list_models()`.

For hyperparameter sets:
  * Register: `registry.register_hparams`
  * List: `registry.list_hparams`
  * Retrieve by name: `registry.hparams`
  * Command-line flag in `trainer.py`: `--hparams_set=name`

For hyperparameter ranges:
  * Register: `registry.register_ranged_hparams`
  * List: `registry.list_ranged_hparams`
  * Retrieve by name: `registry.ranged_hparams`
  * Command-line flag in `trainer.py`: `--hparams_range=name`
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import re

# Dependency imports

from tensor2tensor.utils import t2t_model

import tensorflow as tf

_MODELS = {}
_HPARAMS = {}
_RANGED_HPARAMS = {}

# Camel case to snake case utils
_first_cap_re = re.compile("(.)([A-Z][a-z0-9]+)")
_all_cap_re = re.compile("([a-z])([A-Z])")


def _convert_camel_to_snake(name):
  s1 = _first_cap_re.sub(r"\1_\2", name)
  return _all_cap_re.sub(r"\1_\2", s1).lower()


def _reset():
  for ctr in [_MODELS, _HPARAMS, _RANGED_HPARAMS]:
    ctr.clear()


def _default_name(obj):
  return _convert_camel_to_snake(obj.__name__)


def register_model(name=None):
  """Register a model. name defaults to class name snake-cased."""

  def decorator(model_cls, registration_name=None):
    """Registers & returns model_cls with registration_name or default name."""
    model_name = registration_name or _default_name(model_cls)
    if model_name in _MODELS:
      raise ValueError("Model %s already registered." % model_name)
    if (not inspect.isclass(model_cls) or
        not issubclass(model_cls, t2t_model.T2TModel)):
      tf.logging.warning("Model %s is not an instance of T2TModel. "
                         "Object is expected to abide by its API.", model_name)
    _MODELS[model_name] = model_cls
    return model_cls

  # Handle if decorator was used without parens
  if callable(name):
    model_cls = name
    return decorator(model_cls, registration_name=_default_name(model_cls))

  return lambda model_cls: decorator(model_cls, name)


def model(name):
  if name not in _MODELS:
    raise ValueError("Model %s never registered." % name)
  return _MODELS[name]


def list_models():
  return list(_MODELS)


def register_hparams(name=None):
  """Register an HParams set. name defaults to function name snake-cased."""

  def decorator(hp_fn, registration_name=None):
    """Registers & returns hp_fn with registration_name or default name."""
    hp_name = registration_name or _default_name(hp_fn)
    if hp_name in _HPARAMS:
      raise ValueError("HParams set %s already registered." % hp_name)
    _HPARAMS[hp_name] = hp_fn
    return hp_fn

  # Handle if decorator was used without parens
  if callable(name):
    hp_fn = name
    return decorator(hp_fn, registration_name=_default_name(hp_fn))

  return lambda hp_fn: decorator(hp_fn, name)


def hparams(name):
  if name not in _HPARAMS:
    raise ValueError("HParams set %s never registered." % name)
  return _HPARAMS[name]


def list_hparams():
  return list(_HPARAMS)


def register_ranged_hparams(name=None):
  """Register a RangedHParams set. name defaults to fn name snake-cased."""

  def decorator(rhp_fn, registration_name=None):
    """Registers & returns hp_fn with registration_name or default name."""
    rhp_name = registration_name or _default_name(rhp_fn)
    if rhp_name in _RANGED_HPARAMS:
      raise ValueError("RangedHParams set %s already registered." % rhp_name)
    # Check that the fn takes a single argument
    args, varargs, keywords, _ = inspect.getargspec(rhp_fn)
    if len(args) != 1 or varargs is not None or keywords is not None:
      raise ValueError("RangedHParams set function must take a single "
                       "argument, the RangedHParams object.")

    _RANGED_HPARAMS[rhp_name] = rhp_fn
    return rhp_fn

  # Handle if decorator was used without parens
  if callable(name):
    rhp_fn = name
    return decorator(rhp_fn, registration_name=_default_name(rhp_fn))

  return lambda rhp_fn: decorator(rhp_fn, name)


def ranged_hparams(name):
  if name not in _RANGED_HPARAMS:
    raise ValueError("RangedHParams set %s never registered." % name)
  return _RANGED_HPARAMS[name]


def list_ranged_hparams():
  return list(_RANGED_HPARAMS)


def help_string():
  help_str = """Registry contents:

  Models: %s

  HParams: %s

  RangedHParams: %s
  """
  return help_str % (list_models(), list_hparams(), list_ranged_hparams())
