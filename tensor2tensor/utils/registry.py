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

import collections
import inspect
import re

# Dependency imports

import six

_MODELS = {}
_HPARAMS = {}
_RANGED_HPARAMS = {}


class Modalities(object):
  SYMBOL = "symbol"
  IMAGE = "image"
  AUDIO = "audio"
  CLASS_LABEL = "class_label"
  GENERIC = "generic"


_MODALITIES = {
    Modalities.SYMBOL: {},
    Modalities.IMAGE: {},
    Modalities.AUDIO: {},
    Modalities.CLASS_LABEL: {},
    Modalities.GENERIC: {},
}

# Camel case to snake case utils
_first_cap_re = re.compile("(.)([A-Z][a-z0-9]+)")
_all_cap_re = re.compile("([a-z])([A-Z])")


def _convert_camel_to_snake(name):
  s1 = _first_cap_re.sub(r"\1_\2", name)
  return _all_cap_re.sub(r"\1_\2", s1).lower()


def _reset():
  for ctr in [_MODELS, _HPARAMS, _RANGED_HPARAMS] + list(_MODALITIES.values()):
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


def _internal_get_modality(name, mod_collection, collection_str):
  if name is None:
    name = "default"
  if name not in mod_collection:
    raise ValueError("%s modality %s never registered." % (collection_str,
                                                           name))
  return mod_collection[name]


def symbol_modality(name=None):
  return _internal_get_modality(name, _MODALITIES[Modalities.SYMBOL],
                                Modalities.SYMBOL.capitalize())


def generic_modality(name=None):
  return _internal_get_modality(name, _MODALITIES[Modalities.GENERIC],
                                Modalities.GENERIC.capitalize())


def audio_modality(name=None):
  return _internal_get_modality(name, _MODALITIES[Modalities.AUDIO],
                                Modalities.AUDIO.capitalize())


def image_modality(name=None):
  return _internal_get_modality(name, _MODALITIES[Modalities.IMAGE],
                                Modalities.IMAGE.capitalize())


def class_label_modality(name=None):
  return _internal_get_modality(name, _MODALITIES[Modalities.CLASS_LABEL],
                                Modalities.CLASS_LABEL.capitalize())


def _internal_register_modality(name, mod_collection, collection_str):
  """Register a modality into mod_collection."""

  def decorator(mod_cls, registration_name=None):
    """Registers & returns mod_cls with registration_name or default name."""
    mod_name = registration_name or _default_name(mod_cls)
    if mod_name in mod_collection:
      raise ValueError("%s modality %s already registered." % (collection_str,
                                                               mod_name))
    mod_collection[mod_name] = mod_cls
    return mod_cls

  # Handle if decorator was used without parens
  if callable(name):
    mod_cls = name
    return decorator(mod_cls, registration_name=_default_name(mod_cls))

  return lambda mod_cls: decorator(mod_cls, name)


def register_symbol_modality(name=None):
  """Register a symbol modality. name defaults to class name snake-cased."""
  return _internal_register_modality(name, _MODALITIES[Modalities.SYMBOL],
                                     Modalities.SYMBOL.capitalize())


def register_generic_modality(name=None):
  """Register a generic modality. name defaults to class name snake-cased."""
  return _internal_register_modality(name, _MODALITIES[Modalities.GENERIC],
                                     Modalities.GENERIC.capitalize())


def register_audio_modality(name=None):
  """Register an audio modality. name defaults to class name snake-cased."""
  return _internal_register_modality(name, _MODALITIES[Modalities.AUDIO],
                                     Modalities.AUDIO.capitalize())


def register_image_modality(name=None):
  """Register an image modality. name defaults to class name snake-cased."""
  return _internal_register_modality(name, _MODALITIES[Modalities.IMAGE],
                                     Modalities.IMAGE.capitalize())


def register_class_label_modality(name=None):
  """Register an image modality. name defaults to class name snake-cased."""
  return _internal_register_modality(name, _MODALITIES[Modalities.CLASS_LABEL],
                                     Modalities.CLASS_LABEL.capitalize())


def list_modalities():
  all_modalities = []
  for modality_type, modalities in six.iteritems(_MODALITIES):
    all_modalities.extend([
        "%s:%s" % (mtype, modality)
        for mtype, modality in zip([modality_type] * len(modalities),
                                   modalities)
    ])
  return all_modalities


def parse_modality_name(name):
  name_parts = name.split(":")
  if len(name_parts) < 2:
    name_parts.append("default")
  modality_type, modality_name = name_parts
  return modality_type, modality_name


def create_modality(modality_spec, model_hparams):
  """Create modality.

  Args:
    modality_spec: tuple, ("modality_type:modality_name", vocab_size).
    model_hparams: HParams object.

  Returns:
    Modality instance.

  Raises:
    ValueError: if modality_type is not recognized. See Modalities class for
    accepted types.
  """
  retrieval_fns = {
      Modalities.SYMBOL: symbol_modality,
      Modalities.AUDIO: audio_modality,
      Modalities.IMAGE: image_modality,
      Modalities.CLASS_LABEL: class_label_modality,
      Modalities.GENERIC: generic_modality,
  }

  modality_full_name, vocab_size = modality_spec
  modality_type, modality_name = parse_modality_name(modality_full_name)
  if modality_type not in retrieval_fns:
    raise ValueError("Modality type %s not recognized. Options are: %s" %
                     (modality_type, list(_MODALITIES)))

  return retrieval_fns[modality_type](modality_name)(model_hparams, vocab_size)


def _hparams_help_string():
  hparams_names = list_hparams()
  prefixes = zip([name.split("_")[0] for name in hparams_names], hparams_names)
  names_by_prefix = collections.defaultdict(list)
  for (prefix, full_name) in prefixes:
    names_by_prefix[prefix].append(full_name)
  return "\n".join(
      sorted([
          "    * %s: %s" % (prefix, sorted(names))
          for prefix, names in six.iteritems(names_by_prefix)
      ]))


def help_string():
  """Generate help string with contents of registry."""
  help_str = """
Registry contents:
------------------

  Models: %s

  HParams (by model):
%s

  RangedHParams: %s

  Modalities: %s
  """
  m, rhp, mod = [
      sorted(entries)
      for entries in [list_models(),
                      list_ranged_hparams(),
                      list_modalities()]
  ]
  return help_str % (m, _hparams_help_string(), rhp, mod)
