# Constructing T2T Models.

This directory contains T2T models, their hyperparameters, and a number
of common layers and hyperparameter settings to help construct new models.
Common building blocks are in `common_layers.py` and `common_attention.py`.
Common hyperparameters are in `common_hparams.py`. Models are imported in
`__init__.py`.

## Adding a new model.

To add a model to the built-in set, create a new file (see, e.g.,
`neural_gpu.py`) and write your model class inheriting from `T2TModel` there and
decorate it with `registry.register_model`. Import it in `__init__.py`.

It is now available to use with the trainer binary (`t2t-trainer`) using the
`--model=model_name` flag.
