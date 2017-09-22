# T2T: Life of an Example

[![PyPI
version](https://badge.fury.io/py/tensor2tensor.svg)](https://badge.fury.io/py/tensor2tensor)
[![GitHub
Issues](https://img.shields.io/github/issues/tensorflow/tensor2tensor.svg)](https://github.com/tensorflow/tensor2tensor/issues)
[![Contributions
welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/tensor2tensor/Lobby)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)

This doc explains how a training example flows through T2T, from data generation
to training, evaluation, and decoding. It points out the various hooks available
in the `Problem` and `T2TModel` classes and gives an overview of the T2T code
(key functions, files, hyperparameters, etc.).

Some key files and their functions:

*   [`trainer_utils.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/trainer_utils.py):
    Constructs and runs all the main components of the system (the `Problem`,
    the `HParams`, the `Estimator`, the `Experiment`, the `input_fn`s and
    `model_fn`).
*   [`common_hparams.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/layers/common_hparams.py):
    `basic_params1` serves as the base for all model hyperparameters. Registered
    model hparams functions always start with this default set of
    hyperparameters.
*   [`problem.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/problem.py):
    Every dataset in T2T subclasses `Problem`.
*   [`t2t_model.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/t2t_model.py):
    Every model in T2T subclasses `T2TModel`.

## Data Generation

The `t2t-datagen` binary is the entrypoint for data generation. It simply looks
up the `Problem` specified by `--problem` and calls
`Problem.generate_data(data_dir, tmp_dir)`.

All `Problem`s are expected to generate 2 sharded `TFRecords` files - 1 for
training and 1 for evaluation - with `tensorflow.Example` protocol buffers. The
expected names of the files are given by `Problem.{training, dev}_filepaths`.
Typically, the features in the `Example` will be `"inputs"` and `"targets"`;
however, some tasks have a different on-disk representation that is converted to
`"inputs"` and `"targets"` online in the input pipeline (e.g. image features are
typically stored with features `"image/encoded"` and `"image/format"` and the
decoding happens in the input pipeline).

For tasks that require a vocabulary, this is also the point at which the
vocabulary is generated and all examples are encoded.

There are several utility functions in
[`generator_utils`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/generator_utils.py)
that are commonly used by `Problem`s to generate data. Several are highlighted
below:

*   `generate_dataset_and_shuffle`: given 2 generators, 1 for training and 1 for
    eval, yielding dictionaries of `<feature name, list< int or float or
    string >>`, will produce sharded and shuffled `TFRecords` files with
    `tensorflow.Example` protos.
*   `maybe_download`: downloads a file at a URL to the given directory and
    filename (see `maybe_download_from_drive` if the URL points to Google
    Drive).
*   `get_or_generate_vocab_inner`: given a target vocabulary size and a
    generator that yields lines or tokens from the dataset, will build a
    `SubwordTextEncoder` along with a backing vocabulary file that can be used
    to map input strings to lists of ids.
    [`SubwordTextEncoder`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/text_encoder.py)
    uses word pieces and its encoding is fully invertible.

## Data Input Pipeline

Once the data is produced on disk, training, evaluation, and inference (if
decoding from the dataset) consume it by way of T2T input pipeline. This section
will give an overview of that pipeline with specific attention to the various
hooks in the `Problem` class and the model's `HParams` object (typically
registered in the model's file and specified by the `--hparams_set` flag).

The entire input pipeline is implemented with the new `tf.data.Dataset` API
(previously `tf.contrib.data.Dataset`).

The key function in the codebase for the input pipeline is
[`data_reader.input_pipeline`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/data_reader.py).
The full input function is built in
[`input_fn_builder.build_input_fn`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/input_fn_builder.py)
(which calls `data_reader.input_pipeline`).

### Reading and decoding data

`Problem.dataset_filename` specifies the prefix of the files on disk (they will
be suffixed with `-train` or `-dev` as well as their sharding).

The features read from the files and their decoding is specified by
`Problem.example_reading_spec`, which returns 2 items:

1.  Dict mapping from on-disk feature name to on-disk types (`VarLenFeature` or
    `FixedLenFeature`.
2.  Dict mapping output feature name to decoder. This return value is optional
    and is only needed for tasks whose features may require additional decoding
    (e.g. images). You can find the available decoders in
    `tf.contrib.slim.tfexample_decoder`.

At this point in the input pipeline, the example is a `dict<feature name,
Tensor>`.

### Preprocessing

The read `Example` now runs through `Problem.preprocess_example`, which by
default runs
[`problem.preprocess_example_common`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/problem.py),
which may truncate the inputs/targets or prepend to targets, governed by some
hyperparameters.

### Batching

Examples are bucketed by sequence length and then batched out of those buckets.
This significantly improves performance over a naive batching scheme for
variable length sequences because each example in a batch must be padded to
match the example with the maximum length in the batch.

There are several hyperparameters that affect how examples are batched together:

*   `hp.batch_size`: this is the approximate total number of tokens in the batch
    (i.e. for a sequence problem, long sequences will have smaller actual batch
    size and short sequences will have a larger actual batch size in order to
    generally have an equal number of tokens in the batch).
*   `hp.max_length`: sequences with length longer than this will be dropped
    during training (and also during eval if `hp.eval_drop_long_sequences` is
    `True`). If not set, the maximum length of examples is set to
    `hp.batch_size`.
*   `hp.batch_size_multiplier`: multiplier for the maximum length
*   `hp.min_length_bucket`: example length for the smallest bucket (i.e. the
    smallest bucket will bucket examples up to this length).
*   `hp.length_bucket_step`: controls how spaced out the length buckets are.

## Building the Model

At this point, the input features typically have `"inputs"` and `"targets"`,
each of which is a batched 4-D Tensor (e.g. of shape `[batch_size,
sequence_length, 1, 1]` for text input or `[batch_size, height, width, 3]` for
image input).

A `T2TModel` is composed of transforms of the input features by `Modality`s,
then the body of the model, then transforms of the model output to predictions
by a `Modality`, and then a loss (during training).

The `Modality` types for the various input features and for the target are
specified in `Problem.hparams`. A `Modality` is a feature adapter that enables
models to be agnostic to input/output spaces. You can see the various
`Modality`s in
[`modalities.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/layers/modalities.py).

The sketch structure of a T2T model is as follows:

```python
features = {...}  # output from the input pipeline
input_modaly = ...  # specified in Problem.hparams
target_modality = ...  # specified in Problem.hparams

transformed_features = {}
transformed_features["inputs"] = input_modality.bottom(
    features["inputs"])
transformed_features["targets"] = target_modality.targets_bottom(
    features["targets"])  # for autoregressive models

body_outputs = model.model_fn_body(transformed_features)

predictions = target_modality.top(body_outputs, features["targets"])
loss = target_modality.loss(predictions, features["targets"])
```

Most `T2TModel`s only override `model_fn_body`.

## Training, Eval, Inference modes

Both the input function and model functions take a mode in the form of a
`tf.estimator.ModeKeys`, which allows the functions to behave differently in
different modes.

In training, the model function constructs an optimizer and minimizes the loss.

In evaluation, the model function constructs the evaluation metrics specified by
`Problem.eval_metrics`.

In inference, the model function outputs predictions.

## `Estimator` and `Experiment`

With the input function and model functions constructed, the actual training
loop and related services (checkpointing, summaries, continuous evaluation,
etc.) are all handled by `Estimator` and `Experiment` objects, constructed in
[`trainer_utils.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/trainer_utils.py).

## Decoding

*   [`decoding.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/decoding.py)

TODO(rsepassi): Explain decoding (interactive, from file, and from dataset) and
`Problem.feature_encoders`.
