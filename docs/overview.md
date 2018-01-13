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
to training, evaluation, and decoding.

Some key files and their functions:

*   [`t2t_trainer.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/bin/t2t_trainer.py) and [`trainer_lib.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/trainer_lib.py):
    Main entrypoint for training and evaluation.  Constructs and runs all the
    main components of the system (the `Problem`, the `HParams`, the
    `Estimator`, the `Experiment`, the `input_fn`s and `model_fn`).
*   [`common_hparams.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/layers/common_hparams.py):
    `basic_params1` serves as the base for all model hyperparameters. Registered
    model hparams functions always start with this default set of
    hyperparameters.
*   [`problem.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/problem.py):
    Every dataset in T2T subclasses `Problem`. `Problem.input_fn` is the
    Estimator input function.
*   [`t2t_model.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/t2t_model.py):
    Every model in T2T subclasses `T2TModel`. `T2TModel.estimator_model_fn` is
    the Estimator model function.

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
decoding from the dataset) consume it by way of the T2T input pipeline, defined
by `Problem.input_fn`.

The entire input pipeline is implemented with the new `tf.data.Dataset` API.

The input function has 2 main parts: first, reading and processing individual
examples, which is done is `Problem.dataset`, and second, batching, which is
done in `Problem.input_fn` after the call to `Problem.dataset`.

`Problem` subclasses may override the entire `input_fn` or portions of it (e.g.
`example_reading_spec` to indicate the names, types, and shapes of features on
disk). Typically they only override portions.

### Batching

Problems that have fixed size features (e.g. image problems) can use
`hp.batch_size` to set the batch size.

Variable length Problems are bucketed by sequence length and then batched out of
those buckets.  This significantly improves performance over a naive batching
scheme for variable length sequences because each example in a batch must be
padded to match the example with the maximum length in the batch.

Controlling hparams:

* `hp.batch_size`: the approximate total number of tokens in
  the batch (i.e. long sequences will have smaller actual batch size and short
  sequences will have a larger actual batch size in order to generally have an
  equal number of tokens in the batch).
* `hp.max_length`: For variable length features, sequences with length longer
  than this will be dropped during training (and also during eval if
  `hp.eval_drop_long_sequences` is `True`). If not set, the maximum length of
  examples is set to `hp.batch_size`.
* `hp.batch_size_multiplier`: multiplier for the maximum length
* `hp.min_length_bucket`: example length for the smallest bucket (i.e. the
  smallest bucket will bucket examples up to this length).
* `hp.length_bucket_step`: controls how spaced out the length buckets are.

## Building the Model

At this point, the input features typically have `"inputs"` and `"targets"`,
each of which is a batched 4-D Tensor (e.g. of shape `[batch_size,
sequence_length, 1, 1]` for text input or `[batch_size, height, width, 3]` for
image input).

The Estimator model function is created by `T2TModel.estimator_model_fn`, which
may be overridden in its entirety by subclasses if desired. Typically,
subclasses only override `T2TModel.body`.

The model function constructs a `T2TModel`, calls it, and then calls
`T2TModel.{estimator_spec_train, estimator_spec_eval, estimator_spec_predict}`
depending on the mode.

A call of a `T2TModel` internally calls `bottom`, `body`, `top`, and `loss`, all
of which can be overridden by subclasses (typically only `body` is).

The default implementations of `bottom`, `top`, and `loss` depend on the
`Modality` specified for the input and target features (e.g.
`SymbolModality.bottom` embeds integer tokens and `SymbolModality.loss` is
`softmax_cross_entropy`).

## `Estimator` and `Experiment`

The actual training loop and related services (checkpointing, summaries,
continuous evaluation, etc.) are all handled by `Estimator` and `Experiment`
objects. `t2t_trainer.py` is the main entrypoint and uses `trainer_lib.py`
to construct the various components.

## Decoding

* [`t2t_decoder.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/bin/t2t-decoder)
* [`decoding.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/decoding.py)

## System Overview for Train/Eval

See `t2t_trainer.py` and `trainer_lib.py`.

* Create HParams
* Create `RunConfig`, including `Parallelism` object (i.e. `data_parallelism`)
* Create `Experiment`, including hooks
* Create `Estimator`
  * `T2TModel.estimator_model_fn`
    * `model(features)`
      * `model.model_fn`
        * `model.bottom`
        * `model.body`
        * `model.top`
        * `model.loss`
    * [TRAIN] `model.estimator_spec_train`
      * `train_op = model.optimize`
    * [EVAL] `model.estimator_spec_eval`
      * Create metrics
* Create input functions
  * `Problem.input_fn`
    * `Problem.dataset`
    * Batching
* Create hooks
* Run Experiment --schedule (e.g. `exp.continuous_train_and_eval()`)
  * `estimator.train`
    * `train_op = model_fn(input_fn(mode=TRAIN))`
    * Run train op
  * `estimator.evaluate`
    * `metrics = model_fn(input_fn(mode=EVAL))`
    * Accumulate metrics
