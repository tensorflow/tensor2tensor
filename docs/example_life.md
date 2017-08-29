# T2T: Life of an Example

[![PyPI
version](https://badge.fury.io/py/tensor2tensor.svg)](https://badge.fury.io/py/tensor2tensor)
[![GitHub
Issues](https://img.shields.io/github/issues/tensorflow/tensor2tensor.svg)](https://github.com/tensorflow/tensor2tensor/issues)
[![Contributions
welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/tensor2tensor/Lobby)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)

This document show how a training example passes through the T2T pipeline,
and how all its parts are connected to work together.

## The Life of an Example

A training example passes the following stages in T2T:
* raw input (text from command line or file)
* encoded input after [Problem.feature_encoder](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/problem.py#L173) function `encode` is usually a sparse tensor, e.g., a vector of `tf.int32`s
* batched input after [data input pipeline](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/data_reader.py#L242) where the inputs, after [Problem.preprocess_examples](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/problem.py#L188) are grouped by their length and made into batches.
* dense input after being processed by a [Modality](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/modality.py#L30) function `bottom`.
* dense output after [T2T.model_fn_body](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/t2t_model.py#L542)
* back to sparse output through [Modality](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/modality.py#L30) function `top`.
* if decoding, back through [Problem.feature_encoder](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/problem.py#L173) function `decode` to display on the screen.

We go into these phases step by step below.

## Feature Encoders

TODO: describe [Problem.feature_encoder](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/problem.py#L173) which is a dict of encoders that have `encode` and `decode` functions.

## Modalities

TODO: describe [Modality](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/modality.py#L30) which has `bottom` and `top` but also sharded versions and one for targets.
