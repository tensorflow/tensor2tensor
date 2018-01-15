# Tensor2Tensor

[![PyPI
version](https://badge.fury.io/py/tensor2tensor.svg)](https://badge.fury.io/py/tensor2tensor)
[![GitHub
Issues](https://img.shields.io/github/issues/tensorflow/tensor2tensor.svg)](https://github.com/tensorflow/tensor2tensor/issues)
[![Contributions
welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/tensor2tensor/Lobby)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Travis](https://img.shields.io/travis/tensorflow/tensor2tensor.svg)](https://travis-ci.org/tensorflow/tensor2tensor)

[Tensor2Tensor](https://github.com/tensorflow/tensor2tensor), or
[T2T](https://github.com/tensorflow/tensor2tensor) for short, is a library
of deep learning models and datasets. It has binaries to train the models and
to download and prepare the data for you. T2T is modular and extensible and can
be used in [notebooks](https://goo.gl/wkHexj) for prototyping your own models
or running existing ones on your data. It is actively used and maintained by
researchers and engineers within
the [Google Brain team](https://research.google.com/teams/brain/) and was used
to develop state-of-the-art models for translation (see
[Attention Is All You Need](https://arxiv.org/abs/1706.03762)), summarization,
image generation and other tasks. You can read
more about T2T in the [Google Research Blog post introducing
it](https://research.googleblog.com/2017/06/accelerating-deep-learning-research.html).

We're eager to collaborate with you on extending T2T, so please feel
free to [open an issue on
GitHub](https://github.com/tensorflow/tensor2tensor/issues) or
send along a pull request to add your dataset or model.
See [our contribution
doc](CONTRIBUTING.md) for details and our [open
issues](https://github.com/tensorflow/tensor2tensor/issues).
You can chat with us and other users on
[Gitter](https://gitter.im/tensor2tensor/Lobby) and please join our
[Google Group](https://groups.google.com/forum/#!forum/tensor2tensor) to keep up
with T2T announcements.

### Quick Start

[This iPython notebook](https://goo.gl/wkHexj) explains T2T and runs in your
browser using a free VM from Google, no installation needed.

Alternatively, here is a one-command version that installs T2T, downloads data,
trains an English-German translation model, and evaluates it:

```
pip install tensor2tensor && t2t-trainer \
  --generate_data \
  --data_dir=~/t2t_data \
  --problems=translate_ende_wmt32k \
  --model=transformer \
  --hparams_set=transformer_base_single_gpu \
  --output_dir=~/t2t_train/base
```

You can decode from the model interactively:

```
t2t-decoder \
  --data_dir=~/t2t_data \
  --problems=translate_ende_wmt32k \
  --model=transformer \
  --hparams_set=transformer_base_single_gpu \
  --output_dir=~/t2t_train/base \
  --decode_interactive
```

See the [Walkthrough](#walkthrough) below for more details on each step
and [Suggested Models](#suggested-models) for well performing models
on common tasks.

### Contents

* [Walkthrough](#walkthrough)
* [Suggested Models](#suggested-models)
  * [Translation](#translation)
  * [Summarization](#summarization)
  * [Image Classification](#image-classification)
* [Installation](#installation)
* [Features](#features)
* [T2T Overview](#t2t-overview)
  * [Datasets](#datasets)
  * [Problems and Modalities](#problems-and-modalities)
  * [Models](#models)
  * [Hyperparameter Sets](#hyperparameter-sets)
  * [Trainer](#trainer)
* [Adding your own components](#adding-your-own-components)
* [Adding a dataset](#adding-a-dataset)

---

## Walkthrough

Here's a walkthrough training a good English-to-German translation
model using the Transformer model from [*Attention Is All You
Need*](https://arxiv.org/abs/1706.03762) on WMT data.

```
pip install tensor2tensor

# See what problems, models, and hyperparameter sets are available.
# You can easily swap between them (and add new ones).
t2t-trainer --registry_help

PROBLEM=translate_ende_wmt32k
MODEL=transformer
HPARAMS=transformer_base_single_gpu

DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR

# Decode

DECODE_FILE=$DATA_DIR/decode_this.txt
echo "Hello world" >> $DECODE_FILE
echo "Goodbye world" >> $DECODE_FILE

BEAM_SIZE=4
ALPHA=0.6

t2t-decoder \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FILE

cat $DECODE_FILE.$MODEL.$HPARAMS.beam$BEAM_SIZE.alpha$ALPHA.decodes
```

---

## Suggested Models

Here are some combinations of models, hparams and problems that we found
work well, so we suggest to use them if you're interested in that problem.

### Translation

For translation, esp. English-German and English-French, we suggest to use
the Transformer model in base or big configurations, i.e.
for `--problems=translate_ende_wmt32k` use `--model=transformer` and
`--hparams_set=transformer_base`. When trained on 8 GPUs for 300K steps
this should reach a BLEU score of about 28.

### Summarization

For summarization suggest to use the Transformer model in prepend mode, i.e.
for `--problems=summarize_cnn_dailymail32k` use `--model=transformer` and
`--hparams_set=transformer_prepend`.

### Image Classification

For image classification suggest to use the ResNet or Xception, i.e.
for `--problems=image_imagenet` use `--model=resnet50` and
`--hparams_set=resnet_base` or `--model=xception` and
`--hparams_set=xception_base`.


## Installation

```
# Assumes tensorflow or tensorflow-gpu installed
pip install tensor2tensor

# Installs with tensorflow-gpu requirement
pip install tensor2tensor[tensorflow_gpu]

# Installs with tensorflow (cpu) requirement
pip install tensor2tensor[tensorflow]
```

Binaries:

```
# Data generator
t2t-datagen

# Trainer
t2t-trainer --registry_help
```

Library usage:

```
python -c "from tensor2tensor.models.transformer import Transformer"
```

---

## Features

* Many state of the art and baseline models are built-in and new models can be
  added easily (open an issue or pull request!).
* Many datasets across modalities - text, audio, image - available for
  generation and use, and new ones can be added easily (open an issue or pull
  request for public datasets!).
* Models can be used with any dataset and input mode (or even multiple); all
  modality-specific processing (e.g. embedding lookups for text tokens) is done
  with `Modality` objects, which are specified per-feature in the dataset/task
  specification.
* Support for multi-GPU machines and synchronous (1 master, many workers) and
  asynchronous (independent workers synchronizing through a parameter server)
  [distributed training](https://github.com/tensorflow/tensor2tensor/tree/master/docs/distributed_training.md).
* Easily swap amongst datasets and models by command-line flag with the data
  generation script `t2t-datagen` and the training script `t2t-trainer`.

---

## T2T overview

### Datasets

**Datasets** are all standardized on `TFRecord` files with `tensorflow.Example`
protocol buffers. All datasets are registered and generated with the
[data
generator](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/bin/t2t-datagen)
and many common sequence datasets are already available for generation and use.

### Problems and Modalities

**Problems** define training-time hyperparameters for the dataset and task,
mainly by setting input and output **modalities** (e.g. symbol, image, audio,
label) and vocabularies, if applicable. All problems are defined either in
[`problem_hparams.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/problem_hparams.py)
or are registered with `@registry.register_problem` (run `t2t-datagen` to see
the list of all available problems).
**Modalities**, defined in
[`modality.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/modality.py),
abstract away the input and output data types so that **models** may deal with
modality-independent tensors.

### Models

**`T2TModel`s** define the core tensor-to-tensor transformation, independent of
input/output modality or task. Models take dense tensors in and produce dense
tensors that may then be transformed in a final step by a **modality** depending
on the task (e.g. fed through a final linear transform to produce logits for a
softmax over classes). All models are imported in the
[`models` subpackage](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/models/__init__.py),
inherit from `T2TModel` - defined in
[`t2t_model.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/t2t_model.py) -
and are registered with
[`@registry.register_model`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/registry.py).

### Hyperparameter Sets

**Hyperparameter sets** are defined and registered in code with
[`@registry.register_hparams`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/registry.py)
and are encoded in
[`tf.contrib.training.HParams`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/training/python/training/hparam.py)
objects. The `HParams` are available to both the problem specification and the
model. A basic set of hyperparameters are defined in
[`common_hparams.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/layers/common_hparams.py)
and hyperparameter set functions can compose other hyperparameter set functions.

### Trainer

The **trainer** binary is the main entrypoint for training, evaluation, and
inference. Users can easily switch between problems, models, and hyperparameter
sets by using the `--model`, `--problems`, and `--hparams_set` flags. Specific
hyperparameters can be overridden with the `--hparams` flag. `--schedule` and
related flags control local and distributed training/evaluation
([distributed training documentation](https://github.com/tensorflow/tensor2tensor/tree/master/docs/distributed_training.md)).

---

## Adding your own components

T2T's components are registered using a central registration mechanism that
enables easily adding new ones and easily swapping amongst them by command-line
flag. You can add your own components without editing the T2T codebase by
specifying the `--t2t_usr_dir` flag in `t2t-trainer`.

You can do so for models, hyperparameter sets, modalities, and problems. Please
do submit a pull request if your component might be useful to others.

See the [`example_usr_dir`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/test_data/example_usr_dir)
for an example user directory.

## Adding a dataset

To add a new dataset, subclass
[`Problem`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/problem.py)
and register it with `@registry.register_problem`. See
[`TranslateEndeWmt8k`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/translate_ende.py)
for an example.

Also see the [data generators
README](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/README.md).

---

*Note: This is not an official Google product.*
