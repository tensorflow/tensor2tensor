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
of deep learning models and datasets designed to make deep learning more
accessible and [accelerate ML
research](https://research.googleblog.com/2017/06/accelerating-deep-learning-research.html).
T2T is actively used and maintained by researchers and engineers within the
[Google Brain team](https://research.google.com/teams/brain/) and a community
of users. We're eager to collaborate with you too, so feel free to
[open an issue on GitHub](https://github.com/tensorflow/tensor2tensor/issues)
or send along a pull request (see [our contribution doc](CONTRIBUTING.md)).
You can chat with us on
[Gitter](https://gitter.im/tensor2tensor/Lobby) and join the
[T2T Google Group](https://groups.google.com/forum/#!forum/tensor2tensor).

### Quick Start

[This iPython notebook](https://goo.gl/wkHexj) explains T2T and runs in your
browser using a free VM from Google, no installation needed.
Alternatively, here is a one-command version that installs T2T, downloads MNIST,
trains a model and evaluates it:

```
pip install tensor2tensor && t2t-trainer \
  --generate_data \
  --data_dir=~/t2t_data \
  --output_dir=~/t2t_train/mnist \
  --problems=image_mnist \
  --model=shake_shake \
  --hparams_set=shake_shake_quick \
  --train_steps=1000 \
  --eval_steps=100
```

### Contents

* [Suggested Datasets and Models](#suggested-datasets-and-models)
  * [Image Classification](#image-classification)
  * [Language Modeling](#language-modeling)
  * [Sentiment Analysis](#sentiment-analysis)
  * [Speech Recognition](#speech-recognition)
  * [Summarization](#summarization)
  * [Translation](#translation)
* [Basics](#basics)
  * [Walkthrough](#walkthrough)
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
* [Papers](#papers)

## Suggested Datasets and Models

Below we list a number of tasks that can be solved with T2T when
you train the appropriate model on the appropriate problem.
We give the problem and model below and we suggest a setting of
hyperparameters that we know works well in our setup. We usually
run either on Cloud TPUs or on 8-GPU machines; you might need
to modify the hyperparameters if you run on a different setup.

### Image Classification

For image classification, we have a number of standard data-sets:
* ImageNet (a large data-set): `--problems=image_imagenet`, or one
   of the re-scaled versions (`image_imagenet224`, `image_imagenet64`,
   `image_imagenet32`)
* CIFAR-10: `--problems=image_cifar10` (or
    `--problems=image_cifar10_plain` to turn off data augmentation)
* CIFAR-100: `--problems=image_cifar100`
* MNIST: `--problems=image_mnist`

For ImageNet, we suggest to use the ResNet or Xception, i.e.,
use `--model=resnet --hparams_set=resnet_50` or
`--model=xception --hparams_set=xception_base`.
Resnet should get to above 76% top-1 accuracy on ImageNet.

For CIFAR and MNIST, we suggest to try the shake-shake model:
`--model=shake_shake --hparams_set=shakeshake_big`.
This setting trained for `--train_steps=700000` should yield
close to 97% accuracy on CIFAR-10.

### Language Modeling

For language modeling, we have these data-sets in T2T:
* PTB (a small data-set): `--problems=languagemodel_ptb10k` for
    word-level modeling and `--problems=languagemodel_ptb_characters`
    for character-level modeling.
* LM1B (a billion-word corpus): `--problems=languagemodel_lm1b32k` for
    subword-level modeling and `--problems=languagemodel_lm1b_characters`
    for character-level modeling.

We suggest to start with `--model=transformer` on this task and use
`--hparams_set=transformer_small` for PTB and
`--hparams_set=transformer_base` for LM1B.

### Sentiment Analysis

For the task of recognizing the sentiment of a sentence, use
* the IMDB data-set: `--problems=sentiment_imdb`

We suggest to use `--model=transformer_encoder` here and since it is
a small data-set, try `--hparams_set=transformer_tiny` and train for
few steps (e.g., `--train_steps=2000`).

### Speech Recognition

For speech-to-text, we have these data-sets in T2T:
* Librispeech (English speech to text): `--problems=librispeech` for
    the whole set and `--problems=librispeech_clean` for a smaller
    but nicely filtered part.

### Summarization

For summarizing longer text into shorter one we have these data-sets:
* CNN/DailyMail articles summarized into a few sentences:
  `--problems=summarize_cnn_dailymail32k`

We suggest to use `--model=transformer` and
`--hparams_set=transformer_prepend` for this task.
This yields good ROUGE scores.

### Translation

There are a number of translation data-sets in T2T:
* English-German: `--problems=translate_ende_wmt32k`
* English-French: `--problems=translate_enfr_wmt32k`
* English-Czech: `--problems=translate_encs_wmt32k`
* English-Chinese: `--problems=translate_enzh_wmt32k`

You can get translations in the other direction by appending `_rev` to
the problem name, e.g., for German-English use
`--problems=translate_ende_wmt32k_rev`.

For all translation problems, we suggest to try the Transformer model:
`--model=transformer`. At first it is best to try the base setting,
`--hparams_set=transformer_base`. When trained on 8 GPUs for 300K steps
this should reach a BLEU score of about 28 on the English-German data-set,
which is close to state-of-the art. If training on a single GPU, try the
`--hparams_set=transformer_base_single_gpu` setting. For very good results
or larger data-sets (e.g., for English-French), try the big model
with `--hparams_set=transformer_big`.

## Basics

### Walkthrough

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
echo -e 'Hallo Welt\nAuf Wiedersehen Welt' > ref-translation.de

BEAM_SIZE=4
ALPHA=0.6

t2t-decoder \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=translation.en

# See the translations
cat translation.en

# Evaluate the BLEU score
# Note: Report this BLEU score in papers, not the internal approx_bleu metric.
t2t-bleu --translation=translation.en --reference=ref-translation.de
```

### Installation


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

### Features

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
  [distributed training](https://tensorflow.github.io/tensor2tensor/distributed_training.html).
* Easily swap amongst datasets and models by command-line flag with the data
  generation script `t2t-datagen` and the training script `t2t-trainer`.
* Train on [Google Cloud ML](https://tensorflow.github.io/tensor2tensor/cloud_mlengine.html) and [Cloud TPUs](https://tensorflow.github.io/tensor2tensor/cloud_tpu.html).

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
([distributed training documentation](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/g3doc/distributed_training.md)).

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

## Papers

Tensor2Tensor was used to develop a number of state-of-the-art models
and deep learning methods. Here we list some papers that were based on T2T
from the start and benefited from its features and architecture in ways
described in the [Google Research Blog post introducing
T2T](https://research.googleblog.com/2017/06/accelerating-deep-learning-research.html).

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* [Depthwise Separable Convolutions for Neural Machine
   Translation](https://arxiv.org/abs/1706.03059)
* [One Model To Learn Them All](https://arxiv.org/abs/1706.05137)
* [Discrete Autoencoders for Sequence Models](https://arxiv.org/abs/1801.09797)
* [Generating Wikipedia by Summarizing Long
   Sequences](https://arxiv.org/abs/1801.10198)
* [Image Transformer](https://arxiv.org/abs/1802.05751)
* [Training Tips for the Transformer Model](http://ufallab.ms.mff.cuni.cz/~popel/training-tips-transformer.pdf)

*Note: This is not an official Google product.*
