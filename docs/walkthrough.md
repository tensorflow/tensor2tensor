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
[![Run on FH](https://static.floydhub.com/button/button-small.svg)](https://floydhub.com/run)

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

[This iPython notebook](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)
explains T2T and runs in your browser using a free VM from Google,
no installation needed. Alternatively, here is a one-command version that
installs T2T, downloads MNIST, trains a model and evaluates it:

```
pip install tensor2tensor && t2t-trainer \
  --generate_data \
  --data_dir=~/t2t_data \
  --output_dir=~/t2t_train/mnist \
  --problem=image_mnist \
  --model=shake_shake \
  --hparams_set=shake_shake_quick \
  --train_steps=1000 \
  --eval_steps=100
```

### Contents

* [Suggested Datasets and Models](#suggested-datasets-and-models)
  * [Mathematical Language Understanding](#mathematical-language-understanding)
  * [Story, Question and Answer](#story-question-and-answer)
  * [Image Classification](#image-classification)
  * [Image Generation](#image-generation)
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
* [Run on FloydHub](#run-on-floydhub)

## Suggested Datasets and Models

Below we list a number of tasks that can be solved with T2T when
you train the appropriate model on the appropriate problem.
We give the problem and model below and we suggest a setting of
hyperparameters that we know works well in our setup. We usually
run either on Cloud TPUs or on 8-GPU machines; you might need
to modify the hyperparameters if you run on a different setup.

### Mathematical Language Understanding

For evaluating mathematical expressions at the character level involving addition, subtraction and multiplication of both positive and negative decimal numbers with variable digits assigned to symbolic variables, use

* the [MLU](https://art.wangperawong.com/mathematical_language_understanding_train.tar.gz) data-set:
 `--problem=algorithmic_math_two_variables`

You can try solving the problem with different transformer models and hyperparameters as described in the [paper](https://arxiv.org/abs/1812.02825):
* Standard transformer:
`--model=transformer`
`--hparams_set=transformer_tiny`
* Universal transformer:
`--model=universal_transformer`
`--hparams_set=universal_transformer_tiny`
* Adaptive universal transformer:
`--model=universal_transformer`
`--hparams_set=adaptive_universal_transformer_tiny`

### Story, Question and Answer

For answering questions based on a story, use

* the [bAbi](https://research.fb.com/downloads/babi/) data-set:
 `--problem=babi_qa_concat_task1_1k`

You can choose the bAbi task from the range [1,20] and the subset from 1k or
10k. To combine test data from all tasks into a single test set, use
`--problem=babi_qa_concat_all_tasks_10k`

### Image Classification

For image classification, we have a number of standard data-sets:

* ImageNet (a large data-set): `--problem=image_imagenet`, or one
   of the re-scaled versions (`image_imagenet224`, `image_imagenet64`,
   `image_imagenet32`)
* CIFAR-10: `--problem=image_cifar10` (or
    `--problem=image_cifar10_plain` to turn off data augmentation)
* CIFAR-100: `--problem=image_cifar100`
* MNIST: `--problem=image_mnist`

For ImageNet, we suggest to use the ResNet or Xception, i.e.,
use `--model=resnet --hparams_set=resnet_50` or
`--model=xception --hparams_set=xception_base`.
Resnet should get to above 76% top-1 accuracy on ImageNet.

For CIFAR and MNIST, we suggest to try the shake-shake model:
`--model=shake_shake --hparams_set=shakeshake_big`.
This setting trained for `--train_steps=700000` should yield
close to 97% accuracy on CIFAR-10.

### Image Generation

For (un)conditional image generation, we have a number of standard data-sets:

* CelebA: `--problem=img2img_celeba` for image-to-image translation, namely,
    superresolution from 8x8 to 32x32.
* CelebA-HQ: `--problem=image_celeba256_rev` for a downsampled 256x256.
* CIFAR-10: `--problem=image_cifar10_plain_gen_rev` for class-conditional
    32x32 generation.
* LSUN Bedrooms: `--problem=image_lsun_bedrooms_rev`
* MS-COCO: `--problem=image_text_ms_coco_rev` for text-to-image generation.
* Small ImageNet (a large data-set): `--problem=image_imagenet32_gen_rev` for
    32x32 or `--problem=image_imagenet64_gen_rev` for 64x64.

We suggest to use the Image Transformer, i.e., `--model=imagetransformer`, or
the Image Transformer Plus, i.e., `--model=imagetransformerpp` that uses
discretized mixture of logistics, or variational auto-encoder, i.e.,
`--model=transformer_ae`.
For CIFAR-10, using `--hparams_set=imagetransformer_cifar10_base` or
`--hparams_set=imagetransformer_cifar10_base_dmol` yields 2.90 bits per
dimension. For Imagenet-32, using
`--hparams_set=imagetransformer_imagenet32_base` yields 3.77 bits per dimension.

### Language Modeling

For language modeling, we have these data-sets in T2T:

* PTB (a small data-set): `--problem=languagemodel_ptb10k` for
    word-level modeling and `--problem=languagemodel_ptb_characters`
    for character-level modeling.
* LM1B (a billion-word corpus): `--problem=languagemodel_lm1b32k` for
    subword-level modeling and `--problem=languagemodel_lm1b_characters`
    for character-level modeling.

We suggest to start with `--model=transformer` on this task and use
`--hparams_set=transformer_small` for PTB and
`--hparams_set=transformer_base` for LM1B.

### Sentiment Analysis

For the task of recognizing the sentiment of a sentence, use

* the IMDB data-set: `--problem=sentiment_imdb`

We suggest to use `--model=transformer_encoder` here and since it is
a small data-set, try `--hparams_set=transformer_tiny` and train for
few steps (e.g., `--train_steps=2000`).

### Speech Recognition

For speech-to-text, we have these data-sets in T2T:

* Librispeech (US English): `--problem=librispeech` for
    the whole set and `--problem=librispeech_clean` for a smaller
    but nicely filtered part.

* Mozilla Common Voice (US English): `--problem=common_voice` for the whole set
    `--problem=common_voice_clean` for a quality-checked subset.

### Summarization

For summarizing longer text into shorter one we have these data-sets:

* CNN/DailyMail articles summarized into a few sentences:
  `--problem=summarize_cnn_dailymail32k`

We suggest to use `--model=transformer` and
`--hparams_set=transformer_prepend` for this task.
This yields good ROUGE scores.

### Translation

There are a number of translation data-sets in T2T:

* English-German: `--problem=translate_ende_wmt32k`
* English-French: `--problem=translate_enfr_wmt32k`
* English-Czech: `--problem=translate_encs_wmt32k`
* English-Chinese: `--problem=translate_enzh_wmt32k`
* English-Vietnamese: `--problem=translate_envi_iwslt32k`
* English-Spanish: `--problem=translate_enes_wmt32k`

You can get translations in the other direction by appending `_rev` to
the problem name, e.g., for German-English use
`--problem=translate_ende_wmt32k_rev`
(note that you still need to download the original data with t2t-datagen
`--problem=translate_ende_wmt32k`).

For all translation problems, we suggest to try the Transformer model:
`--model=transformer`. At first it is best to try the base setting,
`--hparams_set=transformer_base`. When trained on 8 GPUs for 300K steps
this should reach a BLEU score of about 28 on the English-German data-set,
which is close to state-of-the art. If training on a single GPU, try the
`--hparams_set=transformer_base_single_gpu` setting. For very good results
or larger data-sets (e.g., for English-French), try the big model
with `--hparams_set=transformer_big`.

See this [example](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/Transformer_translate.ipynb) to know how the translation works.

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
  --problem=$PROBLEM \
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
  --problem=$PROBLEM \
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
  with `bottom` and `top` transformations, which are specified per-feature in the
  model.
* Support for multi-GPU machines and synchronous (1 master, many workers) and
  asynchronous (independent workers synchronizing through a parameter server)
  [distributed training](https://tensorflow.github.io/tensor2tensor/distributed_training.html).
* Easily swap amongst datasets and models by command-line flag with the data
  generation script `t2t-datagen` and the training script `t2t-trainer`.
* Train on [Google Cloud ML](https://tensorflow.github.io/tensor2tensor/cloud_mlengine.html) and [Cloud TPUs](https://tensorflow.github.io/tensor2tensor/cloud_tpu.html).

## T2T overview

### Problems

**Problems** consist of features such as inputs and targets, and metadata such
as each feature's modality (e.g. symbol, image, audio) and vocabularies. Problem
features are given by a dataset, which is stored as a `TFRecord` file with
`tensorflow.Example` protocol buffers. All
problems are imported in
[`all_problems.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/all_problems.py)
or are registered with `@registry.register_problem`. Run
[`t2t-datagen`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/bin/t2t-datagen)
to see the list of available problems and download them.

### Models

**`T2TModel`s** define the core tensor-to-tensor computation. They apply a
default transformation to each input and output so that models may deal with
modality-independent tensors (e.g. embeddings at the input; and a linear
transform at the output to produce logits for a softmax over classes). All
models are imported in the
[`models` subpackage](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/models/__init__.py),
inherit from [`T2TModel`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/t2t_model.py),
and are registered with
[`@registry.register_model`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/registry.py).

### Hyperparameter Sets

**Hyperparameter sets** are encoded in
[`HParams`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/hparam.py)
objects, and are registered with
[`@registry.register_hparams`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/registry.py).
Every model and problem has a `HParams`. A basic set of hyperparameters are
defined in
[`common_hparams.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/layers/common_hparams.py)
and hyperparameter set functions can compose other hyperparameter set functions.

### Trainer

The **trainer** binary is the entrypoint for training, evaluation, and
inference. Users can easily switch between problems, models, and hyperparameter
sets by using the `--model`, `--problem`, and `--hparams_set` flags. Specific
hyperparameters can be overridden with the `--hparams` flag. `--schedule` and
related flags control local and distributed training/evaluation
([distributed training documentation](https://github.com/tensorflow/tensor2tensor/tree/master/docs/distributed_training.md)).

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
for an example. Also see the [data generators
README](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/README.md).

## Run on FloydHub

[![Run on FloydHub](https://static.floydhub.com/button/button.svg)](https://floydhub.com/run)

Click this button to open a [Workspace](https://blog.floydhub.com/workspaces/) on [FloydHub](https://www.floydhub.com/?utm_medium=readme&utm_source=tensor2tensor&utm_campaign=jul_2018). You can use the workspace to develop and test your code on a fully configured cloud GPU machine.

Tensor2Tensor comes preinstalled in the environment, you can simply open a [Terminal](https://docs.floydhub.com/guides/workspace/#using-terminal) and run your code.

```bash
# Test the quick-start on a Workspace's Terminal with this command
t2t-trainer \
  --generate_data \
  --data_dir=./t2t_data \
  --output_dir=./t2t_train/mnist \
  --problem=image_mnist \
  --model=shake_shake \
  --hparams_set=shake_shake_quick \
  --train_steps=1000 \
  --eval_steps=100
```

Note: Ensure compliance with the FloydHub [Terms of Service](https://www.floydhub.com/about/terms).

## Papers

When referencing Tensor2Tensor, please cite [this
paper](https://arxiv.org/abs/1803.07416).

```
@article{tensor2tensor,
  author    = {Ashish Vaswani and Samy Bengio and Eugene Brevdo and
    Francois Chollet and Aidan N. Gomez and Stephan Gouws and Llion Jones and
    \L{}ukasz Kaiser and Nal Kalchbrenner and Niki Parmar and Ryan Sepassi and
    Noam Shazeer and Jakob Uszkoreit},
  title     = {Tensor2Tensor for Neural Machine Translation},
  journal   = {CoRR},
  volume    = {abs/1803.07416},
  year      = {2018},
  url       = {http://arxiv.org/abs/1803.07416},
}
```

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
* [Training Tips for the Transformer Model](https://arxiv.org/abs/1804.00247)
* [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)
* [Fast Decoding in Sequence Models using Discrete Latent Variables](https://arxiv.org/abs/1803.03382)
* [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)
* [Universal Transformers](https://arxiv.org/abs/1807.03819)
* [Attending to Mathematical Language with Transformers](https://arxiv.org/abs/1812.02825)
* [The Evolved Transformer](https://arxiv.org/abs/1901.11117)
* [Model-Based Reinforcement Learning for Atari](https://arxiv.org/abs/1903.00374)
* [VideoFlow: A Flow-Based Generative Model for Video](https://arxiv.org/abs/1903.01434)

*NOTE: This is not an official Google product.*
