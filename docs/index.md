# Tensor2Tensor Documentation

[![PyPI
version](https://badge.fury.io/py/tensor2tensor.svg)](https://badge.fury.io/py/tensor2tensor)
[![GitHub
Issues](https://img.shields.io/github/issues/tensorflow/tensor2tensor.svg)](https://github.com/tensorflow/tensor2tensor/issues)
[![Contributions
welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/tensor2tensor/Lobby)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)

[Tensor2Tensor](https://github.com/tensorflow/tensor2tensor), or
[T2T](https://github.com/tensorflow/tensor2tensor) for short, is a library
of deep learning models and datasets designed to make deep learning more
accessible and [accelerate ML
research](https://research.googleblog.com/2017/06/accelerating-deep-learning-research.html).


## Basics

* [Walkthrough](walkthrough.md): Install and run.
* [IPython notebook](https://goo.gl/wkHexj): Get a hands-on experience.
* [Overview](overview.md): How all parts of T2T code are connected.
* [New Problem](new_problem.md): Train T2T models on your data.
* [New Model](new_model.md): Create your own T2T model.

## Training in the cloud

* [Training on Google Cloud ML](cloud_mlengine.md)
* [Training on Google Cloud TPUs](cloud_tpu.md)
* [Distributed Training](distributed_training.md)

## Solving your task

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
