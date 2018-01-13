# T2T: Train on Your Own Data

[![PyPI
version](https://badge.fury.io/py/tensor2tensor.svg)](https://badge.fury.io/py/tensor2tensor)
[![GitHub
Issues](https://img.shields.io/github/issues/tensorflow/tensor2tensor.svg)](https://github.com/tensorflow/tensor2tensor/issues)
[![Contributions
welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/tensor2tensor/Lobby)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)

Let's add a new dataset together and train the transformer model. We'll be learning to define English words by training the transformer to "translate" between English words and their definitions on a character level.

# About the Problem

For each problem we want to tackle we create a new problem class and register it. Let's call our problem `Word2def`.

Since many text2text problems share similar methods, there's already a class
called `Text2TextProblem` that extends the base problem class, `Problem`
(both found in
[`problem.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/problem.py)).

For our problem, we can go ahead and create the file `word2def.py` in the
[`data_generators`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/)
folder and add our new problem, `Word2def`, which extends
[`Text2TextProblem`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/problem.py).
Let's also register it while we're at it so we can specify the problem through
flags.

```python
@registry.register_problem
class Word2def(problem.Text2TextProblem):
  """Problem spec for English word to dictionary definition."""
  @property
  def is_character_level(self):
    ...
```

We need to implement the following methods from
[`Text2TextProblem`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/problem.py).
in our new class:
* is_character_level
* targeted_vocab_size
* generator
* input_space_id
* target_space_id
* num_shards
* vocab_name
* use_subword_tokenizer

Let's tackle them one by one:

**input_space_id, target_space_id, is_character_level, targeted_vocab_size, use_subword_tokenizer**:

SpaceIDs tell Tensor2Tensor what sort of space the input and target tensors are
in. These are things like, EN_CHR (English character), EN_TOK (English token),
AUDIO_WAV (audio waveform), IMAGE, DNA (genetic bases). The complete list can be
found at
[`data_generators/problem.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/problem.py).
in the class `SpaceID`.

Since we're generating definitions and feeding in words at the character level, we set `is_character_level` to true, and use the same SpaceID, EN_CHR, for both input and target. Additionally, since we aren't using tokens, we don't need to give a `targeted_vocab_size` or define `use_subword_tokenizer`.

**vocab_name**:

`vocab_name` will be used to name your vocabulary files. We can call ours `'vocab.word2def.en'`

**num_shards**:

The number of shards to break data files into.

```python
@registry.register_problem()
class Word2def(problem.Text2TextProblem):
  """Problem spec for English word to dictionary definition."""

  @property
  def is_character_level(self):
    return True

  @property
  def vocab_name(self):
    return "vocab.word2def.en"

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def num_shards(self):
    return 100

  @property
  def use_subword_tokenizer(self):
    return False
```

**generator**:

We're almost done. `generator` generates the training and evaluation data and
stores them in files like "word2def_train.lang1" in your DATA_DIR. Thankfully
several commonly used methods like `character_generator`, and `token_generator`
are already written in the file
[`translate.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/translate.py).
We will import `character_generator` and
[`text_encoder`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/text_encoder.py)
to write:

```python
  def generator(self, data_dir, tmp_dir, train):
    character_vocab = text_encoder.ByteTextEncoder()
    datasets = _WORD2DEF_TRAIN_DATASETS if train else _WORD2DEF_TEST_DATASETS
    return character_generator(datasets[0], datasets[1], character_vocab, EOS)
```

Now our `word2def.py` file looks like the below:

```python
@registry.register_problem()
class Word2def(problem.Text2TextProblem):
  """Problem spec for English word to dictionary definition."""
  @property
  def is_character_level(self):
    return True

  @property
  def vocab_name(self):
    return "vocab.word2def.en"

  def generator(self, data_dir, tmp_dir, train):
    character_vocab = text_encoder.ByteTextEncoder()
    datasets = _WORD2DEF_TRAIN_DATASETS if train else _WORD2DEF_TEST_DATASETS
    return character_generator(datasets[0], datasets[1], character_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def num_shards(self):
    return 100

  @property
  def use_subword_tokenizer(self):
    return False
```

## Data:
Now we need to tell Tensor2Tensor where our data is located.

I've gone ahead and split all words into a train and test set and saved them in files called `words.train.txt`, `words.test.txt`,
`definitions.train.txt`, and `definitions.test.txt` in a directory called `LOCATION_OF_DATA/`. Let's tell T2T where these files are:

```python
# English Word2def datasets
_WORD2DEF_TRAIN_DATASETS = [
    LOCATION_OF_DATA + 'words_train.txt',
    LOCATION_OF_DATA + 'definitions_train.txt'
]

_WORD2DEF_TEST_DATASETS = [
    LOCATION_OF_DATA + 'words_test.txt',
    LOCATION_OF_DATA + 'definitions_test.txt'
]
```

## Putting it all together

Now our `word2def.py` file looks like:

```python
""" Problem definition for word to dictionary definition.
"""

import os

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators.translate import character_generator

from tensor2tensor.utils import registry

# English Word2def datasets
_WORD2DEF_TRAIN_DATASETS = [
    LOCATION_OF_DATA+'words_train.txt',
    LOCATION_OF_DATA+'definitions_train.txt'
]

_WORD2DEF_TEST_DATASETS = [
    LOCATION_OF_DATA+'words_test.txt',
    LOCATION_OF_DATA+'definitions_test.txt'
]

@registry.register_problem()
class Word2def(problem.Text2TextProblem):
  """Problem spec for English word to dictionary definition."""
  @property
  def is_character_level(self):
    return True

  @property
  def vocab_name(self):
    return "vocab.word2def.en"

  def generator(self, data_dir, tmp_dir, train):
    character_vocab = text_encoder.ByteTextEncoder()
    datasets = _WORD2DEF_TRAIN_DATASETS if train else _WORD2DEF_TEST_DATASETS
    return character_generator(datasets[0], datasets[1], character_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def num_shards(self):
    return 100

  @property
  def use_subword_tokenizer(self):
    return False

```

# Hyperparameters
All hyperparamters inherit from `_default_hparams()` in `problem.py.` If you would like to customize your hyperparameters, register a new hyperparameter set in `word2def.py` like the example provided in the walkthrough. For example:

```python
from tensor2tensor.models import transformer

@registry.register_hparams
def word2def_hparams():
    hparams = transformer.transformer_base_single_gpu()  # Or whatever you'd like to build off.
    hparams.batch_size = 1024
    return hparams
```

# Test the data generation

You can test data generation of your a problem in your own project with:

```bash
PROBLEM=word2def
DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
mkdir -p $DATA_DIR $TMP_DIR

t2t-datagen \
  --t2t_usr_dir=$PATH_TO_YOUR_PROBLEM_DIR \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM
```

Where:
* `PROBLEM` is the name of the class that was registered with
  `@registry.register_problem()`, but converted from `CamelCase` to
  `snake_case`.
* `PATH_TO_YOUR_PROBLEM_DIR` is a path to the directory of your python problem
  file.

If you plan to contribute to the tensor2tensor repository, you can install the
local cloned version in developer mode with `pip install -e .` from the
tensor2tensor directory. You can also add your new problem file to
[`all_problems.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/all_problems.py).

# Run the problem
Now that we've gotten our problem set up, let's train a model and generate
definitions.

To train, specify the problem name, the model, and hparams:
```bash
PROBLEM=word2def
MODEL=transformer
HPARAMS=word2def_hparams
```

The rest of the steps are as given in the [walkthrough](walkthrough.md).

What if we wanted to train a model to generate words given definitions? In T2T,
we can change the problem name to be `PROBLEM=word2def_rev`.

All done. Let us know what definitions your model generated.
