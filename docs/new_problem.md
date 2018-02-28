# T2T: Train on Your Own Data

[![PyPI
version](https://badge.fury.io/py/tensor2tensor.svg)](https://badge.fury.io/py/tensor2tensor)
[![GitHub
Issues](https://img.shields.io/github/issues/tensorflow/tensor2tensor.svg)](https://github.com/tensorflow/tensor2tensor/issues)
[![Contributions
welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/tensor2tensor/Lobby)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)

Another good overview of this part together with training is given in
[The Cloud ML Poetry Blog
Post](https://cloud.google.com/blog/big-data/2018/02/cloud-poetry-training-and-hyperparameter-tuning-custom-text-models-on-cloud-ml-engine)

Let's add a new dataset together and train the
[Transformer](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/models/transformer.py)
model on it. We'll give the model a line of poetry, and it will learn to
generate the next line.

# Defining the `Problem`

For each problem we want to tackle we create a new subclass of `Problem` and
register it. Let's call our problem `PoetryLines`.

Since many text-to-text problems share similar methods, there's already a class
called
[`Text2TextProblem`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/text_problems.py)
that extends the base problem class
[`Problem`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/problem.py)
and makes it easy to add text-to-text problems.

In that same file, there are other base classes that make it easy to add text
classification tasks (`Text2ClassProblem`) and language modeling tasks
(`Text2SelfProblem`).

For our problem, let's create the file `poetry_lines.py` and add our new
problem, `PoetryLines`, which extends `Text2TextProblem` and register it so that
it is accessible by command-line flag.

Here's the Problem in full. We'll go step by step through it.

```python
import re

from gutenberg import acquire
from gutenberg import cleanup

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

@registry.register_problem
class PoetryLines(text_problems.Text2TextProblem):
  """Predict next line of poetry from the last line. From Gutenberg texts."""

  @property
  def approx_vocab_size(self):
    return 2**13  # ~8k

  @property
  def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return False

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    # 10% evaluation data
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 9,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del data_dir
    del tmp_dir
    del dataset_split


    books = [
        # bookid, skip N lines
        (19221, 223),
        (15553, 522),
    ]

    for (book_id, toskip) in books:
      text = cleanup.strip_headers(acquire.load_etext(book_id)).strip()
      lines = text.split("\n")[toskip:]
      prev_line = None
      ex_count = 0
      for line in lines:
        # Any line that is all upper case is a title or author name
        if not line or line.upper() == line:
          prev_line = None
          continue

        line = re.sub("[^a-z]+", " ", line.strip().lower())
        if prev_line and line:
          yield {
              "inputs": prev_line,
              "targets": line,
          }
          ex_count += 1
        prev_line = line
```

## Vocabulary specification

The text generated is encoded with a vocabulary for training. By default, it is
a `SubwordTextEncoder` that is built with an approximate vocab size specified by
the user. It's fully invertible (no out-of-vocab tokens) with a fixed-size vocab
which makes it ideal for text problems.

You can also choose to use a character-level encoder or a token encoder where
you provide the vocab file yourself. See `Text2TextProblem.vocab_type`.

Here we specify that we're going to have a vocabulary with approximately 8,000
subwords.

```python
  @property
  def approx_vocab_size(self):
    return 2**13  # ~8k
```

## Splitting data between Train and Eval

By setting `is_generate_per_split=False`, the `generate_samples` method will
only be called once and the data will automatically be split across training and
evaluation data for us. This is useful because for our dataset we don't have
pre-existing "training" and "evaluation" sets. If we did, we'd set
`is_generate_per_split=True` so that `generate_samples` was called once per data
split.

The `dataset_splits` method determines the fraction that goes to each split. The
training data will be generated into 9 files and the evaluation data into 1.
90% of the data will be for training. 10% of the data will be for evaluation.

```python
  @property
  def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return False

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    # 10% evaluation data
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 9,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]
```

## Generating samples

`generate_samples` is the bulk of the code where we actually produce
dictionaries of poetry line pairs ("inputs" and "targets").

Some problems might require downloading, which can be done into `tmp_dir`. Some
problems may use their own token vocabulary file, in which case it can be copied
into `data_dir` before yielding samples.

Here we iterate through the lines of a couple books of poetry and produce pairs
of lines for the model to train against.

```python
  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del data_dir
    del tmp_dir
    del dataset_split

    books = [
        # bookid, skip N lines
        (19221, 223),
        (15553, 522),
    ]

    for (book_id, toskip) in books:
      text = cleanup.strip_headers(acquire.load_etext(book_id)).strip()
      lines = text.split("\n")[toskip:]
      prev_line = None
      ex_count = 0
      for line in lines:
        # Any line that is all upper case is a title or author name
        if not line or line.upper() == line:
          prev_line = None
          continue

        line = re.sub("[^a-z]+", " ", line.strip().lower())
        if prev_line and line:
          yield {
              "inputs": prev_line,
              "targets": line,
          }
          ex_count += 1
        prev_line = line
```

That's all for the problem specification! We're ready to generate the data.

# Run data generation

You can run data generation of your a problem in your own project with
`t2t-datagen` and the `--t2t_usr_dir` flag, which should point to the directory
containing an `__init__.py` file that imports `word2def`, the file we just
wrote.

```bash
USR_DIR=...
PROBLEM=poetry_lines
DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
mkdir -p $DATA_DIR $TMP_DIR

t2t-datagen \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM
```

`PROBLEM` is the name of the class that was registered with
`@registry.register_problem`, but converted from `CamelCase` to `snake_case`.

`USR_DIR` should be a directory with the `poetry_lines.py` file as well as an
`__init__.py` file that imports it (`from . import poetry_lines`).

If you plan to contribute problems to the tensor2tensor repository, you can
clone the repository and install it in developer mode with `pip install -e .`.

# Train!

You can train exactly as you do in the [walkthrough](walkthrough.md) with flags
`--problems=poetry_lines` and `--t2t_usr_dir=$USR_DIR`.

All done. Let us know what amazing poetry your model writes!
