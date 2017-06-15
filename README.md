# T2T: Tensor2Tensor Transformers

[T2T](https://github.com/tensorflow/tensor2tensor) is a modular and extensible
library and binaries for supervised learning with TensorFlow and with a focus on
sequence tasks. Actively used and maintained by researchers and engineers within
Google Brain, T2T strives to maximize idea bandwidth and minimize execution
latency.

T2T is particularly well-suited to researchers working on sequence tasks. We're
eager to collaborate with you on extending T2T's powers, so please feel free to
open an issue on GitHub to kick off a discussion and send along pull requests,
See [our contribution doc](CONTRIBUTING.md) for details and our [open
issues](https://github.com/tensorflow/tensor2tensor/issues).

## T2T overview

```
pip install tensor2tensor

DATA_DIR=$HOME/data
PROBLEM=wmt_ende_tokens_32k
MODEL=transformer
HPARAMS=transformer_base
TRAIN_DIR=$HOME/train

# Generate data
t2t-datagen \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM

# Train
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \

# Decode
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_from_file=$DATA_DIR/decode_this.txt
```

T2T modularizes training into several components, each of which can be seen in
use in the above commands.

### Datasets

**Datasets** are all standardized on TFRecord files with `tensorflow.Example`
protocol buffers. All datasets are registered and generated with the
[data
generator](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/bin/t2t-datagen)
and many common sequence datasets are already available for generation and use.

### Problems and Modalities

**Problems** define training-time hyperparameters for the dataset and task,
mainly by setting input and output **modalities** (e.g. symbol, image, audio,
label) and vocabularies, if applicable. All problems are defined in
[`problem_hparams.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/problem_hparams.py).
**Modalities**, defined in
[`modality.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/modality.py),
abstract away the input and output data types so that **models** may deal with
modality-independent tensors.

### Models

**`T2TModel`s** define the core tensor-to-tensor transformation, independent of
input/output modality or task. Models take dense tensors in and produce dense
tensors that may then be transformed in a final step by a **modality** depending
on the task (e.g. fed through a final linear transform to produce logits for a
softmax over classes). All models are imported in
[`models.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/models/models.py),
inherit from `T2TModel` - defined in
[`t2t_model.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/t2t_model.py)
- and are registered with
[`@registry.register_model`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/registry.py).

### Hyperparameter Sets

**Hyperparameter sets** are defined and registered in code with
[`@registry.register_hparams`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/utils/registry.py)
and are encoded in
[`tf.contrib.training.HParams`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/training/python/training/hparam.py)
objects. The `HParams` are available to both the problem specification and the
model. A basic set of hyperparameters are defined in
[`common_hparams.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/models/common_hparams.py)
and hyperparameter set functions can compose other hyperparameter set functions.

### Trainer

The **trainer** binary is the main entrypoint for training, evaluation, and
inference. Users can easily switch between problems, models, and hyperparameter
sets by using the `--model`, `--problems`, and `--hparams_set` flags. Specific
hyperparameters can be overriden with the `--hparams` flag. `--schedule` and
related flags control local and distributed training/evaluation.

## Adding a dataset

See the data generators
[README](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/README.md).

---

*Note: This is not an official Google product.*
