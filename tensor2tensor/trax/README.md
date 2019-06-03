## `trax`: Train Neural Nets with JAX

![train tracks](https://images.pexels.com/photos/461772/pexels-photo-461772.jpeg?dl&fit=crop&crop=entropy&w=640&h=426)

### `trax`: T2T Radically Simpler with JAX

*Why?* Because T2T has gotten too complex. We are simplifying the main code too,
but we wanted to try a more radical step. So you can write code as in pure
NumPy and debug directly. So you can easily pinpoint each line where things
happen and understand each function. But we also want it to run fast on
accelerators, and that's possible with [JAX](https://github.com/google/jax).

*Status:* preview; things work: models train, checkpoints are saved, TensorBoard
has summaries, you can decode. But we are changing a lot every day for now.
Please let us know what we should add, delete, keep, change. We plan to move
the best parts into core JAX.

*Entrypoints:*

* Script: `trainer.py`
* Main library entrypoint: `trax.train`

### Examples

#### Example Colab

See our example constructing language models from scratch in a GPU-backed colab notebook at
[Trax Demo](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/trax/notebooks/trax_demo_iclr2019.ipynb)

#### MLP on MNIST


```
python -m tensor2tensor.trax.trainer \
  --dataset=mnist \
  --model=MLP \
  --config="train.train_steps=1000"
```

#### Resnet50 on Imagenet


```
python -m tensor2tensor.trax.trainer \
  --config_file=$PWD/trax/configs/resnet50_imagenet_8gb.gin
```

#### TransformerDecoder on LM1B


```
python -m tensor2tensor.trax.trainer \
  --config_file=$PWD/trax/configs/transformer_lm1b_8gb.gin
```

### How `trax` differs from T2T

* Configuration is done with [`gin`](https://github.com/google/gin-config).
  `trainer.py` takes `--config_file` as well as `--config` for file overrides.
* Models are defined with [`stax`](https://github.com/google/jax/blob/master/jax/experimental/stax.py) in
  `models/`. They are made gin-configurable in `models/__init__.py`.
* Datasets are simple iterators over batches. Datasets from
  [`tensorflow/datasets`](https://github.com/tensorflow/datasets)
  and [`tensor2tensor`](https://github.com/tensorflow/tensor2tensor)
  are built-in and can be addressed by name.
