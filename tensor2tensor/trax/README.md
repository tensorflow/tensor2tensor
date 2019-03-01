# `trax`: Train Neural Nets with JAX

![train tracks](https://images.pexels.com/photos/461772/pexels-photo-461772.jpeg?dl&fit=crop&crop=entropy&w=640&h=426)

* Configuration is done with [`gin`](https://github.com/google/gin-config).
  `trainer.py` takes `--config_file` as well as `--config` for file overrides.
* Models are defined with [`stax`](https://github.com/google/jax/blob/master/jax/experimental/stax.py) in
  `models/`. They are made gin-configurable in `models/__init__.py`.
* Datasets are simple iterators over batches. Datasets from
  [`tensorflow/datasets`](https://github.com/tensorflow/datasets)
  and [`tensor2tensor`](https://github.com/tensorflow/tensor2tensor)
  are built-in and can be addressed by name.

Entrypoints:

* Script: `trainer.py`
* Main library entrypoint: `trax.train`

### Examples

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
