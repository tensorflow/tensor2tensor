# Automatic Speech Recognition (ASR) with Transformer

## Data set

This tutorial uses the publicly available
[Librispeech](http://www.openslr.org/12/) ASR corpus.

## Generate the dataset

To generate the dataset use `t2t-datagen`. You need to create environment
variables for a data directory `DATA_DIR` where the data is stored and for a
temporary directory `TMP_DIR` where necessary data is downloaded.

As the audio import in `t2t-datagen` uses `sox` to generate normalized
waveforms, please install it as appropriate (e.g. `apt-get install sox`).

```
t2t-datagen --problem=librispeech --data_dir=$DATA_DIR --tmp_dir=$TMP_DIR
```

You can also use smaller versions of the dataset by replacing `librispeech` with
`librispeech_clean` or `librispeech_clean_small`

## Training on GPUs

To train a model on GPU set up`OUT_DIR` and run the trainer:

```
t2t-trainer \
  --model=transformer \
  --hparams_set=transformer_librispeech \
  --problems=librispeech \
  --train_steps=120000 \
  --eval_steps=3 \
  --local_eval_frequency=100 \
  --data_dir=$DATA_DIR \
  --output_dir=$OUT_DIR
```

This model should achieve approximately 22% accuracy per sequence after
approximately 80,000 steps.

## Training on Cloud TPUs

To train a model on TPU set up `OUT_DIR` and run the trainer:

```
t2t-trainer \
  --model=transformer \
  --hparams_set=transformer_librispeech_tpu \
  --problems=librispeech \
  --train_steps=120000 \
  --eval_steps=3 \
  --local_eval_frequency=100 \
  --data_dir=$DATA_DIR \
  --output_dir=$OUT_DIR \
  --cloud_tpu \
  --cloud_delete_on_done
```

For more information, see [Tensor2Tensor's
documentation](https://github.com/tensorflow/tensor2tensor/tree/master/docs/cloud_tpu.md)
for Tensor2Tensor on Cloud TPUs, or the [official Google Cloud Platform
documentation](https://cloud.google.com/tpu/docs/tutorials/transformer) for
Cloud TPUs.
