# Automatic Speech Recognition (ASR) with Transformer

Check out the [Automatic Speech Recognition notebook](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/asr_transformer.ipynb) to see how the resulting model transcribes your speech to text.

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
# Generate both the full dataset and the small clean version, which we use for
# evaluation.
t2t-datagen --problem=librispeech --data_dir=$DATA_DIR --tmp_dir=$TMP_DIR
t2t-datagen --problem=librispeech_clean --data_dir=$DATA_DIR --tmp_dir=$TMP_DIR
```

The problem `librispeech_train_full_test_clean` will train on the full dataset
but evaluate on the clean dataset.

You can also use `librispeech_clean_small` which is a small version of the
clean dataset.

## Training on Cloud TPUs

To train a model on TPU set up `OUT_DIR` and run the trainer with big batches
and truncated sequences:

```
t2t-trainer \
  --model=transformer \
  --hparams_set=transformer_librispeech_tpu \
  --hparams=max_length=125550,max_input_seq_length=1550,max_target_seq_length=350,batch_size=16 \
  --problem=librispeech_train_full_test_clean \
  --train_steps=210000 \
  --eval_steps=3 \
  --local_eval_frequency=100 \
  --data_dir=$DATA_DIR \
  --output_dir=$OUT_DIR \
  --cloud_tpu \
  --cloud_delete_on_done
```

After this step is compleated run the training again for more steps with smaller
batch size and full sequences:

```
t2t-trainer \
  --model=transformer \
  --hparams_set=transformer_librispeech_tpu \
  --hparams=max_length=295650,max_input_seq_length=3650,max_target_seq_length=650,batch_size=6 \
  --problem=librispeech_train_full_test_clean \
  --train_steps=230000 \
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

## Training on GPUs

To train a model on GPU set up`OUT_DIR` and run the trainer:

```
t2t-trainer \
  --model=transformer \
  --hparams_set=transformer_librispeech_tpu \
  --problem=librispeech \
  --train_steps=120000 \
  --eval_steps=3 \
  --local_eval_frequency=100 \
  --data_dir=$DATA_DIR \
  --output_dir=$OUT_DIR
```

This model should achieve approximately 22% accuracy per sequence after
approximately 80,000 steps.
