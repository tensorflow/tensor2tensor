# Running on Cloud TPUs

Tensor2Tensor supports running on Google Cloud Platforms TPUs, chips specialized
for ML training.

Models and hparams that are known to work on TPU:
* `transformer` with `transformer_tpu`
* `transformer_encoder` with `transformer_tpu`
* `transformer_decoder` with `transformer_tpu`
* `resnet50` with `resnet_base`
* `revnet104` with `revnet_base`

TPU access is currently limited, but access will expand soon, so get excited for
your future ML supercomputers in the cloud.

## Tutorial: Transformer En-De translation on TPU

You'll need TensorFlow 1.5+:
```
pip install 'tensorflow==1.5.0rc1'
```

Configure the `gcloud` CLI:
```
gcloud components update
gcloud auth application-default login
# Set your default zone to a TPU-enabled zone.
gcloud config set compute/zone us-central1-f
```

Generate data to GCS.
If you already have the data, use `gsutil cp` to copy to GCS.
```
GCS_BUCKET=gs://my-bucket
DATA_DIR=$GCS_BUCKET/t2t/data/
t2t-datagen --problem=translate_ende_wmt8k --data_dir=$DATA_DIR
```

Specify an output directory and launch TensorBoard to monitor training:
```
OUT_DIR=$GCS_BUCKET/t2t/training/transformer_v1
tensorboard --logdir=$OUT_DIR
```

Note that both the data and output directories must be Google Cloud Storage
buckets (i.e. start with `gs://`).

Launch! It's as simple as adding the `--cloud_tpu` flag.
```
t2t-trainer \
  --model=transformer \
  --hparams_set=transformer_tpu \
  --problems=translate_ende_wmt8k \
  --train_steps=10 \
  --eval_steps=10 \
  --local_eval_frequency=10 \
  --iterations_per_loop=10 \
  --data_dir=$DATA_DIR \
  --output_dir=$OUT_DIR \
  --cloud_tpu
```

The above command will train for 10 steps, then evaluate for 10 steps. You can
(and should) increase the number of total training steps with the
`--train_steps` flag. Evaluation will happen every `--local_eval_frequency`
steps, each time for `--eval_steps`. When you increase the number of training
steps, also increase `--iterations_per_loop`, which controls how frequently the
TPU machine returns control to the host machine (1000 seems like a fine number),
allowing it to log and save checkpoints.

Voila. Enjoy your new supercomputer.

Note that checkpoints are compatible between CPU, GPU, and TPU models so you can
switch between hardware at will.

## Additional flags

* `--cloud_vm_name`: The name of the VM to use or create. This can be reused
  across multiple concurrent runs.
* `--cloud_tpu_name`: The name of the TPU instance to use or create. If you want
  to launch multiple jobs on TPU, provide different names here for each one.
  Each TPU instance can only be training one model at a time.
* `--cloud_delete_on_done`: Delete the instances when done. By default, it's
  `False` to avoid setup and teardown costs when iterating.
