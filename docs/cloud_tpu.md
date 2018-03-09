# Running on Cloud TPUs

Tensor2Tensor supports running on Google Cloud Platforms TPUs, chips
specialized for ML training. See the official tutorial for [running Transfomer
on Cloud TPUs](https://cloud.google.com/tpu/docs/tutorials/transformer) or
read on for more T2T models on TPUs.

## Models and hparams for TPU:

Transformer:
* `transformer` with `transformer_tpu` (or `transformer_packed_tpu`,
    `transformer_tiny_tpu`, `transformer_big_tpu`)
* `transformer_encoder` with `transformer_tpu` (and the above ones)

You can run the Transformer model on a number of problems,
from translation through language modeling to sentiment analysis.
See the official tutorial for [running Transfomer
on Cloud TPUs](https://cloud.google.com/tpu/docs/tutorials/transformer)
for some examples and try out your own problems.

Image Transformer:
* `imagetransformer` with `imagetransformer_base_tpu` (or
    `imagetransformer_tiny_tpu`)
* `img2img_transformer` with `img2img_transformer_base_tpu` (or
    `img2img_transformer_tiny_tpu`)

You can run the `ImageTransformer` model on problems like unconditional or
conditional Image generation and `Img2ImgTransformer` model on Super Resolution.
We run on datasets like CelebA, CIFAR and ImageNet but they should work with any
other image dataset.

Residual networks:
* `resnet` with `resnet_50` (or `resnet_18` or `resnet_34`)
* `revnet` with `revnet_104` (or `revnet_38_cifar`)
* `shake_shake` with `shakeshake_tpu` (or `shakeshake_small`)

We run residual networks on MNIST, CIFAR and ImageNet, but they should
work on any image classification data-set.

## Tutorial: Transformer En-De translation on TPU

**Note**: You'll need TensorFlow 1.5+.

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
  --data_dir=$DATA_DIR \
  --output_dir=$OUT_DIR \
  --cloud_tpu \
  --cloud_delete_on_done
```

The above command will train for 10 steps, then evaluate for 10 steps. You can
(and should) increase the number of total training steps with the
`--train_steps` flag. Evaluation will happen every `--local_eval_frequency`
steps, each time for `--eval_steps`. The `--cloud_delete_on_done` flag has the
trainer delete the VMs on completion.

Voila. Enjoy your new supercomputer.

Note that checkpoints are compatible between CPU, GPU, and TPU models so you can
switch between hardware at will.

## Additional flags

* `--cloud_vm_name`: The name of the VM to use or create. This can be reused
  across multiple concurrent runs.
* `--cloud_tpu_name`: The name of the TPU instance to use or create. If you want
  to launch multiple jobs on TPU, provide different names here for each one.
  Each TPU instance can only be training one model at a time.

## Other T2T models on TPU

To run other models on TPU, proceed exactly as in the tutorial above,
just with different model, problem and hparams_set (and directories).
For example, to train a shake-shake model on CIFAR you can run this command.
```
t2t-trainer \
  --model=shake_shake \
  --hparams_set=shakeshake_tpu \
  --problems=image_cifar10 \
  --train_steps=180000 \
  --eval_steps=9 \
  --local_eval_frequency=100 \
  --data_dir=$DATA_DIR \
  --output_dir=$OUT_DIR \
  --cloud_tpu \
  --cloud_delete_on_done
```
Note that `eval_steps` should not be too high so as not to run out
of evaluation data.
