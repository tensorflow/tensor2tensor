# Running on Cloud TPUs

Tensor2Tensor supports running on Google Cloud Platforms TPUs, chips specialized
for ML training.

Models and hparams that are known to work on TPU:
* `transformer` with `transformer_tpu`
* `transformer_encoder` with `transformer_tpu`
* `transformer_decoder` with `transformer_tpu`
* `resnet50` with `resnet_base`
* `revnet104` with `revnet_base`

To run on TPUs, you need to be part of the alpha program; if you're not, these
commands won't work for you currently, but access will expand soon, so get
excited for your future ML supercomputers in the cloud.

## Tutorial: Transformer En-De translation on TPU

Update `gcloud`: `gcloud components update`

Set your default zone to a TPU-enabled zone. TPU machines are only available in
certain zones for now.
```
gcloud config set compute/zone us-central1-f
```

Launch a GCE instance; this will run the Python trainer.
```
gcloud compute instances create $USER-vm \
  --machine-type=n1-standard-8 \
  --image-family=tf-nightly \
  --image-project=ml-images \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

Launch the TPU instance; the Python program will connect to this to train on the
TPU device.
```
gcloud alpha compute tpus list
# Make an IP with structure 10.240.X.2 that’s unique in the list
TPU_IP=10.240.0.2
gcloud alpha compute tpus create \
  $USER-tpu \
  --range=${TPU_IP/%2/0}/29 \
  --version=nightly
```

SSH in with port forwarding for TensorBoard
```
gcloud compute ssh $USER-vm -- -L 6006:localhost:6006
```

Now that you're on the cloud instance, install T2T:
```
pip install tensor2tensor --user
# Add the python bin dir to your path
export PATH=$HOME/.local/bin:$PATH
```

Generate data to GCS
If you already have the data, use `gsutil cp` to copy to GCS.
```
GCS_BUCKET=gs://my-bucket
DATA_DIR=$GCS_BUCKET/t2t/data/
t2t-datagen --problem=translate_ende_wmt8k --data_dir=$DATA_DIR
```

Setup some vars used below. `TPU_IP` and `DATA_DIR` should be the same as what
was used above. Note that the `DATA_DIR` and `OUT_DIR` must be GCS buckets.
```
TPU_IP=10.240.0.2
DATA_DIR=$GCS_BUCKET/t2t/data/
OUT_DIR=$GCS_BUCKET/t2t/training/transformer_ende_1
TPU_MASTER=grpc://$TPU_IP:8470
```

Launch TensorBoard in the background so you can monitor training:
```
tensorboard --logdir=$OUT_DIR > /tmp/tensorboard_logs.txt 2>&1 &
```

Train and evaluate.
```
t2t-trainer \
  --model=transformer \
  --hparams_set=transformer_tpu \
  --problems=translate_ende_wmt8k \
  --train_steps=10 \
  --eval_steps=10 \
  --local_eval_frequency=10 \
  --iterations_per_loop=10 \
  --master=$TPU_MASTER \
  --use_tpu=True \
  --data_dir=$DATA_DIR \
  --output_dir=$OUT_DIR
```

The above command will train for 10 steps, then evaluate for 10 steps. You can
(and should) increase the number of total training steps with the
`--train_steps` flag. Evaluation will happen every `--local_eval_frequency`
steps, each time for `--eval_steps`. When you increase then number of training
steps, also increase `--iterations_per_loop`, which controls how frequently the
TPU machine returns control to the host machine (1000 seems like a fine number).

Back on your local machine, open your browser and navigate to `localhost:6006`
for TensorBoard.

Voila. Enjoy your new supercomputer.
