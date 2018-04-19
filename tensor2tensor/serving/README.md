# Serving

Tensor2Tensor and the TensorFlow ecosystem make it easy to serve a model once
trained.

## 1. Export for Serving

First, export it for serving:

```
t2t-exporter \
  --model=transformer \
  --hparams_set=transformer_tiny \
  --problem=translate_ende_wmt8k \
  --data_dir=~/t2t/data \
  --output_dir=/tmp/t2t_train
```

You should have an export directory in `output_dir` now.

## 2. Launch a Server

Install the `tensorflow-model-server`
([instructions](https://www.tensorflow.org/serving/setup#installing_the_modelserver)).

Start the server pointing at the export:

```
tensorflow_model_server \
  --port=9000 \
  --model_name=my_model \
  --model_base_path=/tmp/t2t_train/export/Servo
```

## 3. Query the Server

Install some dependencies:

```
pip install tensorflow-serving-api
```

Query:

```
t2t-query-server \
  --server=localhost:9000 \
  --servable_name=my_model \
  --problem=translate_ende_wmt8k \
  --data_dir=~/t2t/data
```


## Serve Predictions with Cloud ML Engine

Alternatively, you can deploy a model on Cloud ML Engine to serve predictions.
To do so, export the model as in Step 1, then do the following:

[Install gcloud](https://cloud.google.com/sdk/downloads)

#### Copy exported model to Google Cloud Storage

```
ORIGIN=<your_gcs_path>
EXPORTS_PATH=/tmp/t2t_train/export/Servo
LATEST_EXPORT=${EXPORTS_PATH}/$(ls ${EXPORTS_PATH} | tail -1)
gsutil cp -r ${LATEST_EXPORT}/* $ORIGIN
```

#### Create a model

```
MODEL_NAME=t2t_test
gcloud ml-engine models create $MODEL_NAME
```

This step only needs to be performed once.

#### Create a model version

```
VERSION=v0
gcloud ml-engine versions create $VERSION \
  --model $MODEL_NAME \
  --runtime-version 1.6 \
  --origin $ORIGIN
```

**NOTE:** Due to overhead from VM warmup, prediction requests may timeout. To
mitigate this issue, provide a [YAML configuration
file](https://cloud.google.com/sdk/gcloud/reference/ml-engine/versions/create)
via the `--config flag`, with `minNodes > 0`. These nodes are always on, and
will be billed accordingly.

#### Query Cloud ML Engine

```
t2t-query-server \
  --cloud_mlengine_model_name $MODEL_NAME \
  --cloud_mlengine_model_version $VERSION \
  --problem translate_ende_wmt8k \
  --data_dir ~/t2t/data
```
