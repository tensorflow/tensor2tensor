# Serving

Tensor2Tensor and the TensorFlow ecosystem make it easy to serve a model once
trained.

**Note**: The following requires recent features in TensorFlow as so if you get
import errors or the like, try installing `tensorflow==1.5.0rc0`.

## 1. Export for Serving

First, export it for serving:

```
t2t-exporter \
  --model=transformer \
  --hparams_set=transformer_tiny \
  --problems=translate_ende_wmt8k \
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
