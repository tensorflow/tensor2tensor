# Running on Cloud ML Engine

Google Cloud Platform offers a managed training environment for TensorFlow
models called [Cloud ML Engine](https://cloud.google.com/ml-engine/) and
you can easily launch Tensor2Tensor on it, including for hyperparameter tuning.

# Launch

It's the same `t2t-trainer` you know and love with the addition of the
`--cloud_mlengine` flag, which by default will launch on a 1-GPU machine.

```
# Note that both the data dir and output dir have to be on GCS
DATA_DIR=gs://my-bucket/data
OUTPUT_DIR=gs://my-bucket/train
t2t-trainer \
  --problems=translate_ende_wmt32k \
  --model=transformer \
  --hparams_set=transformer_base \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --cloud_mlengine
```

By passing `--worker_gpu=4` or `--worker_gpu=8` it will automatically launch on
machines with 4 or 8 GPUs.

You can additionally pass the `--cloud_mlengine_master_type` to select another
kind of machine (see the [docs for
`masterType`](https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#traininginput)
for your options). If you provide this flag yourself, make sure you pass the
correct value for `--worker_gpu`.

**Note**: `t2t-trainer` only currently supports launching with single machines,
possibly with multiple GPUs. Multi-machine setups are not yet supported out of
the box with the `--cloud_mlengine` flag, though multi-machine should in
principle work just fine. Contributions/testers welcome.

## `--t2t_usr_dir`

Launching on Cloud ML Engine works with `--t2t_usr_dir` as well as long as the
directory is fully self-contained (i.e. the imports only refer to other modules
in the directory). If there are additional PyPI dependencies that you need, you
can include a `requirements.txt` file in the directory specified by
`t2t_usr_dir`.

# Hyperparameter Tuning

Hyperparameter tuning with `t2t-trainer` and Cloud ML Engine is also a breeze
with `--hparams_range` and the `--autotune_*` flags:

```
t2t-trainer \
  --problems=translate_ende_wmt32k \
  --model=transformer \
  --hparams_set=transformer_base \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --cloud_mlengine \
  --hparams_range=transformer_base_range \
  --autotune_objective='metrics-translate_ende_wmt32k/neg_log_perplexity' \
  --autotune_maximize \
  --autotune_max_trials=100 \
  --autotune_parallel_trials=3
```

The `--hparams_range` specifies the search space and should be registered with
`@register_ranged_hparams`. It defines a `RangedHParams` object that sets
search ranges and scales for various parameters. See `transformer_base_range`
in
[`transformer.py`](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py)
for an example.

The metric name passed as `--autotune_objective` should be exactly what you'd
see in TensorBoard. To minimize a metric, set `--autotune_maximize=False`.

You control how many total trials to run with `--autotune_max_trials` and the
number of jobs to launch in parallel with `--autotune_parallel_trials`.

Happy tuning!
