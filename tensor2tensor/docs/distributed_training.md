# Distributed Training

The `t2t-trainer` supports both synchronous and asynchronous distributed
training.

T2T uses TensorFlow Estimators and so distributed training is configured with
the `TF_CONFIG` environment variable that is read by the
[RunConfig](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/estimators/run_config.py)
along with a set of flags.

## `TF_CONFIG`

Both workers and parameter servers must have the `TF_CONFIG` environment
variable set.

The `TF_CONFIG` environment variable is a json-encoded string with the addresses
of the workers and parameter servers (in the `'cluster'` key) and the
identification of the current task (in the `'task'` key).

For example:

```
cluster = {
    'ps': ['host1:2222', 'host2:2222'],
    'worker': ['host3:2222', 'host4:2222', 'host5:2222']
}
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': cluster,
    'task': {'type': 'worker', 'index': 1}
})
```

## Command-line flags

The following T2T command-line flags must also be set on the workers for
distributed training:

- `--master=grpc://$ADDRESS`
- `--worker_replicas=$NUM_WORKERS`
- `--worker_gpu=$NUM_GPUS_PER_WORKER`
- `--worker_id=$WORKER_ID`
- `--ps_replicas=$NUM_PS`
- `--ps_gpu=$NUM_GPUS_PER_PS`
- `--schedule=train`
- `--sync`, if you want synchronous training, i.e. for there to be a single
  master worker coordinating the work across "ps" jobs (yes, the naming is
  unfortunate). If not set, then each worker operates independently while
  variables are shared on the parameter servers.

Parameter servers only need `--schedule=run_std_server`.

## Utility to produce `TF_CONFIG` and flags

[`bin/make_tf_configs.py`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/bin/make_tf_configs.py))
generates the `TF_CONFIG` json strings and the above-mentioned command-line
flags for the workers and parameter servers.

Given a set of worker and parameter server addresses, the script outputs, for
each job, a line with the `TF_CONFIG` environment variable and the command-line
flags necessary for distributed training. For each job, you should invoke the
`t2t-trainer` with the `TF_CONFIG` value and flags that are output.

For example:

```
TF_CONFIG=$JOB_TF_CONFIG t2t-trainer $JOB_FLAGS --model=transformer ...
```

## Command-line flags for eval jobs

Eval jobs should set the following flags and do not need the `TF_CONFIG`
environment variable to be set as the eval jobs run locally and do not
communicate to the other jobs (the eval jobs read the model checkpoints that the
trainer writes out):

- `--schedule=continuous_eval_on_train_data` or
  `--schedule=continuous_eval` (for test data)
- `--worker_job='/job:localhost'`
- `--output_dir=$TRAIN_DIR`
