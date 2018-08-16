# Distributed Training

The `t2t-trainer` supports both synchronous and asynchronous distributed
training.

Note that it is almost always more efficient to train on a single machine with
multiple GPUs/TPUs. Async training is less stable than sync training, and sync
training is much faster on 1 machine than on multiple. For these reasons, we
almost always train on single machines with multiple GPUs/TPUs.

T2T uses TensorFlow Estimators and so distributed training is configured with
the `TF_CONFIG` environment variable that is read by the
[RunConfig](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/estimator/run_config.py)
along with a set of flags that T2T uses to distribute the computation.

## Shared output directory

When using multiple machines, it is necessary that all nodes use the same
`--output_dir`, which means that it should be set to a Google Cloud Storage
bucket (`gs://...`) or a directory on a shared network filesystem.

## Utility to produce `TF_CONFIG` and flags

[`t2t-make-tf-configs`](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/bin/t2t-make-tf-configs)
generates the `TF_CONFIG` json strings and the necessary command-line flags for
the jobs.

Given a set of master and parameter server addresses, the script outputs, for
each job, a line with the `TF_CONFIG` environment variable and the command-line
flags necessary for distributed training. For each job, you should invoke the
`t2t-trainer` with the `TF_CONFIG` value and flags that are output.

## Eval jobs

Eval jobs should set the following flags and do not need the `TF_CONFIG`
environment variable to be set as the eval jobs run locally and do not
communicate to the other jobs (the eval jobs read the model checkpoints that the
trainer writes out):

- `--schedule=continuous_eval_on_train_data` or
  `--schedule=continuous_eval` (for dev data)
- `--worker_job='/job:localhost'`
- `--output_dir=$TRAIN_DIR`

**Note that evaluation does not work distributed.** That is, distributed jobs
should always use `--schedule=train`.

## Examples

### Sync training across multiple workers

In this scenario, you wish to do synchronous training across multiple workers.
Note that it is easier to simply use 1 worker with multiple GPUs and set
`--worker_gpu=8`, but there may be cases where you may want to have multiple
machines.

You will need 1 `ip:port` for the master and then 1 `ip:port` for each worker.

For this example we'll use 2 workers and these addresses:

```
# Master
10.0.0.1:5555

# Worker 1
10.0.0.2:5555

# Worker 2
10.0.0.3:5555
```

Next we generate the `TF_CONFIG` and command-line-flags for each job.

```
$ t2t-make-tf-configs --masters='10.0.0.1:5555' --ps='10.0.0.2:5555,10.0.0.3:5555'
Assuming SYNC distributed training with a single master and 2 workers
'{"cluster": {"master": ["10.0.0.1:5555"], "ps": ["10.0.0.2:5555", "10.0.0.3:5555"]}, "environment": "cloud", "task": {"index": 0, "type": "master"}}'      --master=grpc://10.0.0.1:5555 --ps_replicas=2 --worker_replicas=1 --worker_gpu=0 --worker_id=0 --ps_gpu=1 --sync --schedule=train --worker_job='/job:master'
'{"cluster": {"master": ["10.0.0.1:5555"], "ps": ["10.0.0.2:5555", "10.0.0.3:5555"]}, "environment": "cloud", "task": {"index": 0, "type": "ps"}}'  --schedule=run_std_server
'{"cluster": {"master": ["10.0.0.1:5555"], "ps": ["10.0.0.2:5555", "10.0.0.3:5555"]}, "environment": "cloud", "task": {"index": 1, "type": "ps"}}'  --schedule=run_std_server
```

The output here is 1 line per job. Each line contains the `TF_CONFIG` to set
for that job as well as the command-line flags to set for that job.

It is a bit confusing that the workers are being passed to the `--ps` flag, but
this is correct. When running in `--sync` mode, the `ps` are actually the
workers. You can see in the next example below that when `--sync=False`, i.e.
async mode, that the `ps` are in fact being used as parameter servers.

Here's how we would start each job on their respective machines (the
commands below assume that you're ssh'd into that job's machine):

**Master**:

```
$ export TF_CONFIG='{"cluster": {"master": ["10.0.0.1:5555"], "ps": ["10.0.0.2:5555", "10.0.0.3:5555"]}, "environment": "cloud", "task": {"index": 0, "type": "master"}}'
$ t2t-trainer \
    --master=grpc://10.0.0.1:5555 \
    --ps_replicas=2 \
    --worker_replicas=1 \
    --worker_gpu=0 \
    --worker_id=0 \
    --ps_gpu=1 \
    --sync \
    --schedule=train \
    --worker_job='/job:master' \
    --model=transformer \
    --hparams_set=transformer_base \
    --problem=translate_ende_wmt32k
```

**Worker 1**:

```
$ export TF_CONFIG='{"cluster": {"master": ["10.0.0.1:5555"], "ps": ["10.0.0.2:5555", "10.0.0.3:5555"]}, "environment": "cloud", "task": {"index": 0, "type": "ps"}}'
$ t2t-trainer --schedule=run_std_server
```

**Worker 2**:

```
$ export TF_CONFIG='{"cluster": {"master": ["10.0.0.1:5555"], "ps": ["10.0.0.2:5555", "10.0.0.3:5555"]}, "environment": "cloud", "task": {"index": 1, "type": "ps"}}'
$ t2t-trainer --schedule=run_std_server
```

Note that if you have more than 1 GPU on each worker machine, make sure to
modify the `--ps_gpu` passed to the master.

### Async training across multiple workers

In this scenario, you wish to do asynchronous training across multiple workers
with 1+ shared parameter servers.

Note that async training is usually less stable than sync training and for that
reason we almost always prefer sync training, but there may be cases where you
want to do async distributed training.

For this example we'll use 2 workers and 2 parameter servers:

```
# Worker 1
10.0.0.1:5555

# Worker 2
10.0.0.2:5555

# PS 1
10.0.0.3:5555

# PS 2
10.0.0.4:5555
```

Next we generate the `TF_CONFIG` and command-line-flags for each job.

```
$ t2t-make-tf-configs --masters='10.0.0.1:5555,10.0.0.2:5555' --ps='10.0.0.3:5555,10.0.0.4:5555'
Assuming ASYNC distributed training with 2 workers and 2 parameter servers
'{"task": {"index": 0, "type": "chief"}, "cluster": {"chief": ["10.0.0.1:5555"], "ps": ["10.0.0.3:5555", "10.0.0.4:5555"], "worker": ["10.0.0.2:5555"]}, "environment": "cloud"}' --master=grpc://10.0.0.1:5555 --ps_replicas=2 --worker_replicas=2 --worker_gpu=1 --worker_id=0 --ps_gpu=0  --schedule=train --worker_job='/job:chief'
'{"task": {"index": 0, "type": "worker"}, "cluster": {"chief": ["10.0.0.1:5555"], "ps": ["10.0.0.3:5555", "10.0.0.4:5555"], "worker": ["10.0.0.2:5555"]}, "environment": "cloud"}'        --master=grpc://10.0.0.2:5555 --ps_replicas=2 --worker_replicas=2 --worker_gpu=1 --worker_id=1 --ps_gpu=0 --schedule=train --worker_job='/job:worker'
'{"task": {"index": 0, "type": "ps"}, "cluster": {"chief": ["10.0.0.1:5555"], "ps": ["10.0.0.3:5555", "10.0.0.4:5555"], "worker": ["10.0.0.2:5555"]}, "environment": "cloud"}'    --schedule=run_std_server
'{"task": {"index": 1, "type": "ps"}, "cluster": {"chief": ["10.0.0.1:5555"], "ps": ["10.0.0.3:5555", "10.0.0.4:5555"], "worker": ["10.0.0.2:5555"]}, "environment": "cloud"}'    --schedule=run_std_server
```

Here's how we would start each job on their respective machines (the
commands below assume that you're ssh'd into that job's machine):

**Worker 1**:

```
$ export TF_CONFIG='{"task": {"index": 0, "type": "chief"}, "cluster": {"chief": ["10.0.0.1:5555"], "ps": ["10.0.0.3:5555", "10.0.0.4:5555"], "worker": ["10.0.0.2:5555"]}, "environment": "cloud"}'
$ t2t-trainer \
    --master=grpc://10.0.0.1:5555 \
    --ps_replicas=2 \
    --worker_replicas=2 \
    --worker_gpu=1 \
    --worker_id=0 \
    --ps_gpu=0 \
    --schedule=train \
    --worker_job='/job:chief' \
    --model=transformer \
    --hparams_set=transformer_base \
    --problem=translate_ende_wmt32k
```

**Worker 2**:

```
$ export TF_CONFIG='{"task": {"index": 0, "type": "worker"}, "cluster": {"chief": ["10.0.0.1:5555"], "ps": ["10.0.0.3:5555", "10.0.0.4:5555"], "worker": ["10.0.0.2:5555"]}, "environment": "cloud"}'
$ t2t-trainer \
    --master=grpc://10.0.0.2:5555 \
    --ps_replicas=2 \
    --worker_replicas=2 \
    --worker_gpu=1 \
    --worker_id=1 \
    --ps_gpu=0 \
    --schedule=train \
    --worker_job='/job:worker' \
    --model=transformer \
    --hparams_set=transformer_base \
    --problem=translate_ende_wmt32k
```

**PS 1**:

```
$ export TF_CONFIG='{"task": {"index": 0, "type": "ps"}, "cluster": {"chief": ["10.0.0.1:5555"], "ps": ["10.0.0.3:5555", "10.0.0.4:5555"], "worker": ["10.0.0.2:5555"]}, "environment": "cloud"}'
$ t2t-trainer --schedule=run_std_server
```

**PS 2**:

```
$ export TF_CONFIG='{"task": {"index": 1, "type": "ps"}, "cluster": {"chief": ["10.0.0.1:5555"], "ps": ["10.0.0.3:5555", "10.0.0.4:5555"], "worker": ["10.0.0.2:5555"]}, "environment": "cloud"}'
$ t2t-trainer --schedule=run_std_server
```

Increase `--worker_gpu` on each of the workers if you have multiple GPUs. If the
parameter servers are also using GPUs, set `--ps_gpu` to the number of GPUs on
the parameter servers.
