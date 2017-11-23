# About this Fork

This fork expoeses the [TF-NMT tutorial](https://github.com/tensorflow/nmt) to 
T2T. Note that we generally follow the training schemes of T2T, and only take
model definitions from the tutorial. The goal is more to provide strong RNN 
baselines in T2T rather than replicating the results in the tutorial exactly.


## Staying up-to-date

This section explains how to pull commits to the [TF-NMT Github repository](https://github.com/tensorflow/nmt) 
into this fork.


``` shell
git remote add tfnmt_remote https://github.com/tensorflow/nmt.git
git fetch tfnmt_remote
git checkout -b tfnmt_branch tfnmt_remote/master
git read-tree --prefix=tensor2tensor/models/tfnmt -u tfnmt_branch
```

Pull upstream changes:

``` shell
git pull -s subtree tfnmt_remote master
```

