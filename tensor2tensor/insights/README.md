# Tensor2Tensor Insights

The Insights packages provides an interactive webservice for understanding the
inner workings of a Tensor2Tensor model.  It will provide a series of
visualizations extracted from a requested T2T model that informs model developers
and model users on how to improve or best utilize a model.

## Dependencies

Before using the Insights server, you must install [Bower](https://bower.io/)
which we use to manage our web component dependencies.  You can easily install
this with the [Node Package Manager](https://www.npmjs.com/).

## Setup Instructions

After training a model, such as according to the Quick Start guide, you can run
the `t2t-insights-server` binary and begin querying it.

First, prepare the bower dependencies by navigating into the
`tensor2tensor/insights/polymer` directory and running `bower install`:

```
pushd tensor2tensor/insights/polymer
bower install
popd
```

Then you can run the insights server.  Using the model trained in the Quick
Start guide, the command line would be:

```
t2t-insights-server \
  --t2t_data_dir=~/t2t_data \
  --t2t_model_dir=~/t2t_train/base \
  --problems=translate_ende_wmt32k \
  --model=transformer \
  --hparams_set=transformer_base_single_gpu \
  --static_path=`pwd`/tensor2tensor/insights/polymer
```

This will bring up a minimal [Flask](http://flask.pocoo.org/) REST service
served by a [GUnicorn](http://gunicorn.org/) HTTP Server.

## Features to be developed

This is a minimal web server.  We are in the process of adding additional
exciting features that give insight into a model's behavior:

  * Integrating a multi-head attention visualization.
  * Registering multiple models to compare their behavior.
  * Indexing training data to find examples related to a current query.
  * Tracking interesting query + translation pairs for deeper analysis.
