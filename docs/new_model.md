# T2T: Create Your Own Model

[![PyPI
version](https://badge.fury.io/py/tensor2tensor.svg)](https://badge.fury.io/py/tensor2tensor)
[![GitHub
Issues](https://img.shields.io/github/issues/tensorflow/tensor2tensor.svg)](https://github.com/tensorflow/tensor2tensor/issues)
[![Contributions
welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](../CONTRIBUTING.md)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/tensor2tensor/Lobby)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)

Here we show how to create your own model in T2T.

## The T2TModel class - abstract base class for models

  `T2TModel` has three typical usages:

  1. Estimator: The method `make_estimator_model_fn` builds a `model_fn` for
     the tf.Estimator workflow of training, evaluation, and prediction.
     It performs the method `call`, which performs the core computation,
     followed by `estimator_spec_train`, `estimator_spec_eval`, or
     `estimator_spec_predict` depending on the tf.Estimator mode.
  2. Layer: The method `call` enables `T2TModel` to be used a callable by
     itself. It calls the following methods:

     * `bottom`, which transforms features according to `problem_hparams`' input
       and target `Modality`s;
     * `body`, which takes features and performs the core model computation to
        return output and any auxiliary loss terms;
     * `top`, which takes features and the body output, and transforms them
       according to `problem_hparams`' input and target `Modality`s to return
       the final logits;
     * `loss`, which takes the logits, forms any missing training loss, and sums
       all loss terms.
  3. Inference: The method `infer` enables `T2TModel` to make sequence
     predictions by itself.


## Creating your own model

1. Create class that extends T2TModel
    in this example it will be a copy of existing basic fully connected network:

```python
    from tensor2tensor.utils import t2t_model

    class MyFC(t2t_model.T2TModel):
        pass
```


2. Implement body method:

```python
    class MyFC(t2t_model.T2TModel):
      def body(self, features):
        hparams = self.hparams
        x = features["inputs"]
        shape = common_layers.shape_list(x)
        x = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])  # Flatten input as in T2T they are all 4D vectors
        for i in range(hparams.num_hidden_layers): # create layers
          x = tf.layers.dense(x, hparams.hidden_size, name="layer_%d" % i)
          x = tf.nn.dropout(x, keep_prob=1.0 - hparams.dropout)
          x = tf.nn.relu(x)
        return tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)  # 4D For T2T.
```


Method Signature:

  * Args:
      * features: dict of str to Tensor, where each Tensor has shape [batch_size,
     ..., hidden_size]. It typically contains keys `inputs` and `targets`.

  * Returns one of:
    * output: Tensor of pre-logit activations with shape [batch_size, ...,
           hidden_size].
    * losses: Either single loss as a scalar, a list, a Tensor (to be averaged),
           or a dictionary of losses. If losses is a dictionary with the key
           "training", losses["training"] is considered the final training
           loss and output is considered logits; self.top and self.loss will
           be skipped.

3. Register your model

```python
    from tensor2tensor.utils import registry

    @registry.register_model
    class MyFC(t2t_model.T2TModel):
       # ...
```


3. Use it with t2t tools as any other model

    Have in mind that names are translated from camel case to snake_case `MyFC` -> `my_fc`
    and that you need to point t2t to directory containing your model with `t2t_usr_dir` switch. 
    For example if you want to train model on gcloud with 1 GPU worker on IMDB sentiment task you can run your model
    by executing following command from your model class directory. 

```bash
    t2t-trainer \
      --model=my_fc \
      --t2t_usr_dir=.
      --cloud_mlengine --worker_gpu=1 \
      --generate_data \
      --data_dir='gs://data' \
      --output_dir='gs://out' \
      --problem=sentiment_imdb \
      --hparams_set=basic_fc_small \
      --train_steps=10000 \
      --eval_steps=10 \
```
