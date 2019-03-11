# Stax - Layer Extensions

# Convenience layers and combinators

SLAX implements repeat, residual, and multiplex combinators, parallel
input-tuple sub-selection with Take layer, and graph inputs shape-logging with
LogInputs layer for debugging.

# Name Binding

SLAX implements Share, Bind, Var, Vars, Lambda, and make_apply_fun.
These operators augment the point-free Stax API with name-binding operations.
This provides a concise way of pointifying Stax notation when needed for
complicated neural net models while retaining a very functional style overall.

### Bind

Layer name-binding. Caches the results of the layer on its first application
in the computation DAG so that it can be referred to elsewhere in a model
definition and used as though it were a pointer to the cached variable.

We use the name-bound layer inside the main model just like a normal stax layer:

```python
# bind a layer with Bind:
encoder = Bind(serial(Dense(10), Relu))

# elsewhere in stax definition:
model = serial(
    # ...
    encoder, # evaluated and cached here
    # ...
    encoder, # this always returns the same value
    #...
)

# after training, we can access its params:
encoder.params

# or its last activations:
encoder.value

# or we can re-evaluate it with its trained set of params:
eval_time_result = make_apply_fun(encoder)(inputs, **kwargs)
```

Also note the convenience functions __Var__ and __Vars__(_N_), which are just
bound Identity layers. This is convenient for capturing input values to be used
elsewhere in the model. These can be used with __parallel__ and the helper
__multiplex__ combinators to easily route data around inside a stax model.

### Share

Parameter name-binding, for shared parameters. Just like __Bind__, but __Share__
doesn't bind the cached _results_ of a layer, but only it's _parameters_. This
allows us to create a weight-sharing layer by name. This works transparently
with jax.grad and optimizers as they only ever see one set of real parameters
from the state tree in the traced computations, so there's no inefficiency
introduced.

```python
# bind a layer with Share:
shared_layer = Share(serial(Dense(10), Relu))

# elsewhere in stax definition:
tower_A = serial(..., shared_layer, ...)
tower_B = serial(..., shared_layer, ...)

# after training, we can access its params:
shared_layer.params

# or we can re-evaluate it with its trained set of params:
eval_time_result = make_apply_fun(shared_layer)(inputs, **kwargs)
```

### Lambda

A function wrapper to allow concise function definitions of model layers.
This uses __Bind__ behind the scenes to fill in the values of the named
arguments with an input layer that captures the tuple of inputs, finally it
wraps the output of the function with a special form of __Bind__ that overloads
the `__call__` operator to make it easy to couple this subgraph to inputs as if
it were a normal function call.

```python
# we wrap a normal python function (*args only no **kwargs supported, but they
# always can be fed in from an outer scope.) e.g.:

some_layer_outside = serial(...)
@Lambda
def fun(x, y):
    tmp = serial(x, serial(Dense(10), Relu)))
    return serial(parallel(tmp, y), FanInSum, some_layer_outside)

# Later we can simply call the function with staxlayer arguments, even within
# another Lambda wrapped function:
result = fun(input1, input2)

# or chain them:
result = fun(input3, fun(input1, input2))

# Lambda is doing the "spiritual equivalent" to following:
x, y = Var(), Var()
Bind(
  serial(
    parallel(x, y),
    fun(x, y)
  )
)
# But Lambda also takes care of some annoying technical issues with
# combinators behind the scenes to make this work as well using a
# special Bind that overloads __call__ to make the result act like a
# function.

# after training, we can access its params:
fun.params

# or its last activations:
fun.value

# or we can re-evaluate it with its trained set of params:
eval_time_result = make_apply_fun(fun)(inputs, **kwargs)

```
