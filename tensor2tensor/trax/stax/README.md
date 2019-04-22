# Trax Layers



## Base layer structure

All layers inherit form the Layer class and need to implement 3 functions:

```python
def call(self, params, inputs, **kwargs):
"""Call this layer using the given parameters on the given inputs."""

def output_shape(self, input_shape):
"""The shape of the output given the shape of the input."""

def new_parameters(self, input_shape, rng):
"""Create new parameters given the shape of the input."""
```

The base layer class wraps these functions and provides initialization
and call functions to be used as follows.

```python
input = np.zeros(10)
layer = MyLayer()
params = layer.initialize()
output = layer(params, input)
```

## Parameter sharing

Parameters are shared when the same layer object is used.

```python
standard_mlp = layers.Serial(layers.Dense(10), layers.Dense(10))
layer = Dense(10)
shared_parameters_mlp = layers.Serial(layer, layer)
```

## Core layers

* Dense
* Conv

## Layer composition

* Serial
* Parallel
