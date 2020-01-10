#!/bin/bash

set -v  # print commands as they're executed
set -e  # fail and exit on any command erroring

: "${TF_VERSION:?}"

# Make sure we have the latest pip and setuptools installed.
pip install -q -U pip
pip install -q -U setuptools

# Make sure we have the latest version of numpy - avoid problems we were
# seeing with Python 3
pip install -q -U numpy
pip install -q "tensorflow==$TF_VERSION"

# Just print the version again to make sure.
python -c 'import tensorflow as tf; print(tf.__version__)'

# First ensure that the base dependencies are sufficient for a full import
pip install -q -e .
t2t-trainer --registry_help 2>&1 >/dev/null
t2t-datagen 2>&1 | grep translate_ende 2>&1 >/dev/null && echo passed

# Then install the test dependencies
pip install -q -e .[tests,allen]
# Make sure to install the atari extras for gym
pip install "gym[atari]"
