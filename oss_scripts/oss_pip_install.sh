#!/bin/bash

set -v  # print commands as they're executed
set -e  # fail and exit on any command erroring

: "${TF_VERSION:?}"

# Make sure we have the latest version of numpy - avoid problems we were
# seeing with Python 3
pip install -q -U numpy

if [[ "$TF_VERSION" == "tf-nightly"  ]]
then
  pip install tf-nightly;
else
  pip install -q "tensorflow==$TF_VERSION"
fi

# First ensure that the base dependencies are sufficient for a full import
pip install -q -e .
t2t-trainer --registry_help 2>&1 >/dev/null
t2t-datagen 2>&1 | grep translate_ende 2>&1 >/dev/null && echo passed

# Then install the test dependencies
pip install -q -e .[tests,allen]
# Make sure to install the atari extras for gym
pip install "gym[atari]"
