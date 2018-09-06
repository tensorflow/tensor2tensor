#!/bin/bash

set -v  # print commands as they're executed
set -e  # fail and exit on any command erroring

: "${TF_VERSION:?}"
: "${TF_LATEST:?}"
: "${T2T_DATA_DIR:?}"
: "${T2T_TRAIN_DIR:?}"
: "${T2T_PROBLEM:?}"

# Test --t2t_usr_dir
t2t-trainer --registry_help --t2t_usr_dir=./tensor2tensor/test_data/example_usr_dir 2>&1 | grep my_very_own_hparams && echo passed

# Run data generation, training, and decoding on a dummy problem
t2t-datagen --problem=$T2T_PROBLEM --data_dir=$T2T_DATA_DIR
t2t-trainer --problem=$T2T_PROBLEM --data_dir=$T2T_DATA_DIR --model=transformer --hparams_set=transformer_tiny --train_steps=5 --eval_steps=5 --output_dir=$T2T_TRAIN_DIR
t2t-decoder --problem=$T2T_PROBLEM --data_dir=$T2T_DATA_DIR --model=transformer --hparams_set=transformer_tiny --output_dir=$T2T_TRAIN_DIR --decode_hparams='num_samples=10'

# Test serving
if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]] && [[ "$TF_VERSION" == "$TF_LATEST"  ]]
then
  # Export for serving
  t2t-exporter \
      --problem=$T2T_PROBLEM \
      --data_dir=$T2T_DATA_DIR \
      --model=transformer \
      --hparams_set=transformer_tiny \
      --output_dir=$T2T_TRAIN_DIR

  # Run model server
  server_port=8500
  model_name=my_model
  docker run -d -p $server_port:$server_port \
      --mount type=bind,source=$T2T_TRAIN_DIR/export/Servo,target=/models/$model_name \
      -e MODEL_NAME=$model_name -t tensorflow/serving
  sleep 10

  # Query
  pip install tensorflow-serving-api
  t2t-query-server \
      --server=localhost:$server_port \
      --servable_name=$model_name \
      --problem=$T2T_PROBLEM \
      --data_dir=$T2T_DATA_DIR \
      --inputs_once='1 0 1 0 1 0'
fi
