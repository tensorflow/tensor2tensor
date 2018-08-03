# Runs t2t unit tests.
# Intended strictly to run on circleci--will not work locally.
#
# TODO: converge w/ _local.sh unit test script.

#!/usr/bin/env bash
T2T=tensor2tensor
DT=diseaseTools
IMAGE=gcr.io/fathom-containers/t2t_test
GCS_KEY_NAME=GOOGLE_APPLICATION_CREDENTIALS
GCS_KEY_PATH=/usr/src/diseaseTools/gcloud/keys/google-auth.json

T2T_MOUNTED=/usr/src/t2t

docker pull $IMAGE

docker run -it \
       -v $HOME/$DT:/usr/src/diseaseTools \
       -v $HOME/$T2T:$T2T_MOUNTED \
       -w $T2T_MOUNTED \
       --env PYTHONPATH=$T2T_MOUNTED:/usr/src/diseaseTools \
       --env $GCS_KEY_NAME=$GCS_KEY_PATH \
       $IMAGE \
       python3 -m pytest -vv \
       --ignore=$T2T_MOUNTED/tensor2tensor/utils/registry_test.py \
       --ignore=$T2T_MOUNTED/tensor2tensor/utils/trainer_lib_test.py \
       --ignore=$T2T_MOUNTED/tensor2tensor/visualization/visualization_test.py \
       --ignore=$T2T_MOUNTED/tensor2tensor/problems_test.py \
       --ignore=$T2T_MOUNTED/tensor2tensor/bin/t2t_trainer_test.py \
       --ignore=$T2T_MOUNTED/tensor2tensor/data_generators/algorithmic_math_test.py \
       --ignore=$T2T_MOUNTED/tensor2tensor/models/research/r_transformer_test.py \
       --junitxml=$T2T_MOUNTED/test_results/pytest/unittests.xml \
       $T2T_MOUNTED/tensor2tensor
