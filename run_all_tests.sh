T2T=tensor2tensor
DT=diseaseTools
IMAGE=gcr.io/fathom-containers/t2t_test
GCS_KEY_NAME=GOOGLE_APPLICATION_CREDENTIALS
GCS_KEY_PATH=/usr/src/diseaseTools/gcloud/keys/google-auth.json

docker pull $IMAGE

docker run -it \
       -v $HOME/$DT:/usr/src/diseaseTools \
       -v $HOME/$T2T:/usr/src/tensor2tensor \
       -w /usr/src/tensor2tensor \
       --env PYTHONPATH=/usr/src/tensor2tensor:/usr/src/diseaseTools \
       --env $GCS_KEY_NAME=$GCS_KEY_PATH \
       $IMAGE \
       python3 -m pytest -vv \
       --ignore=tensor2tensor/utils/registry_test.py \
       --ignore=tensor2tensor/utils/trainer_lib_test.py \
       --ignore=tensor2tensor/visualization/visualization_test.py \
       --ignore=tensor2tensor/problems_test.py \
       --ignore=tensor2tensor/bin/t2t_trainer_test.py \
       --ignore=tensor2tensor/data_generators/algorithmic_math_test.py \
       --ignore=tensor2tensor/models/research/r_transformer_test.py \
       --junitxml=/usr/src/t2t/test_results/pytest/unittests.xml \
       /usr/src/tensor2tensor/tensor2tensor
